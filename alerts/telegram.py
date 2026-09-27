import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty
from urllib import parse, request

logger = logging.getLogger(__name__)

MAX_TELEGRAM_ATTEMPTS = 3


@dataclass
class TelegramAlert:
    event_id: str
    message_body: str


_alert_queue = Queue()
_result_queue = Queue()
_queued_event_ids = set()
_queue_lock = threading.Lock()
_worker_thread = None


def is_telegram_configured() -> bool:
    return bool(os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'))


def _send_telegram_request(message_body: str) -> tuple[bool, str | None]:
    if not is_telegram_configured():
        return False, 'not configured'

    try:
        token = os.environ['TELEGRAM_BOT_TOKEN']
        chat_id = os.environ['TELEGRAM_CHAT_ID']
        payload = parse.urlencode({
            'chat_id': chat_id,
            'text': message_body,
        }).encode('utf-8')
        endpoint = f'https://api.telegram.org/bot{token}/sendMessage'
        with request.urlopen(request.Request(endpoint, data=payload, method='POST'), timeout=10) as response:
            result = json.loads(response.read().decode('utf-8'))
        if not result.get('ok'):
            raise RuntimeError(result.get('description', 'Telegram API rejected the message'))
        return True, None
    except Exception as exc:
        return False, str(exc)


def _telegram_worker():
    while True:
        alert = _alert_queue.get()
        try:
            last_error = 'unknown error'
            for attempt in range(1, MAX_TELEGRAM_ATTEMPTS + 1):
                sent, error = _send_telegram_request(alert.message_body)
                if sent:
                    print(f"[TELEGRAM] SENT event_id={alert.event_id}")
                    logger.info('[TELEGRAM] SENT event_id=%s', alert.event_id)
                    _result_queue.put((alert.event_id, 'SENT', None, attempt))
                    break
                last_error = error or last_error
                if attempt < MAX_TELEGRAM_ATTEMPTS:
                    time.sleep(1)
            else:
                print(f"[TELEGRAM] FAILED event_id={alert.event_id} error={last_error}")
                logger.error('[TELEGRAM] FAILED event_id=%s error=%s', alert.event_id, last_error)
                _result_queue.put((alert.event_id, 'FAILED', last_error, MAX_TELEGRAM_ATTEMPTS))
        finally:
            with _queue_lock:
                _queued_event_ids.discard(alert.event_id)
            _alert_queue.task_done()


def _ensure_worker():
    global _worker_thread
    with _queue_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(target=_telegram_worker, name='telegram-alert-worker', daemon=True)
            _worker_thread.start()


def send_telegram(message_body: str, event_id: str | None = None) -> bool:
    if not event_id:
        logger.warning('TELEGRAM event_id is required for queued alerts')
        return False
    with _queue_lock:
        if event_id in _queued_event_ids:
            return True
        _queued_event_ids.add(event_id)
    _ensure_worker()
    _alert_queue.put(TelegramAlert(event_id, message_body))
    return True


def drain_telegram_results():
    results = []
    while True:
        try:
            results.append(_result_queue.get_nowait())
        except Empty:
            return results