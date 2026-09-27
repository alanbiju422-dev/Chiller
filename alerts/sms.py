import os
import logging
from typing import Optional
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

logger = logging.getLogger(__name__)

def is_sms_configured() -> bool:
    return bool(
        os.getenv('TWILIO_ACCOUNT_SID')
        and (os.getenv('TWILIO_AUTH_TOKEN') or (os.getenv('TWILIO_API_KEY') and os.getenv('TWILIO_API_SECRET')))
        and os.getenv('TWILIO_FROM_NUMBER')
        and os.getenv('SMS_RECIPIENT')
    )

def get_missing_config() -> list:
    missing = []
    if not os.getenv('TWILIO_ACCOUNT_SID'):
        missing.append('TWILIO_ACCOUNT_SID')
    if not os.getenv('TWILIO_AUTH_TOKEN') and not (os.getenv('TWILIO_API_KEY') and os.getenv('TWILIO_API_SECRET')):
        missing.append('TWILIO_AUTH_TOKEN or TWILIO_API_KEY/TWILIO_API_SECRET')
    if not os.getenv('TWILIO_FROM_NUMBER'):
        missing.append('TWILIO_FROM_NUMBER')
    if not os.getenv('SMS_RECIPIENT'):
        missing.append('SMS_RECIPIENT')
    return missing

import threading

def _send_sms_task(message_body: str):
    if not is_sms_configured():
        logger.warning(f"SMS NOT CONFIGURED. Missing: {get_missing_config()}")
        return False
        
    try:
        if os.getenv('TWILIO_AUTH_TOKEN'):
            client = Client(os.getenv('TWILIO_ACCOUNT_SID'), os.getenv('TWILIO_AUTH_TOKEN'))
        else:
            client = Client(
                os.getenv('TWILIO_API_KEY'),
                os.getenv('TWILIO_API_SECRET'),
                os.getenv('TWILIO_ACCOUNT_SID')
            )
        
        message = client.messages.create(
            body=message_body,
            from_=os.getenv('TWILIO_FROM_NUMBER'),
            to=os.getenv('SMS_RECIPIENT')
        )
        logger.info(f"SMS_SENT: {message.sid}")
        return True
    except TwilioRestException as e:
        logger.error(f"SMS_FAILED: {e}")
        return False
    except Exception as e:
        logger.error(f"SMS_FAILED_UNEXPECTED: {e}")
        return False

def send_sms(message_body: str):
    thread = threading.Thread(target=_send_sms_task, args=(message_body,))
    thread.start()
    return True

def test_sms_configuration() -> bool:
    return send_sms("VOLTURA TEST MESSAGE: SMS Integration is correctly configured.")

def validate_sms_configuration():
    missing = get_missing_config()
    if missing:
        return False, f"Missing required environment variables: {', '.join(missing)}"
    return True, "SMS Configured"
