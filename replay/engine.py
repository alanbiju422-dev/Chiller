import pandas as pd
import streamlit as st
import time
from ml.preprocessing import Preprocessor
from ml.expected_energy import ExpectedEnergyModel
from ml.anomaly_detector import ContextualAnomalyDetector
from ml.features import create_features
from ml.decision_engine import evaluate_decision
from ml.artifacts import get_artifact_path
from alerts.sms import send_sms
from alerts.telegram import drain_telegram_results, send_telegram
from investigation.events import EventTracker
from replay.state import ELECTRICITY_TARIFF, INTERVENTION_COST, MAX_LIVE_HISTORY

REQUIRED_DECISION_FIELDS = {
    'actual_energy',
    'expected_energy',
    'residual',
    'deviation_pct',
    'excess_energy',
    'context_score',
    'is_anomaly',
    'severity',
    'reason',
}


def _validate_decision_result(decision):
    missing_fields = REQUIRED_DECISION_FIELDS.difference(decision)
    if missing_fields:
        missing = ', '.join(sorted(missing_fields))
        raise RuntimeError(f"Decision result is missing required field(s): {missing}")


def _update_telegram_status(state):
    for event_id, status, error, attempts in drain_telegram_results():
        alert = state['telegram_alerts'].get(event_id)
        if alert is None:
            continue
        alert.update({'status': status, 'error': error, 'attempts': attempts})
        for event in state['anomaly_events']:
            if event['event_id'] == event_id:
                event['telegram_status'] = status
                event['telegram_error'] = error
                break


def _log_telegram_summary(state):
    if state['telegram_summary_logged'] or state['replay_status'] != 'COMPLETED':
        return
    if any(alert['status'] == 'PENDING' for alert in state['telegram_alerts'].values()):
        return
    alerts = state['telegram_alerts'].values()
    print(f"Detected anomalies: {len(state['anomaly_events'])}")
    print(f"Telegram queued: {len(state['telegram_alerts'])}")
    print(f"Telegram sent: {sum(alert['status'] == 'SENT' for alert in alerts)}")
    print(f"Telegram failed: {sum(alert['status'] == 'FAILED' for alert in alerts)}")
    print(f"Pending: {sum(alert['status'] == 'PENDING' for alert in alerts)}")
    state['telegram_summary_logged'] = True


def load_models():
    """Loads frozen artifacts once."""
    preprocessor = Preprocessor.load(get_artifact_path('preprocessor.joblib'))
    energy_model = ExpectedEnergyModel.load(get_artifact_path('energy_model.joblib'))
    anomaly_detector = ContextualAnomalyDetector.load(get_artifact_path('anomaly_detector.joblib'))
    return preprocessor, energy_model, anomaly_detector

def process_next_observation():
    """Processes exactly one observation from the unseen dataset."""
    state = st.session_state.replay_state
    state.setdefault('total_energy', 0.0)
    state.setdefault('cumulative_energy_variance', 0.0)
    state.setdefault('cumulative_energy_saving', 0.0)
    state.setdefault('roi_pct', 0.0)
    state.setdefault('telegram_alerts', {})
    state.setdefault('telegram_summary_logged', False)
    _update_telegram_status(state)
    _log_telegram_summary(state)
    if not state['replay_running']:
        return False
        
    # Non-blocking delay logic (with tolerance for Streamlit scheduling jitter)
    current_time = time.time()
    last_tick = state['last_tick_time']
    delay_required = state['next_tick_delay']
    
    if (current_time - last_tick) < (delay_required - 0.02):
        return False # Not time for next row yet
        
    state['last_tick_time'] = current_time
        
    if state['unseen_data'] is None:
        state['replay_running'] = False
        state['replay_status'] = 'STOPPED'
        return False
        
    unseen_df = state['unseen_data']
    total_rows = len(unseen_df)
    
    if state['current_index'] >= total_rows:
        state['replay_running'] = False
        state['replay_status'] = 'COMPLETED'
        return False

    # Initialize tracker if not exists
    if state['active_event_tracker'] is None:
        state['active_event_tracker'] = EventTracker()
        
    # Get current row
    idx = state['current_index']
    row = unseen_df.iloc[[idx]].copy()
    
    # Extract actual values
    actual_energy = row['Chiller Energy Consumption (kWh)'].values[0]
    equipment_id = row['equipment_id'].values[0] if 'equipment_id' in row.columns else 'UNKNOWN'
    timestamp = row['timestamp'].values[0]
    
    # 1. Feature processing
    features_df = create_features(row)
    
    # Load models if not passed (for simplicity, we assume they can be loaded quickly 
    # or cached by Streamlit. Streamlit @st.cache_resource is best)
    preprocessor, energy_model, anomaly_detector = _get_cached_models()
    
    # 2. Preprocess
    X_processed = preprocessor.transform(features_df)
    
    # 3. Expected prediction
    expected_energy = energy_model.predict(X_processed)[0]
    
    # 4. Context score
    context_score = anomaly_detector.predict_score(X_processed)[0]
    
    # 5. Decision Engine (calculates residual, deviation, etc.)
    decision = evaluate_decision(actual_energy, expected_energy, context_score)
    _validate_decision_result(decision)
    
    # 6. Event persistence
    tracker = state['active_event_tracker']
    row_data = row.iloc[0].to_dict()
    active_event, is_new_event = tracker.process_observation(equipment_id, timestamp, decision, row_data)
    
    # 7. Alert delivery
    if is_new_event and active_event and decision['is_anomaly']:
        event_id = active_event['event_id']
        if event_id not in state['telegram_alerts']:
            msg = (f"🚨 VOLTURA ENERGY-WASTE ALERT\n\n"
                   f"Event: {event_id}\n"
                   f"Equipment: {equipment_id}\n"
                   f"Time: {timestamp}\n\n"
                   f"Actual Energy: {actual_energy} kWh\n"
                   f"Expected Energy: {expected_energy} kWh\n"
                   f"Deviation: {decision['deviation_pct']}%\n"
                   f"Excess Energy: {decision['excess_energy']} kWh\n"
                   f"Severity: {decision['severity']}\n\n"
                   f"Context Score: {decision['context_score']}\n\n"
                   f"What happened:\n{decision['reason']}\n\n"
                   "Action:\nInvestigate this equipment and operating conditions.")
            state['telegram_alerts'][event_id] = {
                'status': 'PENDING',
                'error': None,
                'attempts': 0,
            }
            active_event['telegram_status'] = 'PENDING'
            print(f"[ANOMALY] event_id={event_id}")
            print(f"[TELEGRAM] QUEUED event_id={event_id}")
            send_telegram(msg, event_id)

        if decision['severity'] in ['HIGH', 'CRITICAL'] and event_id not in state['sent_sms_events']:
            send_sms(msg)
            state['sent_sms_events'].add(event_id)
            active_event['sms_sent'] = True
    
    # 8. Update State
    state['current_observation'] = {
        'timestamp': timestamp,
        'equipment_id': equipment_id,
        **decision,
        'row_data': row.iloc[0].to_dict()
    }
    
    # Removed processed_count to sync with replay_index
    if decision['is_anomaly']:
        state['anomaly_count'] += 1
    else:
        state['normal_count'] += 1
        
    state['total_excess_energy'] += decision['excess_energy']
    state['total_energy'] += actual_energy
    variance = actual_energy - expected_energy
    state['cumulative_energy_variance'] += variance
    energy_saving = expected_energy - actual_energy
    state['cumulative_energy_saving'] += energy_saving
    savings_value = state['cumulative_energy_saving'] * ELECTRICITY_TARIFF
    state['roi_pct'] = (savings_value / INTERVENTION_COST) * 100
    
    state['processed_history'].append(state['current_observation'])
    if len(state['processed_history']) > MAX_LIVE_HISTORY:
        state['processed_history'].pop(0)
        
    state['anomaly_events'] = tracker.get_all_events()
    if is_new_event and state['selected_event_id'] is None:
        state['selected_event_id'] = active_event['event_id']
    
    # Set delay for the NEXT tick based on this row's result
    if state['fast_demo_mode']:
        state['next_tick_delay'] = 1.0 if decision['is_anomaly'] else 0.05
    else:
        state['next_tick_delay'] = 1.0
            
    # Advance index
    state['current_index'] += 1
    
    if state['current_index'] >= total_rows:
        state['replay_status'] = 'COMPLETED'
        state['replay_running'] = False

    _update_telegram_status(state)
    _log_telegram_summary(state)

    return True

@st.cache_resource
def _get_cached_models():
    return load_models()
