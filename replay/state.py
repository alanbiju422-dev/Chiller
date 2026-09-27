import streamlit as st

MAX_LIVE_HISTORY = 300
ELECTRICITY_TARIFF = 7.75
INTERVENTION_COST = 10000.0

def init_state():
    """Initialize the single replay state object once per session."""
    if 'replay_state' not in st.session_state:
        st.session_state.replay_state = {
            'current_index': 0,
            'replay_running': False,
            'replay_status': 'READY',
            'current_observation': None,
            'processed_history': [],
            'anomaly_events': [],
            'active_event_tracker': None,
            'normal_count': 0,
            'anomaly_count': 0,
            'total_excess_energy': 0.0,
            'total_energy': 0.0,
            'cumulative_energy_variance': 0.0,
            'cumulative_energy_saving': 0.0,
            'roi_pct': 0.0,
            'sent_sms_events': set(),
            'telegram_alerts': {},
            'telegram_summary_logged': False,
            'unseen_data': None,
            'fast_demo_mode': True,
            'last_tick_time': 0.0,
            'next_tick_delay': 0.0,
            'current_anomaly_index': 0,
            'selected_event_id': None,
        }

def reset_state():
    """Reset replay state while preserving the cached dataset and mode."""
    state = st.session_state.replay_state
    if state['active_event_tracker']:
        state['active_event_tracker'].reset()

    state.update({
        'current_index': 0,
        'replay_running': False,
        'replay_status': 'READY',
        'current_observation': None,
        'processed_history': [],
        'anomaly_events': [],
        'active_event_tracker': None,
        'normal_count': 0,
        'anomaly_count': 0,
        'total_excess_energy': 0.0,
        'total_energy': 0.0,
        'cumulative_energy_variance': 0.0,
        'cumulative_energy_saving': 0.0,
        'roi_pct': 0.0,
        'sent_sms_events': set(),
        'telegram_alerts': {},
        'telegram_summary_logged': False,
        'last_tick_time': 0.0,
        'next_tick_delay': 0.0,
        'current_anomaly_index': 0,
        'selected_event_id': None,
    })
