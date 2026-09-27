from typing import Dict, Any, List

class EventTracker:
    def __init__(self):
        self.anomaly_events = []
        self.event_counter = 1
        
    def process_observation(self, equipment_id: str, timestamp, decision_result: Dict[str, Any], row_data: Dict[str, Any] = None):
        """
        Creates a completely independent event for every single anomalous observation.
        """
        is_anomaly = decision_result['is_anomaly']
        
        if is_anomaly:
            # Create a completely independent event for this single observation
            event_id = f"{equipment_id}-EVT-{self.event_counter:06d}"
            self.event_counter += 1
            
            new_event = {
                'event_id': event_id,
                'equipment_id': equipment_id,
                'start_time': timestamp,
                'end_time': timestamp,
                'timestamp': timestamp,
                'duration': 1,
                'observation_count': 1,
                'maximum_deviation': decision_result['deviation_pct'],
                'average_deviation': decision_result['deviation_pct'],
                'maximum_excess_energy': decision_result['excess_energy'],
                'total_excess_energy': decision_result['excess_energy'],
                'actual_energy': decision_result['actual_energy'],
                'expected_energy': decision_result['expected_energy'],
                'residual': decision_result['residual'],
                'deviation_pct': decision_result['deviation_pct'],
                'excess_energy': decision_result['excess_energy'],
                'severity': decision_result['severity'],
                'context_score': decision_result['context_score'],
                'is_anomaly': decision_result['is_anomaly'],
                'reason': decision_result['reason'],
                'sms_sent': False,
                'status': 'NEW',
                'row_data': row_data
            }
            
            self.anomaly_events.append(new_event)
            return new_event, True # Return the new event, and True indicating it's new
            
        return None, False
        
    def get_all_events(self) -> List[Dict[str, Any]]:
        return list(self.anomaly_events)
        
    def reset(self):
        self.anomaly_events = []
        self.event_counter = 1
