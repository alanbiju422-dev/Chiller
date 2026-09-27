import pytest
from investigation.events import EventTracker

def test_event_persistence():
    tracker = EventTracker()
    
    # 1. Normal observation
    dec1 = {"severity": "NORMAL", "is_anomaly": False, "actual_energy": 105, "expected_energy": 100, "deviation_pct": 5, "excess_energy": 0, "residual": 5, "context_score": 0.5, "reason": ""}
    evt, is_new = tracker.process_observation("CHILLER-01", "2023-01-01 00:00", dec1, {})
    assert evt is None
    
    # 2. First Anomaly -> New Event
    dec2 = {"severity": "HIGH", "is_anomaly": True, "actual_energy": 130, "expected_energy": 100, "deviation_pct": 30, "excess_energy": 30, "residual": 30, "context_score": -0.8, "reason": ""}
    evt, is_new = tracker.process_observation("CHILLER-01", "2023-01-01 00:30", dec2, {})
    assert is_new is True
    assert evt['event_id'].startswith("CHILLER-01-EVT")
    assert evt['duration'] == 1
    
    # 3. Second continuous anomaly -> New Event (no grouping)
    dec3 = {"severity": "CRITICAL", "is_anomaly": True, "actual_energy": 155, "expected_energy": 100, "deviation_pct": 55, "excess_energy": 55, "residual": 55, "context_score": -0.8, "reason": ""}
    evt2, is_new2 = tracker.process_observation("CHILLER-01", "2023-01-01 01:00", dec3, {})
    assert is_new2 is True
    assert evt2['event_id'] != evt['event_id']
    
    # 4. Normal -> nothing happens
    evt3, is_new3 = tracker.process_observation("CHILLER-01", "2023-01-01 01:30", dec1, {})
    assert evt3 is None
    assert len(tracker.get_all_events()) == 2
