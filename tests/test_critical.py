import pytest
from ml.decision_engine import evaluate_decision

def test_decision_engine_rules():
    # Negative deviation
    res = evaluate_decision(actual=80, expected=100, context_score=0.9)
    assert res['deviation_pct'] == -20.0
    assert res['excess_energy'] == 0.0
    assert res['severity'] == 'NORMAL'

    # Zero expected energy
    res = evaluate_decision(actual=10, expected=0, context_score=0.9)
    assert res['severity'] == 'NORMAL' # our zero protection returns NORMAL and 0 for everything
    
    # 10% deviation
    res = evaluate_decision(actual=110, expected=100, context_score=0.9)
    assert res['deviation_pct'] == 10.0
    assert res['severity'] == 'NORMAL'
    
    # 15% deviation
    res = evaluate_decision(actual=115, expected=100, context_score=0.9)
    assert res['severity'] == 'WATCH'

    # CRITICAL TEST: 25% deviation
    res = evaluate_decision(actual=125, expected=100, context_score=-0.5)
    assert res['deviation_pct'] == 25.0
    assert res['severity'] == 'INVESTIGATE'
    assert res['is_anomaly'] is True

    # CRITICAL TEST: 30% deviation
    res = evaluate_decision(actual=130, expected=100, context_score=-0.8)
    assert res['deviation_pct'] == 30.0
    assert res['severity'] == 'HIGH'

    # CRITICAL TEST: 50% deviation
    res = evaluate_decision(actual=150, expected=100, context_score=-0.9)
    assert res['deviation_pct'] == 50.0
    assert res['severity'] in ['HIGH', 'CRITICAL']

def test_zero_division_protection():
    res = evaluate_decision(actual=50, expected=0, context_score=0)
    assert res['severity'] == 'NORMAL'
    
    res = evaluate_decision(actual=None, expected=100, context_score=0)
    assert res['severity'] == 'NORMAL'
    
    res = evaluate_decision(actual=100, expected=None, context_score=0)
    assert res['severity'] == 'NORMAL'
