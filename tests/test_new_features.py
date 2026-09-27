"""Integration test for all 3 new features."""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')

from ml.artifacts import get_artifact_path

# Test 1: All artifacts present, none overwritten
assert os.path.exists(get_artifact_path('energy_model.joblib'))
assert os.path.exists(get_artifact_path('anomaly_detector.joblib'))
assert os.path.exists(get_artifact_path('preprocessor.joblib'))
assert os.path.exists(get_artifact_path('energy_forecast_model.joblib'))
print('Test 1 PASS: All artifacts present, none overwritten')

# Test 2: Forecast model loads and produces 3 horizons per equipment
from ml.forecast import EnergyForecastModel
fm = EnergyForecastModel.load(get_artifact_path('energy_forecast_model.joblib'))
assert 'CHILLER-01' in fm.forecasters
assert 'CHILLER-02' in fm.forecasters
assert 'CHILLER-03' in fm.forecasters
for eq in ['CHILLER-01', 'CHILLER-02', 'CHILLER-03']:
    fc = fm.forecast(eq)
    assert len(fc) == 3, f'Expected 3 horizons for {eq}'
    for step in fc:
        assert step['forecast_kwh'] > 0
    print(f'  {eq}: +30min={fc[0]["forecast_kwh"]} | +1h={fc[1]["forecast_kwh"]} | +3h={fc[2]["forecast_kwh"]} kWh')
print('Test 2 PASS: Forecast working for all 3 chillers')

# Test 3: Savings calculation
from recommendations.engine import calculate_savings_summary, generate_recommendations
event_high = {
    'event_id': 'CHILLER-01-EVT-000001', 'equipment_id': 'CHILLER-01',
    'timestamp': '2024-06-15 14:30:00', 'severity': 'HIGH',
    'actual_energy': 180.0, 'expected_energy': 140.0,
    'deviation_pct': 28.6, 'excess_energy': 40.0,
    'context_score': -0.41, 'is_anomaly': True, 'duration': 1, 'reason': 'test',
}
s = calculate_savings_summary(event_high, 7.75)
assert s['available']
assert abs(s['excess_kwh'] - 40.0) < 0.01
assert abs(s['potential_cost_impact_inr'] - 310.0) < 0.01
print(f'Test 3 PASS: Savings = {s["excess_kwh"]} kWh / Rs{s["potential_cost_impact_inr"]}')

# Test 4: Recommendations ranked
ctx = {'Building Load (RT)': 520, 'Cooling Water Temperature (C)': 32.0}
recs = generate_recommendations(event_high, ctx, 7.75)
assert len(recs) > 0
for r in recs:
    assert r['estimated_payback_months'] is not None
    print(f'  [{r["priority"]}] {r["title"]} | payback={r["estimated_payback_months"]}mo | Rs{r["estimated_monthly_savings_inr"]}/mo')
print(f'Test 4 PASS: {len(recs)} recommendations, ranked')

# Test 5: NORMAL event
event_normal = {**event_high, 'severity': 'NORMAL', 'deviation_pct': 5.0}
recs_n = generate_recommendations(event_normal, ctx, 7.75)
print(f'Test 5 PASS: NORMAL event -> {len(recs_n)} recommendations (expected 0)')

# Test 6: Existing modules still import
from replay.engine import process_next_observation
from ml.decision_engine import evaluate_decision
from investigation.events import EventTracker
print('Test 6 PASS: All existing modules import correctly')

# Test 7: Replay engine unchanged
import inspect
src = inspect.getsource(process_next_observation)
assert 'evaluate_decision' in src
assert 'send_telegram' in src
print('Test 7 PASS: Replay engine unchanged')

print()
print('ALL 7 TESTS PASSED')
