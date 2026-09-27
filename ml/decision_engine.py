from typing import Dict, Any

def evaluate_decision(actual: float, expected: float, context_score: float) -> Dict[str, Any]:
    """
    Determines severity and decision based on Energy Deviation and Context.
    
    NORMAL: 0% to <=10%
    WATCH: 10% to <20%
    INVESTIGATE: >=20% to <30%
    HIGH: >=30%
    CRITICAL: >50% (or combined with very strong negative context_score)
    """
    if expected is None or actual is None or expected != expected or expected <= 0: # Handle NaN/0
        return _build_response("NORMAL", actual, expected, 0, 0, context_score, False, "Invalid expected value")

    residual = actual - expected
    deviation_pct = (residual / expected) * 100
    excess_energy = max(0, residual)

    if deviation_pct <= 10:
        severity = "NORMAL"
        is_anomaly = False
        reason = "Normal operation within expected bounds."
    elif 10 < deviation_pct < 20:
        severity = "WATCH"
        is_anomaly = False
        reason = "Slightly elevated energy usage."
    elif 20 <= deviation_pct < 30:
        severity = "INVESTIGATE"
        is_anomaly = True
        reason = "Significant energy deviation requiring investigation."
    elif 30 <= deviation_pct < 50:
        severity = "HIGH"
        is_anomaly = True
        reason = "High energy-waste anomaly."
    else: # >= 50%
        severity = "CRITICAL"
        is_anomaly = True
        reason = "Critical energy-waste anomaly."

    return _build_response(severity, actual, expected, deviation_pct, excess_energy, context_score, is_anomaly, reason)


def _build_response(severity, actual, expected, deviation_pct, excess_energy, context_score, is_anomaly, reason):
    return {
        "decision": severity,
        "severity": severity,
        "actual_energy": actual,
        "expected_energy": expected,
        "deviation_pct": deviation_pct,
        "residual": actual - expected if actual is not None and expected is not None else 0,
        "excess_energy": excess_energy,
        "is_anomaly": is_anomaly,
        "context_score": context_score,
        "reason": reason
    }
