"""
recommendations/engine.py — NEXYRA ROI / Payback Recommendation Engine
=======================================================================

Generates ranked recommendations based on anomaly evidence.
Uses configurable cost assumptions from config/recommendation_costs.json.

All monetary values, payback periods, and savings are ESTIMATED.
They are clearly labelled as assumptions — NOT guaranteed returns.
"""

import json
import os
from typing import Any, Dict, List, Optional

# ── Config loader ──────────────────────────────────────────────────────────────

_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "recommendation_costs.json"
)

def _load_config() -> Dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Priority ordering ──────────────────────────────────────────────────────────

_PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NORMAL": 4}
_SEVERITY_WEIGHT = {"CRITICAL": 1.5, "HIGH": 1.2, "INVESTIGATE": 1.0,
                    "WATCH": 0.7, "NORMAL": 0.3}


# ── Evidence conditions ────────────────────────────────────────────────────────

def _check_condition(condition: str, event: Dict, context_row: Dict) -> bool:
    """
    Returns True if the recommendation's evidence condition is supported
    by the actual anomaly data. Only surface recommendations with real evidence.
    """
    severity   = event.get("severity", "NORMAL")
    load_rt    = context_row.get("Building Load (RT)")
    cw_temp    = context_row.get("Cooling Water Temperature (C)")
    dev_pct    = event.get("deviation_pct", 0)

    if condition == "high_load":
        return load_rt is not None and load_rt > 450

    if condition == "maintenance":
        # Elevated CW temperature or large deviation suggests possible fouling
        return (cw_temp is not None and cw_temp > 30) or dev_pct > 35

    if condition == "inspection":
        return severity in ("HIGH", "CRITICAL")

    if condition == "setpoint":
        # Any significant deviation that isn't pure load-driven
        return dev_pct >= 20

    if condition == "schedule":
        # Any non-normal event may benefit from schedule review
        return severity in ("WATCH", "INVESTIGATE", "HIGH", "CRITICAL")

    return True


# ── Savings calculations ───────────────────────────────────────────────────────

def _calculate_savings(
    event: Dict,
    base_kwh_per_event: float,
    tariff: float,
) -> Dict[str, float]:
    """
    Estimate monthly savings. Scales the base assumption by
    the observed deviation — larger deviation → proportionally higher estimate.
    Caps at 3x to avoid overclaiming.
    """
    dev_pct = event.get("deviation_pct", 20)
    scale   = min(3.0, max(0.5, dev_pct / 20.0))
    monthly_kwh = round(base_kwh_per_event * scale, 1)
    monthly_inr = round(monthly_kwh * tariff, 2)
    return {"monthly_kwh": monthly_kwh, "monthly_inr": monthly_inr}


def _payback_months(implementation_cost: float, monthly_savings_inr: float) -> Optional[float]:
    if monthly_savings_inr <= 0:
        return None
    return round(implementation_cost / monthly_savings_inr, 1)


# ── Priority assignment ─────────────────────────────────────────────────────────

def _assign_priority(
    base_priority: str,
    severity: str,
    payback_months: Optional[float],
    monthly_savings_inr: float,
) -> str:
    """
    Upgrade priority based on:
    - CRITICAL / HIGH severity → upgrade one level
    - Short payback (< 3 months) → upgrade one level
    - High savings (> 3000 INR/month) → keep or upgrade
    """
    priority_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    idx = priority_levels.index(base_priority) if base_priority in priority_levels else 1

    if severity in ("CRITICAL", "HIGH"):
        idx = min(len(priority_levels) - 1, idx + 1)

    if payback_months is not None and payback_months < 3:
        idx = min(len(priority_levels) - 1, idx + 1)

    if monthly_savings_inr > 3000:
        idx = min(len(priority_levels) - 1, idx + 1)

    return priority_levels[idx]


def _build_reason(
    rec_key: str,
    event: Dict,
    context_row: Dict,
    payback_months: Optional[float],
) -> str:
    """Build a transparent, evidence-based reason string."""
    severity   = event.get("severity", "")
    dev_pct    = event.get("deviation_pct", 0)
    equipment  = event.get("equipment_id", "")
    load_rt    = context_row.get("Building Load (RT)")
    cw_temp    = context_row.get("Cooling Water Temperature (C)")

    parts = [
        f"Observed {dev_pct:.1f}% energy deviation on {equipment} (severity: {severity})."
    ]

    if rec_key == "high_load" and load_rt:
        parts.append(f"Building load of {load_rt:.0f} RT is elevated and may be driving excess energy.")
    if rec_key == "maintenance" and cw_temp:
        parts.append(f"Cooling-water temperature of {cw_temp:.1f} °C suggests possible heat-transfer degradation.")
    if rec_key == "inspection":
        parts.append(f"Severity level '{severity}' warrants physical equipment inspection.")
    if rec_key == "setpoint":
        parts.append("Energy deviation pattern is consistent with suboptimal chilled-water setpoints.")
    if rec_key == "schedule":
        parts.append("Anomaly may reflect misalignment between chiller operation schedule and actual building load.")

    if payback_months is not None:
        parts.append(f"Estimated payback: {payback_months} months.")

    parts.append("All savings figures are estimates based on documented assumptions.")
    return " ".join(parts)


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_recommendations(
    event: Dict[str, Any],
    context_row: Dict[str, Any],
    tariff: float = 7.75,
) -> List[Dict[str, Any]]:
    """
    Generate ranked recommendations for an anomaly event.

    Parameters
    ----------
    event       : Anomaly event dict from investigation.events.EventTracker
    context_row : The raw data row (event['row_data'])
    tariff      : Electricity tariff in INR/kWh

    Returns
    -------
    List of recommendation dicts sorted by priority (most important first).
    """
    config   = _load_config()
    recs_cfg = config["recommendations"]
    severity = event.get("severity", "NORMAL")

    recommendations = []

    for key, cfg in recs_cfg.items():
        # Only surface recommendations triggered by this severity level
        if severity not in cfg.get("triggers", []):
            continue

        # Only surface if evidence condition is met
        condition = cfg.get("evidence_condition", "")
        if not _check_condition(condition, event, context_row):
            continue

        impl_cost   = cfg["estimated_implementation_cost"]
        base_kwh    = cfg["estimated_monthly_energy_saving_kwh_per_event"]
        savings     = _calculate_savings(event, base_kwh, tariff)
        payback     = _payback_months(impl_cost, savings["monthly_inr"])
        priority    = _assign_priority(
            cfg["priority_base"], severity, payback, savings["monthly_inr"]
        )
        reason      = _build_reason(condition, event, context_row, payback)

        recommendations.append({
            "key":                         key,
            "title":                       cfg["title"],
            "description":                 cfg["description"],
            "estimated_monthly_kwh":       savings["monthly_kwh"],
            "estimated_monthly_savings_inr": savings["monthly_inr"],
            "estimated_implementation_cost": impl_cost,
            "estimated_payback_months":    payback,
            "priority":                    priority,
            "reason":                      reason,
        })

    # Sort by priority then by monthly savings descending
    recommendations.sort(
        key=lambda r: (
            _PRIORITY_ORDER.get(r["priority"], 99),
            -r["estimated_monthly_savings_inr"],
        )
    )

    return recommendations


def calculate_savings_summary(
    event: Dict[str, Any],
    tariff: float = 7.75,
) -> Dict[str, Any]:
    """
    Calculate potential energy savings for a single anomaly event.
    Uses only the existing actual vs expected energy values.

    Returns
    -------
    Dict with excess_kwh, potential_cost_impact_inr.
    Clearly labelled as 'estimated' / 'potential'.
    """
    actual   = event.get("actual_energy")
    expected = event.get("expected_energy")

    if actual is None or expected is None:
        return {"available": False}

    excess_kwh = max(0.0, actual - expected)
    potential_cost_inr = round(excess_kwh * tariff, 2)

    return {
        "available":            True,
        "actual_kwh":           round(actual, 2),
        "expected_kwh":         round(expected, 2),
        "excess_kwh":           round(excess_kwh, 2),
        "potential_cost_impact_inr": potential_cost_inr,
        "tariff_per_kwh":       tariff,
        "disclaimer":           (
            "Estimated potential cost impact based on observed excess energy. "
            "Not a guaranteed saving."
        ),
    }
