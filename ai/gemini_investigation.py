"""
VOLTURA Gemini AI Investigation Analysis
----------------------------------------
Reads credentials exclusively from environment variables.
The API key is never printed, displayed, or committed.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ── Model configuration ──────────────────────────────────────────────────────
# Model name is sourced from GEMINI_MODEL env var; falls back to a current
# Gemini text-generation model if not set.  Never hardcoded in multiple places.
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"

# ── System role ───────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = (
    "You are VOLTURA's AI Energy-Waste Investigation Assistant for commercial buildings.\n\n"
    "Analyze the supplied anomaly event using ONLY the supplied data.\n\n"
    "Your purpose is to help a facility manager understand abnormal energy behaviour, "
    "identify possible contributing conditions, estimate potential energy impact, and "
    "recommend practical investigation and corrective actions.\n\n"
    "Distinguish between:\n"
    "- observed facts\n"
    "- AI interpretation\n"
    "- possible contributing conditions\n"
    "- recommended investigation\n"
    "- recommended action\n\n"
    "Never claim that a mechanical failure is confirmed from this data alone.\n\n"
    "If the evidence is insufficient to determine a cause, explicitly state that the "
    "cause cannot be determined from the available data.\n\n"
    "Every anomaly must receive analysis based on its own actual data. "
    "Do NOT use generic reasons that could apply to any event. "
    "Reference the specific numeric values supplied.\n\n"
    "Use wording such as 'possible contributing condition', 'observed deviation', "
    "and 'requires investigation' where appropriate.\n\n"
    "Return EXACTLY these eleven markdown sections with no extra headings:\n"
    "### EXECUTIVE SUMMARY\n"
    "### ENERGY-WASTE ANALYSIS\n"
    "### CONTEXTUAL / ENVIRONMENTAL ANALYSIS\n"
    "### WHY IS IT ANOMALOUS?\n"
    "### POSSIBLE CONTRIBUTING CONDITIONS\n"
    "### WHAT SHOULD BE INVESTIGATED?\n"
    "### RECOMMENDED ACTION\n"
    "### ENERGY SAVINGS OPPORTUNITY\n"
    "### FINANCIAL IMPACT\n"
    "### PRIORITY / ACTIONABILITY\n"
    "### BUSINESS IMPACT\n"
)

# ── Section guide appended to the user prompt ────────────────────────────────
SECTION_GUIDE = """
Please populate each section as follows (use ONLY the data supplied above):

### EXECUTIVE SUMMARY
Summarise: equipment, timestamp, actual energy, expected energy, deviation, severity, and the potential energy-waste significance.

### ENERGY-WASTE ANALYSIS
Explain: actual energy, expected energy, energy deviation (kWh), deviation percentage, excess energy, and what the observed deviation means in operating terms.

### CONTEXTUAL / ENVIRONMENTAL ANALYSIS
Analyse all supplied operating conditions: building load (RT), chilled-water rate (L/sec), cooling-water temperature (°C), outside temperature (°F), dew point (°F), humidity (%), wind speed (mph), pressure (in). State which conditions may plausibly contribute to the observed energy behaviour and which appear unremarkable. Reference the specific numeric values.

### WHY IS IT ANOMALOUS?
Explain the relationship between: Expected Energy + Actual Energy + Energy Deviation + Contextual Anomaly Score + Operating Conditions. Do not reduce the explanation to a simple fixed threshold. Reference the supplied context/isolation-forest score.

### POSSIBLE CONTRIBUTING CONDITIONS
List evidence-supported possibilities only (e.g. higher-than-expected energy under observed load, unusual chilled-water flow, elevated cooling-water temperature, environmental factors, control/staging behaviour, operating-condition deviation). Only mention conditions supported by the supplied numeric data.

### WHAT SHOULD BE INVESTIGATED?
Give specific, actionable checks for the facility manager, referencing the actual values supplied.

### RECOMMENDED ACTION
Provide practical actions separated into:
**Immediate:** ...
**Short-term:** ...
**Longer-term:** ...

### ENERGY SAVINGS OPPORTUNITY
Explain excess energy, potential energy saving, and estimated monetary impact. If a tariff is available, calculate using the supplied financial values. Do not invent savings.

### FINANCIAL IMPACT
If financial data is available, show: Electricity Tariff, Estimated Savings, Intervention Cost, and Estimated ROI. Label ROI clearly as "Estimated ROI" and note it is not a guaranteed financial return.

### PRIORITY / ACTIONABILITY
State the existing VOLTURA severity (NORMAL / WATCH / INVESTIGATE / HIGH / CRITICAL) and explain why it applies to this specific event based on the supplied data. Do not create a new severity system.

### BUSINESS IMPACT
Explain how addressing this specific event can potentially help with: energy efficiency, operating cost, facility management, sustainability, and ESG objectives.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _available(value: Any) -> Any:
    """Return the value if present, else 'Not available'."""
    return value if value is not None else "Not available"


def build_event_payload(
    event: Dict[str, Any],
    tariff: float,
    intervention_cost: float,
) -> Dict[str, Any]:
    """
    Build a structured data payload from the selected anomaly event.
    Uses only the event's own data — no CSV scanning, no model retraining.
    """
    context = event.get("row_data") or {}

    actual = event.get("actual_energy")
    expected = event.get("expected_energy")

    if actual is not None and expected is not None:
        deviation_kwh = actual - expected
        excess_kwh = max(0.0, deviation_kwh)
        savings = excess_kwh * tariff
        roi_pct = (savings / intervention_cost * 100) if intervention_cost else None
    else:
        deviation_kwh = None
        excess_kwh = None
        savings = None
        roi_pct = None

    return {
        "event_information": {
            "event_id": _available(event.get("event_id")),
            "equipment_id": _available(event.get("equipment_id")),
            "timestamp": _available(event.get("timestamp")),
            "severity": _available(event.get("severity")),
            "duration_observations": _available(event.get("duration")),
            "observation_count": _available(event.get("observation_count")),
        },
        "energy": {
            "actual_energy_kwh": _available(actual),
            "expected_energy_kwh": _available(expected),
            "energy_deviation_kwh": _available(deviation_kwh),
            "deviation_pct": _available(event.get("deviation_pct")),
            "excess_energy_kwh": _available(excess_kwh),
        },
        "operating_conditions": {
            "building_load_rt": _available(context.get("Building Load (RT)")),
            "chilled_water_rate_l_sec": _available(context.get("Chilled Water Rate (L/sec)")),
            "cooling_water_temperature_c": _available(context.get("Cooling Water Temperature (C)")),
            "outside_temperature_f": _available(context.get("Outside Temperature (F)")),
            "dew_point_f": _available(context.get("Dew Point (F)")),
            "humidity_pct": _available(context.get("Humidity (%)")),
            "wind_speed_mph": _available(context.get("Wind Speed (mph)")),
            "pressure_in": _available(context.get("Pressure (in)")),
        },
        "ai_ml_evidence": {
            "context_isolation_forest_score": _available(event.get("context_score")),
            "anomaly_decision": _available(event.get("is_anomaly")),
            "severity": _available(event.get("severity")),
            "existing_reason": _available(event.get("reason")),
        },
        "financial": {
            "electricity_tariff_per_kwh": _available(tariff),
            "estimated_savings": _available(savings),
            "intervention_cost": _available(intervention_cost),
            "estimated_roi_pct": _available(roi_pct),
        },
    }


def generate_investigation_analysis(
    event: Dict[str, Any],
    tariff: float,
    intervention_cost: float,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Call Gemini ONCE for the selected anomaly event.

    Returns:
        (analysis_text, None)   on success
        (None, error_message)   on any failure

    Security:
        - API key is read from the environment; never printed or displayed.
        - Called only on explicit user action (button click).
    """
    # ── API key validation ────────────────────────────────────────────────────
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key.upper() == "YOUR_GEMINI_API_KEY":
        return None, (
            "⚠️ **GEMINI_API_KEY is not configured.** "
            "Open `.env`, replace `YOUR_GEMINI_API_KEY` with your real key, then restart the app."
        )

    # ── Model selection (single source of truth) ──────────────────────────────
    model_name = (os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL)

    # ── Build structured payload from selected event only ─────────────────────
    payload = build_event_payload(event, tariff, intervention_cost)

    user_prompt = (
        "Analyze this VOLTURA anomaly event using ONLY the JSON data below. "
        "Do not invent missing values. Use 'Not available' where data is absent.\n\n"
        + json.dumps(payload, indent=2, default=str)
        + "\n\n"
        + SECTION_GUIDE
    )

    # ── Gemini API call ───────────────────────────────────────────────────────
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
            ),
        )
        analysis = (response.text or "").strip()
        if not analysis:
            return None, "⚠️ Gemini returned an empty response. Please try again."
        return analysis, None

    except Exception as exc:
        error_str = str(exc)
        # User-friendly error messages without exposing the key
        if "API_KEY" in error_str.upper() or "INVALID_ARGUMENT" in error_str.upper():
            return None, (
                "⚠️ **Invalid API key.** Please verify your `GEMINI_API_KEY` in `.env` and restart the app."
            )
        if "RESOURCE_EXHAUSTED" in error_str.upper() or "quota" in error_str.lower():
            return None, (
                "⚠️ **Gemini API rate limit reached.** Please wait a moment and try again."
            )
        if "UNAVAILABLE" in error_str.upper() or "connect" in error_str.lower():
            return None, (
                "⚠️ **Network / API error.** Check your internet connection and try again."
            )
        return None, f"⚠️ **Gemini analysis failed:** {error_str}"
