from typing import Dict, Any, List

def _number(context_row: Dict[str, Any], key: str):
    value = context_row.get(key)
    return value if isinstance(value, (int, float)) else None


def _value_text(label: str, value: Any, unit: str = '') -> str:
    if value is None:
        return f"{label} is not available"
    return f"{label} was {value:.1f}{unit}"


def generate_anomaly_explanation(event: Dict[str, Any], context_row: Dict[str, Any]) -> str:
    load = _number(context_row, 'Building Load (RT)')
    water_rate = _number(context_row, 'Chilled Water Rate (L/sec)')
    outside_temperature = _number(context_row, 'Outside Temperature (F)')
    humidity = _number(context_row, 'Humidity (%)')
    dew_point = _number(context_row, 'Dew Point (F)')
    cooling_water_temperature = _number(context_row, 'Cooling Water Temperature (C)')
    explanation = (
        f"At {event['timestamp']}, {event['equipment_id']} used {event['actual_energy']:.1f} kWh "
        f"against {event['expected_energy']:.1f} kWh expected, an observed deviation of "
        f"{event['deviation_pct']:.1f}% and {event['excess_energy']:.1f} kWh excess energy. "
    )
    if load is not None and load > 500:
        explanation += f"The observed building load of {load:.1f} RT was high, so demand may be a possible contributing condition. "
    elif load is not None:
        explanation += f"The observed building load was {load:.1f} RT. "
    if cooling_water_temperature is not None:
        explanation += f"Cooling-water temperature was {cooling_water_temperature:.1f} C and requires comparison with operating conditions. "
    if water_rate is not None:
        explanation += f"Chilled-water rate was {water_rate:.1f} L/sec. "
    if outside_temperature is not None or humidity is not None or dew_point is not None:
        explanation += (
            f"Environmental readings were {_value_text('outside temperature', outside_temperature, ' F')}, "
            f"{_value_text('humidity', humidity, '%')}, and {_value_text('dew point', dew_point, ' F')}."
        )
    return explanation


def generate_investigation_points(event: Dict[str, Any], context_row: Dict[str, Any]) -> List[str]:
    points = [f"Compare {event['equipment_id']} schedule, setpoints, and controls at {event['timestamp']}."]
    load = _number(context_row, 'Building Load (RT)')
    water_rate = _number(context_row, 'Chilled Water Rate (L/sec)')
    cooling_water_temperature = _number(context_row, 'Cooling Water Temperature (C)')
    outside_temperature = _number(context_row, 'Outside Temperature (F)')
    if load is not None:
        points.append(f"Check the {load:.1f} RT building load against the expected energy response.")
    if water_rate is not None:
        points.append(f"Check chilled-water flow at the observed {water_rate:.1f} L/sec rate.")
    if cooling_water_temperature is not None:
        points.append(f"Compare the observed cooling-water temperature of {cooling_water_temperature:.1f} C with normal condenser conditions.")
    if outside_temperature is not None:
        points.append(f"Assess whether the outside temperature of {outside_temperature:.1f} F contributed to the deviation.")
    if len(points) == 1:
        points.append("Available data is insufficient to identify a specific contributing condition; further investigation is required.")
    return points


def generate_corrective_actions(event: Dict[str, Any], context_row: Dict[str, Any]) -> List[str]:
    actions = [
        f"Review {event['equipment_id']} operating conditions and verify the energy meter against the {event['actual_energy']:.1f} kWh observation.",
        "Check controls, staging, setpoints, and sensor quality before assigning a mechanical cause.",
    ]
    water_rate = _number(context_row, 'Chilled Water Rate (L/sec)')
    cooling_water_temperature = _number(context_row, 'Cooling Water Temperature (C)')
    if water_rate is not None:
        actions.append(f"Inspect chilled-water flow and operating restrictions around the observed {water_rate:.1f} L/sec rate.")
    if cooling_water_temperature is not None:
        actions.append(f"Review condenser and cooling-water conditions around {cooling_water_temperature:.1f} C.")
    if event['severity'] in ['HIGH', 'CRITICAL']:
        actions.append("Escalate for prompt operating review; the available evidence does not confirm a mechanical failure.")
    return actions

def generate_explanations(event: Dict[str, Any], context_row: Dict[str, Any]) -> List[str]:
    """
    Generates deterministic recommendations based on evidence and context.
    """
    recs = []
    
    # Basic recommendations based on severity
    if event['severity'] in ['HIGH', 'CRITICAL']:
        recs.append("Immediate review of chiller operating schedule and setpoints.")
        recs.append("Inspect equipment operating efficiency manually.")
    elif event['severity'] == 'INVESTIGATE':
        recs.append("Review chiller operating schedule.")
        recs.append("Investigate load/flow relationship.")
        
    # Contextual recommendations
    if context_row.get('Outside Temperature (F)', 0) > 90:
        recs.append("High outside temperature detected. Verify cooling-tower heat rejection capability.")
    
    if context_row.get('Building Load (RT)', 0) > 500:
        recs.append("High building load observed. Verify if multiple chillers should be staged.")
        
    if event['duration'] > 3:
        recs.append("Deviation is persistent. Check maintenance history for potential fouling or sensor drift.")
        
    if not recs:
        recs.append("Monitor equipment for further deviations.")
        
    return recs
