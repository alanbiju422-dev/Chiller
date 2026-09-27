def gather_evidence(event, context_row):
    """
    Gathers environmental and operational evidence for a given event.
    """
    return {
        "event_id": event['event_id'],
        "equipment": event['equipment_id'],
        "deviation_pct": event['maximum_deviation'],
        "duration": event['duration'],
        "environmental_context": {
            "outside_temp": context_row.get('Outside Temperature (F)'),
            "humidity": context_row.get('Humidity (%)'),
            "wind_speed": context_row.get('Wind Speed (mph)')
        },
        "operating_conditions": {
            "load_rt": context_row.get('Building Load (RT)'),
            "chilled_water_rate": context_row.get('Chilled Water Rate (L/sec)')
        }
    }
