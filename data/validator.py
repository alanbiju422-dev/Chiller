import pandas as pd
import numpy as np

EXPECTED_COLUMNS = [
    'timestamp',
    'equipment_id',
    'Chilled Water Rate (L/sec)',
    'Cooling Water Temperature (C)',
    'Building Load (RT)',
    'Chiller Energy Consumption (kWh)',
    'Outside Temperature (F)',
    'Dew Point (F)',
    'Humidity (%)',
    'Wind Speed (mph)',
    'Pressure (in)'
]

def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates the dataset columns, parses timestamps, sorts chronologically,
    and removes exact duplicate rows or handles duplicate timestamps.
    """
    # 1. Check required columns
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 2. Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # 3. Remove NaT timestamps
    df = df.dropna(subset=['timestamp'])

    # 4. Sort chronologically
    df = df.sort_values('timestamp')

    # 5. Drop exact duplicates
    df = df.drop_duplicates()

    # 6. Handle duplicate timestamps per equipment
    # We will keep the first occurrence
    df = df.drop_duplicates(subset=['timestamp', 'equipment_id'], keep='first')
    
    # 7. Identify missing target rows (can't be used for supervised training/evaluation)
    # We will just mark them or let loader handle it. For now, just drop if target is missing
    # because they cannot be evaluated or trained on.
    df = df.dropna(subset=['Chiller Energy Consumption (kWh)'])

    # Reset index after dropping rows
    df = df.reset_index(drop=True)

    return df
