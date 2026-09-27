import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates temporal and physical features.
    Does NOT use future values or target itself as an input feature.
    """
    X = df.copy()
    
    # Temporal features
    X['hour'] = X['timestamp'].dt.hour
    X['day_of_week'] = X['timestamp'].dt.dayofweek
    X['month'] = X['timestamp'].dt.month
    
    # Physical/interaction features
    # E.g., Temperature Difference if applicable
    # Often Chilled Water Delta T could be useful if Return Temp was provided, but we don't have it.
    # We can create a feature combining load and outside temp.
    if 'Building Load (RT)' in X.columns and 'Outside Temperature (F)' in X.columns:
        X['load_per_temp'] = X['Building Load (RT)'] / (X['Outside Temperature (F)'] + 1e-5)

    # Convert equipment_id to categorical/dummies if needed
    if 'equipment_id' in X.columns:
        X = pd.get_dummies(X, columns=['equipment_id'], drop_first=False)
        
    # Drop columns that shouldn't be features
    cols_to_drop = ['timestamp', 'Chiller Energy Consumption (kWh)']
    X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])
    
    return X
