import pandas as pd
import numpy as np
import json

def analyze():
    df = pd.read_csv('development_dataset.csv')
    
    print("Dataset Analysis:")
    print("=================")
    
    # 1. Row & Col Count
    print(f"Row count: {len(df)}")
    print(f"Column count: {len(df.columns)}")
    
    # 2. Columns & Types
    print("\nColumns and Data Types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
        
    # 3. Missing values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    if missing.sum() == 0:
        print("  None")
        
    # 4. Equipment IDs
    if 'equipment_id' in df.columns:
        print("\nEquipment IDs:")
        print(df['equipment_id'].unique().tolist())
        print("Equipment ID Value Counts:")
        print(df['equipment_id'].value_counts())
        
    # 5. Timestamp analysis
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print("\nTimestamp range:")
            print(f"  Start: {df['timestamp'].min()}")
            print(f"  End: {df['timestamp'].max()}")
            
            # Sampling interval
            df = df.sort_values('timestamp')
            diffs = df['timestamp'].diff()
            print("\nSampling Intervals:")
            print(diffs.value_counts().head())
            
            # Invalid timestamps
            print(f"\nInvalid/NaT timestamps: {df['timestamp'].isnull().sum()}")
            
            # Duplicate rows (exact)
            print(f"Exact Duplicate Rows: {df.duplicated().sum()}")
            
            # Duplicate timestamps
            print(f"Duplicate Timestamps: {df['timestamp'].duplicated().sum()}")
            if 'equipment_id' in df.columns:
                print(f"Duplicate (Timestamp, Equipment): {df.duplicated(subset=['timestamp', 'equipment_id']).sum()}")
        except Exception as e:
            print(f"Timestamp error: {e}")

if __name__ == '__main__':
    analyze()
