import pandas as pd
from typing import Tuple

def chronological_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronologically splits data into training and completely unseen validation sets.
    """
    # Ensure sorted
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    
    train_df = df.iloc[:n_train].copy()
    unseen_df = df.iloc[n_train:].copy()
    
    return train_df, unseen_df
