import pandas as pd
from typing import Tuple
from .validator import validate_and_clean_data
from .splitter import chronological_split

def load_and_prepare_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads dataset, cleans it, and returns a train and unseen set.
    """
    df = pd.read_csv(file_path)
    df_clean = validate_and_clean_data(df)
    train_df, unseen_df = chronological_split(df_clean)
    return train_df, unseen_df
