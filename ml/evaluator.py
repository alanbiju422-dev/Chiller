from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Baseline: predict historical median
    baseline_pred = np.full(shape=y_true.shape, fill_value=y_true.median())
    baseline_mae = mean_absolute_error(y_true, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))
    baseline_r2 = r2_score(y_true, baseline_pred)
    
    return {
        'model': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        },
        'baseline': {
            'mae': baseline_mae,
            'rmse': baseline_rmse,
            'r2': baseline_r2
        }
    }
