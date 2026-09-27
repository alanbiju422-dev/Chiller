import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import numpy as np

class ExpectedEnergyModel:
    def __init__(self, model_type='rf'):
        if model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            self.model = HistGradientBoostingRegressor(random_state=42)

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the regression model."""
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'ExpectedEnergyModel':
        return joblib.load(filepath)

def train_and_select_model(X_train: pd.DataFrame, y_train: pd.Series) -> ExpectedEnergyModel:
    """
    Splits training data for internal validation, compares models, 
    and returns the best one trained on the full train set.
    """
    # Simple internal split for validation (last 20% of training data)
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

    models = {
        'rf': ExpectedEnergyModel(model_type='rf'),
        'hgb': ExpectedEnergyModel(model_type='hgb')
    }

    best_name = None
    best_score = float('inf')

    for name, model in models.items():
        model.train(X_tr, y_tr)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        
        if mae < best_score:
            best_score = mae
            best_name = name

    # Train best model on full training set
    best_model = ExpectedEnergyModel(model_type=best_name)
    best_model.train(X_train, y_train)
    
    return best_model
