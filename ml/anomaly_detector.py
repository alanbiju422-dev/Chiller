import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os
import numpy as np

class ContextualAnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)

    def train(self, X: pd.DataFrame):
        """Train Isolation Forest ONLY on training data."""
        self.model.fit(X)

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns anomaly score. Lower is more anomalous. 
        We map it to a context score where closer to 1 is normal, closer to -1 is anomalous.
        """
        return self.model.decision_function(X)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'ContextualAnomalyDetector':
        return joblib.load(filepath)
