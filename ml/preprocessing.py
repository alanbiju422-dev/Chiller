import pandas as pd
from sklearn.impute import SimpleImputer
import joblib
import os

class Preprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None

    def fit(self, X: pd.DataFrame):
        """Fits preprocessing strictly on training data."""
        self.feature_names = X.columns.tolist()
        self.imputer.fit(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies fitted preprocessing to validation/unseen data."""
        # Ensure columns match
        missing_cols = [col for col in self.feature_names if col not in X.columns]
        for col in missing_cols:
            X[col] = 0 # or appropriate default
        
        X_ordered = X[self.feature_names]
        X_imputed = self.imputer.transform(X_ordered)
        return pd.DataFrame(X_imputed, columns=self.feature_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'Preprocessor':
        return joblib.load(filepath)
