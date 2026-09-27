"""
ml/forecast.py — NEXYRA Energy Forecasting Module
===================================================

PURPOSE
-------
Answers: "What energy consumption is likely in the near future?"

This is SEPARATE from the Expected Energy model (which answers
"What SHOULD energy be under current operating conditions?").

APPROACH
--------
Lag-based Random Forest per equipment:
  - Data resolution : 30-minute intervals
  - Lag features    : t-1 (30min), t-2 (1h), t-4 (2h), t-8 (4h)
  - Rolling feature : 4-step rolling mean (2h)
  - Temporal        : hour-of-day, day-of-week
  - Physical        : Building Load (RT), Outside Temperature (°F)
  - Forecast steps  : +1 (30min), +2 (1h), +6 (3h)

Trained PER equipment on training data only — never touches unseen rows.
Saved to artifacts/energy_forecast_model.joblib (separate artifact).

IMPORTANT
---------
- Does NOT touch or overwrite artifacts/energy_model.joblib
- Does NOT use the existing Expected Energy pipeline
- Does NOT run during the replay loop
- Forecast is generated once, cached in session_state
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
STEP_MINUTES      = 30           # Dataset resolution
FORECAST_STEPS    = [1, 2, 6]   # +30min, +1h, +3h
LAG_STEPS         = [1, 2, 4, 8]
ROLL_WINDOW       = 4            # 2-hour rolling mean
ARTIFACT_NAME     = "energy_forecast_model.joblib"

PHYSICAL_FEATURES = [
    "Building Load (RT)",
    "Outside Temperature (F)",
]


# ── Feature builder ────────────────────────────────────────────────────────────

def _build_lag_features(
    df: pd.DataFrame,
    target_col: str = "Chiller Energy Consumption (kWh)",
) -> pd.DataFrame:
    """
    Build lag + rolling + temporal features for a SINGLE equipment's sorted DataFrame.
    Returns a clean DataFrame with NaN rows (from lag shift) dropped.
    """
    out = df.copy()

    for lag in LAG_STEPS:
        out[f"energy_lag{lag}"] = out[target_col].shift(lag)

    out[f"energy_roll{ROLL_WINDOW}_mean"] = (
        out[target_col].rolling(ROLL_WINDOW).mean()
    )

    ts = pd.to_datetime(out["timestamp"])
    out["hour"]        = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["month"]       = ts.dt.month

    # Physical features (available at prediction time)
    for col in PHYSICAL_FEATURES:
        if col in df.columns:
            out[col] = df[col]

    return out.dropna().reset_index(drop=True)


def _feature_cols() -> List[str]:
    """Return the ordered list of feature column names used by the model."""
    cols = [f"energy_lag{l}" for l in LAG_STEPS]
    cols += [f"energy_roll{ROLL_WINDOW}_mean"]
    cols += ["hour", "day_of_week", "month"]
    cols += PHYSICAL_FEATURES
    return cols


# ── Per-equipment forecaster ──────────────────────────────────────────────────

class EquipmentForecaster:
    """
    Lightweight Random Forest forecaster for a single chiller.
    Trains one model per forecast horizon to avoid recursive compounding errors.
    """
    def __init__(self, equipment_id: str):
        self.equipment_id = equipment_id
        self.models: Dict[int, RandomForestRegressor] = {}   # step -> model
        self.feature_cols: List[str] = _feature_cols()
        self._last_known_rows: Optional[pd.DataFrame] = None  # for prediction seed

    def train(self, train_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train one RF model per forecast step on training data only.
        Returns MAE per step for logging purposes.
        """
        target_col = "Chiller Energy Consumption (kWh)"
        eq_df = (
            train_df[train_df["equipment_id"] == self.equipment_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if len(eq_df) < 50:
            logger.warning("Insufficient data for %s", self.equipment_id)
            return {}

        lag_df = _build_lag_features(eq_df, target_col)

        # Verify all feature columns exist
        available_features = [c for c in self.feature_cols if c in lag_df.columns]
        self.feature_cols = available_features

        maes = {}
        for step in FORECAST_STEPS:
            # Target is the energy `step` rows ahead
            y = lag_df[target_col].shift(-step).dropna()
            X = lag_df.loc[y.index, self.feature_cols]

            if len(X) < 20:
                continue

            # Simple chronological validation (last 15%)
            split = int(len(X) * 0.85)
            X_tr, X_val = X.iloc[:split], X.iloc[split:]
            y_tr, y_val = y.iloc[:split], y.iloc[split:]

            rf = RandomForestRegressor(
                n_estimators=80,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_tr, y_tr)
            preds_val = rf.predict(X_val)
            mae = float(np.mean(np.abs(preds_val - y_val.values)))
            maes[f"step_{step}_mae"] = round(mae, 3)

            # Retrain on full data
            rf_full = RandomForestRegressor(
                n_estimators=80,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
            )
            rf_full.fit(X, y)
            self.models[step] = rf_full

        # Store the last `max(LAG_STEPS)` rows as seed for future prediction
        self._last_known_rows = eq_df.tail(max(LAG_STEPS) + ROLL_WINDOW + 1).copy()

        return maes

    def forecast(
        self,
        seed_rows: Optional[pd.DataFrame] = None,
        current_load_rt: Optional[float] = None,
        current_outside_temp_f: Optional[float] = None,
    ) -> List[Dict]:
        """
        Generate forecasts for all configured steps.

        seed_rows   : Recent rows for the equipment (used to build lag features).
                      If None, uses training-time tail.
        Returns a list of dicts: {step, minutes_ahead, label, forecast_kwh}
        """
        if not self.models:
            return []

        target_col = "Chiller Energy Consumption (kWh)"
        seed = seed_rows if seed_rows is not None else self._last_known_rows
        if seed is None or len(seed) == 0:
            return []

        seed = seed.sort_values("timestamp").tail(max(LAG_STEPS) + ROLL_WINDOW + 1).copy()

        # Build features from seed
        lag_df = _build_lag_features(seed, target_col)
        if lag_df.empty:
            return []

        # Use the most recent complete row
        last_row = lag_df.iloc[[-1]][self.feature_cols].copy()

        # Override with live operating conditions if provided
        if current_load_rt is not None and "Building Load (RT)" in last_row.columns:
            last_row["Building Load (RT)"] = current_load_rt
        if current_outside_temp_f is not None and "Outside Temperature (F)" in last_row.columns:
            last_row["Outside Temperature (F)"] = current_outside_temp_f

        results = []
        last_ts = pd.to_datetime(seed["timestamp"].max())
        for step in FORECAST_STEPS:
            if step not in self.models:
                continue
            pred = float(self.models[step].predict(last_row)[0])
            pred = max(0.0, round(pred, 1))
            forecast_ts = last_ts + pd.Timedelta(minutes=STEP_MINUTES * step)
            results.append({
                "step":          step,
                "minutes_ahead": STEP_MINUTES * step,
                "label":         forecast_ts.strftime("%H:%M"),
                "forecast_kwh":  pred,
            })
        return results


# ── Multi-equipment container ─────────────────────────────────────────────────

class EnergyForecastModel:
    """Container holding one EquipmentForecaster per chiller."""

    def __init__(self):
        self.forecasters: Dict[str, EquipmentForecaster] = {}
        self.step_minutes = STEP_MINUTES
        self.forecast_steps = FORECAST_STEPS

    def train(self, train_df: pd.DataFrame) -> Dict:
        """Train all equipment forecasters. Returns training metrics dict."""
        all_metrics = {}
        for eq_id in train_df["equipment_id"].unique():
            forecaster = EquipmentForecaster(eq_id)
            maes = forecaster.train(train_df)
            self.forecasters[eq_id] = forecaster
            all_metrics[eq_id] = maes
            logger.info("Forecast model trained for %s: %s", eq_id, maes)
        return all_metrics

    def forecast(
        self,
        equipment_id: str,
        seed_rows: Optional[pd.DataFrame] = None,
        current_load_rt: Optional[float] = None,
        current_outside_temp_f: Optional[float] = None,
    ) -> List[Dict]:
        """Return forecast for a specific equipment."""
        fc = self.forecasters.get(equipment_id)
        if fc is None:
            return []
        return fc.forecast(seed_rows, current_load_rt, current_outside_temp_f)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("Forecast model saved to %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> "EnergyForecastModel":
        return joblib.load(filepath)


# ── Training entry-point ─────────────────────────────────────────────────────

def train_forecast_model(train_df: pd.DataFrame, save_path: str) -> Dict:
    """
    Train and save the forecast model.  Called from forecast_pipeline.py.
    Does NOT touch energy_model.joblib.
    """
    model = EnergyForecastModel()
    metrics = model.train(train_df)
    model.save(save_path)
    return metrics
