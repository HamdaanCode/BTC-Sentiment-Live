"""Load training CSV, build sliding windows, temporal train/test split."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    COLS_TO_SCALE,
    COLS_TO_SCALE_PHASE2,
    FEATURES_BY_MODEL,
    FEATURES_BY_MODEL_PHASE2,
    TARGET,
    TIMESTEPS,
    TRAINING_CSV,
    TRAINING_CSV_BY_MODEL_PHASE2,
    TRAINING_CSV_OVERRIDES,
    TRAIN_RATIO,
)


@dataclass
class Windowed:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler
    df: pd.DataFrame  # scaled frame, kept for date lookup
    cols_to_scale: list[str]  # kept so inverse_price can undo scaling correctly


def load_scaled(
    csv_path: Path = TRAINING_CSV,
    cols_to_scale: list[str] | None = None,
) -> tuple[pd.DataFrame, MinMaxScaler, list[str]]:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.fillna(0)

    if cols_to_scale is None:
        cols_to_scale = COLS_TO_SCALE

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df_scaled, scaler, cols_to_scale


def create_windows(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = TARGET,
    window_size: int = TIMESTEPS,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[feature_cols].iloc[i : i + window_size].values)
        y.append(data[target_col].iloc[i + window_size])
    return np.array(X), np.array(y)


def temporal_split(
    X: np.ndarray, y: np.ndarray, train_ratio: float = TRAIN_RATIO
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = int(len(X) * train_ratio)
    return X[:split], X[split:], y[:split], y[split:]


def prepare(model_key: str, csv_path: Path | None = None) -> Windowed:
    """Build windowed train/test tensors for one model key.

    Model keys A/B/C use the Phase 1 training CSV + VADER sentiment features.
    Model keys D/E use their own Phase 2 training CSVs + FinBERT sentiment.
    """
    if model_key in FEATURES_BY_MODEL:
        features = FEATURES_BY_MODEL[model_key]
        cols = COLS_TO_SCALE
        path = csv_path or TRAINING_CSV_OVERRIDES.get(model_key, TRAINING_CSV)
    elif model_key in FEATURES_BY_MODEL_PHASE2:
        features = FEATURES_BY_MODEL_PHASE2[model_key]
        cols = COLS_TO_SCALE_PHASE2
        path = csv_path or TRAINING_CSV_BY_MODEL_PHASE2[model_key]
    else:
        known = list(FEATURES_BY_MODEL) + list(FEATURES_BY_MODEL_PHASE2)
        raise ValueError(f"unknown model {model_key!r}; expected one of {known}")

    df_scaled, scaler, cols_used = load_scaled(path, cols_to_scale=cols)
    X, y = create_windows(df_scaled, features)
    X_tr, X_te, y_tr, y_te = temporal_split(X, y)
    return Windowed(X_tr, X_te, y_tr, y_te, scaler, df_scaled, cols_used)


def inverse_price(
    scaled_values: np.ndarray,
    scaler: MinMaxScaler,
    cols_to_scale: list[str] | None = None,
) -> np.ndarray:
    """Invert MinMax scaling for the target column (close sits at index 0)."""
    cols = cols_to_scale if cols_to_scale is not None else COLS_TO_SCALE
    dummy = np.zeros((len(scaled_values), len(cols)))
    dummy[:, 0] = np.asarray(scaled_values).flatten()
    return scaler.inverse_transform(dummy)[:, 0]
