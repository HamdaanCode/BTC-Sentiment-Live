"""Load training CSV, build sliding windows, temporal train/test split."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    COLS_TO_SCALE,
    FEATURES_BY_MODEL,
    TARGET,
    TIMESTEPS,
    TRAINING_CSV,
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


def load_scaled(csv_path: Path = TRAINING_CSV) -> tuple[pd.DataFrame, MinMaxScaler]:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.fillna(0)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = df.copy()
    df_scaled[COLS_TO_SCALE] = scaler.fit_transform(df[COLS_TO_SCALE])
    return df_scaled, scaler


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


def prepare(model_key: str, csv_path: Path = TRAINING_CSV) -> Windowed:
    if model_key not in FEATURES_BY_MODEL:
        raise ValueError(f"unknown model {model_key!r}; expected one of {list(FEATURES_BY_MODEL)}")
    df_scaled, scaler = load_scaled(csv_path)
    X, y = create_windows(df_scaled, FEATURES_BY_MODEL[model_key])
    X_tr, X_te, y_tr, y_te = temporal_split(X, y)
    return Windowed(X_tr, X_te, y_tr, y_te, scaler, df_scaled)


def inverse_price(scaled_values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Invert MinMax scaling for the target column (close sits at index 0)."""
    dummy = np.zeros((len(scaled_values), len(COLS_TO_SCALE)))
    dummy[:, 0] = np.asarray(scaled_values).flatten()
    return scaler.inverse_transform(dummy)[:, 0]
