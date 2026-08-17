"""LSTM architecture used across models A/B/C.

Two stacked LSTM layers with dropout, single scalar output. Mirrors the
dissertation's original build so numbers can be reproduced before we swap in
FinBERT for the 2026 headline model.
"""
from __future__ import annotations

from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from src.config import TIMESTEPS


def build_lstm(n_features: int, name: str) -> Sequential:
    model = Sequential(
        [
            Input(shape=(TIMESTEPS, n_features), name=f"{name}_input"),
            LSTM(32, return_sequences=True, name=f"{name}_lstm1"),
            Dropout(0.2),
            LSTM(16, return_sequences=False, name=f"{name}_lstm2"),
            Dropout(0.2),
            Dense(1, name=f"{name}_output"),
        ],
        name=name,
    )
    model.compile(optimizer="adam", loss="mse")
    return model
