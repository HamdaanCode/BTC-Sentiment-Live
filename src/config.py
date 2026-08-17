"""Central config: paths, hyperparams, seed control."""
from __future__ import annotations

import os
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "reference" / "data"
TRAINING_CSV = DATA_DIR / "NEW_lstm_training_data.csv"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

TIMESTEPS = 3
TRAIN_RATIO = 0.7
EPOCHS = 200
BATCH_SIZE = 4
EARLY_STOP_PATIENCE = 20

COLS_TO_SCALE = [
    "close",
    "volume",
    "price_ma_7",
    "price_change_pct",
    "price_volatility_7",
    "sentiment_a_mean",
    "sentiment_b_mean",
    "sentiment_c_mean",
]

FEATURES_BY_MODEL = {
    "A": ["close", "volume", "price_ma_7", "price_change_pct", "sentiment_a_mean"],
    "B": ["close", "volume", "price_ma_7", "price_change_pct", "sentiment_b_mean"],
    "C": ["close", "volume", "price_ma_7", "price_change_pct", "sentiment_c_mean"],
}
TARGET = "close"

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed python, numpy, and tensorflow for reproducible runs.

    Must be called before any tf ops (import order matters for TF determinism).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        pass
