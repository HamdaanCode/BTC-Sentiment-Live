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
    # Phase 2 extension: identical feature set to Model B, but trained on the
    # 2022 bear-market tweet dataset. Controls the sentiment-method variable
    # (VADER + crypto lexicon stays fixed) so any delta is attributable to
    # the time regime, not the preprocessing.
    "B2": ["close", "volume", "price_ma_7", "price_change_pct", "sentiment_b_mean"],
}
TARGET = "close"

# Training CSV overrides for Phase 1-family models that train on different
# windows (e.g. B2 trains on 2022 instead of Sept-Nov 2021).
TRAINING_CSV_2022 = DATA_DIR / "NEW_lstm_training_data_2022.csv"
TRAINING_CSV_OVERRIDES = {
    "B2": TRAINING_CSV_2022,
}

# Phase 2: FinBERT sentiment, separate training CSVs per window.
# Model D: FinBERT on Sept-Nov 2021 (same window as A/B/C, direct comparison)
# Model E: FinBERT on May-June 2022 (bear-regime generalisation test)
PHASE2_DIR = ARTIFACTS_DIR / "phase2"
TRAINING_CSV_D = PHASE2_DIR / "finbert_2021_training.csv"
TRAINING_CSV_E = PHASE2_DIR / "finbert_2022_training.csv"

COLS_TO_SCALE_PHASE2 = [
    "close",
    "volume",
    "price_ma_7",
    "price_change_pct",
    "price_volatility_7",
    "sentiment_finbert_mean",
]

FEATURES_BY_MODEL_PHASE2 = {
    "D": ["close", "volume", "price_ma_7", "price_change_pct", "sentiment_finbert_mean"],
    "E": ["close", "volume", "price_ma_7", "price_change_pct", "sentiment_finbert_mean"],
}

TRAINING_CSV_BY_MODEL_PHASE2 = {
    "D": TRAINING_CSV_D,
    "E": TRAINING_CSV_E,
}

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
