"""Train one or all LSTM models and persist weights + metrics + scaler.

Usage:
    python -m src.train --model A --seed 42
    python -m src.train --model all --seed 42
    python -m src.train --model B --seed 7 --epochs 100

Artifacts written to artifacts/{model}_seed{seed}/:
    - model.keras         Keras 3 native format, load with keras.models.load_model
    - scaler.pkl          fitted MinMaxScaler, needed to invert predictions
    - metrics.json        train/val loss, epochs run, features used
    - history.json        per-epoch loss / val_loss, for plotting later
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

# Seed before any tf import — set_global_seed handles ordering internally.
from src.config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    DEFAULT_SEED,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    FEATURES_BY_MODEL,
    FEATURES_BY_MODEL_PHASE2,
    set_global_seed,
)


def _features_for(model_key: str) -> list[str]:
    if model_key in FEATURES_BY_MODEL:
        return FEATURES_BY_MODEL[model_key]
    if model_key in FEATURES_BY_MODEL_PHASE2:
        return FEATURES_BY_MODEL_PHASE2[model_key]
    raise ValueError(f"unknown model {model_key!r}")


def train_one(model_key: str, seed: int, epochs: int, batch_size: int, patience: int) -> dict:
    set_global_seed(seed)

    # Imports must follow set_global_seed so tf sees the deterministic env vars.
    from tensorflow.keras.callbacks import EarlyStopping

    from src.data import prepare
    from src.model import build_lstm

    w = prepare(model_key)
    features = _features_for(model_key)
    model = build_lstm(n_features=len(features), name=f"LSTM_{model_key}")

    history = model.fit(
        w.X_train,
        w.y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(w.X_test, w.y_test),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        ],
        verbose=0,
    )

    out_dir = ARTIFACTS_DIR / f"{model_key}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save(out_dir / "model.keras")
    with open(out_dir / "scaler.pkl", "wb") as fh:
        pickle.dump(w.scaler, fh)

    epochs_run = len(history.history["loss"])
    metrics = {
        "model": model_key,
        "seed": seed,
        "features": features,
        "epochs_run": epochs_run,
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "best_val_loss": float(min(history.history["val_loss"])),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "history.json").write_text(
        json.dumps(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            indent=2,
        )
    )

    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--model",
        choices=["A", "B", "C", "B2", "D", "E", "all", "phase2"],
        default="all",
        help="'all' = A/B/C (dissertation baseline); 'phase2' = D/E (FinBERT); B2 = VADER on 2022 bear-market data",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.model == "all":
        keys = ["A", "B", "C"]
    elif args.model == "phase2":
        keys = ["D", "E"]
    else:
        keys = [args.model]

    results = []
    for k in keys:
        print(f"\n--- Training Model {k} (seed={args.seed}) ---", flush=True)
        m = train_one(
            model_key=k,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )
        print(
            f"   epochs_run={m['epochs_run']}  "
            f"train_loss={m['final_train_loss']:.6f}  "
            f"val_loss={m['final_val_loss']:.6f}  "
            f"best_val_loss={m['best_val_loss']:.6f}",
            flush=True,
        )
        results.append(m)

    summary_path = ARTIFACTS_DIR / f"summary_seed{args.seed}.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSummary written to {summary_path.relative_to(Path.cwd()) if summary_path.is_relative_to(Path.cwd()) else summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
