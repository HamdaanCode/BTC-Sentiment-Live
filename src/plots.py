"""4-panel comparison figure: actual vs pred, loss curves, R^2 bars, error boxplots.

Reads eval.json + history.json written by src.evaluate and src.train (so it
runs without touching the model). Writes to artifacts/plots/comparison_seed{seed}.png.

Usage:
    python -m src.plots --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, DEFAULT_SEED, TIMESTEPS, TRAIN_RATIO, TRAINING_CSV

MODELS = ["A", "B", "C"]
COLORS = {"A": "blue", "B": "green", "C": "red"}
LABELS = {"A": "A (Baseline)", "B": "B (Crypto)", "C": "C (Intensity)"}


def _load_eval(seed: int, model_key: str) -> dict:
    p = ARTIFACTS_DIR / f"{model_key}_seed{seed}" / "eval.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run `python -m src.evaluate --seed {seed}` first")
    return json.loads(p.read_text())


def _load_history(seed: int, model_key: str) -> dict:
    p = ARTIFACTS_DIR / f"{model_key}_seed{seed}" / "history.json"
    return json.loads(p.read_text())


def _test_dates() -> pd.DatetimeIndex:
    """Recover the test-window dates the same way train did."""
    df = pd.read_csv(TRAINING_CSV)
    df["date"] = pd.to_datetime(df["date"])
    total_windows = len(df) - TIMESTEPS
    train_windows = int(total_windows * TRAIN_RATIO)
    test_start = TIMESTEPS + train_windows
    return pd.to_datetime(df["date"].iloc[test_start : test_start + (total_windows - train_windows)].values)


def build_figure(seed: int) -> Path:
    evals = {k: _load_eval(seed, k) for k in MODELS}
    hists = {k: _load_history(seed, k) for k in MODELS}

    actual = np.array(evals["A"]["actual"])
    test_dates = _test_dates()
    if len(test_dates) != len(actual):
        # fall back to plain integer index if dates cannot be aligned
        test_dates = np.arange(len(actual))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"LSTM Bitcoin Price Prediction — Sentiment Model Comparison (seed={seed})",
        fontsize=14,
        fontweight="bold",
    )

    # Panel 1: actual vs predicted
    ax = axes[0, 0]
    ax.plot(test_dates, actual, color="black", linewidth=2, label="Actual Price")
    for k in MODELS:
        e = evals[k]
        ax.plot(
            test_dates,
            e["predicted"],
            color=COLORS[k],
            linewidth=1.2,
            linestyle="--",
            label=f"Model {k} (R^2={e['r2']:.3f})",
        )
    ax.set_title("Actual vs Predicted Price")
    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=8)
    if isinstance(test_dates, pd.DatetimeIndex):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # Panel 2: train + val loss
    ax = axes[0, 1]
    for k in MODELS:
        h = hists[k]
        ax.plot(h["loss"], color=COLORS[k], linewidth=1.2, label=f"{LABELS[k]} train")
        ax.plot(h["val_loss"], color=COLORS[k], linewidth=1.2, linestyle=":", label=f"{LABELS[k]} val")
    ax.set_title("Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (scaled)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3: R^2 bars
    ax = axes[1, 0]
    xs = np.arange(len(MODELS))
    r2s = [evals[k]["r2"] for k in MODELS]
    bars = ax.bar(xs, r2s, width=0.5, color=[COLORS[k] for k in MODELS], alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[k].replace(" (", "\n(") for k in MODELS])
    ax.set_ylabel("R^2")
    ax.set_title("R^2 Comparison")
    lo = min(min(r2s), 0) - 0.05
    ax.set_ylim(lo, 1.0)
    ax.axhline(y=0, color="black", linewidth=0.5)
    for bar, val in zip(bars, r2s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: error boxplots
    ax = axes[1, 1]
    errors = [np.array(evals[k]["actual"]) - np.array(evals[k]["predicted"]) for k in MODELS]
    ax.boxplot(
        errors,
        tick_labels=[f"Model {k}" for k in MODELS],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
    )
    ax.axhline(y=0, color="red", linewidth=1, linestyle="--")
    ax.set_title("Prediction Error Distribution")
    ax.set_ylabel("Error (USD)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    out_dir = ARTIFACTS_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"comparison_seed{seed}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = build_figure(args.seed)
    print(f"Wrote {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
