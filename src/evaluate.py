"""Evaluate saved models: real-money metrics + paired t-tests.

Loads model.keras + scaler.pkl from artifacts/{model}_seed{seed}/, runs
prediction on the same temporal test split the training script used, inverts
scaling back to USD, and reports RMSE / MAE / MAPE / R^2 / directional
accuracy. For --model all, also runs paired t-tests on absolute errors.

Usage:
    python -m src.evaluate --seed 42                # all three, comparison table
    python -m src.evaluate --model A --seed 42
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from src.config import ARTIFACTS_DIR, DEFAULT_SEED, set_global_seed
from src.data import inverse_price, prepare


def _artifact_dir(model_key: str, seed: int) -> Path:
    d = ARTIFACTS_DIR / f"{model_key}_seed{seed}"
    if not d.exists():
        raise FileNotFoundError(
            f"no artifacts at {d}; run `python -m src.train --model {model_key} --seed {seed}` first"
        )
    return d


def evaluate_one(model_key: str, seed: int) -> dict:
    """Load artifacts for one model and compute metrics on its test split."""
    from tensorflow.keras.models import load_model

    out_dir = _artifact_dir(model_key, seed)
    model = load_model(out_dir / "model.keras")
    with open(out_dir / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)

    w = prepare(model_key)
    y_pred_scaled = model.predict(w.X_test, verbose=0).flatten()

    actual = inverse_price(w.y_test, scaler)
    predicted = inverse_price(y_pred_scaled, scaler)

    errors = actual - predicted
    abs_errors = np.abs(errors)
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(abs_errors))
    mape = float(np.mean(np.abs(errors / actual)) * 100)
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = float(1 - (ss_res / ss_tot))
    dir_acc = float(
        np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))) * 100
    )

    result = {
        "model": model_key,
        "seed": seed,
        "n_test": int(len(actual)),
        "rmse_usd": rmse,
        "mae_usd": mae,
        "mape_pct": mape,
        "r2": r2,
        "directional_accuracy_pct": dir_acc,
        "actual": actual.tolist(),
        "predicted": predicted.tolist(),
        "abs_errors": abs_errors.tolist(),
    }
    (out_dir / "eval.json").write_text(json.dumps(result, indent=2))
    return result


def _fmt(val: float, is_r2: bool) -> str:
    return f"{val:.4f}" if is_r2 else f"{val:.2f}"


def print_table(results: list[dict]) -> None:
    header = f"{'Metric':<22}" + "".join(f"{'Model ' + r['model']:>14}" for r in results)
    print()
    print(header)
    print("-" * len(header))
    rows = [
        ("RMSE ($)", "rmse_usd", False),
        ("MAE ($)", "mae_usd", False),
        ("MAPE (%)", "mape_pct", False),
        ("R^2", "r2", True),
        ("Dir. Accuracy (%)", "directional_accuracy_pct", False),
    ]
    for label, key, is_r2 in rows:
        line = f"{label:<22}" + "".join(f"{_fmt(r[key], is_r2):>14}" for r in results)
        print(line)


def print_ttests(results: list[dict]) -> list[dict]:
    """Paired t-tests on absolute errors between every pair of models."""
    print("\n--- Paired t-tests on |prediction error| ---")
    out = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i], results[j]
            t, p = stats.ttest_rel(a["abs_errors"], b["abs_errors"])
            marker = "* significant" if p < 0.05 else "ns"
            print(f"  {a['model']} vs {b['model']}: t={t:+.4f}, p={p:.4f}  {marker}")
            out.append({"pair": f"{a['model']}_vs_{b['model']}", "t": float(t), "p": float(p)})
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=["A", "B", "C", "all"], default="all")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_global_seed(args.seed)  # deterministic dropout at predict time

    keys = ["A", "B", "C"] if args.model == "all" else [args.model]
    results = [evaluate_one(k, args.seed) for k in keys]
    print_table(results)

    ttests = print_ttests(results) if len(results) > 1 else []

    if len(results) > 1:
        best = max(results, key=lambda r: r["r2"])
        print(f"\nBest R^2: Model {best['model']} ({best['r2']:.4f})")
        comparison_path = ARTIFACTS_DIR / f"comparison_seed{args.seed}.json"
        comparison_path.write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "per_model": [
                        {k: v for k, v in r.items() if k not in ("actual", "predicted", "abs_errors")}
                        for r in results
                    ],
                    "paired_ttests": ttests,
                    "best_by_r2": best["model"],
                },
                indent=2,
            )
        )
        print(f"Comparison written to {comparison_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
