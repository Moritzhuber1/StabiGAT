from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Paths / folders
# ----------------------------
# The script expects this structure:
# runs/<method>/<run_name>/summary.json
# runs/<method>/<run_name>/config.json
#
# It will write outputs to:
# runs/_report/
ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
REPORT_DIR = RUNS_DIR / "_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Helper functions
# ----------------------------
def read_json(path: Path) -> Dict[str, Any]:
    """
    Read a JSON file and return it as a Python dictionary.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_runs(runs_dir: Path) -> pd.DataFrame:
    """
    Collect information from all runs into one table (DataFrame).

    It searches for:
      runs/<method>/<run>/summary.json
    and reads config.json from the same folder.

    Returned table includes:
    - CV settings (method, splits)
    - model hyperparameters
    - aggregated metrics (micro and mean/std over folds)
    """
    rows: List[Dict[str, Any]] = []

    for summary_path in runs_dir.glob("*/*/summary.json"):
        run_dir = summary_path.parent
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            # Skip runs without config.json because we cannot interpret them well.
            continue

        summary = read_json(summary_path)
        cfg = read_json(cfg_path)

        cv_method = cfg["cv"]["method"]

        row = {
            "cv_method": cv_method,
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "cut_off": cfg["data"]["cut_off"],
            "num_layers": cfg["model"]["num_layers"],
            "hidden_dim": cfg["model"]["hidden_dim"],
            "dropout": cfg["model"]["dropout_rate"],
            "n_splits": cfg["cv"]["n_splits"],
            # Aggregated metrics (use NaN if missing)
            "micro_f1": summary.get("f1", np.nan),
            "micro_acc": summary.get("accuracy", np.nan),
            "micro_precision": summary.get("precision", np.nan),
            "micro_recall": summary.get("recall", np.nan),
            "mean_f1": summary.get("folds", {}).get("mean_f1", np.nan),
            "std_f1": summary.get("folds", {}).get("std_f1", np.nan),
            "mean_acc": summary.get("folds", {}).get("mean_acc", np.nan),
            "std_acc": summary.get("folds", {}).get("std_acc", np.nan),
            "n_samples": summary.get("n_samples", np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"No runs found under {runs_dir}. "
            "Expected runs/<method>/<run>/summary.json"
        )

    return df


def load_learning_curves(run_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate learning curves for all folds of a run.

    Expected files:
      fold0_learning.csv, fold1_learning.csv, ...

    The output DataFrame has columns like:
      epoch, train_loss, val_loss, val_acc, val_f1, ...
    plus an extra column 'fold'.
    """
    frames: List[pd.DataFrame] = []

    for csv_path in sorted(run_dir.glob("fold*_learning.csv")):
        # Example name: fold3_learning.csv -> fold = 3
        fold_str = csv_path.stem.replace("_learning", "").replace("fold", "")
        try:
            fold = int(fold_str)
        except ValueError:
            # If the file name is unexpected, skip it to avoid crashing.
            continue

        df = pd.read_csv(csv_path)
        df["fold"] = fold
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def summarize_learning_curves(lc: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize learning curves: mean and standard deviation over folds per epoch.

    This helps to see the general training behavior without fold-to-fold noise.
    """
    if lc.empty:
        return lc

    grouped = lc.groupby("epoch", as_index=False)
    out = grouped.agg(
        train_loss_mean=("train_loss", "mean"),
        train_loss_std=("train_loss", "std"),
        val_loss_mean=("val_loss", "mean"),
        val_loss_std=("val_loss", "std"),
        val_f1_mean=("val_f1", "mean"),
        val_f1_std=("val_f1", "std"),
        val_acc_mean=("val_acc", "mean"),
        val_acc_std=("val_acc", "std"),
    )
    return out


def compute_overfit_scores(lc: pd.DataFrame) -> Dict[str, float]:
    """
    Compute simple overfitting indicators from learning curves.

    Metrics:
    - gap_last: (val_loss - train_loss) at the last epoch (mean over folds)
    - gap_auc: average (val_loss - train_loss) over epochs (a simple "area" proxy)
    - best_val_f1: maximum of mean(val_f1) over epochs
    - epoch_best_f1: epoch index where mean(val_f1) is maximal
    """
    if lc.empty:
        return {
            "gap_last": np.nan,
            "gap_auc": np.nan,
            "best_val_f1": np.nan,
            "epoch_best_f1": np.nan,
        }

    s = summarize_learning_curves(lc)

    gap = s["val_loss_mean"] - s["train_loss_mean"]
    gap_last = float(gap.iloc[-1])
    gap_auc = float(gap.mean())

    # Find best epoch by validation F1
    val_f1_values = s["val_f1_mean"].to_numpy()
    best_idx = int(np.nanargmax(val_f1_values))
    best_val_f1 = float(s.loc[best_idx, "val_f1_mean"])
    epoch_best_f1 = float(s.loc[best_idx, "epoch"])

    return {
        "gap_last": gap_last,
        "gap_auc": gap_auc,
        "best_val_f1": best_val_f1,
        "epoch_best_f1": epoch_best_f1,
    }


# ----------------------------
# Plot functions
# ----------------------------
def plot_leaderboard(df: pd.DataFrame, out_png: Path, metric: str, topn: int = 20) -> None:
    """
    Plot a horizontal bar chart of the top runs by a selected metric.
    """
    d = df.sort_values(metric, ascending=False).head(topn).copy()
    labels = d["cv_method"].astype(str) + " | " + d["run_name"].astype(str)
    values = d[metric].to_numpy()

    plt.figure(figsize=(12, max(4, 0.35 * len(d))))
    plt.barh(range(len(d)), values)
    plt.yticks(range(len(d)), labels)
    plt.gca().invert_yaxis()
    plt.xlabel(metric)
    plt.title(f"Top {topn} runs by {metric}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_overfit_scatter(df: pd.DataFrame, out_png: Path) -> None:
    """
    Scatter plot:
      x-axis: overfit gap (lower is better)
      y-axis: mean_f1 (higher is better)
    """
    d = df.dropna(subset=["gap_auc", "mean_f1"]).copy()

    plt.figure(figsize=(10, 7))
    plt.scatter(d["gap_auc"], d["mean_f1"])
    plt.xlabel("Overfit gap: mean(val_loss - train_loss) over epochs (lower is better)")
    plt.ylabel("mean_f1 (higher is better)")
    plt.title("Overfitting vs Performance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_learning_curve(summary_lc: pd.DataFrame, title: str, out_prefix: Path) -> None:
    """
    Plot learning curves (mean over folds):
    - Loss curves: train_loss_mean and val_loss_mean
    - Validation F1 curve: val_f1_mean

    The function writes two files:
      <out_prefix>_loss.png
      <out_prefix>_valf1.png
    """
    if summary_lc.empty:
        return

    # Loss plot
    plt.figure(figsize=(9, 6))
    plt.plot(summary_lc["epoch"], summary_lc["train_loss_mean"], label="train_loss_mean")
    plt.plot(summary_lc["epoch"], summary_lc["val_loss_mean"], label="val_loss_mean")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"{title} | Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.stem + "_loss.png"), dpi=300)
    plt.close()

    # Validation F1 plot
    plt.figure(figsize=(9, 6))
    plt.plot(summary_lc["epoch"], summary_lc["val_f1_mean"], label="val_f1_mean")
    plt.xlabel("epoch")
    plt.ylabel("val_f1")
    plt.title(f"{title} | Validation F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.stem + "_valf1.png"), dpi=300)
    plt.close()


# ----------------------------
# Main execution
# ----------------------------
def main() -> None:
    """
    Main report pipeline:
    1) Collect all runs into a table
    2) For each run, compute overfitting scores from learning curves
    3) Save the full table as CSV
    4) Create leaderboard plots and an overfit scatter plot
    5) For each CV method, plot learning curves for the best run (by mean_f1)
    """
    df = collect_runs(RUNS_DIR)

    # Add overfitting indicators for each run
    overfit_scores: List[Dict[str, float]] = []
    for run_dir_str in df["run_dir"]:
        run_dir = Path(run_dir_str)
        lc = load_learning_curves(run_dir)
        scores = compute_overfit_scores(lc)
        overfit_scores.append(scores)

    overfit_df = pd.DataFrame(overfit_scores)
    df = pd.concat([df.reset_index(drop=True), overfit_df.reset_index(drop=True)], axis=1)

    # Save the merged table
    out_csv = REPORT_DIR / "all_runs_table.csv"
    df.to_csv(out_csv, index=False)

    # Leaderboards
    plot_leaderboard(df, REPORT_DIR / "leaderboard_mean_f1.png", metric="mean_f1", topn=25)
    plot_leaderboard(df, REPORT_DIR / "leaderboard_micro_f1.png", metric="micro_f1", topn=25)

    # Overfitting vs performance
    plot_overfit_scatter(df, REPORT_DIR / "overfit_vs_meanf1.png")

    # Learning curves for best run per CV method
    for method in ["sss", "skf", "gkf"]:
        sub = df[df["cv_method"] == method].copy()
        if sub.empty:
            continue

        best = sub.sort_values("mean_f1", ascending=False).iloc[0]
        run_dir = Path(best["run_dir"])
        lc = load_learning_curves(run_dir)
        s = summarize_learning_curves(lc)

        title = f"BEST {method} | mean_f1={best['mean_f1']:.3f} | {best['run_name']}"
        plot_learning_curve(s, title=title, out_prefix=REPORT_DIR / f"learning_{method}.png")

    print(f"Report written to: {REPORT_DIR}")
    print(f"Table saved to: {out_csv}")


if __name__ == "__main__":
    main()