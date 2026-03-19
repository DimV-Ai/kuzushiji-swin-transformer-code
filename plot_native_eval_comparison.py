#!/usr/bin/env python3
"""
Create comparison tables and figures for native evaluation results across kuzushiji models.

What it does
------------
- loads one or more native_summary.csv files
- optionally loads per-model threshold CSVs
- combines everything into one comparison dataframe
- marks invalid / caution models
- saves clean CSV tables
- saves easy-to-read figures for discussion with a supervisor

Typical usage
-------------
python plot_native_eval_comparison.py \
  --summary_csv ./native_eval_outputs/native_summary.csv \
  --summary_csv ./native_eval_codhogihan_outputs/native_summary.csv \
  --out_dir ./native_eval_plots

Notes
-----
By default, the script marks codhogihan_min10 as invalid because its native evaluation
was reconstructed on a severely mismatched label space. It is still included in the raw
combined table, but excluded from the "valid_only" figures unless you override that.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from pandas import isna


INVALID_DEFAULT_MODELS = {"codhogihan_min10"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_csv",
        action="append",
        required=True,
        help="Path to a native_summary.csv file. Pass this flag multiple times for multiple files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./native_eval_plots",
        help="Directory where combined tables and figures will be saved.",
    )
    parser.add_argument(
        "--exclude_model",
        action="append",
        default=[],
        help="Model name to exclude from valid-only figures. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include_invalid_in_main_figures",
        action="store_true",
        help="If set, do not exclude invalid models from the main comparison plots.",
    )
    return parser.parse_args()


def load_and_combine_summaries(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["source_summary_csv"] = str(Path(p).resolve())
        frames.append(df)
    if not frames:
        raise ValueError("No summary CSVs were loaded.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["model"], keep="last").reset_index(drop=True)

    for col in ["n_model_only_labels", "n_rebuilt_only_labels"]:
        if col not in combined.columns:
            combined[col] = 0
        combined[col] = combined[col].fillna(0)

    combined["regime"] = combined["model"].map(infer_regime)
    combined["support_setting"] = combined["min_count"].map(lambda x: f"min{x}")
    combined["display_name"] = combined.apply(
        lambda r: f"{r['regime']} ({r['support_setting']})", axis=1
    )
    combined["is_invalid_default"] = combined["model"].isin(INVALID_DEFAULT_MODELS)
    combined["notes"] = combined.apply(make_note, axis=1)
    return combined


def infer_regime(model_name: str) -> str:
    if model_name.startswith("ogihan_"):
        return "Ogihan only"
    if model_name.startswith("codh_") and "codhogihan" not in model_name:
        return "CODH only"
    if model_name.startswith("codhogihan_"):
        return "CODH + Ogihan"
    return "Other"


def make_note(row: pd.Series) -> str:
    notes = []
    if bool(row.get("is_invalid_default", False)):
        notes.append("exclude from main comparison")

    rebuilt_val = row.get("n_rebuilt_only_labels", 0)
    model_val = row.get("n_model_only_labels", 0)

    rebuilt_only = 0 if isna(rebuilt_val) else int(rebuilt_val)
    model_only = 0 if isna(model_val) else int(model_val)

    if rebuilt_only or model_only:
        notes.append(
            f"label mismatch: rebuilt_only={rebuilt_only}, model_only={model_only}"
        )
    return "; ".join(notes)


def load_threshold_data(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary_df.iterrows():
        model_name = row["model"]
        source_summary = Path(row["source_summary_csv"])
        eval_root = source_summary.parent
        threshold_csv = eval_root / model_name / "macro_metrics_by_support_threshold.csv"

        if threshold_csv.exists():
            tdf = pd.read_csv(threshold_csv)
            tdf["model"] = model_name
            tdf["display_name"] = row["display_name"]
            tdf["regime"] = row["regime"]
            tdf["min_count_setting"] = row["min_count"]
            rows.append(tdf)

    if not rows:
        return pd.DataFrame(
            columns=[
                "support_threshold", "n_classes", "macro_precision", "macro_recall",
                "macro_f1", "model", "display_name", "regime", "min_count_setting"
            ]
        )
    return pd.concat(rows, ignore_index=True)


def save_tables(summary_df: pd.DataFrame, out_dir: Path, valid_only_df: pd.DataFrame) -> None:
    raw_cols = [
        "model", "display_name", "regime", "min_count", "n_samples", "n_classes",
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
        "top5_accuracy", "total_errors",
        "n_model_only_labels", "n_rebuilt_only_labels", "notes", "model_dir"
    ]
    summary_df[raw_cols].to_csv(out_dir / "combined_native_summary_all.csv", index=False, encoding="utf-8")

    present_cols = [
        "display_name", "regime", "min_count", "n_samples", "n_classes",
        "accuracy", "recall_macro", "f1_macro", "top5_accuracy",
        "n_model_only_labels", "n_rebuilt_only_labels", "notes"
    ]
    present_df = summary_df[present_cols].copy()
    present_df = present_df.sort_values(["regime", "min_count"]).reset_index(drop=True)
    present_df.to_csv(out_dir / "presentation_table_all_models.csv", index=False, encoding="utf-8")

    valid_df = valid_only_df[present_cols].copy()
    valid_df = valid_df.sort_values(["regime", "min_count"]).reset_index(drop=True)
    valid_df.to_csv(out_dir / "presentation_table_valid_models_only.csv", index=False, encoding="utf-8")

    rank_df = valid_only_df[
        ["display_name", "regime", "min_count", "accuracy", "recall_macro", "f1_macro", "top5_accuracy"]
    ].copy()
    rank_df = rank_df.sort_values(["f1_macro", "accuracy"], ascending=[False, False]).reset_index(drop=True)
    rank_df.insert(0, "rank_by_macro_f1", range(1, len(rank_df) + 1))
    rank_df.to_csv(out_dir / "ranking_by_macro_f1_valid_models.csv", index=False, encoding="utf-8")


def plot_metric_bar(summary_df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    df = summary_df.sort_values(["regime", "min_count"]).reset_index(drop=True)
    x = range(len(df))
    ax.bar(x, df[metric])
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["display_name"], rotation=25, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_main_metrics(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    df = summary_df.sort_values(["regime", "min_count"]).reset_index(drop=True)

    metrics = ["accuracy", "recall_macro", "f1_macro", "top5_accuracy"]
    width = 0.18
    x = list(range(len(df)))

    for i, metric in enumerate(metrics):
        x_pos = [v + (i - 1.5) * width for v in x]
        ax.bar(x_pos, df[metric], width=width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(df["display_name"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Main native evaluation metrics across models")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_label_mismatch(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    df = summary_df.sort_values(["regime", "min_count"]).reset_index(drop=True)

    width = 0.35
    x = list(range(len(df)))
    ax.bar([v - width / 2 for v in x], df["n_model_only_labels"], width=width, label="model-only labels")
    ax.bar([v + width / 2 for v in x], df["n_rebuilt_only_labels"], width=width, label="rebuilt-only labels")

    ax.set_xticks(x)
    ax.set_xticklabels(df["display_name"], rotation=25, ha="right")
    ax.set_ylabel("count")
    ax.set_title("Label-space mismatch diagnostics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_curves(threshold_df: pd.DataFrame, out_path: Path) -> None:
    if threshold_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for display_name, sub in threshold_df.groupby("display_name", sort=False):
        sub = sub.sort_values("support_threshold")
        ax.plot(sub["support_threshold"], sub["macro_f1"], marker="o", label=display_name)

    ax.set_xlabel("minimum support threshold for per-class aggregation")
    ax.set_ylabel("macro F1")
    ax.set_title("Macro F1 by support threshold")
    ax.set_xticks(sorted(threshold_df["support_threshold"].unique()))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_regime_best(summary_df: pd.DataFrame, out_path: Path) -> None:
    df = (
        summary_df.sort_values(["regime", "f1_macro", "accuracy"], ascending=[True, False, False])
        .groupby("regime", as_index=False)
        .first()
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = range(len(df))
    ax.bar(x, df["f1_macro"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["display_name"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("macro F1")
    ax.set_title("Best valid model per training regime")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = load_and_combine_summaries(args.summary_csv)

    excluded_models = set(args.exclude_model)
    excluded_models.update(INVALID_DEFAULT_MODELS)

    if args.include_invalid_in_main_figures:
        valid_only_df = summary_df.copy()
    else:
        valid_only_df = summary_df[~summary_df["model"].isin(excluded_models)].copy()

    threshold_df = load_threshold_data(summary_df)

    save_tables(summary_df, out_dir, valid_only_df)

    plot_label_mismatch(summary_df, out_dir / "label_mismatch_diagnostics_all_models.png")

    plot_grouped_main_metrics(valid_only_df, out_dir / "main_metrics_valid_models.png")
    plot_metric_bar(valid_only_df, "f1_macro", "Macro F1 across valid models", out_dir / "macro_f1_valid_models.png")
    plot_metric_bar(valid_only_df, "accuracy", "Accuracy across valid models", out_dir / "accuracy_valid_models.png")
    plot_metric_bar(valid_only_df, "recall_macro", "Macro recall across valid models", out_dir / "macro_recall_valid_models.png")
    plot_regime_best(valid_only_df, out_dir / "best_model_per_regime_macro_f1.png")

    if not threshold_df.empty:
        if args.include_invalid_in_main_figures:
            threshold_plot_df = threshold_df.copy()
        else:
            threshold_plot_df = threshold_df[~threshold_df["model"].isin(excluded_models)].copy()
        plot_threshold_curves(threshold_plot_df, out_dir / "macro_f1_by_support_threshold_valid_models.png")

    print(f"Saved tables and figures to: {out_dir}")


if __name__ == "__main__":
    main()
