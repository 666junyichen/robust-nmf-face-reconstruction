"""Validation and plotting helpers for completed experiment summaries."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


SUMMARY_COLUMNS = [
    "dataset",
    "corruption",
    "salt_ratio",
    "method",
    "rre_mean",
    "rre_std",
    "accuracy_mean",
    "accuracy_std",
    "nmi_mean",
    "nmi_std",
]

_METRIC_LABELS = {"rre": "RRE", "accuracy": "Accuracy", "nmi": "NMI"}
_NUMERIC_COLUMNS = SUMMARY_COLUMNS[1:3] + SUMMARY_COLUMNS[4:]


def validate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a validated, sorted copy of a completed-results summary."""
    if list(frame.columns) != SUMMARY_COLUMNS:
        raise ValueError(f"summary columns must be exactly {SUMMARY_COLUMNS}")
    if frame.empty:
        raise ValueError("summary must contain at least one row")

    result = frame.copy()
    for column in _NUMERIC_COLUMNS:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must be numeric") from exc
    if not np.isfinite(result[_NUMERIC_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("summary numeric values must be finite")
    if result[["dataset", "method"]].isna().any().any():
        raise ValueError("dataset and method values must be present")
    if (result[["dataset", "method"]].astype(str).apply(lambda col: col.str.strip() == "")).any().any():
        raise ValueError("dataset and method values must be nonempty")
    if not result["corruption"].between(0.0, 1.0).all():
        raise ValueError("corruption must be in [0, 1]")
    if not result["salt_ratio"].between(0.0, 1.0).all():
        raise ValueError("salt_ratio must be in [0, 1]")
    for metric in ("accuracy", "nmi"):
        if not result[f"{metric}_mean"].between(0.0, 1.0).all():
            raise ValueError(f"{metric}_mean must be in [0, 1]")
    if (result[["rre_mean", "rre_std", "accuracy_std", "nmi_std"]] < 0).any().any():
        raise ValueError("metric means and standard deviations must be nonnegative")
    if result.duplicated(["dataset", "corruption", "salt_ratio", "method"]).any():
        raise ValueError("summary contains duplicate experiment rows")
    return result.sort_values(
        ["dataset", "salt_ratio", "corruption", "method"]
    ).reset_index(drop=True)


def load_summary(path: str | Path) -> pd.DataFrame:
    """Load and validate a completed-results CSV."""
    return validate_summary(pd.read_csv(Path(path)))


def plot_metric_comparison(
    frame: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("rre",),
) -> Figure:
    """Plot method-versus-corruption panels, separated by dataset and metric."""
    data = validate_summary(frame)
    requested = tuple(metrics)
    if not requested or any(metric not in _METRIC_LABELS for metric in requested):
        raise ValueError(f"metrics must be selected from {tuple(_METRIC_LABELS)}")

    datasets = tuple(data["dataset"].drop_duplicates())
    figure, axes = plt.subplots(
        len(datasets),
        len(requested),
        figsize=(5.2 * len(requested), 3.8 * len(datasets)),
        squeeze=False,
        sharex=True,
    )
    colors = dict(zip(data["method"].drop_duplicates(), plt.cm.Set2.colors))
    line_styles = ("-", "--", ":", "-.")
    salt_values = tuple(sorted(data["salt_ratio"].unique()))

    for row, dataset in enumerate(datasets):
        dataset_rows = data[data["dataset"] == dataset]
        for column, metric in enumerate(requested):
            axis = axes[row, column]
            for method in data["method"].drop_duplicates():
                for salt_index, salt_ratio in enumerate(salt_values):
                    subset = dataset_rows[
                        (dataset_rows["method"] == method)
                        & (dataset_rows["salt_ratio"] == salt_ratio)
                    ].sort_values("corruption")
                    if subset.empty:
                        continue
                    axis.errorbar(
                        subset["corruption"],
                        subset[f"{metric}_mean"],
                        yerr=subset[f"{metric}_std"],
                        marker="o",
                        capsize=3,
                        color=colors[method],
                        linestyle=line_styles[salt_index % len(line_styles)],
                        label=f"{method}; salt={salt_ratio:g}",
                    )
            axis.set_title(str(dataset))
            axis.set_xlabel("Corruption fraction")
            axis.set_ylabel(_METRIC_LABELS[metric])
            axis.grid(alpha=0.25)
            axis.legend(fontsize="small")

    figure.tight_layout()
    return figure


__all__ = ["SUMMARY_COLUMNS", "load_summary", "plot_metric_comparison", "validate_summary"]
