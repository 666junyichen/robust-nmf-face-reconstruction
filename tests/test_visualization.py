import matplotlib
import pandas as pd
import pytest
from pathlib import Path

matplotlib.use("Agg")

from robust_nmf.visualization import load_summary, plot_metric_comparison


COLUMNS = [
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


def _valid_frame() -> pd.DataFrame:
    rows = []
    for dataset in ("Dataset A", "Dataset B"):
        for corruption in (0.2, 0.4):
            for method, offset in (("L2-NMF", 0.1), ("L21-NMF", 0.0)):
                rows.append(
                    {
                        "dataset": dataset,
                        "corruption": corruption,
                        "salt_ratio": 0.1,
                        "method": method,
                        "rre_mean": corruption + offset,
                        "rre_std": 0.01,
                        "accuracy_mean": 0.7 - corruption - offset,
                        "accuracy_std": 0.02,
                        "nmi_mean": 0.8 - corruption - offset,
                        "nmi_std": 0.03,
                    }
                )
    return pd.DataFrame(rows, columns=COLUMNS)


def test_load_summary_validates_and_sorts_rows(tmp_path):
    path = tmp_path / "summary.csv"
    _valid_frame().sample(frac=1, random_state=3).to_csv(path, index=False)

    loaded = load_summary(path)

    assert list(loaded.columns) == COLUMNS
    assert loaded.equals(
        loaded.sort_values(
            ["dataset", "salt_ratio", "corruption", "method"]
        ).reset_index(drop=True)
    )


def test_load_summary_rejects_missing_columns(tmp_path):
    path = tmp_path / "summary.csv"
    _valid_frame().drop(columns="nmi_std").to_csv(path, index=False)

    with pytest.raises(ValueError, match="columns"):
        load_summary(path)


def test_plot_metric_comparison_creates_headless_metric_panels():
    figure = plot_metric_comparison(_valid_frame(), metrics=("rre", "accuracy"))

    try:
        assert len(figure.axes) == 4
        labels = {axis.get_ylabel() for axis in figure.axes}
        assert {"RRE", "Accuracy"} <= labels
        assert all(axis.get_xlabel() == "Corruption fraction" for axis in figure.axes)
    finally:
        matplotlib.pyplot.close(figure)


def test_checked_in_summary_has_complete_noisy_grid():
    root = Path(__file__).resolve().parents[1]
    summary = load_summary(root / "results" / "metrics" / "summary.csv")

    assert len(summary) == 24
    assert set(summary["dataset"]) == {"ORL", "Extended YaleB"}
    assert set(summary["method"]) == {"L2-NMF", "L21-NMF"}
    assert set(summary["corruption"]) == {0.2, 0.4, 0.6}
    assert set(summary["salt_ratio"]) == {0.1, 0.7}
    assert not (summary["corruption"] == 0.0).any()
