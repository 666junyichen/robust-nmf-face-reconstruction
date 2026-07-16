"""Regenerate curated figures from the stored completed-results summary."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robust_nmf.visualization import load_summary, plot_metric_comparison


def main() -> int:
    summary = load_summary(ROOT / "results" / "metrics" / "summary.csv")
    output_dir = ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    specifications = {
        "rre_comparison.png": ("rre",),
        "clustering_comparison.png": ("accuracy", "nmi"),
    }
    for filename, metrics in specifications.items():
        figure = plot_metric_comparison(summary, metrics=metrics)
        figure.savefig(output_dir / filename, dpi=160, bbox_inches="tight")
        matplotlib.pyplot.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
