"""Static checks for the portfolio documentation."""

from __future__ import annotations

import csv
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
ENGLISH = ROOT / "README.md"
CHINESE = ROOT / "README.zh-CN.md"
DATA_GUIDE = ROOT / "data" / "README.md"
LANGUAGE_NAV = "[English](README.md) | [简体中文](README.zh-CN.md)"
RESULT_START = "<!-- aggregate-results:start -->"
RESULT_END = "<!-- aggregate-results:end -->"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _local_targets(path: Path) -> set[str]:
    markdown = _text(path)
    links = re.findall(r"!?\[[^\]]*]\(([^)]+)\)", markdown)
    return {
        target
        for target in links
        if not re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE)
        and not target.startswith("#")
    }


def _result_rows(path: Path) -> dict[tuple[str, str], tuple[float, float, float]]:
    block = _text(path).split(RESULT_START, 1)[1].split(RESULT_END, 1)[0]
    rows: dict[tuple[str, str], tuple[float, float, float]] = {}
    for line in block.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[1] not in {"L2-NMF", "L21-NMF"}:
            continue
        rows[(cells[0], cells[1])] = tuple(float(value) for value in cells[2:])
    return rows


def _summary_aggregates() -> dict[tuple[str, str], tuple[float, float, float]]:
    grouped: dict[tuple[str, str], list[tuple[Decimal, Decimal, Decimal]]] = {}
    with (ROOT / "results" / "metrics" / "summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault((row["dataset"], row["method"]), []).append(
                (
                    Decimal(row["rre_mean"]),
                    Decimal(row["accuracy_mean"]),
                    Decimal(row["nmi_mean"]),
                )
            )
    return {
        key: tuple(
            float(
                (sum(values) / len(values)).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
            )
            for values in zip(*rows)
        )
        for key, rows in grouped.items()
    }


def test_language_navigation_and_link_parity() -> None:
    assert _text(ENGLISH).splitlines()[0] == LANGUAGE_NAV
    assert _text(CHINESE).splitlines()[0] == LANGUAGE_NAV
    counterpart_links = {"README.md", "README.zh-CN.md"}
    assert _local_targets(ENGLISH) - counterpart_links == (
        _local_targets(CHINESE) - counterpart_links
    )


def test_all_referenced_local_files_exist() -> None:
    for document in (ENGLISH, CHINESE, DATA_GUIDE):
        for target in _local_targets(document):
            assert (document.parent / target).resolve().exists(), (
                f"{document.relative_to(ROOT)} references missing {target}"
            )


def test_documented_repository_paths_and_commands_are_valid() -> None:
    expected = (
        "notebooks/robust_nmf_experiments.ipynb",
        "scripts/smoke_experiment.py",
        "scripts/generate_result_figures.py",
        "scripts/verify_result_provenance.py",
        "results/metrics/summary.csv",
        "results/metrics/PROVENANCE.md",
        "results/figures/rre_comparison.png",
        "results/figures/clustering_comparison.png",
        "docs/robust_nmf_technical_report.pdf",
    )
    assert all((ROOT / path).is_file() for path in expected)
    pyproject = _text(ROOT / "pyproject.toml")
    assert "dev = [" in pyproject and "report = [" in pyproject
    for readme in (ENGLISH, CHINESE):
        text = _text(readme)
        assert 'python -m pip install -e ".[dev,report]"' in text
        assert "python -m pytest" in text
        assert "python scripts/smoke_experiment.py" in text
        assert "python scripts/generate_result_figures.py" in text


def test_notebook_uses_documented_dataset_roots_and_is_clean() -> None:
    notebook_path = ROOT / "notebooks" / "robust_nmf_experiments.ipynb"
    raw = _text(notebook_path)
    notebook = json.loads(raw)
    source = "\n".join(
        "".join(cell.get("source", ())) for cell in notebook["cells"]
    )
    assert "ROOT / 'data' / 'ORL'" in source
    assert "ROOT / 'data' / 'CroppedYaleB'" in source
    assert "ROOT / 'data' / 'orl'" not in source
    assert "extended_yaleb" not in source
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


def test_personal_contributions_are_precise_and_bounded() -> None:
    english = _text(ENGLISH)
    required = (
        "Implemented and validated the salt-and-pepper noise generator.",
        "Implemented the L2-NMF baseline and checked its objective and update rules against the literature.",
        "Researched and organized references supporting the background and motivation.",
        "Contributed the theoretical framing and explanation of the compared methods.",
    )
    assert all(english.count(statement) == 1 for statement in required)
    section = english.split("## Team Project and My Contributions", 1)[1].split(
        "## Architecture", 1
    )[0]
    assert "four-person team project" in section
    assert "solely my work" in section
    assert "Implemented L21-NMF" not in section


def test_sensitive_context_is_absent_from_documentation() -> None:
    fragments = (
        "stu" + "dent[ _-]*id",
        "tu" + "tor",
        "team" + "mate",
        "assess" + "ment",
        "course" + "work",
        "letter[ _-]*grade",
    )
    combined = "\n".join(_text(path) for path in (ENGLISH, CHINESE, DATA_GUIDE))
    for fragment in fragments:
        assert re.search(fragment, combined, re.IGNORECASE) is None


def test_readme_result_tables_match_summary_aggregates() -> None:
    expected = _summary_aggregates()
    assert _result_rows(ENGLISH) == expected
    assert _result_rows(CHINESE) == expected
