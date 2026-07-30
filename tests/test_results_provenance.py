import csv
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts" / "verify_result_provenance.py"
VERIFIER_SPEC = importlib.util.spec_from_file_location(
    "verify_result_provenance", VERIFIER_PATH
)
if VERIFIER_SPEC is None or VERIFIER_SPEC.loader is None:
    raise ImportError(f"could not load provenance verifier from {VERIFIER_PATH}")
verify_result_provenance = importlib.util.module_from_spec(VERIFIER_SPEC)
VERIFIER_SPEC.loader.exec_module(verify_result_provenance)


RECORDED_REPORT_SHA256 = (
    "f731e87a11f0456adf354b88532bf37f42247f9833f58f68dabc53e29aea7493"
)


def _write_synthetic_summary(path: Path) -> list[dict[str, str]]:
    rows = []
    index = 0
    for dataset in ("ORL", "Extended YaleB"):
        for salt_ratio in ("0.1", "0.7"):
            for corruption in ("0.2", "0.4", "0.6"):
                for method in ("L21-NMF", "L2-NMF"):
                    base = 100 + index * 6
                    rows.append(
                        {
                            "dataset": dataset,
                            "corruption": corruption,
                            "salt_ratio": salt_ratio,
                            "method": method,
                            "rre_mean": f"0.{base:03d}",
                            "rre_std": f"0.{base + 1:03d}",
                            "accuracy_mean": f"0.{base + 2:03d}",
                            "accuracy_std": f"0.{base + 3:03d}",
                            "nmi_mean": f"0.{base + 4:03d}",
                            "nmi_std": f"0.{base + 5:03d}",
                        }
                    )
                    index += 1

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=verify_result_provenance.EXPECTED_COLUMNS
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _synthetic_report_text(rows: list[dict[str, str]]) -> str:
    sections = []
    for table_number, dataset in ((3, "ORL"), (4, "Extended YaleB")):
        lines = [f"Table {table_number}: Detailed results on {dataset}"]
        for row in (item for item in rows if item["dataset"] == dataset):
            method = (
                "L 2,1-norm" if row["method"] == "L21-NMF" else "L2-norm"
            )
            prefix = (
                f"({row['corruption']},{row['salt_ratio']})"
                if row["method"] == "L21-NMF"
                else ""
            )
            tokens = " ".join(
                f"{row[f'{metric}_mean']}±{row[f'{metric}_std']}"
                for metric in ("rre", "accuracy", "nmi")
            )
            lines.append(f"{prefix}{method} {tokens}")
        sections.append("\n".join(lines))
    return "\n".join(sections) + "\nDiscussion. The L 2,1-norm NMF is robust."


def _split_line_table_text() -> str:
    return """Table 3: Detailed results on ORL
(0.2, 0.1)
L 2,1-norm
0.100卤0.101
0.102卤0.103
0.104卤0.105
L2-norm
0.106卤0.107
0.108卤0.109
0.110卤0.111
(0.4,0.7)
L2, 1-norm
0.200卤0.201 0.202卤0.203
0.204卤0.205
L 2-norm
0.206卤0.207
0.208卤0.209
0.210卤0.211
Table 4: Detailed results on Extended YaleB
"""


def test_parser_accumulates_split_line_metric_tokens_until_row_boundaries():
    parsed = verify_result_provenance._parse_table_rows(
        _split_line_table_text(), 3
    )

    assert parsed == {
        ("0.2", "0.1", "L21-NMF"): (
            "0.100",
            "0.101",
            "0.102",
            "0.103",
            "0.104",
            "0.105",
        ),
        ("0.2", "0.1", "L2-NMF"): (
            "0.106",
            "0.107",
            "0.108",
            "0.109",
            "0.110",
            "0.111",
        ),
        ("0.4", "0.7", "L21-NMF"): (
            "0.200",
            "0.201",
            "0.202",
            "0.203",
            "0.204",
            "0.205",
        ),
        ("0.4", "0.7", "L2-NMF"): (
            "0.206",
            "0.207",
            "0.208",
            "0.209",
            "0.210",
            "0.211",
        ),
    }


def test_parser_rejects_incomplete_split_line_row_at_method_boundary():
    text = _split_line_table_text().replace(
        "0.104卤0.105\nL2-norm", "0.104\nL2-norm", 1
    )

    with pytest.raises(
        ValueError,
        match=r"Table 3.*exactly six metric tokens.*found 5.*L21-NMF",
    ):
        verify_result_provenance._parse_table_rows(text, 3)


def test_parser_rejects_extra_split_line_token_at_pair_boundary():
    text = _split_line_table_text().replace(
        "0.110卤0.111\n(0.4,0.7)",
        "0.110卤0.111\n0.112\n(0.4,0.7)",
    )

    with pytest.raises(
        ValueError,
        match=r"Table 3.*exactly six metric tokens.*found 7.*L2-NMF",
    ):
        verify_result_provenance._parse_table_rows(text, 3)


def test_manifest_identifies_archived_snapshot_by_digest_without_private_locator():
    manifest = (ROOT / "results" / "metrics" / "PROVENANCE.md").read_text(
        encoding="utf-8"
    )
    digest = re.search(r"\b[0-9a-f]{64}\b", manifest)

    assert digest is not None
    assert digest.group() == RECORDED_REPORT_SHA256
    assert "final archived team technical report snapshot" in manifest.lower()
    assert "Tables 3–4" in manifest
    assert "--report" in manifest
    assert "--csv" in manifest
    assert "\\" not in manifest


def test_verifier_accepts_valid_synthetic_extracted_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    report = tmp_path / "snapshot.pdf"
    report.write_bytes(b"synthetic archived report")
    summary = tmp_path / "summary.csv"
    rows = _write_synthetic_summary(summary)
    monkeypatch.setattr(
        verify_result_provenance,
        "extract_pdf_text",
        lambda _path: _synthetic_report_text(rows),
    )

    verify_result_provenance.verify_provenance(
        report,
        summary,
        hashlib.sha256(report.read_bytes()).hexdigest(),
    )


def test_verifier_reports_the_unmatched_csv_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    report = tmp_path / "snapshot.pdf"
    report.write_bytes(b"synthetic archived report")
    summary = tmp_path / "summary.csv"
    rows = _write_synthetic_summary(summary)
    text = _synthetic_report_text(rows).replace("0.100±0.101", "0.999±0.998", 1)
    monkeypatch.setattr(
        verify_result_provenance, "extract_pdf_text", lambda _path: text
    )

    with pytest.raises(ValueError, match=r"ORL.*0\.2.*0\.1.*L21-NMF"):
        verify_result_provenance.verify_provenance(
            report,
            summary,
            hashlib.sha256(report.read_bytes()).hexdigest(),
        )


def test_notebook_prerequisites_precede_first_code_import():
    notebook = json.loads(
        (ROOT / "notebooks" / "robust_nmf_experiments.ipynb").read_text(
            encoding="utf-8"
        )
    )
    first_code_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
    )
    earlier_markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"][:first_code_index]
        if cell["cell_type"] == "markdown"
    )

    assert 'python -m pip install -e ".[dev,report]"' in earlier_markdown
    assert "repository root" in earlier_markdown.lower()
    assert "launch jupyter" in earlier_markdown.lower()
    assert not any(
        "pip install" in "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_notebook_documents_historical_clean_data_refit_asymmetry():
    notebook = json.loads(
        (ROOT / "notebooks" / "robust_nmf_experiments.ipynb").read_text(
            encoding="utf-8"
        )
    )
    limitations = next(
        cell for cell in notebook["cells"] if cell.get("id") == "limitations"
    )
    text = "".join(limitations["source"]).lower()

    assert "l2,1-nmf" in text
    assert "coefficient matrix `h` on clean data" in text
    assert "keeping `w` fixed" in text
    assert "l2-nmf" in text
    assert "factors learned on noisy data" in text
    assert "without clean-data refitting" in text
    assert "not a perfectly symmetric protocol" in text


def test_figure_readme_links_detailed_provenance():
    text = (ROOT / "results" / "figures" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "../metrics/PROVENANCE.md" in text
    lowered = text.lower()
    assert "archived team technical report" in lowered
    assert "not generated by the current refactored implementation" in lowered
    assert "noisy-data factors without clean-data refitting" in lowered
