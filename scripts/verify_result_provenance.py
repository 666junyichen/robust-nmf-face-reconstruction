"""Verify that stored aggregate metrics match the archived report tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import unicodedata
from itertools import product
from pathlib import Path


EXPECTED_COLUMNS = [
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
DATASETS = ("ORL", "Extended YaleB")
CORRUPTIONS = ("0.2", "0.4", "0.6")
SALT_RATIOS = ("0.1", "0.7")
METHODS = ("L21-NMF", "L2-NMF")
METRIC_COLUMNS = EXPECTED_COLUMNS[4:]
THREE_DECIMAL = re.compile(r"^(?:0|1)\.\d{3}$")
SHA256 = re.compile(r"\b[0-9a-f]{64}\b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_recorded_digest(manifest_path: Path) -> str:
    digests = SHA256.findall(manifest_path.read_text(encoding="utf-8"))
    if len(digests) != 1:
        raise ValueError(
            f"{manifest_path}: expected exactly one recorded SHA-256 digest, "
            f"found {len(digests)}"
        )
    return digests[0]


def load_and_validate_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"{path}: schema mismatch; expected {EXPECTED_COLUMNS}, "
                f"found {reader.fieldnames}"
            )
        rows = list(reader)

    if len(rows) != 24:
        raise ValueError(f"{path}: expected 24 data rows, found {len(rows)}")

    observed_keys: set[tuple[str, str, str, str]] = set()
    for line_number, row in enumerate(rows, start=2):
        key = (
            row["dataset"],
            row["corruption"],
            row["salt_ratio"],
            row["method"],
        )
        if key in observed_keys:
            raise ValueError(f"{path}:{line_number}: duplicate combination {key}")
        observed_keys.add(key)

        for column in METRIC_COLUMNS:
            raw = row[column]
            if not THREE_DECIMAL.fullmatch(raw):
                raise ValueError(
                    f"{path}:{line_number}: {column}={raw!r} must use "
                    "three-decimal rounded precision"
                )
            value = float(raw)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{path}:{line_number}: {column}={raw!r} is outside [0, 1]"
                )

    expected_keys = set(product(DATASETS, CORRUPTIONS, SALT_RATIOS, METHODS))
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(observed_keys - expected_keys)
        raise ValueError(
            f"{path}: invalid experiment combinations; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return rows


def extract_pdf_text(path: Path) -> str:
    try:
        import fitz
    except ImportError:
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError(
                'PDF extraction requires the "report" dependency group; run '
                'python -m pip install -e ".[dev,report]"'
            ) from error
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)

    with fitz.open(path) as document:
        return "\n".join(page.get_text() for page in document)


def _table_block(text: str, table_number: int) -> str:
    start = re.search(
        rf"table\s*{table_number}\s*:", text, flags=re.IGNORECASE
    )
    if start is None:
        raise ValueError(f"extracted PDF text does not contain Table {table_number}")
    next_table = re.search(
        rf"table\s*{table_number + 1}\s*:",
        text[start.end() :],
        flags=re.IGNORECASE,
    )
    end = (
        start.end() + next_table.start()
        if next_table is not None
        else len(text)
    )
    return text[start.start() : end]


def _parse_table_rows(
    text: str, table_number: int
) -> dict[tuple[str, str, str], tuple[str, ...]]:
    normalized_lines = [
        re.sub(r"\s+", " ", unicodedata.normalize("NFKC", line).lower()).strip()
        for line in _table_block(text, table_number).splitlines()
    ]
    parsed: dict[tuple[str, str, str], tuple[str, ...]] = {}
    current_pair: tuple[str, str] | None = None
    pending_key: tuple[str, str, str] | None = None
    pending_tokens: list[str] = []

    def finish_pending_row() -> None:
        nonlocal pending_key, pending_tokens
        if pending_key is None:
            return
        if len(pending_tokens) != 6:
            raise ValueError(
                f"Table {table_number}: expected exactly six metric tokens, "
                f"found {len(pending_tokens)} for {pending_key}"
            )
        if pending_key in parsed:
            raise ValueError(
                f"Table {table_number}: duplicate extracted row {pending_key}"
            )
        parsed[pending_key] = tuple(pending_tokens)
        pending_key = None
        pending_tokens = []

    for line in normalized_lines:
        compact = re.sub(r"\s+", "", line)
        pair = re.search(r"\((0\.[246]),(0\.[17])\)", compact)
        if pair:
            finish_pending_row()
            if len(parsed) == 12:
                break
            current_pair = (pair.group(1), pair.group(2))

        if "l2,1-norm" in compact:
            method = "L21-NMF"
        elif "l2-norm" in compact:
            method = "L2-NMF"
        else:
            method = None

        if method is not None:
            finish_pending_row()
            if len(parsed) == 12:
                break
            if current_pair is None:
                continue
            pending_key = (*current_pair, method)

        if pending_key is not None:
            pending_tokens.extend(
                re.findall(r"(?<![\d.])(?:0|1)\.\d{3}(?!\d)", line)
            )

    finish_pending_row()
    return parsed


def verify_table_values(text: str, rows: list[dict[str, str]]) -> None:
    parsed_by_dataset = {
        "ORL": _parse_table_rows(text, 3),
        "Extended YaleB": _parse_table_rows(text, 4),
    }
    mismatches = []
    for row in rows:
        key = (row["corruption"], row["salt_ratio"], row["method"])
        expected = tuple(row[column] for column in METRIC_COLUMNS)
        actual = parsed_by_dataset[row["dataset"]].get(key)
        if actual != expected:
            mismatches.append(
                f"{row['dataset']} corruption={row['corruption']} "
                f"salt_ratio={row['salt_ratio']} method={row['method']}: "
                f"CSV={expected}, report={actual}"
            )
    if mismatches:
        raise ValueError(
            "CSV rows did not match Tables 3–4:\n  " + "\n  ".join(mismatches)
        )


def verify_provenance(
    report_path: Path, csv_path: Path, expected_digest: str
) -> None:
    actual_digest = sha256_file(report_path)
    if actual_digest != expected_digest:
        raise ValueError(
            "report SHA-256 mismatch: "
            f"expected {expected_digest}, found {actual_digest}"
        )
    rows = load_and_validate_csv(csv_path)
    verify_table_values(extract_pdf_text(report_path), rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify archived report bytes and stored result-table values."
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected_digest = load_recorded_digest(
        root / "results" / "metrics" / "PROVENANCE.md"
    )
    try:
        verify_provenance(args.report, args.csv, expected_digest)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"provenance verification failed: {error}\n")
    print(
        "PASS: report digest, CSV contract, and all Tables 3–4 metric tokens match"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
