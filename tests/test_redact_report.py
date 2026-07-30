from __future__ import annotations

import importlib.util
from pathlib import Path

import fitz
from pypdf import PdfReader


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "redact_report.py"
SPEC = importlib.util.spec_from_file_location("redact_report", SCRIPT_PATH)
assert SPEC and SPEC.loader
redact_report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(redact_report)


def _save_synthetic_private_report(path: Path) -> None:
    doc = fitz.open()
    first = doc.new_page()
    first.insert_text((72, 100), "Assignment 1")
    first.insert_text((72, 140), "Tutors: Example Person")
    first.insert_text((72, 160), "Group members: Sample Student (12345678)")
    first.insert_text((72, 260), "Abstract")
    first.insert_text((72, 290), "Technical abstract remains legible.")

    methods = doc.new_page()
    methods.insert_textbox(
        fitz.Rect(72, 100, 500, 170),
        "We used two benchmark face datasets: the ORL and Extended YaleB datasets provided "
        "with the assignment package [8]. Each dataset consists of grayscale facial images "
        "under varying illumination and expressions. Our pre-processing procedure standardizes "
        "image format and ensures numerical stability for matrix factorization.",
        fontsize=10,
    )
    methods.insert_text((72, 230), "Technical result: robust loss reduced reconstruction error.")
    methods.insert_text((72, 280), "University of Example. COMP1234 assignment 1. Course handout.")
    methods.insert_link(
        {"kind": fitz.LINK_URI, "from": fitz.Rect(72, 300, 180, 320), "uri": "https://example.edu"}
    )

    appendix = doc.new_page()
    appendix.insert_text((72, 100), "B Team Contribution Details")
    appendix.insert_text((72, 130), "Name SID ContributionDetails")
    appendix.insert_text((72, 160), "Sample Student 12345678 100%")

    doc.set_metadata({"title": "Assignment 1", "author": "Sample Student"})
    doc.embfile_add("private.txt", b"private")
    appendix_xref = appendix.xref
    doc.xref_set_key(
        doc.pdf_catalog(),
        "Names",
        (
            "<< /Dests << /Names [ "
            f"(appendix.B) [{appendix_xref} 0 R /Fit] "
            f"(cite.COMP1234_Assignment1) [{appendix_xref} 0 R /Fit] "
            "] >> >>"
        ),
    )
    doc.xref_set_key(doc.pdf_catalog(), "OpenAction", f"[{appendix_xref} 0 R /Fit]")
    doc.save(path)
    doc.close()


def _save_clean_report(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Robust matrix factorization technical report")
    doc.set_metadata({})
    doc.save(path)
    doc.close()


def test_real_redaction_removes_content_and_document_extras(tmp_path: Path) -> None:
    source = tmp_path / "private.pdf"
    output = tmp_path / "public.pdf"
    _save_synthetic_private_report(source)

    redact_report.redact_report(source, output)

    reader = PdfReader(output)
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    lowered = extracted.lower()
    assert len(reader.pages) == 2
    assert "sample student" not in lowered
    assert "12345678" not in extracted
    assert "assignment" not in lowered
    assert "university of example" not in lowered
    assert "technical abstract remains legible" in lowered
    assert "technical result" in lowered
    assert not {k: v for k, v in (reader.metadata or {}).items() if v}

    with fitz.open(output) as doc:
        raw_streams = b"".join(
            doc.xref_stream(xref) or b""
            for page in doc
            for xref in page.get_contents()
        ).lower()
        assert b"sample student" not in raw_streams
        assert b"12345678" not in raw_streams
        assert not doc.embfile_names()
        assert all(not list(page.annots() or ()) for page in doc)
        assert all(not page.get_links() for page in doc)
        catalog = doc.pdf_catalog()
        assert set(doc.xref_get_keys(catalog)) <= {"Type", "Pages", "Info"}
        assert doc.xref_get_key(catalog, "Info")[1] == "null"
        all_xref_material = []
        for xref in range(1, doc.xref_length()):
            all_xref_material.append(doc.xref_object(xref, compressed=False))
            if doc.xref_is_stream(xref):
                all_xref_material.append((doc.xref_stream(xref) or b"").decode("latin-1", "ignore"))
        all_xref_text = "\n".join(all_xref_material).lower()
        assert "appendix.b" not in all_xref_text
        assert "comp1234" not in all_xref_text
        assert "12345678" not in all_xref_text


def test_audit_detects_generic_sensitive_patterns(tmp_path: Path) -> None:
    path = tmp_path / "candidate.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(
        fitz.Rect(72, 72, 520, 200),
        "University of Example - COMP1234\n"
        "Tutor: Example Person\n"
        "Group members: A Student 123456789\n"
        "AI disclosure and submission instructions\n"
        "Team Contribution Details",
        fontsize=11,
    )
    doc.save(path)
    doc.close()

    kinds = {finding.kind for finding in redact_report.audit_pdf(path)}
    assert {
        "institution",
        "course-code",
        "tutor-or-staff",
        "group-members",
        "student-id",
        "ai-disclosure",
        "submission-or-marking",
        "contribution-heading",
    } <= kinds


def test_audit_detects_sensitive_orphan_xref_stream(tmp_path: Path) -> None:
    base = tmp_path / "base.pdf"
    leaked = tmp_path / "leaked.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "Public technical result")
    hidden = doc.new_page()
    hidden.insert_text((72, 100), "Team Contribution Details")
    hidden.insert_text((72, 130), "Sample Student 12345678")
    doc.save(base)
    doc.close()

    with fitz.open(base) as doc:
        doc.delete_page(1)
        doc.save(leaked, garbage=0, clean=False, deflate=True)

    assert fitz.open(leaked).page_count == 1
    findings = redact_report.audit_pdf(leaked)
    assert any(
        finding.location.startswith("xref ")
        and finding.kind in {"student-id", "contribution-heading"}
        for finding in findings
    )


def test_audit_only_returns_status_without_writing(tmp_path: Path) -> None:
    clean = tmp_path / "clean.pdf"
    dirty = tmp_path / "dirty-assignment.pdf"
    nonexistent_output = tmp_path / "unused.pdf"
    _save_clean_report(clean)
    _save_clean_report(dirty)

    assert redact_report.main(["--input", str(clean), "--audit-only"]) == 0
    assert (
        redact_report.main(["--input", str(dirty), "--audit-only", "--output", str(nonexistent_output)])
        == 1
    )
    assert not nonexistent_output.exists()


def test_rendering_is_optional(tmp_path: Path) -> None:
    clean = tmp_path / "clean.pdf"
    render_dir = tmp_path / "render"
    _save_clean_report(clean)

    assert redact_report.main(["--input", str(clean), "--audit-only"]) == 0
    assert not render_dir.exists()
    assert (
        redact_report.main(
            ["--input", str(clean), "--audit-only", "--render-dir", str(render_dir)]
        )
        == 0
    )
    assert (render_dir / "page-001.png").is_file()
    assert (render_dir / "contact-sheet.png").is_file()
