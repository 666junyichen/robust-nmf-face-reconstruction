"""Create and audit a public, identity-free technical report PDF.

The transformation uses PDF redaction annotations followed by
``Page.apply_redactions``. It intentionally does not use cosmetic overlays.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import fitz
from PIL import Image, ImageDraw


PUBLIC_TITLE = "Robust NMF Face Reconstruction Under Pixel Corruption"
PUBLIC_SUBTITLE = "Team Technical Report"

STUDENT_ID_RE = re.compile(r"(?<![\d.])\d{8,9}(?![\d.])")
COURSE_CODE_RE = re.compile(r"\b(?:[A-Z]{4,8}|[A-Z][a-z]{3,7})\s*[-_]?\s*\d{4}\b")
DOMAIN_RE = re.compile(r"\b[\w.-]+\.(?:edu|edu\.au|ac\.[a-z]{2})\b", re.IGNORECASE)
ALLOWED_CATALOG_KEYS = {"Type", "Pages", "Info"}

TEXT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("institution", re.compile(r"\b(?:university|college|school)\s+of\b", re.IGNORECASE)),
    ("tutor-or-staff", re.compile(r"\b(?:tutors?|lecturers?|teaching\s+staff)\s*:", re.IGNORECASE)),
    ("group-members", re.compile(r"\bgroup\s+members?\s*:", re.IGNORECASE)),
    ("assignment-branding", re.compile(r"\b(?:assignment|assessment)\b", re.IGNORECASE)),
    (
        "ai-disclosure",
        re.compile(
            r"\b(?:artificial\s+intelligence|generative\s+AI|AI\s+(?:use|disclosure)|ChatGPT)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "submission-or-marking",
        re.compile(r"\b(?:submission|marking\s+(?:guide|criteria)|grading\s+rubric)\b", re.IGNORECASE),
    ),
    (
        "contribution-heading",
        re.compile(r"\b(?:team|group|member)\s+contribution(?:s|\s+details)?\b", re.IGNORECASE),
    ),
    (
        "identity-table-heading",
        re.compile(r"\bname\s+(?:sid|student\s+id)\s+contribution", re.IGNORECASE),
    ),
)


class Finding(NamedTuple):
    kind: str
    location: str
    context: str


def _normalized_text(text: str) -> str:
    # Join line-wrapped hyphenation so "assign-\nment" remains auditable.
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _context(text: str, start: int, end: int, radius: int = 55) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    return text[lo:hi].replace("\n", " ").strip()


def _iter_metadata_values(doc: fitz.Document) -> Iterable[tuple[str, str]]:
    ignored = {"format", "encryption"}
    for key, value in (doc.metadata or {}).items():
        if key not in ignored and value:
            yield key, str(value)


def _scan_sensitive_text(text: str, location: str) -> list[Finding]:
    normalized = _normalized_text(text)
    findings: list[Finding] = []
    for kind, pattern in (
        ("student-id", STUDENT_ID_RE),
        ("course-code", COURSE_CODE_RE),
        ("school-domain", DOMAIN_RE),
        *TEXT_PATTERNS,
    ):
        for match in pattern.finditer(normalized):
            findings.append(
                Finding(kind, location, _context(normalized, match.start(), match.end()))
            )
    return findings


def _pdf_syntax_as_text(text: str) -> str:
    """Make common PDF string/TJ syntax auditable without parsing binary assets."""

    def decode_hex_string(match: re.Match[str]) -> str:
        compact = re.sub(r"\s+", "", match.group(1))
        if len(compact) % 2:
            compact += "0"
        try:
            return bytes.fromhex(compact).decode("latin-1", "replace")
        except ValueError:
            return match.group(0)

    text = re.sub(r"<([0-9A-Fa-f\s]{2,})>", decode_hex_string, text)

    def decode_octal(match: re.Match[str]) -> str:
        return chr(int(match.group(1), 8))

    text = re.sub(r"\\([0-7]{1,3})", decode_octal, text)
    text = re.sub(r"\\([()\\])", r"\1", text)

    # Negative TJ adjustments usually represent visible word spacing; small
    # positive adjustments join glyph fragments within a word.
    def join_tj_fragments(match: re.Match[str]) -> str:
        return " " if float(match.group(1)) < -100 else ""

    text = re.sub(
        r"\)\s*(-?\d+(?:\.\d+)?)\s*\(",
        join_tj_fragments,
        text,
    )
    return _normalized_text(re.sub(r"[\[\]()<>{}/]", " ", text))


def _is_auditable_stream(object_text: str, data: bytes) -> bool:
    if not data:
        return False
    if re.search(
        r"/Subtype\s*/Image|/FontFile\d?\b|/Subtype\s*/(?:Type1C|CIDFontType0C)\b",
        object_text,
    ):
        return False
    if b"begincmap" in data[:4096].lower():
        return False
    if b"BT" in data and (b"TJ" in data or b"Tj" in data):
        return True
    sample = data[: min(len(data), 16384)]
    printable = sum(byte in (9, 10, 13) or 32 <= byte <= 126 for byte in sample)
    return printable / len(sample) >= 0.88


def _object_strings_for_audit(object_text: str) -> str:
    """Extract human-bearing PDF strings/names, excluding structural numbers."""
    literal_strings = re.findall(r"\((?:\\.|[^\\)])*\)", object_text)
    hex_strings = re.findall(r"<[0-9A-Fa-f\s]{2,}>", object_text)
    pdf_names = re.findall(r"/[A-Za-z][^\s<>\[\]()]+", object_text)
    return _pdf_syntax_as_text(" ".join((*literal_strings, *hex_strings, *pdf_names)))


def _audit_all_xrefs(doc: fitz.Document) -> list[Finding]:
    findings: list[Finding] = []
    for xref in range(1, doc.xref_length()):
        location = f"xref {xref}"
        object_text = doc.xref_object(xref, compressed=False)
        findings.extend(_scan_sensitive_text(_object_strings_for_audit(object_text), location))
        if not doc.xref_is_stream(xref):
            continue
        data = doc.xref_stream(xref) or b""
        if _is_auditable_stream(object_text, data):
            decoded = data.decode("latin-1", "replace")
            findings.extend(_scan_sensitive_text(_pdf_syntax_as_text(decoded), location))
    return findings


def audit_pdf(path: str | Path) -> list[Finding]:
    """Return all public-release policy violations with page/context details."""
    pdf_path = Path(path)
    findings: list[Finding] = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            text = _normalized_text(page.get_text())
            location = f"page {page_index + 1}"
            findings.extend(_scan_sensitive_text(text, location))

            annotations = list(page.annots() or ())
            if annotations:
                findings.append(
                    Finding("annotation", location, f"{len(annotations)} annotation(s) remain")
                )
            links = page.get_links()
            if links:
                findings.append(Finding("link", location, f"{len(links)} link(s) remain"))

        for key, value in _iter_metadata_values(doc):
            findings.append(Finding("metadata", f"metadata/{key}", value[:140]))

        xml_metadata = doc.get_xml_metadata()
        if xml_metadata.strip():
            findings.append(Finding("xmp-metadata", "document catalog", xml_metadata[:140]))

        attachment_names = doc.embfile_names()
        if attachment_names:
            findings.append(
                Finding("attachment", "document catalog", ", ".join(attachment_names[:5]))
            )

        toc = doc.get_toc(simple=True)
        if toc:
            findings.append(Finding("outline", "document catalog", f"{len(toc)} outline item(s)"))

        catalog = doc.pdf_catalog()
        for key in doc.xref_get_keys(catalog):
            key_type, key_value = doc.xref_get_key(catalog, key)
            if key not in ALLOWED_CATALOG_KEYS:
                findings.append(
                    Finding("catalog-key", f"catalog/{key}", f"{key_type}: {key_value[:140]}")
                )

        findings.extend(_audit_all_xrefs(doc))

    filename = pdf_path.name
    for kind, pattern in (
        ("student-id-in-filename", STUDENT_ID_RE),
        ("course-code-in-filename", COURSE_CODE_RE),
        ("assignment-in-filename", re.compile(r"assignment|assessment", re.IGNORECASE)),
    ):
        match = pattern.search(filename)
        if match:
            findings.append(Finding(kind, "filename", filename))
    return findings


def _redact_rect(page: fitz.Page, rect: fitz.Rect) -> None:
    page.add_redact_annot(rect, fill=(1, 1, 1))


def _sensitive_block_replacement(text: str) -> str | None:
    normalized = _normalized_text(text).lower()
    if "provided with the assignment package" in normalized:
        return (
            "We used two benchmark face datasets: ORL and Extended YaleB, obtained from standard "
            "benchmark distributions. Each contains grayscale facial images under varying "
            "illumination and expressions. Pre-processing standardizes image format and ensures "
            "numerical stability for matrix factorization."
        )
    if "following the assignment guideline" in normalized:
        return (
            "Figure 1: Examples of salt-and-pepper noise at different intensity levels (p, r)."
        )
    return None


def _redact_first_page_identity(page: fitz.Page) -> None:
    abstract_rects = page.search_for("Abstract")
    if not abstract_rects:
        raise ValueError("First page has no Abstract heading; refusing an unsafe layout guess")
    cutoff = min(rect.y0 for rect in abstract_rects) - 4
    identity_rect = fitz.Rect(72, 72, page.rect.width - 72, cutoff)
    _redact_rect(page, identity_rect)
    page.apply_redactions(
        images=fitz.PDF_REDACT_IMAGE_NONE,
        graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        text=fitz.PDF_REDACT_TEXT_REMOVE,
    )

    title_box = fitz.Rect(82, 102, page.rect.width - 82, 180)
    subtitle_box = fitz.Rect(82, 190, page.rect.width - 82, 216)
    title_result = page.insert_textbox(
        title_box,
        PUBLIC_TITLE,
        fontname="hebo",
        fontsize=17,
        color=(0.08, 0.12, 0.18),
        align=fitz.TEXT_ALIGN_CENTER,
    )
    subtitle_result = page.insert_textbox(
        subtitle_box,
        PUBLIC_SUBTITLE,
        fontname="helv",
        fontsize=11,
        color=(0.25, 0.28, 0.33),
        align=fitz.TEXT_ALIGN_CENTER,
    )
    if title_result < 0 or subtitle_result < 0:
        raise ValueError("Neutral first-page title did not fit its reserved area")


def _redact_sensitive_blocks(page: fitz.Page) -> None:
    replacements: list[tuple[fitz.Rect, str, float]] = []
    for block in page.get_text("blocks"):
        rect = fitz.Rect(block[:4])
        text = block[4]
        normalized = _normalized_text(text)
        replacement = _sensitive_block_replacement(text)
        is_course_reference = bool(
            re.search(r"\buniversity\b", normalized, re.IGNORECASE)
            and re.search(r"\bcourse\s+handout\b", normalized, re.IGNORECASE)
        )
        if replacement is not None or is_course_reference:
            padded = fitz.Rect(rect.x0 - 1, rect.y0 - 1, rect.x1 + 1, rect.y1 + 1)
            _redact_rect(page, padded)
            if replacement is not None:
                font_size = 9 if normalized.lower().startswith("figure") else 9.5
                replacements.append((padded, replacement, font_size))

    if not replacements and not any(page.first_annot for _ in [0]):
        return

    # apply_redactions removes the original text operators from the content stream.
    page.apply_redactions(
        images=fitz.PDF_REDACT_IMAGE_NONE,
        graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        text=fitz.PDF_REDACT_TEXT_REMOVE,
    )
    for rect, text, font_size in replacements:
        result = page.insert_textbox(
            rect,
            text,
            fontname="tiro",
            fontsize=font_size,
            lineheight=1.18,
            color=(0, 0, 0),
            align=fitz.TEXT_ALIGN_JUSTIFY if font_size > 9 else fitz.TEXT_ALIGN_LEFT,
        )
        if result < 0:
            raise ValueError(f"Replacement text did not fit on page {page.number + 1}")


def _remove_document_extras(doc: fitz.Document) -> None:
    for page in doc:
        for annot in list(page.annots() or ()):
            page.delete_annot(annot)
        for link in list(page.get_links()):
            page.delete_link(link)
    for name in list(doc.embfile_names()):
        doc.embfile_del(name)
    doc.set_toc([])
    doc.set_metadata({})
    if doc.get_xml_metadata():
        doc.set_xml_metadata("")


def _clear_catalog_extras(doc: fitz.Document) -> None:
    catalog = doc.pdf_catalog()
    for key in doc.xref_get_keys(catalog):
        if key not in {"Type", "Pages"}:
            doc.xref_set_key(catalog, key, "null")


def _is_disposable_contribution_page(page: fitz.Page) -> bool:
    text = _normalized_text(page.get_text()).lower()
    contribution = "team contribution" in text or "member contributions" in text
    identity_columns = bool(re.search(r"\bname\s+sid\s+contribution", text))
    return contribution and identity_columns


def redact_report(input_path: str | Path, output_path: str | Path) -> None:
    """Create the public report, raising if the result does not pass the audit."""
    source = Path(input_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with fitz.open(source) as doc:
        disposable_pages = [
            index for index, page in enumerate(doc) if _is_disposable_contribution_page(page)
        ]
        if not disposable_pages:
            raise ValueError("No team-contribution appendix page found; refusing partial release")
        for index in reversed(disposable_pages):
            doc.delete_page(index)

        _redact_first_page_identity(doc[0])
        for page in doc:
            _redact_sensitive_blocks(page)
            page.clean_contents(sanitize=True)
        _remove_document_extras(doc)

        # Rebuild from sanitized live pages only. Saving the mutated source
        # directly can preserve catalog name trees and orphaned deleted pages.
        with fitz.open() as public_doc:
            public_doc.insert_pdf(doc, links=0, annots=0, widgets=0)
            _remove_document_extras(public_doc)
            _clear_catalog_extras(public_doc)
            public_doc.save(destination, garbage=4, clean=True, deflate=True)

    findings = audit_pdf(destination)
    if findings:
        destination.unlink(missing_ok=True)
        raise ValueError(_format_findings(findings))


def render_pdf(path: str | Path, render_dir: str | Path, dpi: int = 150) -> list[Path]:
    """Render every page plus a contact sheet into an explicitly requested directory."""
    destination = Path(render_dir)
    destination.mkdir(parents=True, exist_ok=True)
    page_paths: list[Path] = []
    scale = dpi / 72
    with fitz.open(path) as doc:
        for index, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            page_path = destination / f"page-{index + 1:03d}.png"
            pix.save(page_path)
            page_paths.append(page_path)

    thumbs: list[Image.Image] = []
    for page_path in page_paths:
        with Image.open(page_path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((260, 370))
            thumbs.append(thumb.copy())
    columns = min(4, max(1, len(thumbs)))
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 280, rows * 410), "white")
    draw = ImageDraw.Draw(sheet)
    for index, thumb in enumerate(thumbs):
        x = (index % columns) * 280 + 10
        y = (index // columns) * 410 + 25
        sheet.paste(thumb, (x, y))
        draw.text((x, 5 + (index // columns) * 410), f"Page {index + 1}", fill="black")
    sheet.save(destination / "contact-sheet.png")
    return page_paths


def _format_findings(findings: Sequence[Finding]) -> str:
    lines = ["PDF public-release audit failed:"]
    lines.extend(
        f"- {finding.kind} at {finding.location}: {finding.context}" for finding in findings
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Source or audit-target PDF")
    parser.add_argument("--output", type=Path, help="Public output PDF")
    parser.add_argument(
        "--audit-only", action="store_true", help="Audit --input without modifying any file"
    )
    parser.add_argument(
        "--render-dir", type=Path, help="Optional directory for page PNGs and contact sheet"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.audit_only:
        findings = audit_pdf(args.input)
        if findings:
            print(_format_findings(findings), file=sys.stderr)
            return 1
        if args.render_dir:
            render_pdf(args.input, args.render_dir)
        print(f"Audit passed: {args.input}")
        return 0

    if args.output is None:
        print("--output is required unless --audit-only is used", file=sys.stderr)
        return 2
    try:
        redact_report(args.input, args.output)
        if args.render_dir:
            render_pdf(args.output, args.render_dir)
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Created and audited: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
