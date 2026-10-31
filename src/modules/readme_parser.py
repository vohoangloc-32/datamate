"""
readme_parser.py — README.md Parser for DataMate

Extracts business descriptions, column definitions, and domain-specific
notes from each business's README.md file. Produces a structured context
string for LLM prompt construction.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import unicodedata

logger = logging.getLogger(__name__)


class BusinessContext:
    """Parsed context from a business's README.md."""

    def __init__(self):
        self.raw_text: str = ""
        self.business_name: str = ""
        self.business_type: str = ""
        self.description: str = ""
        self.column_definitions: Dict[str, str] = {}
        self.notes: List[str] = []
        self.sections: Dict[str, str] = {}

    def to_prompt_string(self, max_length: int = 2000) -> str:
        """Generate a prompt-friendly summary of the business context."""
        parts = []

        if self.business_name:
            parts.append(f"Business: {self.business_name}")
        if self.business_type:
            parts.append(f"Type: {self.business_type}")
        if self.description:
            parts.append(f"Description: {self.description}")

        if self.column_definitions:
            parts.append("\nColumn definitions:")
            for col, defn in self.column_definitions.items():
                parts.append(f"  - {col}: {defn}")

        if self.notes:
            parts.append("\nNotes:")
            for note in self.notes:
                parts.append(f"  - {note}")

        result = "\n".join(parts)
        if len(result) > max_length:
            result = result[:max_length - 3] + "..."
        return result


def _extract_sections(text: str) -> Dict[str, str]:
    """Split markdown text into sections by headers."""
    sections = {}
    current_header = "intro"
    current_lines: List[str] = []

    for line in text.split("\n"):
        header_match = re.match(r"^#{1,4}\s+(.+)", line)
        if header_match:
            if current_lines:
                sections[current_header] = "\n".join(current_lines).strip()
            current_header = header_match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_header] = "\n".join(current_lines).strip()

    return sections


def _extract_column_definitions(text: str) -> Dict[str, str]:
    """
    Try to extract column definitions from markdown text.
    Handles formats like:
    - `column_name`: description
    - **column_name**: description
    - | column | description |  (table format)
    - column_name — description
    """
    definitions = {}

    # Pattern 1: backtick or bold column name followed by colon/dash
    pattern1 = re.compile(
        r"(?:`([^`]+)`|\*\*([^*]+)\*\*)\s*[:–—-]\s*(.+)", re.UNICODE
    )
    for m in pattern1.finditer(text):
        col_name = m.group(1) or m.group(2)
        description = m.group(3).strip()
        definitions[col_name.strip()] = description

    # Pattern 2: simple "- column_name: description" list items
    pattern2 = re.compile(
        r"^[-*]\s+(\w[\w\s]*?)\s*[:–—]\s*(.+)", re.MULTILINE | re.UNICODE
    )
    for m in pattern2.finditer(text):
        col_name = m.group(1).strip()
        description = m.group(2).strip()
        if col_name not in definitions:
            definitions[col_name] = description

    # Pattern 3: markdown table rows (| col | desc |)
    table_pattern = re.compile(
        r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", re.MULTILINE
    )
    rows = list(table_pattern.finditer(text))
    if len(rows) >= 3:  # header + separator + at least 1 data row
        for row in rows[2:]:  # skip header and separator
            col_name = row.group(1).strip()
            description = row.group(2).strip()
            if col_name and description and not col_name.startswith("-"):
                definitions[col_name] = description

    return definitions


def _extract_business_info(text: str) -> Tuple[str, str]:
    """Try to extract business name and type from the README."""
    name = ""
    biz_type = ""

    # Look for common patterns
    name_patterns = [
        re.compile(r"(?:tên|name|cửa hàng|shop)\s*[:：]\s*(.+)", re.IGNORECASE),
        re.compile(r"^#\s+(.+)", re.MULTILINE),
    ]
    for p in name_patterns:
        m = p.search(text)
        if m:
            name = m.group(1).strip()
            break

    type_patterns = [
        re.compile(r"(?:loại|type|ngành|lĩnh vực)\s*[:：]\s*(.+)", re.IGNORECASE),
    ]
    for p in type_patterns:
        m = p.search(text)
        if m:
            biz_type = m.group(1).strip()
            break

    return name, biz_type


def parse_readme(readme_path: str) -> BusinessContext:
    """
    Parse a business README.md file and extract structured context.

    Parameters
    ----------
    readme_path : str
        Path to the README.md file.

    Returns
    -------
    BusinessContext
        Structured context extracted from the README.
    """
    ctx = BusinessContext()
    path = Path(readme_path)

    if not path.exists():
        logger.warning("README not found: %s", readme_path)
        return ctx

    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            raw = path.read_text(encoding="utf-8-sig")
        except Exception:
            raw = path.read_text(encoding="latin-1")

    raw = unicodedata.normalize("NFC", raw)
    ctx.raw_text = raw

    # Extract sections
    ctx.sections = _extract_sections(raw)

    # Extract business info
    ctx.business_name, ctx.business_type = _extract_business_info(raw)

    # Use intro or first section as description
    if "intro" in ctx.sections and ctx.sections["intro"]:
        ctx.description = ctx.sections["intro"][:500]
    elif ctx.sections:
        first_key = next(iter(ctx.sections))
        ctx.description = ctx.sections[first_key][:500]

    # Extract column definitions
    ctx.column_definitions = _extract_column_definitions(raw)

    # Extract notes (bullet points that look like special instructions)
    note_pattern = re.compile(
        r"(?:lưu ý|note|ghi chú|chú ý|đặc biệt|special)\s*[:：]?\s*(.+)",
        re.IGNORECASE
    )
    for m in note_pattern.finditer(raw):
        ctx.notes.append(m.group(1).strip())

    logger.info("Parsed README %s: name='%s', type='%s', %d col defs, %d notes",
                readme_path, ctx.business_name, ctx.business_type,
                len(ctx.column_definitions), len(ctx.notes))

    return ctx
