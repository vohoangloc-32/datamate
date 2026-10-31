"""
rulebook_interpreter.py — Rulebook Parser for DataMate

Parses rulebook.md to extract default business rules and vocabulary
definitions that the agent must apply when answering questions.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import unicodedata

logger = logging.getLogger(__name__)


class BusinessRule:
    """A single business rule extracted from the rulebook."""

    def __init__(self, keyword: str, definition: str, category: str = "general"):
        self.keyword = keyword
        self.definition = definition
        self.category = category  # e.g., "revenue", "cost", "time", "ranking", "general"

    def __repr__(self):
        return f"BusinessRule({self.keyword!r}: {self.definition[:50]}...)"


class Rulebook:
    """Parsed rulebook containing all default business rules."""

    def __init__(self):
        self.raw_text: str = ""
        self.rules: List[BusinessRule] = []
        self.vocabulary: Dict[str, str] = {}
        self.sections: Dict[str, str] = {}

    def get_rules_by_category(self, category: str) -> List[BusinessRule]:
        return [r for r in self.rules if r.category == category]

    def get_relevant_rules(self, question: str) -> List[BusinessRule]:
        """Find rules relevant to a given question based on keyword matching."""
        question_lower = question.lower()
        relevant = []
        for rule in self.rules:
            # Check if the rule keyword appears in the question
            if rule.keyword.lower() in question_lower:
                relevant.append(rule)
        return relevant

    def get_vocabulary_match(self, term: str) -> Optional[str]:
        """Look up a term in the rulebook vocabulary."""
        term_lower = term.strip().lower()
        for key, value in self.vocabulary.items():
            if key.lower() == term_lower:
                return value
        return None

    def to_prompt_string(self, relevant_only: Optional[str] = None,
                          max_length: int = 3000) -> str:
        """
        Generate a prompt-friendly representation of the rules.

        Parameters
        ----------
        relevant_only : str, optional
            If provided, only include rules relevant to this question text.
        max_length : int
            Maximum length of the output string.
        """
        if relevant_only:
            rules = self.get_relevant_rules(relevant_only)
            if not rules:
                # Fall back to all rules if none match
                rules = self.rules
        else:
            rules = self.rules

        parts = ["Default business rules (apply unless the question says otherwise):"]
        for rule in rules:
            parts.append(f"- {rule.keyword}: {rule.definition}")

        result = "\n".join(parts)
        if len(result) > max_length:
            result = result[:max_length - 3] + "..."
        return result

    def to_full_prompt_string(self) -> str:
        """Return the complete rulebook text for inclusion in prompts."""
        return self.raw_text


# ---------------------------------------------------------------------------
# Keyword categories for auto-classification
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS = {
    "revenue": [
        "doanh thu", "revenue", "doanh số", "tổng tiền", "thành tiền",
        "giá bán", "tiền bán", "bán được",
    ],
    "cost": [
        "giá vốn", "chi phí", "cost", "giá nhập", "vốn", "tiền nhập",
        "phí", "giá mua", "weighted-average", "trung bình",
    ],
    "time": [
        "tháng", "tuần", "ngày", "năm", "quý", "week", "month",
        "monday", "sunday", "thứ",
    ],
    "ranking": [
        "bán chạy", "top", "nhiều nhất", "ít nhất", "cao nhất", "thấp nhất",
        "xếp hạng", "ranking",
    ],
    "order": [
        "đơn hàng", "order", "hủy", "cancelled", "trả", "returned",
        "hoàn", "giao",
    ],
    "discount": [
        "giảm giá", "discount", "voucher", "khuyến mãi", "chiết khấu",
    ],
    "tax": [
        "vat", "thuế", "tax", "phí giao", "delivery",
    ],
    "tet": [
        "tết", "chạp", "giêng", "mùng", "lunar", "âm lịch",
    ],
}


def _classify_rule(keyword: str, definition: str) -> str:
    """Auto-classify a rule into a category based on its content."""
    combined = f"{keyword} {definition}".lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                return category
    return "general"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_rules_from_text(text: str) -> List[BusinessRule]:
    """Extract rules from rulebook text."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    rules = []

    # Pattern 1: "term" = definition (or "term" — definition)
    pattern1 = re.compile(
        r'["""]([^"""]+)["""]'
        r'\s*[=:–—]\s*'
        r'(.+?)(?=\n["""]|\n\n|\Z)',
        re.DOTALL | re.UNICODE
    )
    for m in pattern1.finditer(text):
        keyword = m.group(1).strip()
        definition = m.group(2).strip()
        definition = re.sub(r'\s+', ' ', definition)
        category = _classify_rule(keyword, definition)
        rules.append(BusinessRule(keyword, definition, category))

    # Pattern 2: bullet-point rules: - **keyword**: definition
    pattern2 = re.compile(
        r'[-*]\s+\*?\*?([^*:\n]+?)\*?\*?\s*[:：]\s*(.+?)(?=\n[-*]|\n\n|\Z)',
        re.DOTALL | re.UNICODE
    )
    for m in pattern2.finditer(text):
        keyword = m.group(1).strip()
        definition = m.group(2).strip()
        definition = re.sub(r'\s+', ' ', definition)
        # Avoid duplicates
        if not any(r.keyword == keyword for r in rules):
            category = _classify_rule(keyword, definition)
            rules.append(BusinessRule(keyword, definition, category))

    # Pattern 3: simple lines with = or : separator
    pattern3 = re.compile(
        r'^([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\s]*?)\s*[=]\s*(.+)$',
        re.MULTILINE | re.UNICODE
    )
    for m in pattern3.finditer(text):
        keyword = m.group(1).strip()
        definition = m.group(2).strip()
        if not any(r.keyword == keyword for r in rules):
            category = _classify_rule(keyword, definition)
            rules.append(BusinessRule(keyword, definition, category))

    # Pattern 4: Markdown tables (| Luật | Mặc định |, | Cách nói | Nghĩa |, etc.)
    table_pattern = re.compile(
        r"^\|\s*([^|\n]+?)\s*\|\s*([^|\n]+?)\s*\|",
        re.MULTILINE | re.UNICODE
    )
    for m in table_pattern.finditer(text):
        col1 = m.group(1).strip()
        col2 = m.group(2).strip()
        if not col1 or not col2:
            continue
        col1_clean = col1.lower()
        if col1_clean in ("luật", "cách nói", "cách ghi", "từ", "type", "ngành", "đúng", "sai") or "---" in col1 or "---" in col2:
            continue
        keyword = col1
        definition = re.sub(r'\s+', ' ', col2)
        if not any(r.keyword.lower() == keyword.lower() for r in rules):
            category = _classify_rule(keyword, definition)
            rules.append(BusinessRule(keyword, definition, category))

    return rules


def _parse_vocabulary(text: str) -> Dict[str, str]:
    """Extract vocabulary/terminology definitions from the rulebook."""
    vocab = {}

    # Look for vocabulary/glossary sections
    sections = re.split(r'^#{1,3}\s+', text, flags=re.MULTILINE)
    for section in sections:
        # Find definition patterns
        for m in re.finditer(
            r'[-*]\s*["""]?([^""":\n]+?)["""]?\s*[:=–—]\s*(.+?)(?=\n[-*]|\n\n|\Z)',
            section, re.DOTALL | re.UNICODE
        ):
            term = m.group(1).strip()
            definition = m.group(2).strip()
            definition = re.sub(r'\s+', ' ', definition)
            if len(term) < 50:  # reasonable term length
                vocab[term] = definition

    return vocab


def parse_rulebook(rulebook_path: str) -> Rulebook:
    """
    Parse a rulebook.md file and extract structured business rules.

    Parameters
    ----------
    rulebook_path : str
        Path to the rulebook.md file.

    Returns
    -------
    Rulebook
        Parsed rulebook with rules and vocabulary.
    """
    rulebook = Rulebook()
    path = Path(rulebook_path)

    if not path.exists():
        logger.warning("Rulebook not found: %s", rulebook_path)
        return rulebook

    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="utf-8-sig")

    raw = unicodedata.normalize("NFC", raw)
    rulebook.raw_text = raw

    # Extract sections
    current_header = "intro"
    current_lines = []
    for line in raw.split("\n"):
        header_match = re.match(r"^#{1,3}\s+(.+)", line)
        if header_match:
            if current_lines:
                rulebook.sections[current_header] = "\n".join(current_lines).strip()
            current_header = header_match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        rulebook.sections[current_header] = "\n".join(current_lines).strip()

    # Parse rules
    rulebook.rules = _parse_rules_from_text(raw)

    # Parse vocabulary
    rulebook.vocabulary = _parse_vocabulary(raw)

    logger.info("Parsed rulebook %s: %d rules, %d vocabulary entries",
                rulebook_path, len(rulebook.rules), len(rulebook.vocabulary))

    return rulebook
