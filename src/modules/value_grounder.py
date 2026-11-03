"""
value_grounder.py — Cell-level value grounding and fuzzy entity resolution.

Inspired by XiYan-SQL (M-Schema) and CHESS Information Retriever.
Extracts distinct values from tabular columns and maps user question terms
(including unaccented Vietnamese, abbreviations, teencode) to actual database cell values.
"""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

_STOPWORDS = {
    "la", "va", "cua", "trong", "ngay", "thang", "nam", "cho", "cac",
    "nhung", "co", "duoc", "dc", "sp", "don", "hang", "tien", "khach",
    "mua", "ban", "bao", "nhieu", "may", "nao", "gi", "sao", "voi",
}


def remove_accents(text: str) -> str:
    """Remove Vietnamese diacritics for flexible fuzzy matching."""
    if not isinstance(text, str):
        return ""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_token_str(text: str) -> str:
    """Normalize text into lowercase whitespace-collapsed tokens."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text).lower().strip()
    return re.sub(r"\s+", " ", text)


class ValueGrounder:
    """Indexes column cell values across tables and resolves question entities."""

    def __init__(
        self,
        tables: Dict[str, pd.DataFrame],
        max_cardinality: int = 1500,
    ):
        self.tables = tables
        self.max_cardinality = max_cardinality
        # Mapping: table_name -> column_name -> list of original distinct string values
        self.column_values: Dict[str, Dict[str, List[str]]] = {}
        # Precomputed normalized forms: (table, col, original_val, norm_val, unaccented_val)
        self.indexed_items: List[Tuple[str, str, str, str, str]] = []
        self._build_index()

    def _build_index(self) -> None:
        for table_name, df in self.tables.items():
            self.column_values[table_name] = {}
            for col in df.columns:
                series = df[col]
                # Only index string/object/categorical columns
                if not (
                    pd.api.types.is_string_dtype(series)
                    or pd.api.types.is_object_dtype(series)
                    or isinstance(series.dtype, pd.CategoricalDtype)
                ):
                    continue

                distinct = series.dropna().unique()
                if len(distinct) == 0 or len(distinct) > self.max_cardinality:
                    continue

                val_list = [str(v).strip() for v in distinct if str(v).strip()]
                self.column_values[table_name][col] = val_list

                for val in val_list:
                    # Ignore purely numeric strings
                    if len(val) <= 1 or val.isdigit():
                        continue
                    norm_val = normalize_token_str(val)
                    unaccent_val = remove_accents(norm_val)
                    self.indexed_items.append((table_name, col, val, norm_val, unaccent_val))

    def _calculate_score(
        self,
        unaccent_val: str,
        q_unaccent: str,
        q_words: Set[str],
    ) -> float:
        """Calculate relevance score between database cell value and question text."""
        if unaccent_val in q_unaccent:
            return 1.0

        v_words = [w for w in unaccent_val.split() if w not in _STOPWORDS and len(w) > 1]
        if not v_words:
            return 0.0

        # Check consecutive n-grams of v_words in q_unaccent (e.g. 'xi mang ha tien')
        max_matched_words = 0
        for n in range(len(v_words), 1, -1):
            for i in range(len(v_words) - n + 1):
                gram = " ".join(v_words[i : i + n])
                if gram in q_unaccent:
                    max_matched_words = max(max_matched_words, n)
                    break
            if max_matched_words > 0:
                break

        if max_matched_words >= 2:
            return 0.70 + 0.30 * (max_matched_words / len(v_words))

        # Check token intersection
        inter = set(v_words) & q_words
        if len(inter) >= 2:
            return 0.65 + 0.35 * (len(inter) / len(v_words))
        elif len(inter) == 1:
            w = list(inter)[0]
            # Distinctive single word tokens (e.g. brand name or product code)
            if len(w) >= 4 or w in {"c2", "m", "l", "xl"}:
                return 0.75

        # Fallback to character similarity if string is reasonably short
        if len(unaccent_val) <= 25 and len(q_unaccent) <= 50:
            ratio = SequenceMatcher(None, unaccent_val, q_unaccent).quick_ratio()
            if ratio >= 0.75:
                return ratio

        return 0.0

    def resolve_entities(
        self,
        question: str,
        top_k: int = 3,
        min_similarity: float = 0.72,
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Resolve terms in the question to actual database cell values.

        Returns
        -------
        Dict[str, Dict[str, List[str]]]
            table_name -> column_name -> list of matched original values
        """
        if not question or not self.indexed_items:
            return {}

        q_norm = normalize_token_str(question)
        q_unaccent = remove_accents(q_norm)
        q_words = {w for w in q_unaccent.split() if w not in _STOPWORDS and len(w) > 1}

        # Collect candidate matches: (table, col) -> list of (val, score)
        candidate_matches: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}

        for table, col, original_val, norm_val, unaccent_val in self.indexed_items:
            score = self._calculate_score(unaccent_val, q_unaccent, q_words)
            if score >= min_similarity:
                key = (table, col)
                if key not in candidate_matches:
                    candidate_matches[key] = []
                candidate_matches[key].append((original_val, score))

        # Filter and rank top_k per (table, col)
        results: Dict[str, Dict[str, List[str]]] = {}
        for (table, col), matches in candidate_matches.items():
            matches.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
            seen: Set[str] = set()
            top_vals: List[str] = []
            for val, _ in matches:
                if val not in seen:
                    seen.add(val)
                    top_vals.append(val)
                    if len(top_vals) >= top_k:
                        break
            if top_vals:
                if table not in results:
                    results[table] = {}
                results[table][col] = top_vals

        return results

    def format_grounded_context(
        self,
        question: str,
        top_k: int = 3,
    ) -> str:
        """Format resolved entities into a concise prompt block for LLMs."""
        resolved = self.resolve_entities(question, top_k=top_k)
        if not resolved:
            return ""

        lines = ["【Gợi ý giá trị thực tế trong dữ liệu (Entity Grounding)】"]
        for table, col_map in resolved.items():
            for col, vals in col_map.items():
                val_str = ", ".join(f'"{v}"' for v in vals)
                lines.append(f"- Bảng '{table}', cột '{col}': [{val_str}]")

        return "\n".join(lines)
