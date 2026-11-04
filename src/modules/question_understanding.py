"""
question_understanding.py — Vietnamese Question Analyzer for DataMate

Classifies question intent, expected answer type, and extracts key entities
(time expressions, aggregation intent, ranking criteria, filter predicates).
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

from .time_resolver import LunarCalendar, TimeResolver

logger = logging.getLogger(__name__)


def _normalize_for_matching(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.casefold())
    normalized = "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    )
    return normalized.replace("đ", "d")


def _pattern_matches(pattern: str, question: str) -> bool:
    if re.search(pattern, question, re.IGNORECASE):
        return True
    return re.search(
        _normalize_for_matching(pattern),
        _normalize_for_matching(question),
        re.IGNORECASE,
    ) is not None


# ---------------------------------------------------------------------------
# Answer type classification
# ---------------------------------------------------------------------------

# Keywords strongly associated with each answer type
_TYPE_INDICATORS = {
    "number": [
        r"bao nhiêu(?!\s*%|\s*phần trăm)",
        r"\bbao\s+nhiu\b",
        r"\b(?:bnhiu|bn)\b",
        r"tổng\s+(?:số|tiền|doanh|chi|lượng|giá)",
        r"số lượng",
        r"(?:có|đã|được)\s+bao nhiêu",
        r"mấy\s+(?:cái|chiếc|đơn|sản phẩm|mặt hàng|khách|đĩa|món|bill|hóa đơn)",
        r"số\s+(?:đơn|bill|hóa đơn)",
        r"\b(?:bán|ban)\s+(?:được|dc|duoc)\s+nhiêu\s+tiền\b",
        r"\bbill\b.*\bla\s+nhieu\b",
        r"đếm",
        r"count",
        r"tổng cộng",
    ],
    "percent": [
        r"bao nhiêu\s*(?:%|phần trăm)",
        r"tỷ lệ",
        r"phần trăm",
        r"(?:tăng|giảm)\s+bao nhiêu\s*%",
        r"chiếm\s+bao nhiêu",
        r"(?:tăng|giảm)\s+(?:bao nhiêu\s+)?(?:%|phần trăm)",
        r"\b(?:tỷ lệ|ti le)\b",
    ],
    "string": [
        r"(?:là\s+)?(?:gì|nào)\s*\??$",
        r"tên\s+(?:gì|nào)",
        r"(?:sản phẩm|mặt hàng|khách|khách hàng|nhà cung cấp)\s+nào",
        r"(?:khách|khách hàng|sản phẩm|mặt hàng)\s+nào.*\b(?:nhất|nhì)\b",
        r"\bai\b.*\b(?:mua|chi tiêu)\b.*\btien\b.*\bnhat\b",
        r"(?:bán chạy|nhiều|ít)\s+nhất",
        r"loại\s+(?:gì|nào)",
    ],
    "date": [
        r"ngày\s+nào",
        r"khi\s+nào",
        r"bao giờ",
        r"thời điểm",
        r"(?:ngày|tháng|năm)\s+(?:bao nhiêu|mấy|nào)",
        r"từ\s+ngày\s+nào",
        r"lần\s+(?:cuối|gần nhất|đầu tiên)",
    ],
    "boolean": [
        r"^(?:có|liệu)\s+.+\s+không\s*\??$",
        r"\bcó\b.+\b(?:không|ko|khong)\s*\??$",
        r"đúng\s+hay\s+sai",
        r"(?:có\s+)?đúng không",
        r"^.+\s+(?:có không|phải không|chưa)\s*\??$",
    ],
    "list": [
        r"(?:liệt kê|danh sách|list|kể tên|những)",
        r"(?:top|xếp hạng)\s+\d+",
        r"(?:các|những)\s+(?:sản phẩm|mặt hàng|khách hàng)",
        r"(?:\d+)\s+(?:sản phẩm|mặt hàng|khách hàng)\s+(?:bán chạy|nhiều|ít)",
        r"(?<![/\d])\b\d+\s+(?:sản phẩm|mặt hàng|khách|khách hàng)\s+nào\b",
        r"\b\d+\s+tháng\b.*\bmỗi tháng\b",
        r"\blần lượt\b",
    ],
    "table": [
        r"bảng",
        r"(?:thống kê|tổng hợp|chi tiết)\s+(?:theo|từng)",
        r"(?:theo|từng)\s+(?:tháng|tuần|ngày|sản phẩm|loại|nhóm)",
        r"(?:so sánh|breakdown)",
    ],
}

# Keywords indicating unanswerable
_UNANSWERABLE_INDICATORS = [
    r"không\s+có\s+(?:dữ liệu|thông tin|data)",
    r"ngoài\s+phạm vi",
]


@dataclass
class QuestionInfo:
    """Structured analysis of a question."""
    original: str = ""
    expected_type: str = "string"  # default fallback
    confidence: float = 0.5

    # Extracted entities
    time_expressions: List[Tuple[str, date, date]] = field(default_factory=list)
    aggregation: Optional[str] = None  # sum, avg, count, max, min
    ranking_direction: Optional[str] = None  # "desc" or "asc"
    ranking_k: Optional[int] = None  # top-k
    ranking_criterion: Optional[str] = None  # e.g., "quantity", "revenue"
    filter_entities: List[str] = field(default_factory=list)
    is_comparison: bool = False
    comparison_periods: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "expected_type": self.expected_type,
            "confidence": self.confidence,
            "aggregation": self.aggregation,
            "ranking": {
                "direction": self.ranking_direction,
                "k": self.ranking_k,
                "criterion": self.ranking_criterion,
            } if self.ranking_direction else None,
            "is_comparison": self.is_comparison,
            "time_ranges": [
                {"expr": e, "start": str(s), "end": str(d)}
                for e, s, d in self.time_expressions
            ],
        }


# ---------------------------------------------------------------------------
# Aggregation detection
# ---------------------------------------------------------------------------

_AGGREGATION_PATTERNS = {
    "sum": [r"tổng", r"cộng", r"sum", r"tổng cộng"],
    "avg": [r"trung bình", r"bình quân", r"average", r"avg", r"tb"],
    "count": [r"đếm", r"số lượng", r"bao nhiêu", r"mấy", r"count"],
    "max": [r"lớn nhất", r"cao nhất", r"nhiều nhất", r"max", r"đắt nhất"],
    "min": [r"nhỏ nhất", r"thấp nhất", r"ít nhất", r"min", r"rẻ nhất"],
}


# ---------------------------------------------------------------------------
# Ranking detection
# ---------------------------------------------------------------------------

_RANKING_DESC_KEYWORDS = [
    r"bán chạy nhất", r"nhiều nhất", r"cao nhất", r"lớn nhất",
    r"đắt nhất", r"top", r"doanh thu cao nhất",
]

_RANKING_ASC_KEYWORDS = [
    r"ít nhất", r"thấp nhất", r"nhỏ nhất", r"rẻ nhất", r"bán ít nhất",
]


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class QuestionAnalyzer:
    """Analyze Vietnamese questions to extract structured intent."""

    def __init__(self, calendar: Optional[LunarCalendar] = None):
        self.time_resolver = TimeResolver(calendar)

    def analyze(self, question: str, as_of_date: date) -> QuestionInfo:
        """
        Analyze a Vietnamese question and extract structured information.

        Parameters
        ----------
        question : str
            The question text in Vietnamese.
        as_of_date : date
            The as-of date for resolving time expressions.

        Returns
        -------
        QuestionInfo
            Structured analysis of the question.
        """
        normalized_question = unicodedata.normalize("NFC", question)
        normalized_question = re.sub(r"\s+", " ", normalized_question).strip()
        info = QuestionInfo(original=normalized_question)
        q_lower = normalized_question.lower()

        # 1. Classify expected answer type
        info.expected_type, info.confidence = self._classify_type(q_lower)

        # 2. Extract time expressions
        info.time_expressions = self.time_resolver.extract_and_resolve(
            normalized_question, as_of_date
        )

        # 3. Detect aggregation
        info.aggregation = self._detect_aggregation(q_lower)

        # 4. Detect ranking
        self._detect_ranking(q_lower, info)

        # 5. Detect comparison
        info.is_comparison = bool(re.search(
            r"so\s+với|so\s+sánh|tăng|giảm|chênh\s+lệch|cùng\s+kỳ",
            q_lower
        ))

        return info

    def _classify_type(self, question: str) -> Tuple[str, float]:
        """Classify the expected answer type."""
        # Check for unanswerable indicators first
        for pattern in _UNANSWERABLE_INDICATORS:
            if _pattern_matches(pattern, question):
                return ("unanswerable", 0.7)

        # Check for strong boolean sentence patterns (e.g., "Có ... không?")
        if (
            _pattern_matches(
                r"^(?:có|liệu)\s+.+\s+(?:không|chưa|ko|khong)\s*\??$",
                question,
            )
            or _pattern_matches(
                r"(?:đúng\s+hay\s+sai|đúng\s+không|phải\s+không)\s*\??$",
                question,
            )
            or _pattern_matches(
                r"\bcó\b.+\b(?:không|ko|khong)\s*\??$",
                question,
            )
        ):
            return ("boolean", 0.9)

        normalized_question = _normalize_for_matching(question).strip()
        if re.match(r"^(?:cho\s+)?bang\s+(?:tong hop|chi tiet|thong ke|\d+)\b", normalized_question):
            return ("table", 0.9)
        if _pattern_matches(
            r"\b(?:ngày|tháng|năm)\s+nào\b|\bkhi nào\b|bao giờ",
            question,
        ):
            return ("date", 0.9)
        if (
            re.search(r"\blan luot\b", normalized_question)
            or len(re.findall(r"\b(?:bao nhieu|bao nhiu|bnhiu|bn)\b", normalized_question)) >= 2
            or re.search(r"\b\d+\s+thang\b.*\bmoi thang\b", normalized_question)
            or re.search(
                r"(?<![/\d])\b\d+\s+(?:khach(?: hang)?|san pham|mat hang)\b"
                r".*\b(?:mua|chi tieu|ban chay|nhat|top|best seller)\b",
                normalized_question,
            )
            or (
                "best seller" in normalized_question
                and re.search(r"(?<![/\d])\b\d+\s+best seller\b", normalized_question)
            )
            or re.search(
                r"\bmay\s+\w+\b.*\bva\b.*\bmay\s+\w+\b",
                normalized_question,
            )
        ):
            return ("list", 0.9)
        if re.search(
            r"(?<![/\d])\b\d+\s+(?:san pham|mat hang|khach(?: hang)?)\s+nao\b",
            normalized_question,
        ):
            return ("list", 0.9)

        percent_shorthand = re.search(
            r"\b(?:bn|bnhiu|bao nhieu|bao nhiu)\s*%",
            normalized_question,
        )
        changed_percentage = re.search(
            r"\b(?:tang|giam|thay doi)\b.*\b(?:bao nhieu|nhieu|nhiu|bnhiu|bn)\s*%",
            normalized_question,
        )
        if (
            _pattern_matches(r"\b(?:tỷ lệ|phần trăm|ti le|phan tram)\b", question)
            or _pattern_matches(r"\bchiếm\b", question)
            or percent_shorthand is not None
            or changed_percentage is not None
            or re.search(r"%\s*\??$", question)
        ):
            return ("percent", 0.9)
        if (
            re.search(r"\b(?:doanh thu|dthu|revenue)\b", normalized_question)
            and not re.search(
                r"\b(?:ai|nao|nhat|nhieu nhat|cao nhat|lon nhat|it nhat|thap nhat)\b",
                normalized_question,
            )
        ):
            return ("number", 0.75)

        scores: Dict[str, float] = {}

        for answer_type, patterns in _TYPE_INDICATORS.items():
            score = 0.0
            for pattern in patterns:
                if _pattern_matches(pattern, question):
                    score += 1.0
            if score > 0:
                scores[answer_type] = score

        if not scores:
            return ("string", 0.3)

        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]
        confidence = min(max_score / 3.0, 1.0)

        return (best_type, confidence)

    def _detect_aggregation(self, question: str) -> Optional[str]:
        """Detect aggregation type from the question."""
        if re.search(r"\b(?:doanh thu|dthu|revenue)\b", question, re.IGNORECASE):
            return "sum"
        for agg_type, patterns in _AGGREGATION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question, re.IGNORECASE):
                    return agg_type
        return None

    def _detect_ranking(self, question: str, info: QuestionInfo):
        """Detect ranking intent and extract k."""
        normalized_question = _normalize_for_matching(question)

        # Check descending (top/most)
        for pattern in _RANKING_DESC_KEYWORDS:
            if _pattern_matches(pattern, question):
                info.ranking_direction = "desc"
                break

        # Check ascending (least/lowest)
        if not info.ranking_direction:
            for pattern in _RANKING_ASC_KEYWORDS:
                if _pattern_matches(pattern, question):
                    info.ranking_direction = "asc"
                    break

        if info.ranking_direction:
            # Extract k (top-k)
            m = re.search(
                r"(?<![/\d])\b(?:top\s+)?(\d+)\s+(?:san pham|mat hang|khach(?: hang)?|don|bill|mon|dia)\b",
                normalized_question,
            )
            if not m:
                m = re.search(r"top\s+(\d+)", normalized_question)
            if m:
                info.ranking_k = int(m.group(1))
            else:
                # "nhất" implies k=1
                if "nhat" in normalized_question:
                    info.ranking_k = 1

            # Try to detect ranking criterion
            if re.search(r"ban chay|so luong|quantity", normalized_question):
                info.ranking_criterion = "quantity"
            elif re.search(r"doanh thu|revenue|tien", normalized_question):
                info.ranking_criterion = "revenue"

            # If ranking with k=1 and direction, answer type is likely string (unless date/list/table)
            if info.ranking_k == 1 and info.expected_type not in ("list", "table", "date"):
                info.expected_type = "string"
            elif info.ranking_k and info.ranking_k > 1 and info.expected_type != "table":
                info.expected_type = "list"
