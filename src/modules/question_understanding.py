"""
question_understanding.py — Vietnamese Question Analyzer for DataMate

Classifies question intent, expected answer type, and extracts key entities
(time expressions, aggregation intent, ranking criteria, filter predicates).
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

from .time_resolver import LunarCalendar, TimeResolver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Answer type classification
# ---------------------------------------------------------------------------

# Keywords strongly associated with each answer type
_TYPE_INDICATORS = {
    "number": [
        r"bao nhiêu(?!\s*%|\s*phần trăm)",
        r"tổng\s+(?:số|tiền|doanh|chi|lượng|giá)",
        r"số lượng",
        r"(?:có|đã|được)\s+bao nhiêu",
        r"mấy\s+(?:cái|chiếc|đơn|sản phẩm|khách)",
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
    ],
    "string": [
        r"(?:là\s+)?(?:gì|nào)\s*\??$",
        r"tên\s+(?:gì|nào)",
        r"(?:sản phẩm|mặt hàng|khách hàng|nhà cung cấp)\s+nào",
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
        r"đúng\s+hay\s+sai",
        r"(?:có\s+)?đúng không",
        r"^.+\s+(?:có không|phải không|chưa)\s*\??$",
    ],
    "list": [
        r"(?:liệt kê|danh sách|list|kể tên|những)",
        r"(?:top|xếp hạng)\s+\d+",
        r"(?:các|những)\s+(?:sản phẩm|mặt hàng|khách hàng)",
        r"(?:\d+)\s+(?:sản phẩm|mặt hàng|khách hàng)\s+(?:bán chạy|nhiều|ít)",
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
        info = QuestionInfo(original=question)
        q_lower = question.lower().strip()

        # 1. Classify expected answer type
        info.expected_type, info.confidence = self._classify_type(q_lower)

        # 2. Extract time expressions
        info.time_expressions = self.time_resolver.extract_and_resolve(
            question, as_of_date
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
            if re.search(pattern, question, re.IGNORECASE):
                return ("unanswerable", 0.7)

        # Check for strong boolean sentence patterns (e.g., "Có ... không?")
        if re.search(r"^(?:có|liệu)\s+.+\s+(?:không|chưa)\s*\??$", question, re.IGNORECASE) or \
           re.search(r"(?:đúng\s+hay\s+sai|đúng\s+không|phải\s+không)\s*\??$", question, re.IGNORECASE):
            return ("boolean", 0.9)

        scores: Dict[str, float] = {}

        for answer_type, patterns in _TYPE_INDICATORS.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, question, re.IGNORECASE):
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
        for agg_type, patterns in _AGGREGATION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question, re.IGNORECASE):
                    return agg_type
        return None

    def _detect_ranking(self, question: str, info: QuestionInfo):
        """Detect ranking intent and extract k."""
        # Check descending (top/most)
        for pattern in _RANKING_DESC_KEYWORDS:
            if re.search(pattern, question, re.IGNORECASE):
                info.ranking_direction = "desc"
                break

        # Check ascending (least/lowest)
        if not info.ranking_direction:
            for pattern in _RANKING_ASC_KEYWORDS:
                if re.search(pattern, question, re.IGNORECASE):
                    info.ranking_direction = "asc"
                    break

        if info.ranking_direction:
            # Extract k (top-k)
            m = re.search(r"top\s+(\d+)", question)
            if m:
                info.ranking_k = int(m.group(1))
            else:
                # "nhất" implies k=1
                if "nhất" in question:
                    info.ranking_k = 1

            # Try to detect ranking criterion
            if re.search(r"bán chạy|số lượng|quantity", question, re.IGNORECASE):
                info.ranking_criterion = "quantity"
            elif re.search(r"doanh thu|revenue|tiền", question, re.IGNORECASE):
                info.ranking_criterion = "revenue"

            # If ranking with k=1 and direction, answer type is likely string (unless date/list/table)
            if info.ranking_k == 1 and info.expected_type not in ("list", "table", "date"):
                info.expected_type = "string"
            elif info.ranking_k and info.ranking_k > 1:
                info.expected_type = "list"
