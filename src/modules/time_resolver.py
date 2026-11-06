"""
time_resolver.py — Vietnamese Time Expression Resolver for DataMate

Resolves relative Vietnamese time expressions (e.g., "tháng này", "tuần rồi",
"cùng kỳ năm ngoái") into concrete (start_date, end_date) ranges using
the as_of_date and the solar-lunar calendar.
"""

import csv
import logging
import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _normalize_for_matching(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.casefold())
    normalized = "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    )
    return normalized.replace("đ", "d")


# ---------------------------------------------------------------------------
# Lunar calendar loader
# ---------------------------------------------------------------------------

class LunarCalendar:
    """
    Solar-lunar calendar loaded from calendar.csv.
    Provides mappings between solar and lunar dates for Tết, tháng Chạp, etc.
    """

    def __init__(self):
        # solar_date -> (lunar_year, lunar_month, lunar_day)
        self.solar_to_lunar: Dict[date, Tuple[int, int, int]] = {}
        # (lunar_year, lunar_month, lunar_day) -> solar_date
        self.lunar_to_solar: Dict[Tuple[int, int, int], date] = {}

    def load(self, calendar_path: str):
        """Load calendar.csv mapping solar ↔ lunar dates."""
        try:
            with open(calendar_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        solar = date.fromisoformat(row.get("solar_date", row.get("date", "")))
                        lunar_y = int(row.get("lunar_year", 0))
                        lunar_m = int(row.get("lunar_month", 0))
                        lunar_d = int(row.get("lunar_day", 0))
                        if lunar_y and lunar_m and lunar_d:
                            self.solar_to_lunar[solar] = (lunar_y, lunar_m, lunar_d)
                            self.lunar_to_solar[(lunar_y, lunar_m, lunar_d)] = solar
                    except (ValueError, KeyError):
                        continue
            logger.info("Loaded lunar calendar: %d entries", len(self.solar_to_lunar))
        except FileNotFoundError:
            logger.warning("Calendar file not found: %s", calendar_path)

    def get_solar_date(self, lunar_year: int, lunar_month: int, lunar_day: int) -> Optional[date]:
        """Convert lunar date to solar date."""
        return self.lunar_to_solar.get((lunar_year, lunar_month, lunar_day))

    def get_lunar_date(self, solar_date: date) -> Optional[Tuple[int, int, int]]:
        """Convert solar date to lunar date."""
        return self.solar_to_lunar.get(solar_date)

    def find_tet_window(self, solar_year: int) -> Optional[Tuple[date, date]]:
        """
        Find the Tết window: 23 tháng Chạp to mùng 10 tháng Giêng.
        tháng Chạp = lunar month 12, tháng Giêng = lunar month 1.
        The Tết of solar_year means the one that falls in/around Jan-Feb of solar_year.
        """
        # 23 tháng Chạp of lunar year (solar_year - 1)
        start = self.get_solar_date(solar_year - 1, 12, 23)
        # mùng 10 tháng Giêng of lunar year solar_year
        end = self.get_solar_date(solar_year, 1, 10)

        if start and end:
            return (start, end)
        return None


# ---------------------------------------------------------------------------
# Time expression patterns
# ---------------------------------------------------------------------------

# Vietnamese month names to month numbers
_MONTH_NAMES = {
    "giêng": 1, "một": 1, "hai": 2, "ba": 3, "tư": 4, "bốn": 4,
    "năm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10,
    "mười một": 11, "mười hai": 12, "chạp": 12,
}

# Ordinal to number
_ORDINAL = {
    "nhất": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4,
    "năm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10,
}


def _monday_of_week(d: date) -> date:
    """Return the Monday of the week containing d (week = Mon–Sun per rulebook)."""
    return d - timedelta(days=d.weekday())


def _month_range(year: int, month: int, as_of: date) -> Tuple[date, date]:
    """
    Return (first_day, last_day_or_as_of) for a given month.
    If the month is the current month, end at as_of_date (inclusive).
    """
    first = date(year, month, 1)
    if month == 12:
        last = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)

    # If this is the current month, cap at as_of_date
    if year == as_of.year and month == as_of.month:
        last = min(last, as_of)

    return first, last


def _quarter_range(year: int, quarter: int, as_of: date) -> Tuple[date, date]:
    """Return (first_day, last_day_or_as_of) for a given quarter."""
    start_month = (quarter - 1) * 3 + 1
    end_month = start_month + 2
    first = date(year, start_month, 1)
    if end_month == 12:
        last = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(year, end_month + 1, 1) - timedelta(days=1)

    if first <= as_of <= last:
        last = min(last, as_of)
    return first, last


def _year_range(year: int, as_of: date) -> Tuple[date, date]:
    """Return (first_day, last_day_or_as_of) for a given year."""
    first = date(year, 1, 1)
    last = date(year, 12, 31)
    if year == as_of.year:
        last = min(last, as_of)
    return first, last


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------

class TimeResolver:
    """
    Resolve Vietnamese time expressions to date ranges.

    Usage:
        resolver = TimeResolver(calendar)
        start, end = resolver.resolve("tháng trước", as_of_date)
    """

    def __init__(self, calendar: Optional[LunarCalendar] = None):
        self.calendar = calendar or LunarCalendar()

    def resolve(self, expression: str, as_of: date) -> Optional[Tuple[date, date]]:
        """
        Resolve a Vietnamese time expression to a (start_date, end_date) range.

        Parameters
        ----------
        expression : str
            Vietnamese time expression (e.g., "tháng này", "tuần rồi").
        as_of : date
            The as-of date for resolving relative expressions.

        Returns
        -------
        Optional[Tuple[date, date]]
            (start_date, end_date) inclusive, or None if not recognized.
        """
        expr = expression.strip().lower()

        # --- Exact date patterns ---
        # "hôm nay"
        if expr in ("hôm nay", "ngày hôm nay", "today", "hnay"):
            return (as_of, as_of)

        # "hôm qua"
        if expr in ("hôm qua", "ngày hôm qua", "hqua", "bữa qua", "bua qua"):
            yesterday = as_of - timedelta(days=1)
            return (yesterday, yesterday)

        # --- Year-to-date and Month-to-date patterns ---
        if re.search(r"từ\s+đầu\s+năm", expr):
            return (date(as_of.year, 1, 1), as_of)

        if re.search(r"từ\s+đầu\s+tháng", expr):
            return (date(as_of.year, as_of.month, 1), as_of)

        # --- Week patterns ---
        if re.search(r"tuần\s*(này|nay)", expr):
            monday = _monday_of_week(as_of)
            return (monday, as_of)

        if re.search(r"tuần\s*(rồi|trước|qua|vừa rồi|vừa qua)", expr):
            prev_monday = _monday_of_week(as_of) - timedelta(days=7)
            prev_sunday = prev_monday + timedelta(days=6)
            return (prev_monday, prev_sunday)

        # --- "cùng kỳ" (Same period) ---
        # Rulebook §2: "cùng kỳ tháng trước" = from 1st to same day min(as_of.day, last_day_prev)
        if re.search(r"cùng\s*kỳ\s*tháng\s*(trước|rồi)", expr):
            prev_y = as_of.year if as_of.month > 1 else as_of.year - 1
            prev_m = as_of.month - 1 if as_of.month > 1 else 12
            first_prev, last_prev = _month_range(prev_y, prev_m, as_of)
            cap_day = min(as_of.day, last_prev.day)
            return (first_prev, date(prev_y, prev_m, cap_day))

        # "cùng kỳ năm ngoái" = same month, last year
        if re.search(r"cùng\s*kỳ\s*(năm\s*)?(ngoái|trước|rồi)", expr):
            return _month_range(as_of.year - 1, as_of.month, as_of)

        # --- Month patterns ---
        # Explicit month and year: "tháng M/YYYY" or "tháng M năm YYYY"
        m_my = re.search(r"tháng\s+(\d{1,2})\s*(?:[/\-]|năm\s*)(\d{4})", expr)
        if m_my:
            m_num = int(m_my.group(1))
            y_num = int(m_my.group(2))
            if 1 <= m_num <= 12:
                return _month_range(y_num, m_num, as_of)

        if re.search(r"tháng\s*(này|nay)", expr):
            return _month_range(as_of.year, as_of.month, as_of)

        if re.search(r"tháng\s*(rồi|trước|qua|vừa rồi|vừa qua)", expr):
            if as_of.month == 1:
                return _month_range(as_of.year - 1, 12, as_of)
            else:
                return _month_range(as_of.year, as_of.month - 1, as_of)

        # "tháng N" or "tháng <name>"
        m = re.match(r"tháng\s+(\d{1,2})", expr)
        if m:
            month_num = int(m.group(1))
            if 1 <= month_num <= 12:
                return _month_range(as_of.year, month_num, as_of)

        for name, num in _MONTH_NAMES.items():
            if f"tháng {name}" in expr:
                return _month_range(as_of.year, num, as_of)

        # --- Quarter patterns ---
        # Explicit quarter and year: "quý Q/YYYY" or "quý Q năm YYYY"
        m_qy = re.search(r"quý\s+(\d)\s*(?:[/\-]|năm\s*)(\d{4})", expr)
        if m_qy:
            q_num = int(m_qy.group(1))
            y_num = int(m_qy.group(2))
            if 1 <= q_num <= 4:
                return _quarter_range(y_num, q_num, as_of)

        m = re.match(r"quý\s+(\d)", expr)
        if m:
            q = int(m.group(1))
            if 1 <= q <= 4:
                return _quarter_range(as_of.year, q, as_of)

        if re.search(r"quý\s*(này|nay)", expr):
            q = (as_of.month - 1) // 3 + 1
            return _quarter_range(as_of.year, q, as_of)

        if re.search(r"quý\s*(rồi|trước|qua)", expr):
            q = (as_of.month - 1) // 3 + 1
            if q == 1:
                return _quarter_range(as_of.year - 1, 4, as_of)
            else:
                return _quarter_range(as_of.year, q - 1, as_of)

        # --- Year patterns ---
        if re.search(r"năm\s*(nay|này)", expr):
            return _year_range(as_of.year, as_of)

        if re.search(r"năm\s*(rồi|trước|ngoái|qua)", expr):
            return _year_range(as_of.year - 1, as_of)

        m = re.match(r"năm\s+(\d{4})", expr)
        if m:
            year = int(m.group(1))
            return _year_range(year, as_of)

        # --- Tết / Lunar patterns ---
        if "tết" in expr:
            window = self.calendar.find_tet_window(as_of.year)
            if window:
                return window

        # "mùng N" (lunar new year day N)
        m = re.search(r"mùng\s+(\d{1,2})", expr)
        if m:
            day = int(m.group(1))
            solar = self.calendar.get_solar_date(as_of.year, 1, day)
            if solar:
                return (solar, solar)

        # --- N days ago ---
        m = re.search(r"(\d+)\s*ngày\s*(trước|qua)", expr)
        if m:
            n = int(m.group(1))
            start = as_of - timedelta(days=n)
            return (start, as_of)

        m = re.search(
            r"(\d+)\s*ngay\s*gan\s*nhat",
            _normalize_for_matching(expr),
        )
        if m:
            n = int(m.group(1))
            if n < 1:
                return None
            return (as_of - timedelta(days=n - 1), as_of)

        # --- Specific date ---
        m = re.search(r"ngày\s+(\d{1,2})[/\-.](\d{1,2})(?:[/\-.](\d{2,4}))?", expr)
        if m:
            day = int(m.group(1))
            month = int(m.group(2))
            year = int(m.group(3)) if m.group(3) else as_of.year
            if year < 100:
                year += 2000
            try:
                d = date(year, month, day)
                return (d, d)
            except ValueError:
                pass

        logger.debug("Could not resolve time expression: '%s'", expression)
        return None

    def extract_and_resolve(self, question: str, as_of: date) -> List[Tuple[str, date, date]]:
        """
        Extract all time expressions from a question and resolve them.

        Returns a list of (expression, start_date, end_date) tuples.
        """
        results = []
        patterns = [
            r"(tháng\s+(?:này|nay|rồi|trước|qua|vừa rồi|vừa qua|\d{1,2}(?:\s*(?:[/\-]|năm\s*)\d{4})?|\w+))",
            r"(tuần\s+(?:này|nay|rồi|trước|qua|vừa rồi|vừa qua))",
            r"(quý\s+(?:này|nay|rồi|trước|qua|\d(?:\s*(?:[/\-]|năm\s*)\d{4})?))",
            r"(cùng\s*kỳ\s*(?:tháng|năm)?\s*(?:ngoái|trước|rồi)?)",
            r"(từ\s+đầu\s+(?:năm|tháng)(?:\s+(?:này|nay))?(?:\s+đến\s+hết\s+(?:hôm\s+nay|hnay))?)",
            r"(năm\s+(?:nay|này|rồi|trước|ngoái|qua|\d{4}))",
            r"(hôm\s+(?:nay|qua)|bữa\s+qua|bua\s+qua|hqua|hnay)",
            r"(tết)",
            r"(\d+\s*ngày\s*(?:trước|qua))",
        ]

        normalized_question = _normalize_for_matching(question)
        for match in re.finditer(
            r"\d+\s*ngay\s*gan\s*nhat",
            normalized_question,
        ):
            expr = match.group(0)
            resolved = self.resolve(expr, as_of)
            if resolved:
                results.append((expr, resolved[0], resolved[1]))

        for pattern in patterns:
            for m in re.finditer(pattern, question.lower()):
                expr = m.group(1)
                resolved = self.resolve(expr, as_of)
                if resolved:
                    results.append((expr, resolved[0], resolved[1]))

        return results
