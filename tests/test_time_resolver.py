"""
Tests for time_resolver module.
"""

import pytest
from datetime import date, timedelta
from src.modules.time_resolver import (
    TimeResolver,
    LunarCalendar,
    _monday_of_week,
    _month_range,
    _quarter_range,
    _year_range,
)


@pytest.fixture
def as_of_date():
    # 2024-05-18 is a Saturday
    return date(2024, 5, 18)


@pytest.fixture
def calendar():
    cal = LunarCalendar()
    # Mock lunar mappings for 2024 Tết
    # 23 tháng Chạp năm Quý Mão (2023 lunar) -> 2024-02-02
    # mùng 1 Tết Giáp Thìn -> 2024-02-10
    # mùng 10 Tết -> 2024-02-19
    cal.lunar_to_solar[(2023, 12, 23)] = date(2024, 2, 2)
    cal.solar_to_lunar[date(2024, 2, 2)] = (2023, 12, 23)

    cal.lunar_to_solar[(2024, 1, 1)] = date(2024, 2, 10)
    cal.solar_to_lunar[date(2024, 2, 10)] = (2024, 1, 1)

    cal.lunar_to_solar[(2024, 1, 10)] = date(2024, 2, 19)
    cal.solar_to_lunar[date(2024, 2, 19)] = (2024, 1, 10)
    return cal


def test_monday_of_week():
    # 2024-05-18 is Saturday, Monday is 2024-05-13
    d = date(2024, 5, 18)
    assert _monday_of_week(d) == date(2024, 5, 13)

    # Monday should return itself
    mon = date(2024, 5, 13)
    assert _monday_of_week(mon) == mon


def test_resolve_today_yesterday(as_of_date):
    resolver = TimeResolver()
    assert resolver.resolve("hôm nay", as_of_date) == (as_of_date, as_of_date)
    assert resolver.resolve("hôm qua", as_of_date) == (date(2024, 5, 17), date(2024, 5, 17))


def test_resolve_week(as_of_date):
    resolver = TimeResolver()
    # Tuần này: Mon 2024-05-13 to Sat 2024-05-18 (as_of_date)
    assert resolver.resolve("tuần này", as_of_date) == (date(2024, 5, 13), as_of_date)

    # Tuần rồi: Mon 2024-05-06 to Sun 2024-05-12
    assert resolver.resolve("tuần trước", as_of_date) == (date(2024, 5, 6), date(2024, 5, 12))


def test_resolve_month(as_of_date):
    resolver = TimeResolver()
    # Tháng này: 2024-05-01 to 2024-05-18 (capped at as_of_date per rulebook)
    assert resolver.resolve("tháng này", as_of_date) == (date(2024, 5, 1), as_of_date)

    # Tháng trước: 2024-04-01 to 2024-04-30
    assert resolver.resolve("tháng trước", as_of_date) == (date(2024, 4, 1), date(2024, 4, 30))

    # Tháng 3: 2024-03-01 to 2024-03-31
    assert resolver.resolve("tháng 3", as_of_date) == (date(2024, 3, 1), date(2024, 3, 31))


def test_resolve_quarter(as_of_date):
    resolver = TimeResolver()
    # Quý 1: 2024-01-01 to 2024-03-31
    assert resolver.resolve("quý 1", as_of_date) == (date(2024, 1, 1), date(2024, 3, 31))

    # Quý này (as_of is May -> Q2): 2024-04-01 to 2024-05-18 (capped at as_of)
    assert resolver.resolve("quý này", as_of_date) == (date(2024, 4, 1), as_of_date)


def test_resolve_year(as_of_date):
    resolver = TimeResolver()
    # Năm nay: 2024-01-01 to 2024-05-18
    assert resolver.resolve("năm nay", as_of_date) == (date(2024, 1, 1), as_of_date)

    # Năm ngoái: 2023-01-01 to 2023-12-31
    assert resolver.resolve("năm ngoái", as_of_date) == (date(2023, 1, 1), date(2023, 12, 31))


def test_resolve_cung_ky_nam_ngoai(as_of_date):
    resolver = TimeResolver()
    # as_of is 2024-05-18 -> same month last year = May 2023
    assert resolver.resolve("cùng kỳ năm ngoái", as_of_date) == (date(2023, 5, 1), date(2023, 5, 31))


def test_resolve_tet(calendar, as_of_date):
    resolver = TimeResolver(calendar)
    # Tết window 2024: 23 tháng Chạp (2024-02-02) to mùng 10 (2024-02-19)
    assert resolver.resolve("tết", as_of_date) == (date(2024, 2, 2), date(2024, 2, 19))


def test_resolve_specific_date(as_of_date):
    resolver = TimeResolver()
    assert resolver.resolve("ngày 15/04/2024", as_of_date) == (date(2024, 4, 15), date(2024, 4, 15))


@pytest.mark.parametrize(
    "expression",
    ["7 ngày gần nhất", "7 ngay gan nhat"],
)
def test_resolve_recent_days_including_today(expression, as_of_date):
    resolver = TimeResolver()

    assert resolver.resolve(expression, as_of_date) == (
        date(2024, 5, 12),
        as_of_date,
    )


def test_resolve_one_recent_day(as_of_date):
    assert TimeResolver().resolve("1 ngày gần nhất", as_of_date) == (
        as_of_date,
        as_of_date,
    )


def test_extract_recent_days_unaccented():
    resolver = TimeResolver()

    assert resolver.extract_and_resolve(
        "7 ngay gan nhat cua hang ban duoc bao nhieu",
        date(2024, 5, 18),
    ) == [("7 ngay gan nhat", date(2024, 5, 12), date(2024, 5, 18))]


def test_extract_and_resolve(as_of_date):
    resolver = TimeResolver()
    q = "Doanh thu trong tháng này và tuần rồi so với năm ngoái như thế nào?"
    results = resolver.extract_and_resolve(q, as_of_date)

    # Should detect "tháng này", "tuần rồi", "năm ngoái"
    expressions = [r[0] for r in results]
    assert any("tháng này" in e for e in expressions)
    assert any("tuần rồi" in e for e in expressions)
    assert any("năm ngoái" in e for e in expressions)
