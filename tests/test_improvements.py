"""Tests for engine improvements: temporal resolution, domain parser fixes, and gating."""

import datetime as dt
from pathlib import Path

from src.canonical import normalize_question_text
from src.executor import ExecutionResult
from src.parser import ParsedQuestion, parse_question
from src.temporal import LunarSolarCalendar, TimeWindow, resolve_question_time_window
from src.verifier import verify_and_gate


def test_shorthand_month_year_temporal():
    cal = LunarSolarCalendar(Path("train/calendar.csv"))
    as_of = dt.date(2026, 7, 22)

    # Single month shorthand 8/26 with shift
    tw = resolve_question_time_window("8/26 tiền bán ca chiều là bnhiu?", as_of, cal)
    assert tw.label == "2026-08"
    assert tw.start_date == dt.date(2026, 8, 1)
    assert tw.end_date == dt.date(2026, 8, 31)
    assert tw.shift_filter == "afternoon"

    # Single month shorthand 10/25 with two shifts compared
    tw2 = resolve_question_time_window("10/25 doanh thu ca chieu hon hay kem ca sang may %?", as_of, cal)
    assert tw2.label == "2025-10"
    assert tw2.shift_filter == "afternoon"
    assert tw2.compare_window is not None
    assert tw2.compare_window.shift_filter == "morning"


def test_bare_month_and_multi_months():
    cal = LunarSolarCalendar(Path("train/calendar.csv"))
    as_of = dt.date(2026, 7, 22)

    # Bare month Th3
    tw_bare = resolve_question_time_window("Th3 trung bình mỗi đơn được bao nhiêu doanh thu?", as_of, cal)
    assert tw_bare.label == "2026-03"
    assert tw_bare.start_date == dt.date(2026, 3, 1)
    assert tw_bare.end_date == dt.date(2026, 3, 31)

    # Multi-month bare "th4, th5, th6"
    tw_multi = resolve_question_time_window("Dthu th4, th5, th6 mỗi tháng được nhiêu z?", as_of, cal)
    assert tw_multi.label == "multi_months"
    assert len(tw_multi.multi_windows) == 3
    assert [w.label for w in tw_multi.multi_windows] == ["2026-04", "2026-05", "2026-06"]

    # Multi-month shorthand "3/25, 4/25 và 5/25"
    tw_multi2 = resolve_question_time_window("So sánh 3/25, 4/25 và 5/25 xem tháng nào cửa hàng bán đc nhiều tiền nhất", as_of, cal)
    assert tw_multi2.label == "multi_months"
    assert [w.label for w in tw_multi2.multi_windows] == ["2025-03", "2025-04", "2025-05"]

    # Cross-year multi-month range
    tw_rng = resolve_question_time_window("Doanh thu tu 12/25 den 2/26 co thang sau luon cao hon thang truoc khong?", as_of, cal)
    assert tw_rng.label == "multi_months"
    assert [w.label for w in tw_rng.multi_windows] == ["2025-12", "2026-01", "2026-02"]


def test_relative_slang_temporal():
    cal = LunarSolarCalendar(Path("train/calendar.csv"))
    as_of = dt.date(2026, 7, 22)

    tw = resolve_question_time_window("tuan trc doanh thu quan duoc bnhiu?", as_of, cal)
    assert tw.label == "last_week"


def test_medical_advice_unanswerable():
    cal = LunarSolarCalendar(Path("train/calendar.csv"))
    as_of = dt.date(2026, 7, 22)

    # Mock empty or simple biz package
    from src.loader import load_business_package
    from src.canonical import CanonicalBusiness

    # Try loading dev_006 or biz_101
    pkg_dir = Path("public_test/dev_006")
    if pkg_dir.exists():
        pkg = load_business_package(pkg_dir)
        cb = CanonicalBusiness.from_package(pkg)
        pq = parse_question("q_fa1084dc1d3f", "dev_006", "Bà bầu dùng Acemuc Kids 200mg hộp 30 gói được không?", as_of, cal, cb)
        assert pq.intent == "unanswerable_subjective"
        assert pq.expected_type == "unanswerable"


def test_auto_repair_avg_ticket_parsing():
    cal = LunarSolarCalendar(Path("train/calendar.csv"))
    as_of = dt.date(2026, 7, 22)

    pkg_dir = Path("public_test/dev_012")
    if pkg_dir.exists():
        from src.loader import load_business_package
        from src.canonical import CanonicalBusiness
        pkg = load_business_package(pkg_dir)
        cb = CanonicalBusiness.from_package(pkg)
        pq = parse_question(
            "q_test",
            "dev_012",
            "không tính xe quay lại bảo hành, bình quân 1 phiếu sửa hết bnhiu tiền?",
            as_of,
            cal,
            cb,
        )
        assert pq.intent == "repair_avg_paid_ticket"
        assert pq.expected_type == "number"


def test_verifier_domain_plausibility():
    # Percentage > 1000% is gated to abstain
    res_bad_pct = ExecutionResult("percent", 1450.5, confidence=1.0)
    gated = verify_and_gate(res_bad_pct)
    assert gated.answer_type == "abstain"

    # Normal percentage passes
    res_ok_pct = ExecutionResult("percent", 25.5, confidence=1.0)
    assert verify_and_gate(res_ok_pct).answer_type == "percent"

    # Low confidence < 0.65 gates to abstain
    res_low = ExecutionResult("number", 100000, confidence=0.5)
    assert verify_and_gate(res_low).answer_type == "abstain"
