import pytest
from src.modules.confidence_gate import ConfidenceGate
from src.modules.question_understanding import QuestionInfo


def test_confidence_gate_abstains_on_type_mismatch():
    gate = ConfidenceGate(min_confidence=0.45)
    q_info = QuestionInfo(
        expected_type="number",
        confidence=0.9,
    )
    answer = {"type": "string", "answer": "Sản phẩm A"}
    result = gate.evaluate(answer, question_info=q_info)
    assert result["type"] == "abstain"
    assert "type_mismatch" in result["reason"]


def test_confidence_gate_abstains_on_zero_after_retries():
    gate = ConfidenceGate(min_confidence=0.45)
    q_info = QuestionInfo(
        expected_type="number",
        confidence=0.8,
        aggregation="sum",
    )
    answer = {"type": "number", "answer": 0}
    # 2 retries + zero numeric -> confidence drops below 0.45
    result = gate.evaluate(answer, question_info=q_info, execution_retries=2)
    assert result["type"] == "abstain"
    assert "zero_numeric" in result["reason"]


def test_confidence_gate_accepts_clean_high_confidence_answer():
    gate = ConfidenceGate(min_confidence=0.45)
    q_info = QuestionInfo(
        expected_type="number",
        confidence=0.8,
    )
    answer = {"type": "number", "answer": 4500000}
    result = gate.evaluate(answer, question_info=q_info, execution_retries=0)
    assert result["type"] == "number"
    assert result["answer"] == 4500000


def test_confidence_gate_abstains_on_empty_list():
    gate = ConfidenceGate(min_confidence=0.45)
    q_info = QuestionInfo(
        expected_type="list",
        confidence=0.8,
    )
    answer = {"type": "list", "answer": []}
    result = gate.evaluate(answer, question_info=q_info)
    assert result["type"] == "abstain"


def test_confidence_gate_schema_answerability_layer1():
    import pandas as pd
    gate = ConfidenceGate(min_confidence=0.45)
    tables = {
        "don_hang": pd.DataFrame({"Mã đơn": ["ORD1"], "Tổng tiền": [100000]}),
        "khach_hang": pd.DataFrame({"Mã KH": ["KH1"], "Tên KH": ["Nguyễn Văn A"]}),
    }

    # Gender not recorded
    res_gender = gate.evaluate(
        {"type": "number", "answer": 5},
        question="Có bao nhiêu khách hàng nam?",
        tables=tables,
    )
    assert res_gender["type"] == "unanswerable"
    assert "gender" in res_gender["reason"]

    # Cancel reason not recorded
    res_cancel = gate.evaluate(
        {"type": "number", "answer": 2},
        question="Có bao nhiêu đơn hàng bị hủy vì giao trễ?",
        tables=tables,
    )
    assert res_cancel["type"] == "unanswerable"
    assert "cancel_reason" in res_cancel["reason"]

    # Import table not recorded
    res_import = gate.evaluate(
        {"type": "number", "answer": 5000000},
        question="Tổng tiền nhập hàng trong tháng qua là bao nhiêu?",
        tables=tables,
    )
    assert res_import["type"] == "unanswerable"
    assert "import_table" in res_import["reason"]


def test_confidence_gate_static_ast_audit_layer2():
    from datetime import date
    gate = ConfidenceGate(min_confidence=0.45)
    q_info = QuestionInfo(
        expected_type="number",
        confidence=0.9,
        time_expressions=[("tháng 10", date(2025, 10, 1), date(2025, 10, 31))],
    )

    # 1. Missing temporal filter on time-constrained question
    code_no_time = "result = int(df['Tổng tiền'].sum())\nanswer_type = 'number'"
    res_time = gate.evaluate(
        {"type": "number", "answer": 10000000},
        question_info=q_info,
        question="Doanh thu tháng 10 là bao nhiêu?",
        code=code_no_time,
    )
    assert res_time["type"] == "abstain"
    assert "missing_temporal_filter" in res_time["reason"]

    # 2. Unauthorized unit division (e.g. division by 1e9 because of slang 'bn')
    code_div = "result = df['Tổng tiền'].sum() / 1_000_000_000\nanswer_type = 'number'"
    res_div = gate.evaluate(
        {"type": "number", "answer": 0.26},
        question="Doanh thu tháng 10 bn?",
        code=code_div,
    )
    assert res_div["type"] == "abstain"
    assert "unauthorized_unit_division" in res_div["reason"]

    # 3. Count vs Sum mismatch: question asks for money but code used len()
    code_len = "result = len(df)\nanswer_type = 'number'"
    res_len = gate.evaluate(
        {"type": "number", "answer": 42},
        question="Tổng doanh thu cửa hàng là bao nhiêu tiền?",
        code=code_len,
    )
    assert res_len["type"] == "abstain"
    assert "count_used_for_monetary" in res_len["reason"]


def test_confidence_gate_domain_plausibility_layer3():
    gate = ConfidenceGate(min_confidence=0.45)

    # 1. Tiny VND revenue for a store (e.g. 0.26 VND)
    res_vnd = gate.evaluate(
        {"type": "number", "answer": 0.26},
        question="Doanh thu tháng 10 là bao nhiêu?",
    )
    assert res_vnd["type"] == "abstain"
    assert "implausible_vnd_revenue" in res_vnd["reason"]

    # 2. Implausible single-store order count > 500k
    res_orders = gate.evaluate(
        {"type": "number", "answer": 950000},
        question="Tổng số đơn hàng là bao nhiêu?",
    )
    assert res_orders["type"] == "abstain"
    assert "implausible_order_count" in res_orders["reason"]

    # 3. Rate outside [0, 100]%
    res_rate = gate.evaluate(
        {"type": "percent", "answer": 125.5},
        question="Tỉ lệ hủy đơn hàng là bao nhiêu %?",
    )
    assert res_rate["type"] == "abstain"
    assert "rate_out_of_bounds" in res_rate["reason"]
