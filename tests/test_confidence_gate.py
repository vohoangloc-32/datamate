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
