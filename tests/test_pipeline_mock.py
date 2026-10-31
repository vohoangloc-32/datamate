"""
End-to-end integration test for DataMatePipeline using mock LLM generator.
"""

import json
import pytest
from pathlib import Path
from src.predict import DataMatePipeline
from src.evaluate import compute_pa


def mock_llm(prompt: str) -> str:
    """Mock LLM code generator that produces valid python code."""
    task_section = prompt.split("## YOUR TASK")[-1] if "## YOUR TASK" in prompt else prompt
    if "doanh thu" in task_section.lower():
        return """```python
result = float(orders['amount'].sum())
answer_type = "number"
```"""
    elif "bán chạy" in task_section.lower():
        return """```python
top_item = orders.sort_values(by='qty', ascending=False).iloc[0]['product']
result = str(top_item)
answer_type = "string"
```"""
    else:
        return """```python
result = None
answer_type = "abstain"
```"""


def test_pipeline_end_to_end(tmp_path):
    data_dir = tmp_path / "input_data"
    output_dir = tmp_path / "result"
    data_dir.mkdir()
    output_dir.mkdir()

    # 1. Setup mock business data
    biz_dir = data_dir / "data" / "biz_test"
    biz_dir.mkdir(parents=True)

    csv_path = biz_dir / "orders.csv"
    csv_path.write_text(
        "order_id,product,qty,amount\n"
        "1,Cà phê sữa,10,250000\n"
        "2,Bánh mì,20,100000\n",
        encoding="utf-8"
    )

    readme_path = biz_dir / "README.md"
    readme_path.write_text("# Tiệm Cà Phê & Bánh Mì Test\nBán đồ ăn sáng và nước uống.", encoding="utf-8")

    # 2. Setup rulebook and calendar
    rulebook_path = data_dir / "rulebook.md"
    rulebook_path.write_text('"Bán chạy" = Theo số lượng (qty).\n', encoding="utf-8")

    calendar_path = data_dir / "calendar.csv"
    calendar_path.write_text("solar_date,lunar_year,lunar_month,lunar_day\n", encoding="utf-8")

    # 3. Setup questions.jsonl
    questions_path = data_dir / "questions.jsonl"
    q1 = {
        "id": "q_001",
        "business_id": "biz_test",
        "question": "Tổng doanh thu bán hàng là bao nhiêu?",
        "as_of_date": "2024-05-18"
    }
    q2 = {
        "id": "q_002",
        "business_id": "biz_test",
        "question": "Sản phẩm nào bán chạy nhất?",
        "as_of_date": "2024-05-18"
    }
    with open(questions_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(q1, ensure_ascii=False) + "\n")
        f.write(json.dumps(q2, ensure_ascii=False) + "\n")

    # 4. Instantiate and run pipeline with mock LLM
    pipeline = DataMatePipeline(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        llm_generate_fn=mock_llm,
    )
    pipeline.run()

    # 5. Verify submission.jsonl
    submission_path = output_dir / "submission.jsonl"
    assert submission_path.exists()

    lines = [json.loads(line) for line in submission_path.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) == 2

    # Map by question id
    preds = {line["id"]: line for line in lines}

    assert preds["q_001"]["type"] == "number"
    assert preds["q_001"]["answer"] == 350000.0

    assert preds["q_002"]["type"] == "string"
    assert preds["q_002"]["answer"] == "Bánh mì"

    # 6. Verify logs were written
    log_dir = output_dir / "logs"
    assert (log_dir / "q_001.json").exists()
    assert (log_dir / "q_002.json").exists()

    # 7. Evaluate with evaluate.py compute_pa
    refs = {
        "q_001": {"type": "number", "answer": 350000},
        "q_002": {"type": "string", "answer": "Bánh mì"},
    }
    eval_results = compute_pa(preds, refs)
    assert eval_results["correct"] == 2
    assert eval_results["wrong"] == 0
    assert eval_results["penalized_accuracy"] == 1.0
