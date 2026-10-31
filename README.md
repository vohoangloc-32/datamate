# DataMate — AI Agent for Vietnamese Tabular QA

> **Zalo AI Challenge 2026 — DataMate Track**  
> An autonomous agent answering natural-language Vietnamese business questions directly over structured heterogeneous tabular data (CSV, Excel, SQLite).

---

## 🚀 Key Highlights & Architecture

- **Universal Multi-Format Ingestion**: Auto-detects and loads CSV, multi-sheet XLSX, and SQLite databases with Vietnamese character encoding detection (`charset-normalizer`) and NFC Unicode normalization.
- **Vietnamese Business Context & Time Resolver**: Resolves relative temporal queries ("tháng này", "tuần rồi", "cùng kỳ năm ngoái", "quý này") and lunar calendar events (Tết, tháng Chạp, mùng 10) against `as_of_date` and `calendar.csv`.
- **Rulebook Interpreter**: Automatically extracts and enforces competition business logic (revenue formulas, cost calculations, "bán chạy" = quantity ranking, VAT & delivery fee exclusion, cancelled order handling).
- **Sandboxed Code Execution & Self-Correction**: Translates questions into executable pandas/Python code executed in a restricted subprocess with automated retry and syntax/runtime error feedback.
- **Strict Format Compliance**: Guarantees typed and normalized answers (`number`, `percent`, `string`, `date`, `boolean`, `list`, `table`, `unanswerable`, `abstain`) matching the competition evaluation metric.
- **Risk-Calibrated Abstention**: Optimized for Penalized Accuracy (PA = `(correct - wrong) / total`), abstaining (score = 0) rather than guessing when data is missing or confidence is low.
- **Offline & Multi-GPU Ready**: Fully self-contained Docker container optimized for 2× NVIDIA RTX 5090 (Blackwell architecture, CUDA 12.8+, vLLM / HuggingFace Transformers).

---

## 📁 Repository Structure

```
d:\datamate\
├── data/
│   ├── train/                  # Training businesses data + answers.jsonl
│   └── public_test/            # Public test data
├── docker/
│   ├── Dockerfile              # Offline CUDA 12.8+ submission image
│   └── predict.sh              # Container entrypoint script
├── prompts/
│   └── templates/
│       ├── system_prompt.txt   # Base system instruction for code generation
│       └── few_shot_examples.json # Curated few-shot demonstrations by question type
├── src/
│   ├── modules/
│   │   ├── answer_formatter.py       # Type-safe normalization & formatting
│   │   ├── code_executor.py          # Sandboxed Python code executor
│   │   ├── confidence_gate.py        # Uncertainty & abstention decision gate
│   │   ├── data_loader.py            # CSV / XLSX / SQLite reader & cache
│   │   ├── prompt_builder.py         # Dynamic prompt constructor
│   │   ├── question_logger.py        # Per-question JSON execution trace logger
│   │   ├── question_understanding.py # Intent & entity classification
│   │   ├── readme_parser.py          # Business README.md extractor
│   │   ├── retry_engine.py           # LLM generation & execution retry loop
│   │   ├── rulebook_interpreter.py   # Rulebook parser & rule matcher
│   │   ├── schema_registry.py        # Unified table schema & metadata store
│   │   └── time_resolver.py          # Vietnamese temporal & lunar resolver
│   ├── predict.py              # Main orchestrator CLI
│   └── evaluate.py             # Local Penalized Accuracy (PA) scorer
├── tests/                      # Full pytest unit & integration test suite (55 tests)
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup & Installation

### Local Environment
Requires Python 3.10+ (tested on Python 3.11 & 3.14):

```bash
pip install -r requirements.txt
```

### Running Tests
Execute the entire test suite (all 55 unit and integration tests):

```bash
pytest tests/ -v
```

---

## ⚡ Inference & Submission Pipeline

### 1. Local Prediction CLI
```bash
python src/predict.py \
    --data-dir data/train \
    --output-dir result/ \
    --model-dir /path/to/qwen2.5-coder-14b \
    --max-retries 3 \
    --confidence-threshold 0.35
```

### 2. Local Penalized Accuracy (PA) Evaluation
Evaluate predictions on the training split:
```bash
python src/evaluate.py \
    --submission result/submission.jsonl \
    --answers data/train/answers.jsonl \
    --output result/eval_summary.json
```

---

## 🐳 Docker Deployment

Build the offline container image:
```bash
docker build -t datamate-agent:latest -f docker/Dockerfile .
```

Run inference using the official competition volume mounts:
```bash
docker run --gpus all --rm \
    -v /path/to/host/data:/data:ro \
    -v /path/to/host/result:/result:rw \
    -v /path/to/host/models:/app/models:ro \
    datamate-agent:latest
```

Entrypoint `/app/predict.sh` automatically reads questions from `/data/questions.jsonl`, queries data in `/data/data/<biz_id>/`, and writes the formatted predictions to `/result/submission.jsonl` and detailed execution traces to `/result/logs/`.
