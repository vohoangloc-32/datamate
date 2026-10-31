#!/bin/bash
set -e

# DataMate Zalo AI Challenge 2026 - Container Entrypoint
echo "=========================================================="
echo " Starting DataMate QA Agent"
echo "=========================================================="

# Default directories per competition specification
DATA_DIR=${1:-/data}
OUTPUT_DIR=${2:-/result}
MODEL_DIR=${MODEL_DIR:-/app/models/qwen2.5-coder-14b}

echo "Input Data Directory:   $DATA_DIR"
echo "Output Directory:       $OUTPUT_DIR"
echo "Model Directory:        $MODEL_DIR"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Verify input data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory '$DATA_DIR' not found!"
    exit 1
fi

cd /app

# Ensure src is in PYTHONPATH
export PYTHONPATH="/app:${PYTHONPATH}"

# Run prediction pipeline
echo "Running prediction pipeline..."
python -u src/predict.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --model-dir "$MODEL_DIR" \
    --max-retries 3 \
    --confidence-threshold 0.35 \
    --log-level INFO

echo "=========================================================="
echo " DataMate QA Agent Completed Successfully"
echo " Submission file written to $OUTPUT_DIR/submission.jsonl"
echo "=========================================================="
