#!/usr/bin/env bash
set -e

echo "============================================================"
echo "🚀 DataMate 1-Click Runner for Dual RTX 5090 (Vast.ai)"
echo "============================================================"

# 1. Hardware verification & Pre-flight checks
nvidia-smi

# Check free disk space (in GB)
FREE_DISK_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
echo "💾 Available Disk Space: ${FREE_DISK_GB} GB"
if [ "$FREE_DISK_GB" -lt 25 ]; then
    echo "⚠️ WARNING: Free disk space is ${FREE_DISK_GB} GB (< 25 GB)!"
    echo "👉 If your instance has only 16GB disk, please increase Disk Space to 60-80 GB in Vast.ai configuration."
fi

# Detect GPU count for Tensor Parallelism
GPU_COUNT=$(nvidia-smi -L | wc -l)
if [ "$GPU_COUNT" -ge 2 ]; then
    VLLM_TP=2
    echo "🔥 Detected $GPU_COUNT GPUs: Tensor Parallelism VLLM_TP=2 enabled!"
else
    VLLM_TP=1
    echo "⚡ Detected single GPU: Tensor Parallelism VLLM_TP=1 enabled."
fi

# 2. Check / update code
echo "📦 Checking Git repository..."
REPO_URL="https://github.com/vohoangloc-32/datamate.git"
BRANCH="loc"

if [ -d "datamate/.git" ]; then
    echo "Switching to datamate directory and updating..."
    cd datamate
    git fetch origin "$BRANCH" --depth 1
    git checkout "$BRANCH"
    git reset --hard "origin/$BRANCH"
elif [ -d ".git" ]; then
    echo "Updating latest code from branch $BRANCH..."
    git fetch origin "$BRANCH" --depth 1
    git checkout "$BRANCH"
    git reset --hard "origin/$BRANCH"
else
    echo "Cloning repository into datamate directory..."
    rm -rf datamate
    git clone --depth 1 -b "$BRANCH" "$REPO_URL" datamate
    cd datamate
fi

# 3. Dependencies & Unit Tests
echo "🔧 Installing requirements and verifying test suite..."
pip uninstall -y torchaudio 2>/dev/null || true
pip install -q vllm openpyxl xlrd tabulate pytest huggingface_hub pandas hf-transfer
pytest tests/ -q

# 4. Download model weights directly from Hugging Face
MODEL_REPO="Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int8"
LOCAL_MODEL_DIR="$HOME/model_cache/Qwen2.5-Coder-14B-Instruct-GPTQ-Int8"

export HF_HUB_ENABLE_HF_TRANSFER="1"

if [ ! -d "$LOCAL_MODEL_DIR" ] || [ $(ls -1 "$LOCAL_MODEL_DIR" | wc -l) -lt 5 ]; then
    echo "⬇️ Downloading $MODEL_REPO to $LOCAL_MODEL_DIR..."
    python3 -c "
from huggingface_hub import snapshot_download
import os
os.makedirs('$LOCAL_MODEL_DIR', exist_ok=True)
snapshot_download(repo_id='$MODEL_REPO', local_dir='$LOCAL_MODEL_DIR', local_dir_use_symlinks=False, max_workers=8)
"
    echo "✅ Model download complete!"
else
    echo "✅ Model weights already present in $LOCAL_MODEL_DIR"
fi

# 5. Execution parameters
MODE=${1:-"train_unseen_500"}
CONFIDENCE=${2:-"0.45"}
OUTPUT_DIR="results"
mkdir -p "$OUTPUT_DIR"

if [ "$MODE" = "public_test" ]; then
    DATA_DIR="data/public_test"
    QUESTIONS_FILE="data/public_test/questions.jsonl"
elif [ "$MODE" = "train_unseen_500" ]; then
    DATA_DIR="data/train"
    QUESTIONS_FILE="data/sample_questions_unseen_500.jsonl"
elif [ "$MODE" = "train_500" ]; then
    DATA_DIR="data/train"
    QUESTIONS_FILE="data/sample_questions_500.jsonl"
elif [ "$MODE" = "train_200" ]; then
    DATA_DIR="data/train"
    QUESTIONS_FILE="data/sample_questions_200.jsonl"
else
    echo "Unknown mode: $MODE"
    exit 1
fi

echo "============================================================"
echo "Starting Inference (Dual RTX 5090 Optimized)..."
echo "Mode           : $MODE"
echo "Data Dir       : $DATA_DIR"
echo "Questions File : $QUESTIONS_FILE"
echo "Confidence     : $CONFIDENCE"
echo "VLLM_TP        : $VLLM_TP"
echo "============================================================"

export VLLM_TP="$VLLM_TP"
export VLLM_DTYPE=auto
export VLLM_MAX_MODEL_LEN=16384
export VLLM_MAX_NEW_TOKENS=1024

python3 src/predict.py \
    --data-dir "$DATA_DIR" \
    --questions-file "$QUESTIONS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --model-dir "$LOCAL_MODEL_DIR" \
    --confidence-threshold "$CONFIDENCE" \
    --log-level INFO

# 6. Evaluation (if ground truth answers exist)
if [ -f "$DATA_DIR/answers.jsonl" ]; then
    echo "============================================================"
    echo "🏆 Computing Official Penalized Accuracy (PA)..."
    echo "============================================================"
    python3 -c "
import json, pandas as pd
from src.evaluate import compute_pa, score_answer

sub_path = '$OUTPUT_DIR/submission.jsonl'
ans_path = '$DATA_DIR/answers.jsonl'
q_path = '$QUESTIONS_FILE'

sub_records = [json.loads(l) for l in open(sub_path, encoding='utf-8-sig') if l.strip()]
preds = {r['id']: r for r in sub_records}
refs = {json.loads(l)['id']: json.loads(l) for l in open(ans_path, encoding='utf-8-sig') if l.strip() and json.loads(l)['id'] in preds}

metrics = compute_pa(preds, refs)
total = metrics['total']
correct = metrics['correct']
wrong = metrics['wrong']
abstain = metrics['abstain']
pa = metrics['pa']

print('\n' + '=' * 60)
print('🏆 DATAMATE OFFICIAL BENCHMARK SCORE (EVALUATED SUBSET)')
print('=' * 60)
print(f'Total Questions  : {total}')
print(f'Correct (+1 pt)  : {correct} ({correct/total*100:.1f}%)')
print(f'Wrong   (-1 pt)  : {wrong} ({wrong/total*100:.1f}%)')
print(f'Abstained (0 pt) : {abstain} ({abstain/total*100:.1f}%)')
print('-' * 60)
print(f'🔥 Penalized Accuracy (PA): {pa:+.4f} ({pa*100:+.2f}%)')
print('=' * 60)

type_rows = []
for t, s in sorted(metrics.get('per_type', {}).items()):
    tot = s['total']
    cor = s['correct']
    wro = s['wrong']
    abs_ = s['abstain']
    sub_pa = (cor - wro) / tot if tot > 0 else 0.0
    type_rows.append({
        'Type': t,
        'Total': tot,
        'Correct': cor,
        'Wrong': wro,
        'Abstain': abs_,
        'PA': f'{sub_pa:+.4f}',
        'Acc%': f'{cor/tot*100:.1f}%' if tot > 0 else '0.0%'
    })
print('\n📊 DETAILED BREAKDOWN BY QUESTION TYPE:')
print(pd.DataFrame(type_rows).to_string(index=False))
print('=' * 60)

# Generate error_analysis.csv & abstain_analysis.csv
q_map = {json.loads(l)['id']: json.loads(l) for l in open(q_path, encoding='utf-8-sig') if l.strip()}

errors, abstains = [], []
for qid, gt in refs.items():
    if qid not in preds: continue
    sub = preds[qid]
    sc = score_answer(sub, gt)
    q_meta = q_map.get(qid, {})
    row = {
        'id': qid,
        'business_id': q_meta.get('business_id', ''),
        'question': q_meta.get('question', ''),
        'pred_type': sub.get('type'),
        'gt_type': gt.get('type'),
        'pred_val': sub.get('value'),
        'gt_val': gt.get('value'),
        'as_of_date': q_meta.get('as_of_date', '')
    }
    if sc < 0: errors.append(row)
    elif sc == 0: abstains.append(row)

pd.DataFrame(errors).to_csv('$OUTPUT_DIR/error_analysis.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(abstains).to_csv('$OUTPUT_DIR/abstain_analysis.csv', index=False, encoding='utf-8-sig')
print(f'📊 Generated $OUTPUT_DIR/error_analysis.csv ({len(errors)} errors)')
print(f'📊 Generated $OUTPUT_DIR/abstain_analysis.csv ({len(abstains)} abstains)')
"
fi

echo "============================================================"
echo "✅ Finished successfully! Results are saved in $OUTPUT_DIR"
echo "============================================================"
