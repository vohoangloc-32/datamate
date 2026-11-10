#!/usr/bin/env bash
set -e

echo "============================================================"
echo "🚀 DataMate 1-Click Runner for Vast.ai / Linux Workstation"
echo "============================================================"

# 1. Hardware verification
nvidia-smi

# 2. Check / update code
echo "📦 Checking Git repository..."
git fetch origin
git checkout loc
git reset --hard origin/loc

# 3. Dependencies & Unit Tests
echo "🔧 Installing requirements and verifying test suite..."
pip install -q vllm openpyxl xlrd tabulate pytest huggingface_hub pandas
pytest tests/ -q

# 4. Download model weights
MODEL_REPO="Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int8"
LOCAL_MODEL_DIR="$HOME/model_cache/Qwen2.5-Coder-14B-Instruct-GPTQ-Int8"

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
echo "Starting Inference..."
echo "Mode           : $MODE"
echo "Data Dir       : $DATA_DIR"
echo "Questions File : $QUESTIONS_FILE"
echo "Confidence     : $CONFIDENCE"
echo "============================================================"

export VLLM_TP=1
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

metrics = compute_pa(sub_path, ans_path)
print('Total Questions  :', metrics.total_questions)
print(f'Correct (+1 pt)  : {metrics.correct_count} ({metrics.accuracy*100:.1f}%)')
print(f'Wrong   (-1 pt)  : {metrics.wrong_count} ({metrics.wrong_rate*100:.1f}%)')
print(f'Abstained (0 pt) : {metrics.abstain_count} ({metrics.abstain_rate*100:.1f}%)')
print('-' * 60)
print(f'🔥 Penalized Accuracy (PA): {metrics.penalized_accuracy:+.4f} ({metrics.penalized_accuracy*100:+.2f}%)')
print('=' * 60)

if metrics.breakdown_by_type:
    df = pd.DataFrame.from_dict(metrics.breakdown_by_type, orient='index').reset_index()
    df['PA'] = df['pa'].apply(lambda x: f'{x:+.4f}')
    df['Acc%'] = (df['correct'] / df['total'] * 100).apply(lambda x: f'{x:.1f}%')
    print(df[['index', 'total', 'correct', 'wrong', 'abstain', 'PA', 'Acc%']].to_string(index=False))

# Generate error_analysis.csv
sub_map = {json.loads(l)['id']: json.loads(l) for l in open(sub_path, encoding='utf-8-sig') if l.strip()}
gt_map = {json.loads(l)['id']: json.loads(l) for l in open(ans_path, encoding='utf-8-sig') if l.strip()}
q_map = {json.loads(l)['id']: json.loads(l) for l in open(q_path, encoding='utf-8-sig') if l.strip()}

errors, abstains = [], []
for qid, gt in gt_map.items():
    if qid not in sub_map: continue
    sub = sub_map[qid]
    sc = score_answer(sub, gt)
    row = {'id': qid, 'business_id': q_map.get(qid, {}).get('business_id', ''), 'question': q_map.get(qid, {}).get('question', ''), 'pred_type': sub.get('type'), 'gt_type': gt.get('type'), 'pred_val': sub.get('value'), 'gt_val': gt.get('value')}
    if sc < 0: errors.append(row)
    elif sc == 0: abstains.append(row)

pd.DataFrame(errors).to_csv('$OUTPUT_DIR/error_analysis.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(abstains).to_csv('$OUTPUT_DIR/abstain_analysis.csv', index=False, encoding='utf-8-sig')
print(f'📊 Generated $OUTPUT_DIR/error_analysis.csv ({len(errors)} errors)')
"
fi

echo "============================================================"
echo "✅ Finished successfully! Results are saved in $OUTPUT_DIR"
echo "============================================================"
