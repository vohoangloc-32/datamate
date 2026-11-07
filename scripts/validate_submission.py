import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('submission.jsonl', encoding='utf-8') as f:
    lines = [json.loads(line) for line in f]

assert len(lines) == 1000, f'Expected 1000 lines, got {len(lines)}'
valid_types = {'number', 'percent', 'string', 'list', 'date', 'unanswerable', 'table', 'boolean', 'abstain'}
type_counts = {}

for i, r in enumerate(lines, 1):
    assert 'id' in r and 'type' in r and 'answer' in r, f"Line {i} missing key: {r}"
    t = r['type']
    assert t in valid_types, f"Line {i} invalid type: {t}"
    type_counts[t] = type_counts.get(t, 0) + 1
    if t in ('number', 'percent'):
        assert isinstance(r['answer'], (int, float)), f"Line {i} expected numeric answer, got {type(r['answer'])}: {r['answer']}"
    elif t == 'list':
        assert isinstance(r['answer'], list), f"Line {i} expected list answer, got {type(r['answer'])}"
    elif t == 'table':
        assert isinstance(r['answer'], dict) and 'columns' in r['answer'] and 'rows' in r['answer'], f"Line {i} invalid table: {r['answer']}"
    elif t == 'boolean':
        assert isinstance(r['answer'], bool), f"Line {i} expected bool, got {type(r['answer'])}"
    elif t in ('unanswerable', 'abstain'):
        assert r['answer'] is None, f"Line {i} expected None answer, got {r['answer']}"

print("Validation passed: all 1,000 records strictly compliant with competition contract!")
print("Type breakdown:", type_counts)
