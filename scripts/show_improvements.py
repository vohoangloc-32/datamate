import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

v2_traces = {}
with open("artifacts/traces_public_test_v2.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        v2_traces[d["id"]] = d

v8_traces = {}
with open("artifacts/traces_public_test_v8.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        v8_traces[d["id"]] = d

improved = []
for qid, t8 in v8_traces.items():
    t2 = v2_traces.get(qid, {})
    ans2 = t2.get("pred_answer") if "pred_answer" in t2 else t2.get("answer")
    ans8 = t8.get("pred_answer")
    type2 = t2.get("pred_type") if "pred_type" in t2 else t2.get("type")
    type8 = t8.get("pred_type")
    if (ans2 != ans8) or (type2 != type8):
        improved.append((qid, t8.get("question"), type2, ans2, type8, ans8, t8.get("intent")))

print(f"Total improved/changed questions: {len(improved)} / 1000")
for i, (qid, q, t2, a2, t8, a8, intent) in enumerate(improved, 1):
    print(f"{i}. [{qid}] ({intent}) {q[:80]}")
    print(f"   BEFORE: {type2} = {str(a2)[:60]}")
    print(f"   AFTER:  {type8} = {str(a8)[:60]}")
    print()
