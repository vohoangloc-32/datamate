import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

v2_traces = {}
with open("artifacts/traces_public_test_v2.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        v2_traces[d["id"]] = d

v6_traces = {}
with open("artifacts/traces_public_test_v6.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        v6_traces[d["id"]] = d

improvements = []
for qid, t6 in v6_traces.items():
    t2 = v2_traces.get(qid, {})
    ans2 = t2.get("pred_answer") if "pred_answer" in t2 else t2.get("answer")
    ans6 = t6.get("pred_answer")
    type2 = t2.get("pred_type") if "pred_type" in t2 else t2.get("type")
    type6 = t6.get("pred_type")
    if (ans2 != ans6) or (type2 != type6):
        improvements.append((qid, t6.get("question"), type2, ans2, type6, ans6, t6.get("intent")))

print(f"Total improved/changed questions: {len(improvements)} / 1000")
for i, (qid, q, t2, a2, t6, a6, intent) in enumerate(improvements, 1):
    print(f"{i}. [{qid}] ({intent}) {q[:80]}")
    print(f"   BEFORE: {type2} = {str(a2)[:60]}")
    print(f"   AFTER:  {type6} = {str(a6)[:60]}")
    print()
