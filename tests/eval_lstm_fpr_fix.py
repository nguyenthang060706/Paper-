"""
Quick evaluation script to verify FPR on benign sessions with LSTM ON after our fixes.
"""
import json, os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TIER05_LSTM_ENABLED"] = "true"
from core.pipeline import UnifiedFirewallPipeline

pipeline = UnifiedFirewallPipeline(use_synthetic_iat=True)

dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "benchmark_5000_comprehensive.jsonl")

benign_records = []
with open(dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            if r.get("attack_type") == "benign":
                benign_records.append(r)

print(f"Testing on {len(benign_records)} benign records with LSTM ON...")

blocked_total = 0
blocked_by_lstm = 0
blocked_by_heuristics = 0
blocked_by_tier0 = 0
blocked_by_v61 = 0

for i, r in enumerate(benign_records):
    action = r.get("action", "")
    sid = r.get("session_id", "default")
    action_type = "tool_call" if re.match(r'^[a-zA-Z_]\w*\s*\(', action.strip()) else "prompt"
    
    res = pipeline.scan(action, session_id=sid, action_type=action_type)
    
    if res.get("decision") == "BLOCK":
        blocked_total += 1
        layer = res.get("layer", "")
        if "LSTM" in str(layer) or "LSTM" in str(res.get("reason", "")):
            blocked_by_lstm += 1
        elif "Heuristics" in str(layer):
            blocked_by_heuristics += 1
        elif "Tier0" in str(layer):
            blocked_by_tier0 += 1
        elif "V61" in str(layer):
            blocked_by_v61 += 1
            
    if (i + 1) % 500 == 0:
        print(f"  Processed {i+1}/{len(benign_records)}: current blocked = {blocked_total} (LSTM blocks: {blocked_by_lstm})")

fpr = (blocked_total / len(benign_records)) * 100
print("\n" + "=" * 60)
print(f"BENIGN FPR TEST (LSTM ENABLED)")
print("=" * 60)
print(f"Total Benign Records: {len(benign_records)}")
print(f"Total Blocked (FP): {blocked_total} ({fpr:.2f}%)")
print(f"  - Tier 0.5-LSTM Blocks: {blocked_by_lstm}")
print(f"  - Heuristics Blocks: {blocked_by_heuristics}")
print(f"  - Tier 0 / Other Blocks: {blocked_by_tier0}")
print(f"  - V61 Blocks: {blocked_by_v61}")
print("=" * 60)
