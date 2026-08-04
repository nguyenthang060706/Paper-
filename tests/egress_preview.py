import json
import time
import os
import sys

# Ensure imports work
sys.path.insert(0, r"d:\DEMO_GROUP_1")

from core.pipeline import UnifiedFirewallPipeline

def preview_egress_benchmark():
    pipeline = UnifiedFirewallPipeline()
    records = []
    with open("d:/DEMO_GROUP_1/Benchmark_Datasets/output/evo_pca_full_neuralchemy_backup_20260702.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 1000:  # just 1000 samples for a quick preview
                break
            records.append(json.loads(line))
            
    benign = [r for r in records if r.get("label", "").lower() == "benign"]
    malicious = [r for r in records if r.get("label", "").lower() == "malicious"]
    
    fp_count = 0
    for r in benign:
        res = pipeline.sanitize(r["action"], tool_name="benchmark")
        if res["decision"] in ["STRIP_AND_WRAP", "QUARANTINE"]:
            fp_count += 1
            
    tp_count = 0
    for r in malicious:
        res = pipeline.sanitize(r["action"], tool_name="benchmark")
        if res["decision"] != "PASS":
            tp_count += 1
            
    fpr = (fp_count / len(benign) * 100) if benign else 0
    tpr = (tp_count / len(malicious) * 100) if malicious else 0
    
    print(f"PREVIEW (1000 samples) -> Benign: {len(benign)}, Malicious: {len(malicious)}")
    print(f"FPR: {fpr:.2f}% | TPR: {tpr:.2f}%")

if __name__ == '__main__':
    preview_egress_benchmark()
