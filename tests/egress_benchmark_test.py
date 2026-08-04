import json
import time
import os
import sys

# Ensure imports work
sys.path.insert(0, r"d:\DEMO_GROUP_1")

from core.pipeline import UnifiedFirewallPipeline

def run_egress_benchmark():
    pipeline = UnifiedFirewallPipeline()
    records = []
    try:
        with open("D:/DEMO_GROUP_1/Benchmark_Datasets/output/evo_pca_11k_balanced.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    except Exception as e:
        print(f"Egress benchmark failed to load data: {e}")
        return
        
    benign = [r for r in records if r.get("label", "").lower() == "benign"]
    malicious = [r for r in records if r.get("label", "").lower() == "malicious"]
    
    print(f"Loaded {len(benign)} benign and {len(malicious)} malicious Egress samples.")
    
    t0 = time.time()
    
    fp_count = 0
    benign_texts = [r["action"] for r in benign]
    benign_results = pipeline.sanitize_batch(benign_texts, tool_name="benchmark")
    for res in benign_results:
        if res["decision"] in ["STRIP_AND_WRAP", "QUARANTINE"]:
            fp_count += 1
            
    tp_count = 0
    malicious_texts = [r["action"] for r in malicious]
    malicious_results = pipeline.sanitize_batch(malicious_texts, tool_name="benchmark")
    for res in malicious_results:
        if res["decision"] != "PASS":
            tp_count += 1
            
    latency = time.time() - t0
    
    fpr = (fp_count / len(benign) * 100) if benign else 0
    tpr = (tp_count / len(malicious) * 100) if malicious else 0
    
    print("=" * 80)
    print("  EGRESS BENCHMARK (ContextSanitizer)")
    print("=" * 80)
    print(f"FPR (Destructive on Benign): {fpr:.2f}%")
    print(f"TPR (Detection of Malicious): {tpr:.2f}%")
    print(f"Avg Latency: {(latency / len(records) * 1000):.2f} ms")
    print("=" * 80)

if __name__ == '__main__':
    run_egress_benchmark()
