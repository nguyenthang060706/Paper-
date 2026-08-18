def run_egress_benchmark():
    import json, time, os
    from core.pipeline import UnifiedFirewallPipeline
    pipeline = UnifiedFirewallPipeline()
    records = []
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = os.path.join(project_root, "ablation", "output", "evo_pca_full.jsonl")
        with open(dataset_path, "r", encoding="utf-8") as f:
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
    for r in benign:
        res = pipeline.sanitize(r["action"], tool_name="benchmark")
        if res["decision"] in ["STRIP_AND_WRAP", "QUARANTINE"]:
            fp_count += 1
            
    tp_count = 0
    for r in malicious:
        res = pipeline.sanitize(r["action"], tool_name="benchmark")
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

