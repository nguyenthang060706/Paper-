import os
import sys
import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_latency_breakdown(evaluation_records):
    """
    Computes fine-grained latency breakdown and path correctness.
    
    evaluation_records: list of dicts with keys:
      'latency_ms', 'decision', 'ground_truth', 'path_taken', 'layer'
    """
    groups = {
        "Overall": [],
        "Allowed": [],
        "Blocked": [],
        "Fast-Path (All)": [],
        "Fast-Path (Tier0)": [],
        "Fast-Path (LSTM)": [],
        "Fast-Path (V61-ML)": [],
        "Slow-Path (LLM-Judge)": []
    }
    
    correctness_by_group = {k: [] for k in groups.keys()}

    for r in evaluation_records:
        lat = float(r.get("latency_ms", 0.0))
        dec = r.get("decision", "ALLOW")
        gt = r.get("ground_truth", "benign")
        path = r.get("path_taken", "fast-path-ml")
        layer = r.get("layer", "Unknown")

        is_malicious_gt = (gt != "benign")
        is_blocked_dec = (dec in ("DENY", "BLOCK", "QUARANTINE"))
        is_correct = (is_malicious_gt == is_blocked_dec)

        # Overall
        groups["Overall"].append(lat)
        correctness_by_group["Overall"].append(is_correct)

        # Allowed vs Blocked
        if is_blocked_dec:
            groups["Blocked"].append(lat)
            correctness_by_group["Blocked"].append(is_correct)
        else:
            groups["Allowed"].append(lat)
            correctness_by_group["Allowed"].append(is_correct)

        # Path categories
        if path == "slow-path-llm" or layer == "V61-Slow":
            groups["Slow-Path (LLM-Judge)"].append(lat)
            correctness_by_group["Slow-Path (LLM-Judge)"].append(is_correct)
        else:
            groups["Fast-Path (All)"].append(lat)
            correctness_by_group["Fast-Path (All)"].append(is_correct)

            if layer == "Tier0":
                groups["Fast-Path (Tier0)"].append(lat)
                correctness_by_group["Fast-Path (Tier0)"].append(is_correct)
            elif layer == "Tier0.5-LSTM":
                groups["Fast-Path (LSTM)"].append(lat)
                correctness_by_group["Fast-Path (LSTM)"].append(is_correct)
            else:
                groups["Fast-Path (V61-ML)"].append(lat)
                correctness_by_group["Fast-Path (V61-ML)"].append(is_correct)

    rows = []
    for gname, lats in groups.items():
        if not lats:
            continue
        corr_list = correctness_by_group[gname]
        acc = (sum(corr_list) / len(corr_list) * 100.0) if corr_list else 0.0
        rows.append({
            "Path / Group": gname,
            "Count": len(lats),
            "Mean (ms)": round(float(np.mean(lats)), 2),
            "Std (ms)": round(float(np.std(lats)), 2),
            "p50 Median (ms)": round(float(np.percentile(lats, 50)), 2),
            "p90 (ms)": round(float(np.percentile(lats, 90)), 2),
            "p95 (ms)": round(float(np.percentile(lats, 95)), 2),
            "p99 (ms)": round(float(np.percentile(lats, 99)), 2),
            "Min (ms)": round(float(np.min(lats)), 2),
            "Max (ms)": round(float(np.max(lats)), 2),
            "Decision Correctness (%)": round(acc, 2)
        })

    df = pd.DataFrame(rows)
    return df

def test_latency_breakdown_module():
    print("=" * 80)
    print("  E5: KIỂM THỬ MODULE PHÂN RÃ ĐỘ TRỄ & PATH CORRECTNESS")
    print("=" * 80)

    # Tạo synthetic evaluation data đa dạng các path
    np.random.seed(42)
    sample_records = []
    
    # 1. Tier 0 Fast-path (0.5 - 2ms)
    for _ in range(500):
        sample_records.append({
            "latency_ms": np.random.uniform(0.5, 2.0),
            "decision": "DENY",
            "ground_truth": "malicious",
            "path_taken": "fast-path-regex",
            "layer": "Tier0"
        })

    # 2. Tier 0.5-LSTM Fast-path (2 - 8ms)
    for _ in range(1200):
        is_mal = np.random.rand() > 0.1
        sample_records.append({
            "latency_ms": np.random.normal(5.0, 1.5),
            "decision": "DENY" if is_mal else "ALLOW",
            "ground_truth": "malicious" if is_mal else "benign",
            "path_taken": "fast-path-lstm",
            "layer": "Tier0.5-LSTM"
        })

    # 3. V61 ML Fast-path (5 - 15ms)
    for _ in range(2500):
        is_mal = np.random.rand() > 0.8
        sample_records.append({
            "latency_ms": np.random.normal(10.0, 2.5),
            "decision": "DENY" if is_mal else "ALLOW",
            "ground_truth": "malicious" if is_mal else "benign",
            "path_taken": "fast-path-ml",
            "layer": "V61"
        })

    # 4. Slow-path LLM Judge (150 - 450ms)
    for _ in range(150):
        is_mal = np.random.rand() > 0.2
        sample_records.append({
            "latency_ms": np.random.normal(250.0, 50.0),
            "decision": "DENY" if is_mal else "ALLOW",
            "ground_truth": "malicious" if is_mal else "benign",
            "path_taken": "slow-path-llm",
            "layer": "V61-Slow"
        })

    df = compute_latency_breakdown(sample_records)
    print("\n" + df.to_string(index=False))

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "benchmark_latency_breakdown.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved latency breakdown report to: {csv_path}")

    # Acceptance Criteria validation
    overall_p95 = df.loc[df["Path / Group"] == "Overall", "p95 (ms)"].values[0]
    print(f"\n  Overall p95 Latency: {overall_p95:.2f}ms (Target <= 300ms)")
    assert overall_p95 <= 300.0, f"Overall p95 must be <= 300ms, got {overall_p95}"

    slow_acc = df.loc[df["Path / Group"] == "Slow-Path (LLM-Judge)", "Decision Correctness (%)"].values[0]
    fast_acc = df.loc[df["Path / Group"] == "Fast-Path (All)", "Decision Correctness (%)"].values[0]
    diff = abs(slow_acc - fast_acc)
    print(f"  Slow vs Fast Path Correctness Diff: {diff:.2f}% (Target <= 10%)")
    assert diff <= 10.0, f"Correctness diff should be <= 10%, got {diff:.2f}%"

    print("\n" + "=" * 80)
    print("  => [PASS] E5 LATENCY BREAKDOWN MODULE VERIFIED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_latency_breakdown_module()
