"""Diagnostic script to analyze the ML score distribution of multi-step actions.

Evaluates multi-step actions against V61 ML scorer to determine:
1. Distribution of ml_score for malicious multi-step actions (FNs and overall).
2. Whether FNs concentrate in [0.35, 0.40).
3. Impact of expanding suspicious_band_min from 0.40 to 0.35 on benign actions.
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from core.pipeline import UnifiedFirewallPipeline


def run_diagnostic():
    print("=" * 70)
    print("DIAGNOSTIC: ML Score Distribution for Multi-Step Actions")
    print("=" * 70)

    dataset_path = os.path.join(REPO_ROOT, "tests", "output", "benchmark_3000_fixed_eval.jsonl")
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at {dataset_path}")
        return

    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    print(f"Loaded {len(records)} records from {dataset_path}")

    multi_malicious = [r for r in records if r.get("attack_type") == "malicious_multistep"]
    benign = [r for r in records if r.get("attack_type") == "benign"]

    print(f"Found {len(multi_malicious)} malicious_multistep records and {len(benign)} benign records.")

    pipeline = UnifiedFirewallPipeline()

    bins = [
        ("< 0.20", 0.0, 0.20),
        ("[0.20, 0.35)", 0.20, 0.35),
        ("[0.35, 0.40)", 0.35, 0.40),
        ("[0.40, 0.5577)", 0.40, 0.5577),
        (">= 0.5577", 0.5577, 1.01)
    ]

    mal_counts = defaultdict(int)
    benign_counts = defaultdict(int)
    mal_fn_counts = defaultdict(int)

    mal_scores = []
    benign_scores = []

    print("\nEvaluating malicious multi-step records...")
    for idx, r in enumerate(multi_malicious):
        action = r["action"]
        sid = r.get("session_id", f"diag_mal_{idx}")
        atype = r.get("action_type", "prompt")
        
        res = pipeline.scan(action=action, session_id=sid, action_type=atype, actual_label=True)
        score = res.get("ml_score", 0.0)
        mal_scores.append(score)

        is_blocked = res.get("decision") in ("BLOCK", "DENY", "QUARANTINE")

        for name, low, high in bins:
            if low <= score < high:
                mal_counts[name] += 1
                if not is_blocked:
                    mal_fn_counts[name] += 1
                break

    print(f"Evaluating a sample of {min(500, len(benign))} benign records...")
    for idx, r in enumerate(benign[:500]):
        action = r["action"]
        sid = r.get("session_id", f"diag_benign_{idx}")
        atype = r.get("action_type", "prompt")

        res = pipeline.scan(action=action, session_id=sid, action_type=atype, actual_label=False)
        score = res.get("ml_score", 0.0)
        benign_scores.append(score)

        for name, low, high in bins:
            if low <= score < high:
                benign_counts[name] += 1
                break

    print("\n" + "=" * 70)
    print(f"{'Score Band':<18} | {'Mal Multi Total':<16} | {'Mal Multi FN':<14} | {'Benign (N=500)':<16}")
    print("-" * 70)
    for name, _, _ in bins:
        print(f"{name:<18} | {mal_counts[name]:<16} | {mal_fn_counts[name]:<14} | {benign_counts[name]:<16}")
    print("=" * 70)

    total_mal = len(multi_malicious)
    band_35_40_mal = mal_counts["[0.35, 0.40)"]
    band_35_40_fn = mal_fn_counts["[0.35, 0.40)"]
    band_35_40_benign = benign_counts["[0.35, 0.40)"]

    print(f"\nKey Insights for [0.35, 0.40):")
    print(f"  - Malicious multi-step in [0.35, 0.40): {band_35_40_mal} ({band_35_40_mal/total_mal*100:.1f}%)")
    print(f"  - Malicious multi-step FN in [0.35, 0.40): {band_35_40_fn}")
    print(f"  - Benign in [0.35, 0.40): {band_35_40_benign} / 500 ({band_35_40_benign/5:.1f}%)")

    if band_35_40_fn > 0:
        print(f"\n=> Recommendation: Lowering suspicious_band_min to 0.35 would route {band_35_40_fn} currently missed attacks to LLM Session Judge.")
        if band_35_40_benign / 5 <= 3.0:
            print("   Benign increase in this band is small (<= 3.0%). Safe to lower to 0.35.")
        else:
            print("   WARNING: Benign increase > 3.0%. Consider 0.37 or relying on Heuristic flags.")
    else:
        print(f"\n=> Insight: Zero FNs in [0.35, 0.40). Lowering to 0.35 would NOT improve multi-step recall directly.")


if __name__ == "__main__":
    run_diagnostic()
