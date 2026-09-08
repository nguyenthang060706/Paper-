"""
scripts/calibrate_lstm_threshold.py — Phase 0.1 + 0.2
=====================================================
Đo phân phối LSTM probability trên tập benign thuần từ benchmark dataset.

Bước 0.1: Thu thập prob theo từng step position trong session.
Bước 0.2: Chạy 2 kịch bản — upstream=0.0 (sạch) vs upstream từ pipeline thật.

Output: Bảng percentile (p50/p90/p95/p99/max) → dùng để chọn block_threshold.

Usage:
    python scripts/calibrate_lstm_threshold.py
    python scripts/calibrate_lstm_threshold.py --dataset path/to/data.jsonl
"""
import json
import os
import sys
import re
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tier_lstm import SessionAwareLSTMRisk, _extract_tool_and_resource

# ─── Default dataset ───
DEFAULT_DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "benchmark_5000_comprehensive.jsonl",
)


def infer_action_type(action: str) -> str:
    """Infer action_type from action string (same logic as benchmark_5000.py)."""
    if re.match(r'^[a-zA-Z_]\w*\s*\(', action.strip()):
        return 'tool_call'
    return 'prompt'


def percentile(sorted_vals: list, p: float) -> float:
    """Get p-th percentile from a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    idx = min(int(p * (len(sorted_vals) - 1)), len(sorted_vals) - 1)
    return sorted_vals[idx]


def load_benign_sessions(path: str) -> dict:
    """Load benign records grouped by session_id, sorted by step_index."""
    sessions = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            # Dataset uses attack_type == 'benign', NOT is_malicious
            if record.get("attack_type") != "benign":
                continue
            sid = record.get("session_id", "default")
            sessions[sid].append(record)
    # Sort each session by step_index for correct replay order
    for sid in sessions:
        sessions[sid].sort(key=lambda r: r.get("step_index", 0))
    return dict(sessions)


def get_upstream_risk_score_real(record: dict, tier05_instance, session_id: str) -> float:
    """Tái tạo upstream_risk_score bằng cách chạy Tier0.5 regex thật."""
    action = record.get("action", "")
    action_type = record.get("action_type") or infer_action_type(action)
    tool_name, _ = _extract_tool_and_resource(action, action_type)
    skip_rce = (tool_name in {"Read", "Glob", "Grep", "ls", "cat", "find", "stat",
                               "read_file", "search_dir"})
    try:
        res = tier05_instance.scan(
            action, session_id=session_id, action_type=action_type, skip_rce=skip_rce
        )
        return float(getattr(res, "confidence", 0.0))
    except Exception as e:
        print(f"  [WARN] Tier0.5 scan error: {e}")
        return 0.0


def run_calibration(sessions: dict, upstream_mode: str = "zero",
                    tier05_instance=None) -> list:
    """
    Run LSTM on all benign sessions, collect probabilities.
    
    upstream_mode:
        "zero" → upstream_risk_score = 0.0 for all steps (isolate LSTM behavior)
        "real" → upstream_risk_score from Tier0.5 regex pipeline
    """
    # block_threshold=1.1 → never triggers is_blocked, only measures prob
    lstm = SessionAwareLSTMRisk(block_threshold=1.1, use_synthetic_iat=True)

    probs_by_step = defaultdict(list)
    all_probs = []
    blocked_count = 0
    total_steps = 0

    t0 = time.time()
    for i, (session_id, records) in enumerate(sessions.items()):
        lstm.reset_session(session_id)
        for step_idx, record in enumerate(records, start=1):
            action = record.get("action", "")
            action_type = record.get("action_type") or infer_action_type(action)

            if upstream_mode == "zero":
                upstream = 0.0
            else:
                upstream = get_upstream_risk_score_real(record, tier05_instance, session_id)

            res = lstm.update_and_score(session_id, action, action_type, upstream)
            prob = res["probability"]
            probs_by_step[step_idx].append(prob)
            all_probs.append(prob)
            total_steps += 1

            if res.get("is_blocked"):
                blocked_count += 1

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{upstream_mode}] Processed {i+1}/{len(sessions)} sessions "
                  f"({total_steps} steps, {elapsed:.1f}s)")

    elapsed = time.time() - t0

    # ─── Report ───
    print(f"\n{'='*72}")
    print(f"CALIBRATION REPORT — upstream_mode={upstream_mode}")
    print(f"{'='*72}")
    print(f"Total sessions: {len(sessions)}")
    print(f"Total steps: {total_steps}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Would-be-blocked (at 0.98): {sum(1 for p in all_probs if p >= 0.98)} "
          f"({100*sum(1 for p in all_probs if p >= 0.98)/max(total_steps,1):.1f}%)")
    print(f"Would-be-blocked (at 0.999): {sum(1 for p in all_probs if p >= 0.999)} "
          f"({100*sum(1 for p in all_probs if p >= 0.999)/max(total_steps,1):.1f}%)")
    print(f"Would-be-blocked (at 0.9999): {sum(1 for p in all_probs if p >= 0.9999)} "
          f"({100*sum(1 for p in all_probs if p >= 0.9999)/max(total_steps,1):.1f}%)")

    print(f"\n{'Step':>6} {'N':>6} {'p50':>8} {'p90':>8} {'p95':>8} {'p99':>8} {'max':>8}")
    print("-" * 58)
    for step_idx in sorted(probs_by_step.keys())[:30]:  # show first 30 steps
        vals = sorted(probs_by_step[step_idx])
        n = len(vals)
        print(f"{step_idx:>6} {n:>6} {percentile(vals, 0.50):>8.4f} "
              f"{percentile(vals, 0.90):>8.4f} {percentile(vals, 0.95):>8.4f} "
              f"{percentile(vals, 0.99):>8.4f} {vals[-1]:>8.4f}")

    all_sorted = sorted(all_probs)
    n = len(all_sorted)
    print(f"\nAGGREGATE (all steps):")
    print(f"  n={n}")
    print(f"  p50  = {percentile(all_sorted, 0.50):.6f}")
    print(f"  p90  = {percentile(all_sorted, 0.90):.6f}")
    print(f"  p95  = {percentile(all_sorted, 0.95):.6f}")
    print(f"  p99  = {percentile(all_sorted, 0.99):.6f}")
    print(f"  p99.5= {percentile(all_sorted, 0.995):.6f}")
    print(f"  p99.9= {percentile(all_sorted, 0.999):.6f}")
    print(f"  max  = {all_sorted[-1]:.6f}")

    return all_sorted


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calibrate LSTM block threshold on benign data")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to benchmark JSONL")
    parser.add_argument("--skip-real", action="store_true",
                        help="Skip real-upstream scenario (faster, but no comparison)")
    args = parser.parse_args()

    print(f"[calibrate] Loading benign sessions from {args.dataset}")
    sessions = load_benign_sessions(args.dataset)
    print(f"[calibrate] Loaded {len(sessions)} benign sessions "
          f"({sum(len(v) for v in sessions.values())} total steps)")

    if not sessions:
        print("[ERROR] No benign sessions found in dataset!")
        sys.exit(1)

    # ─── Kịch bản 1: upstream = 0.0 (isolate LSTM behavior) ───
    print("\n>>> Scenario 1: upstream_risk_score = 0.0 (clean)")
    probs_clean = run_calibration(sessions, upstream_mode="zero")

    if not args.skip_real:
        # ─── Kịch bản 2: upstream từ pipeline thật ───
        print("\n>>> Scenario 2: upstream_risk_score from real Tier0.5 pipeline")
        try:
            from core.tier05 import SessionAwareTier05
            from core.tier0 import LlamaFirewallTier0
            tier05 = SessionAwareTier05(tier0=LlamaFirewallTier0())
            probs_real = run_calibration(sessions, upstream_mode="real",
                                         tier05_instance=tier05)
        except Exception as e:
            print(f"[ERROR] Cannot initialize Tier0.5 for real upstream: {e}")
            print("[SKIP] Skipping real-upstream scenario")
            probs_real = None

        # ─── Comparison ───
        if probs_real:
            print(f"\n{'='*72}")
            print("COMPARISON: Clean (upstream=0) vs Real (upstream=pipeline)")
            print(f"{'='*72}")
            for p_label, p_val in [("p50", 0.50), ("p90", 0.90), ("p95", 0.95),
                                    ("p99", 0.99), ("p99.5", 0.995), ("p99.9", 0.999)]:
                clean_v = percentile(probs_clean, p_val)
                real_v = percentile(probs_real, p_val)
                delta = real_v - clean_v
                flag = " ⚠️" if abs(delta) > 0.05 else ""
                print(f"  {p_label:>6}: clean={clean_v:.6f}, real={real_v:.6f}, "
                      f"delta={delta:+.6f}{flag}")

    print("\n[calibrate] Done. Use the p99/p99.5 value + margin as LSTM block_threshold.")


if __name__ == "__main__":
    main()
