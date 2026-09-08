"""
scripts/audit_heuristics_triggers.py — Phase 0 / Phase 1 support
=================================================================
Đo tỷ lệ trigger của PermissionGate signals trên benign set.
Xác định combo/signal nào gây FP nhiều nhất.
Phát hiện double-counting check_function_signature.

Usage:
    python scripts/audit_heuristics_triggers.py
    python scripts/audit_heuristics_triggers.py --dataset path/to/data.jsonl
"""
import json
import os
import sys
import re
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.security.advanced_heuristics import (
    PermissionGate, Canonicalizer, VotingAggregator, RiskSignal
)
from models.security.function_risk_registry import check_function_signature
from core.tier_lstm import _extract_tool_and_resource

DEFAULT_DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "benchmark_5000_comprehensive.jsonl",
)


def infer_action_type(action: str) -> str:
    if re.match(r'^[a-zA-Z_]\w*\s*\(', action.strip()):
        return 'tool_call'
    return 'prompt'


def load_benign_records(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("attack_type") == "benign":
                records.append(record)
    return records


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Audit heuristics trigger rates on benign data")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to benchmark JSONL")
    args = parser.parse_args()

    print(f"[audit] Loading benign records from {args.dataset}")
    records = load_benign_records(args.dataset)
    print(f"[audit] Loaded {len(records)} benign records")

    if not records:
        print("[ERROR] No benign records found!")
        sys.exit(1)

    gate = PermissionGate()

    # Counters
    signal_counter = Counter()            # signal name → trigger count
    combo_counter = Counter()             # combo description → trigger count
    double_count_cases = 0                # both places fired for same action
    per_tool_double = Counter()           # tool_name → double-count occurrences
    upstream_scores = []                  # track upstream_risk_score from Tier0.5
    high_risk_tool_triggers = Counter()   # tool_name → times check_function_signature fires
    records_with_any_signal = 0
    records_with_critical = 0

    for i, record in enumerate(records):
        action = record.get("action", "")
        action_type = record.get("action_type") or infer_action_type(action)
        tool_name, _ = _extract_tool_and_resource(action, action_type)

        try:
            canonicalized = Canonicalizer.canonicalize(action)
        except Exception:
            canonicalized = action

        skip_rce = (tool_name in {"Read", "Glob", "Grep", "ls", "cat", "find", "stat",
                                   "read_file", "search_dir"})

        # 1) PermissionGate.detect() — includes its OWN check_function_signature call
        try:
            gate_signals = gate.detect(
                canonicalized,
                session=None,
                enable_provenance=False,
                skip_rce=skip_rce,
                tool_name=tool_name,
            )
        except Exception as e:
            print(f"  [WARN] gate.detect error at record {i}: {e}")
            gate_signals = []

        # 2) pipeline.py's SECOND check_function_signature call (simulating current behavior)
        pipeline_func_signal = None
        if tool_name:
            pipeline_func_signal = check_function_signature(tool_name)

        # Count signals from gate
        if gate_signals:
            records_with_any_signal += 1
        if any(s.is_critical for s in gate_signals):
            records_with_critical += 1

        for sig in gate_signals:
            signal_counter[sig.name] += 1
            if sig.name == "high_risk_capability_combo" and sig.evidence:
                combo_counter[sig.evidence[0]] += 1

        # Detect double-counting
        gate_has_func_sig = any(
            s.name.startswith("high_risk_function:") for s in gate_signals
        )
        if gate_has_func_sig and pipeline_func_signal is not None:
            double_count_cases += 1
            if tool_name:
                per_tool_double[tool_name] += 1

        # Track which tools trigger function_risk_registry
        if pipeline_func_signal is not None and tool_name:
            high_risk_tool_triggers[tool_name] += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(records)}")

    # ─── Report ───
    print(f"\n{'='*72}")
    print("HEURISTICS TRIGGER AUDIT — BENIGN SET")
    print(f"{'='*72}")
    print(f"Total benign records: {len(records)}")
    print(f"Records with ANY signal: {records_with_any_signal} "
          f"({100*records_with_any_signal/max(len(records),1):.1f}%)")
    print(f"Records with CRITICAL signal: {records_with_critical} "
          f"({100*records_with_critical/max(len(records),1):.1f}%)")

    print(f"\n--- Double-counting diagnosis ---")
    print(f"Cases where check_function_signature fires in BOTH "
          f"PermissionGate.detect() AND pipeline.py: "
          f"{double_count_cases} ({100*double_count_cases/max(len(records),1):.1f}%)")

    print(f"\n--- Signal trigger counts (top 20) ---")
    for name, count in signal_counter.most_common(20):
        pct = 100 * count / len(records)
        print(f"  {name}: {count} ({pct:.1f}%)")

    print(f"\n--- HIGH_RISK_COMBO triggers on benign ---")
    if combo_counter:
        for combo, count in combo_counter.most_common(10):
            print(f"  {combo}: {count}")
    else:
        print("  (none)")

    print(f"\n--- High-risk function triggers on benign (from registry) ---")
    for tool, count in high_risk_tool_triggers.most_common(15):
        print(f"  {tool}: {count}")

    print(f"\n--- Double-counted tools (fire in BOTH places) ---")
    if per_tool_double:
        for tool, count in per_tool_double.most_common(10):
            print(f"  {tool}: {count}")
    else:
        print("  (none — no overlap detected)")

    # Simulate VotingAggregator impact
    print(f"\n--- VotingAggregator impact simulation ---")
    would_deny_single = 0
    would_deny_double = 0
    for record in records:
        action = record.get("action", "")
        action_type = record.get("action_type") or infer_action_type(action)
        tool_name, _ = _extract_tool_and_resource(action, action_type)
        try:
            canonicalized = Canonicalizer.canonicalize(action)
        except Exception:
            canonicalized = action
        skip_rce = (tool_name in {"Read", "Glob", "Grep", "ls", "cat", "find", "stat",
                                   "read_file", "search_dir"})
        try:
            gate_signals = gate.detect(canonicalized, session=None,
                                        enable_provenance=False, skip_rce=skip_rce,
                                        tool_name=tool_name)
        except Exception:
            gate_signals = []

        # Single-count: only gate signals
        tier_single, _, *_ = VotingAggregator.vote(gate_signals)
        if tier_single.value in ("DENY", "QUARANTINE"):
            would_deny_single += 1

        # Double-count: gate signals + pipeline duplicate
        double_signals = list(gate_signals)
        if tool_name:
            dup = check_function_signature(tool_name)
            if dup:
                double_signals.append(dup)
        tier_double, _, *_ = VotingAggregator.vote(double_signals)
        if tier_double.value in ("DENY", "QUARANTINE"):
            would_deny_double += 1

    print(f"  Would DENY/QUARANTINE (single-count): {would_deny_single} "
          f"({100*would_deny_single/max(len(records),1):.1f}%)")
    print(f"  Would DENY/QUARANTINE (double-count): {would_deny_double} "
          f"({100*would_deny_double/max(len(records),1):.1f}%)")
    print(f"  Delta (double - single): {would_deny_double - would_deny_single}")


if __name__ == "__main__":
    main()
