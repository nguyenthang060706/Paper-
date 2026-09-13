"""
tests/diagnose_tool_call_escalation.py
======================================
Diagnostic Script: Tool-Call Escalation Gap Analysis (Bước 2).

PURPOSE:
  Diagnose tool-call escalation, False Positives (FP), False Negatives (FN),
  layer breakdown, and force_review impact for tool calls under the firewall pipeline.

OUTPUT:
  - Console summary:
      * Tool-call FPR riêng (HIGH-RISK vs NORMAL)
      * Top 10 tool_name bị FP nhiều nhất
      * Top 10 tool_name bị FN (bỏ lọt) nhiều nhất
      * FP breakdown theo layer (V61 / Heuristics / LLM Judge)
      * force_review impact: bao nhiêu benign bị escalate nhầm
  - File: tests/logs/diagnose_tool_call_escalation_report.json
"""

import os
import sys
import json
import re
import time
import uuid
import random
from collections import defaultdict, Counter

# Ensure repo root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("TIER05_LSTM_ENABLED", "false")
os.environ.setdefault("ESCALATION_FEEDBACK_MODE", "decoupled")
os.environ.setdefault("MULTI_STEP_HEURISTICS_ENABLED", "true")
os.environ.setdefault("LLM_SESSION_JUDGE_ENABLED", "true")

try:
    from core.config_loader import load_settings
    load_settings(override_existing=False)
except ImportError:
    pass

from core.pipeline import UnifiedFirewallPipeline
from models.security.function_risk_registry import HIGH_RISK_FUNCTIONS, check_function_signature
from models.security.advanced_heuristics import PermissionGate


def infer_action_type(action):
    if isinstance(action, dict):
        return 'tool_call'
    if re.match(r'^[a-zA-Z_]\w*\s*\(', str(action).strip()):
        return 'tool_call'
    return 'prompt'


def extract_tool_name(action):
    if isinstance(action, dict):
        return action.get("name") or action.get("tool") or "unknown"
    m = re.match(r'^([a-zA-Z_]\w*)\s*\(', str(action).strip())
    if m:
        return m.group(1)
    return "unknown"


def load_tool_call_dataset(use_full_dataset=False):
    dataset_path = os.path.join(PROJECT_ROOT, "ablation", "output", "evo_pca_full.jsonl")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(PROJECT_ROOT, "output", "evo_pca_full.jsonl")
    
    print(f"Loading dataset from {dataset_path}...")
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"  Total records in dataset: {len(data)}")

    for r in data:
        if 'action_type' not in r or not r['action_type']:
            r['action_type'] = infer_action_type(r.get('action', ''))

    if use_full_dataset:
        tool_calls = [r for r in data if r.get('action_type') == 'tool_call']
        for i, r in enumerate(tool_calls):
            r['session_id'] = r.get('session_id') or f"tc_full_{i}_{uuid.uuid4().hex[:6]}"
            r['step_num'] = r.get('step_num', 1)
        return tool_calls

    # Match benchmark_5000.py stratified sampling exactly
    random.seed(42)
    benign = [r for r in data if r.get('attack_type') == 'benign']
    single = [r for r in data if r.get('attack_type') == 'malicious_single']
    multi  = [r for r in data if r.get('attack_type') == 'malicious_multistep']

    benign_by_source = defaultdict(list)
    for r in benign:
        benign_by_source[r.get('source_dataset', 'unknown')].append(r)
    TARGET_BENIGN = 2000
    sampled_benign = []
    for src, records in benign_by_source.items():
        proportion = len(records) / len(benign)
        n = max(1, round(proportion * TARGET_BENIGN))
        n = min(n, len(records))
        sampled_benign.extend(random.sample(records, n))
    if len(sampled_benign) > TARGET_BENIGN:
        sampled_benign = random.sample(sampled_benign, TARGET_BENIGN)
    for r in sampled_benign:
        r['session_id'] = f"bench_benign_{uuid.uuid4().hex[:8]}"
        r['step_num'] = 1

    single_by_source = defaultdict(list)
    for r in single:
        single_by_source[r.get('source_dataset', 'unknown')].append(r)
    TARGET_SINGLE = 2000
    sampled_single = []
    for src, records in single_by_source.items():
        proportion = len(records) / len(single)
        n = max(1, round(proportion * TARGET_SINGLE))
        n = min(n, len(records))
        sampled_single.extend(random.sample(records, n))
    if len(sampled_single) > TARGET_SINGLE:
        sampled_single = random.sample(sampled_single, TARGET_SINGLE)
    for r in sampled_single:
        r['session_id'] = f"bench_single_{uuid.uuid4().hex[:8]}"
        r['step_num'] = 1

    multi_sessions = defaultdict(list)
    for r in multi:
        multi_sessions[r['session_id']].append(r)
    for sid in multi_sessions:
        multi_sessions[sid].sort(key=lambda x: x.get('step_index', 0))
    sampled_multi = []
    used_sessions = 0
    for sid, session_records in multi_sessions.items():
        new_sid = f"bench_multi_{used_sessions:03d}_{uuid.uuid4().hex[:6]}"
        for step_num, r in enumerate(session_records, start=1):
            r = dict(r)
            r['session_id'] = new_sid
            r['step_num'] = step_num
            sampled_multi.append(r)
        used_sessions += 1

    bench_5k = sampled_benign + sampled_single + sampled_multi
    tool_calls = [r for r in bench_5k if r.get('action_type') == 'tool_call']
    print(f"  Sampled benchmark subset tool_calls: {len(tool_calls)} "
          f"(benign={sum(1 for r in tool_calls if r.get('label') == 'benign')}, "
          f"malicious={sum(1 for r in tool_calls if r.get('label') != 'benign')})")
    return tool_calls


def run_diagnose():
    use_all = "--all" in sys.argv
    tool_calls = load_tool_call_dataset(use_full_dataset=use_all)

    print("\n" + "=" * 80)
    print("  INITIALIZING FIREWALL PIPELINE FOR TOOL-CALL DIAGNOSIS...")
    print("=" * 80)
    pipeline = UnifiedFirewallPipeline(use_synthetic_iat=True)

    # Track metrics
    total_calls = len(tool_calls)
    benign_calls = [r for r in tool_calls if r.get('label') == 'benign']
    malicious_calls = [r for r in tool_calls if r.get('label') != 'benign']

    # FPR metrics
    high_risk_benign_total = 0
    high_risk_benign_fp = 0
    normal_benign_total = 0
    normal_benign_fp = 0

    # Tool-level counters
    tool_fp_count = Counter()
    tool_benign_total = Counter()
    tool_fn_count = Counter()
    tool_malicious_total = Counter()

    # Layer breakdown for FPs
    fp_layer_breakdown = Counter()

    # force_review tracking
    force_review_benign_total = 0
    force_review_benign_fp = 0
    force_review_benign_recovered = 0

    detailed_eval = []

    print(f"\nEvaluating {total_calls} tool_call records...")
    t_start = time.perf_counter()

    for idx, r in enumerate(tool_calls, 1):
        action = r['action']
        session_id = r.get('session_id', 'diag_tool_session')
        label = r.get('label', 'benign')
        is_benign = (label == 'benign')

        tool_name = extract_tool_name(action)
        is_high_risk = tool_name in HIGH_RISK_FUNCTIONS

        # Check pre-scan force_review potential
        session_ref = pipeline.tier05.get_session_ref(session_id)
        user_goal = getattr(session_ref, 'user_goal_text', '') if session_ref else ''
        report = pipeline.tier05.get_session_report(session_id)
        active_flags = report.get('active_flags', [])
        
        would_force_review = False
        if is_high_risk:
            has_taint = PermissionGate.check_taint_overlap(action, session_ref)
            has_exfil = PermissionGate.has_explicit_exfil_target(action, user_goal)
            if len(active_flags) > 0 or has_taint or has_exfil:
                would_force_review = True

        res = pipeline.scan(
            action=action,
            session_id=session_id,
            action_type="tool_call"
        )

        decision = res.get("decision", "ALLOW")
        blocked = decision in ("BLOCK", "QUARANTINE")
        layer = res.get("layer", "Unknown")
        reason = res.get("reason", "")
        was_review = res.get("was_review", False) or res.get("path") == "slow-path-llm"

        # Categorize normalized layer
        if layer in ("Tier0", "Tier0.5"):
            norm_layer = "Heuristics (Tier0/0.5)"
        elif layer in ("Heuristics", "MultiStep-Heuristics"):
            norm_layer = "Heuristics"
        elif layer == "LLM-Session-Judge":
            norm_layer = "LLM Judge (Session)"
        elif layer == "V61":
            if was_review:
                norm_layer = "LLM Judge (V61 Action)"
            else:
                norm_layer = "V61 (Fast ML)"
        else:
            norm_layer = layer or "Unknown"

        record_stat = {
            "tool_name": tool_name,
            "is_high_risk": is_high_risk,
            "label": label,
            "decision": decision,
            "blocked": blocked,
            "layer": norm_layer,
            "raw_layer": layer,
            "reason": reason,
            "was_review": was_review,
            "would_force_review": would_force_review
        }
        detailed_eval.append(record_stat)

        if is_benign:
            tool_benign_total[tool_name] += 1
            if is_high_risk:
                high_risk_benign_total += 1
                if blocked:
                    high_risk_benign_fp += 1
            else:
                normal_benign_total += 1
                if blocked:
                    normal_benign_fp += 1

            if would_force_review:
                force_review_benign_total += 1
                if blocked:
                    force_review_benign_fp += 1
                else:
                    force_review_benign_recovered += 1

            if blocked:
                tool_fp_count[tool_name] += 1
                # Simplified 3-category layer breakdown
                if "LLM Judge" in norm_layer:
                    fp_layer_breakdown["LLM Judge"] += 1
                elif "Heuristics" in norm_layer:
                    fp_layer_breakdown["Heuristics"] += 1
                elif "V61" in norm_layer:
                    fp_layer_breakdown["V61"] += 1
                else:
                    fp_layer_breakdown["Other"] += 1

        else:
            tool_malicious_total[tool_name] += 1
            if not blocked:
                tool_fn_count[tool_name] += 1

        if idx % 200 == 0 or idx == total_calls:
            elapsed = time.perf_counter() - t_start
            print(f"  Processed {idx}/{total_calls} records ({idx/elapsed:.1f} rec/s)...")

    # Metrics Calculation
    total_benign = len(benign_calls)
    total_fp = high_risk_benign_fp + normal_benign_fp
    overall_fpr = (total_fp / total_benign * 100) if total_benign > 0 else 0.0
    high_risk_fpr = (high_risk_benign_fp / high_risk_benign_total * 100) if high_risk_benign_total > 0 else 0.0
    normal_fpr = (normal_benign_fp / normal_benign_total * 100) if normal_benign_total > 0 else 0.0

    # Top 10 FP
    top_10_fp = []
    for t_name, fp_c in tool_fp_count.most_common(10):
        tot = tool_benign_total[t_name]
        rate = (fp_c / tot * 100) if tot > 0 else 0.0
        top_10_fp.append({
            "tool_name": t_name,
            "fp_count": fp_c,
            "total_benign": tot,
            "fp_rate_pct": round(rate, 2),
            "is_high_risk": t_name in HIGH_RISK_FUNCTIONS
        })

    # Top 10 FN
    top_10_fn = []
    for t_name, fn_c in tool_fn_count.most_common(10):
        tot = tool_malicious_total[t_name]
        rate = (fn_c / tot * 100) if tot > 0 else 0.0
        top_10_fn.append({
            "tool_name": t_name,
            "fn_count": fn_c,
            "total_malicious": tot,
            "fn_rate_pct": round(rate, 2),
            "is_high_risk": t_name in HIGH_RISK_FUNCTIONS
        })

    # Layer Breakdown Percentages
    fp_layer_report = {}
    for l_key in ["V61", "Heuristics", "LLM Judge", "Other"]:
        cnt = fp_layer_breakdown[l_key]
        pct = (cnt / total_fp * 100) if total_fp > 0 else 0.0
        fp_layer_report[l_key] = {
            "count": cnt,
            "percentage": round(pct, 2)
        }

    report_data = {
        "summary": {
            "total_tool_calls_evaluated": total_calls,
            "total_benign_tool_calls": total_benign,
            "total_malicious_tool_calls": len(malicious_calls),
            "total_tool_call_fp": total_fp,
            "overall_tool_call_fpr_pct": round(overall_fpr, 2),
        },
        "fpr_by_risk_tier": {
            "high_risk": {
                "benign_total": high_risk_benign_total,
                "fp_count": high_risk_benign_fp,
                "fpr_pct": round(high_risk_fpr, 2)
            },
            "normal": {
                "benign_total": normal_benign_total,
                "fp_count": normal_benign_fp,
                "fpr_pct": round(normal_fpr, 2)
            }
        },
        "top_10_fp_tools": top_10_fp,
        "top_10_fn_tools": top_10_fn,
        "fp_breakdown_by_layer": fp_layer_report,
        "force_review_impact": {
            "benign_force_review_triggered": force_review_benign_total,
            "benign_force_review_fp_blocked": force_review_benign_fp,
            "benign_force_review_recovered_allow": force_review_benign_recovered,
            "fp_rate_among_forced_pct": round((force_review_benign_fp / force_review_benign_total * 100) if force_review_benign_total > 0 else 0.0, 2)
        }
    }

    # Console Output
    print("\n" + "=" * 80)
    print("  DIAGNOSE TOOL-CALL ESCALATION REPORT")
    print("=" * 80)
    print(f"\n1. Tool-Call FPR (HIGH-RISK vs NORMAL):")
    print(f"   - HIGH-RISK Tools : {high_risk_benign_fp}/{high_risk_benign_total} (FPR = {high_risk_fpr:.2f}%)")
    print(f"   - NORMAL Tools    : {normal_benign_fp}/{normal_benign_total} (FPR = {normal_fpr:.2f}%)")
    print(f"   - Overall Tool FPR: {total_fp}/{total_benign} ({overall_fpr:.2f}%)")

    print(f"\n2. Top 10 tool_name bi FP nhieu nhat:")
    print(f"   {'#':<3} {'Tool Name':<30} {'Risk':<10} {'FP / Total':<15} {'FP Rate':<10}")
    print(f"   {'-'*70}")
    for i, item in enumerate(top_10_fp, 1):
        risk_str = "HIGH" if item['is_high_risk'] else "NORMAL"
        print(f"   {i:<3} {item['tool_name']:<30} {risk_str:<10} {item['fp_count']}/{item['total_benign']:<12} {item['fp_rate_pct']:.2f}%")

    print(f"\n3. Top 10 tool_name bi FN (bo lot) nhieu nhat:")
    print(f"   {'#':<3} {'Tool Name':<30} {'Risk':<10} {'FN / Total':<15} {'FN Rate':<10}")
    print(f"   {'-'*70}")
    for i, item in enumerate(top_10_fn, 1):
        risk_str = "HIGH" if item['is_high_risk'] else "NORMAL"
        print(f"   {i:<3} {item['tool_name']:<30} {risk_str:<10} {item['fn_count']}/{item['total_malicious']:<12} {item['fn_rate_pct']:.2f}%")

    print(f"\n4. FP Breakdown theo Layer:")
    for l_key, l_data in fp_layer_report.items():
        print(f"   - {l_key:<15}: {l_data['count']} FP ({l_data['percentage']:.2f}%)")

    print(f"\n5. force_review Impact (Benign Escalate Nham):")
    print(f"   - Benign tools triggered force_review : {force_review_benign_total}")
    print(f"   - Benign bi chan nham (FP)             : {force_review_benign_fp}")
    print(f"   - Benign duoc LLM Judge cuu (ALLOW)   : {force_review_benign_recovered}")
    if force_review_benign_total > 0:
        print(f"   - Ty le chan nham trong nhom force   : {report_data['force_review_impact']['fp_rate_among_forced_pct']:.2f}%")

    # Save to file
    logs_dir = os.path.join(PROJECT_ROOT, "tests", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    report_file = os.path.join(logs_dir, "diagnose_tool_call_escalation_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[OK] Report saved to: {report_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_diagnose()
