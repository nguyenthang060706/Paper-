"""
tests/benchmark_3000.py
========================
Benchmark Configuration C (Dual Shield + Multi-Step Heuristics + LLM Session Judge)
on 3,000 diverse samples from evo_pca_full.jsonl.

Configuration C:
- Tier 0 (Fast Regex) = ON
- Tier 0.5 LSTM (Temporal sequence risk with decoupled confidence) = ON (TIER05_LSTM_ENABLED=true)
- Multi-Step Heuristics (State-Machine Kill-Chain Detection) = ON
- LLM Session Judge (Cross-step Session Judge) = ON
- V61 Router (ML Risk Model + Action Judge) = ON
- Permission Gate & Advanced Heuristics Voting = ON

Sampling Strategy:
- 1,200 Benign records (stratified by source_dataset)
- 1,200 Malicious Single-Step records (stratified by source_dataset)
- ~600 Malicious Multi-Step records (100% complete sessions preserved)
Total = 3,000 records.
"""

import os
import sys
import json
import re
import time
import uuid
import random
import gc
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

# Set Environment Variables for Configuration C
os.environ["TIER05_LSTM_ENABLED"] = "true"
os.environ["ESCALATION_FEEDBACK_MODE"] = "decoupled"
os.environ["MULTI_STEP_HEURISTICS_ENABLED"] = "true"
os.environ["LLM_SESSION_JUDGE_ENABLED"] = "true"

random.seed(42)
np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Load Config ───
try:
    from core.config_loader import load_settings
    load_settings(override_existing=True)
except ImportError:
    print("[WARN] core.config_loader not found, skipping config load")

# ─── Pre-flight health check ───
try:
    from scripts.preflight_check import run_preflight, write_benchmark_metadata
    preflight_meta = run_preflight()
    print(f"[Preflight] Ollama status: {preflight_meta.get('ollama_status')} (model: {preflight_meta.get('ollama_model')})")
    if preflight_meta['ollama_status'] == 'OFFLINE':
        print("\n" + "!" * 80)
        print("  WARNING: Ollama is OFFLINE! LLM Judge will timeout.")
        print("!" * 80)
except ImportError:
    print("[WARN] scripts/preflight_check.py not found, skipping pre-flight")
    preflight_meta = None

# ─── Load dataset ───
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(project_root, "ablation", "output", "evo_pca_full.jsonl")
if not os.path.exists(dataset_path):
    dataset_path = os.path.join(project_root, "output", "evo_pca_full.jsonl")

print(f"\nLoading dataset from {dataset_path}...")
data = []
with open(dataset_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))
print(f"  Total records loaded: {len(data)}")


# ─── Infer action_type ───
def infer_action_type(action):
    """Infer action_type from action string/dict if missing."""
    if isinstance(action, dict):
        return 'tool_call'
    if isinstance(action, str) and re.match(r'^[a-zA-Z_]\w*\s*\(', action.strip()):
        return 'tool_call'
    return 'prompt'


for r in data:
    if 'action_type' not in r or not r['action_type']:
        r['action_type'] = infer_action_type(r.get('action', ''))

# ─── Group by category ───
benign = [r for r in data if r.get('attack_type') == 'benign']
single = [r for r in data if r.get('attack_type') == 'malicious_single']
multi  = [r for r in data if r.get('attack_type') == 'malicious_multistep']

print(f"  Available records: Benign={len(benign)}, Single-Step={len(single)}, Multi-Step={len(multi)}")

# ─── 1. Sample Multi-Step Sessions (Complete sessions) ───
multi_sessions = defaultdict(list)
for r in multi:
    multi_sessions[r['session_id']].append(r)

for sid in multi_sessions:
    multi_sessions[sid].sort(key=lambda x: x.get('step_index', 0))

session_ids = list(multi_sessions.keys())
sampled_multi_sessions = []
sampled_multi = []
used_sessions = 0

for sid in session_ids:
    session_records = multi_sessions[sid]
    new_sid = f"bench_multi_{used_sessions:03d}_{uuid.uuid4().hex[:6]}"
    processed_records = []
    for step_num, r in enumerate(session_records, start=1):
        r_copy = dict(r)
        r_copy['session_id'] = new_sid
        r_copy['step_num'] = step_num
        sampled_multi.append(r_copy)
        processed_records.append(r_copy)
    sampled_multi_sessions.append(processed_records)
    used_sessions += 1

total_multi_records = len(sampled_multi)
print(f"  Multi-Step sampled: {total_multi_records} records across {used_sessions} sessions")

# ─── 2. Stratified Sampling for Benign and Single-Step ───
# We want exactly 3,000 total records:
# TARGET_RECORDS = 3000
# Remainder to split between benign and single:
rem_target = 3000 - total_multi_records
TARGET_BENIGN = rem_target // 2
TARGET_SINGLE = rem_target - TARGET_BENIGN

# Sample benign: stratified by source
benign_by_source = defaultdict(list)
for r in benign:
    benign_by_source[r.get('source_dataset', 'unknown')].append(r)

sampled_benign = []
for src, records in benign_by_source.items():
    prop = len(records) / len(benign)
    n = max(1, round(prop * TARGET_BENIGN))
    n = min(n, len(records))
    sampled_benign.extend(random.sample(records, n))

if len(sampled_benign) > TARGET_BENIGN:
    sampled_benign = random.sample(sampled_benign, TARGET_BENIGN)
elif len(sampled_benign) < TARGET_BENIGN:
    diff = TARGET_BENIGN - len(sampled_benign)
    pool = [r for r in benign if r not in sampled_benign]
    sampled_benign.extend(random.sample(pool, min(diff, len(pool))))

for r in sampled_benign:
    r['session_id'] = f"bench_benign_{uuid.uuid4().hex[:8]}"
    r['step_num'] = 1

# Sample single: stratified by source
single_by_source = defaultdict(list)
for r in single:
    single_by_source[r.get('source_dataset', 'unknown')].append(r)

sampled_single = []
for src, records in single_by_source.items():
    prop = len(records) / len(single)
    n = max(1, round(prop * TARGET_SINGLE))
    n = min(n, len(records))
    sampled_single.extend(random.sample(records, n))

if len(sampled_single) > TARGET_SINGLE:
    sampled_single = random.sample(sampled_single, TARGET_SINGLE)
elif len(sampled_single) < TARGET_SINGLE:
    diff = TARGET_SINGLE - len(sampled_single)
    pool = [r for r in single if r not in sampled_single]
    sampled_single.extend(random.sample(pool, min(diff, len(pool))))

for r in sampled_single:
    r['session_id'] = f"bench_single_{uuid.uuid4().hex[:8]}"
    r['step_num'] = 1

fixed_eval_path = os.path.join(project_root, "tests", "output", "benchmark_3000_fixed_eval.jsonl")
if os.path.exists(fixed_eval_path):
    print(f"\n[Fixed Eval] Loading frozen 3,000 dataset from {fixed_eval_path}...")
    dataset_shuffled = []
    with open(fixed_eval_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset_shuffled.append(json.loads(line))
    print(f"  Loaded {len(dataset_shuffled)} frozen records.")
else:
    dataset = sampled_benign + sampled_single + sampled_multi
    print(f"\n=== Final Benchmark 3,000 Dataset ===")
    print(f"  Benign:     {len(sampled_benign)}")
    print(f"  Single:     {len(sampled_single)}")
    print(f"  Multi:      {len(sampled_multi)} ({used_sessions} sessions)")
    print(f"  Total:      {len(dataset)} records")

    # ─── Group by session (preserve sequential within-session order) ───
    session_groups = defaultdict(list)
    for r in dataset:
        session_groups[r['session_id']].append(r)
    for grp in session_groups.values():
        grp.sort(key=lambda x: x.get('step_num', 1))

    session_ids_final = list(session_groups.keys())
    random.shuffle(session_ids_final)
    dataset_shuffled = [r for sid in session_ids_final for r in session_groups[sid]]

    os.makedirs(os.path.dirname(fixed_eval_path), exist_ok=True)
    with open(fixed_eval_path, 'w', encoding='utf-8') as f:
        for r in dataset_shuffled:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"[Fixed Eval] Saved frozen 3,000 dataset to {fixed_eval_path}")

# ─── Initialize Unified Firewall Pipeline (Config C) ───
print("\n" + "=" * 80)
print("  INITIALIZING UNIFIED FIREWALL PIPELINE (CONFIGURATION C)...")
print("=" * 80)

from core.pipeline import UnifiedFirewallPipeline
from models.security.v61_inference_router import V61SecurityRouter
from models.security.global_threat_tracker import GlobalThreatTracker

V61SecurityRouter.reset_instance()
GlobalThreatTracker.reset_instance()

pipeline = UnifiedFirewallPipeline(use_synthetic_iat=True)

# ─── Benchmark Execution Loop ───
print(f"\nEvaluating {len(dataset_shuffled)} records (Sequential / Memory-Safe)...")

latencies = []
tp_single = fp = tn = fn_single = 0
tp_multi = fn_multi = 0

tier0_blocks = 0
lstm_blocks = 0
multistep_heuristics_blocks = 0
session_judge_blocks = 0
v61_blocks = 0
heuristics_blocks = 0
llm_down_count = 0

WARMUP_SESSIONS = 20
seen_sessions = set()
fp_ss = tn_ss = 0

multi_sessions_seen = set()
multi_session_blocked = defaultdict(bool)
multi_session_blocked_s1 = defaultdict(bool)

fp_records = []
fn_records = []
all_evaluated_results = []

from tqdm.auto import tqdm
import psutil

process = psutil.Process()
pbar = tqdm(total=len(dataset_shuffled), desc="Benchmarking Config C")

t_start = time.time()
gc_counter = 0

for record in dataset_shuffled:
    action = record['action']
    session_id = record.get('session_id', 'default')
    action_type = record.get('action_type', 'prompt')
    label = record.get('label')
    atype = record.get('attack_type', 'benign')
    step_num = record.get('step_num', 1)
    actual_label = (label != 'benign')

    t0 = time.perf_counter()
    res = pipeline.scan(
        action=action,
        session_id=session_id,
        action_type=action_type,
        actual_label=actual_label
    )
    lat_ms = (time.perf_counter() - t0) * 1000
    latencies.append(lat_ms)

    decision_str = str(res.get('decision', 'ALLOW')).upper()
    blocked = decision_str in ('DENY', 'QUARANTINE', 'BLOCK')
    layer = res.get('layer') or 'Unknown'
    reason = res.get('reason') or ''

    if res.get('llm_down'):
        llm_down_count += 1

    # Record block layers
    if blocked:
        if layer == 'Tier0' or res.get('tier0') == 'BLOCK':
            tier0_blocks += 1
        elif layer == 'Tier0.5-LSTM':
            lstm_blocks += 1
        elif layer == 'MultiStep-Heuristics':
            multistep_heuristics_blocks += 1
        elif layer == 'LLM-Session-Judge':
            session_judge_blocks += 1
        elif layer == 'V61':
            v61_blocks += 1
        elif layer == 'Heuristics':
            heuristics_blocks += 1

    # Steady state tracking
    seen_sessions.add(session_id)
    in_steady_state = (len(seen_sessions) > WARMUP_SESSIONS)

    record_eval = {
        'record': record,
        'decision': 'DENY' if blocked else 'ALLOW',
        'layer': layer,
        'reason': reason,
        'latency_ms': lat_ms
    }
    all_evaluated_results.append(record_eval)

    if label == 'benign':
        if blocked:
            fp += 1
            fp_records.append(record_eval)
            if in_steady_state:
                fp_ss += 1
        else:
            tn += 1
            if in_steady_state:
                tn_ss += 1
    elif atype == 'malicious_single':
        if blocked:
            tp_single += 1
        else:
            fn_single += 1
            fn_records.append(record_eval)
    elif atype == 'malicious_multistep':
        multi_sessions_seen.add(session_id)
        if blocked:
            tp_multi += 1
            multi_session_blocked[session_id] = True
            if step_num == 1:
                multi_session_blocked_s1[session_id] = True
        else:
            fn_multi += 1
            fn_records.append(record_eval)

    pbar.update(1)
    gc_counter += 1
    if gc_counter % 200 == 0:
        ram_mb = process.memory_info().rss / (1024 * 1024)
        pbar.set_postfix({'RAM': f'{ram_mb:.0f}MB', 'FP': fp, 'TP': tp_single + tp_multi})
        gc.collect()

pbar.close()
total_time = time.time() - t_start
print(f"\nCompleted evaluation in {total_time:.2f}s ({total_time/len(dataset_shuffled)*1000:.2f}ms/record)")

# ─── Compute Metrics ───
tot_b = fp + tn
tot_s = tp_single + fn_single
tot_m = tp_multi + fn_multi
tot_mal = tot_s + tot_m
tot_block = tp_single + tp_multi

fpr = (fp / tot_b * 100) if tot_b else 0.0
tot_b_ss = fp_ss + tn_ss
fpr_ss = (fp_ss / tot_b_ss * 100) if tot_b_ss else 0.0
absr_total = (tot_block / tot_mal * 100) if tot_mal else 0.0
absr_single = (tp_single / tot_s * 100) if tot_s else 0.0
absr_multi = (tp_multi / tot_m * 100) if tot_m else 0.0

n_multi_sessions = len(multi_sessions_seen)
absr_multi_session = (
    sum(1 for s in multi_sessions_seen if multi_session_blocked[s]) /
    n_multi_sessions * 100 if n_multi_sessions else 0.0
)
absr_multi_session_early = (
    sum(1 for s in multi_sessions_seen if multi_session_blocked_s1[s]) /
    n_multi_sessions * 100 if n_multi_sessions else 0.0
)
avg_lat = float(np.mean(latencies)) if latencies else 0.0

# ─── Print Summary Table ───
print("\n" + "=" * 80)
print(f"  BENCHMARK 3000 RESULTS — CONFIGURATION C (ALL LAYERS ON)")
print("=" * 80)
print(f"  FPR Total:                           {fpr:.2f}% (ss: {fpr_ss:.2f}%)")
print(f"  ABSR Total:                          {absr_total:.2f}%")
print(f"  ABSR Single-step:                    {absr_single:.2f}%")
print(f"  ABSR Multi-step Action:              {absr_multi:.2f}%")
print(f"  ABSR Multi-step Session (any-step):  {absr_multi_session:.2f}%")
print(f"  ABSR Multi-step Session (step1-only):{absr_multi_session_early:.2f}%")
print(f"  Avg Latency:                         {avg_lat:.2f} ms")
print("-" * 80)
print(f"  Layer Block Breakdown:")
print(f"    Tier 0 Blocks:                     {tier0_blocks}")
print(f"    Tier 0.5-LSTM Blocks:              {lstm_blocks}")
print(f"    Multi-Step Heuristics Blocks:      {multistep_heuristics_blocks}")
print(f"    LLM Session Judge Blocks:          {session_judge_blocks}")
print(f"    V61 Router Blocks:                 {v61_blocks}")
print(f"    Heuristics Blocks:                 {heuristics_blocks}")
print(f"    LLM_DOWN Incidents:                {llm_down_count}")
print("=" * 80)

# ─── Save Results ───
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_dir, exist_ok=True)

model_name = "EVO-PCA Config C (Tier0 + Tier0.5-LSTM + State-Machine + V61 + Session Judge)"

results_summary = {
    model_name: {
        'FPR (%) lower-better': round(fpr, 2),
        'FPR Steady-State (%) lower-better': round(fpr_ss, 2),
        'ABSR Total (%) higher-better': round(absr_total, 2),
        'ABSR Single-step (%)': round(absr_single, 2),
        'ABSR Multi-step Action (%)': round(absr_multi, 2),
        'ABSR Multi-step Session (any-step)%': round(absr_multi_session, 2),
        'ABSR Multi-step Session (step1-only)%': round(absr_multi_session_early, 2),
        'Avg Latency (ms) lower-better': round(avg_lat, 2),
        'Tier 0 Blocks': tier0_blocks,
        'Tier 0.5-LSTM Blocks': lstm_blocks,
        'Multi-Step Heuristics Blocks': multistep_heuristics_blocks,
        'LLM Session Judge Blocks': session_judge_blocks,
        'V61 Blocks': v61_blocks,
        'Heuristics Blocks': heuristics_blocks,
    }
}

summary_df = pd.DataFrame(results_summary)

csv_out_3000 = os.path.join(output_dir, 'benchmark_3000_results.csv')
json_out_3000 = os.path.join(output_dir, 'benchmark_3000_results.json')
csv_out_main = os.path.join(output_dir, 'benchmark_results.csv')
json_out_main = os.path.join(output_dir, 'benchmark_results.json')

summary_df.T.to_csv(csv_out_3000, encoding='utf-8')
summary_df.to_json(json_out_3000, indent=2, force_ascii=False)
summary_df.T.to_csv(csv_out_main, encoding='utf-8')
summary_df.to_json(json_out_main, indent=2, force_ascii=False)

print(f"\n[OUTPUT] Benchmark results saved to:")
print(f"  - {csv_out_3000}")
print(f"  - {json_out_3000}")
print(f"  - {csv_out_main}")

# ─── Save FP/FN Diagnostic Report ───
report_path = os.path.join(output_dir, 'fp_fn_report_config_c_3000.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(f"=== DIAGNOSTIC REPORT: CONFIGURATION C (3,000 SAMPLES) ===\n\n")
    f.write(f"FPR={fpr:.2f}% (ss={fpr_ss:.2f}%) | ABSR Total={absr_total:.2f}% | ABSR Single={absr_single:.2f}% | ABSR Multi={absr_multi:.2f}%\n\n")
    f.write(f"Block Breakdown: Tier0={tier0_blocks}, LSTM={lstm_blocks}, MultiStep={multistep_heuristics_blocks}, SessionJudge={session_judge_blocks}, V61={v61_blocks}, Heuristics={heuristics_blocks}\n\n")
    
    f.write(f"--- FALSE POSITIVES ({len(fp_records)} records) ---\n")
    for i, item in enumerate(fp_records[:40], 1):
        rec = item['record']
        f.write(f"{i}. [Layer: {item['layer']}] [Reason: {item['reason']}] [Action: {rec.get('action_type')}] {rec.get('action')[:250]}\n\n")

    f.write(f"\n--- FALSE NEGATIVES ({len(fn_records)} records) ---\n")
    for i, item in enumerate(fn_records[:40], 1):
        rec = item['record']
        f.write(f"{i}. [Type: {rec.get('attack_type')}] [Action: {rec.get('action_type')}] {rec.get('action')[:250]}\n\n")

print(f"[OUTPUT] Diagnostic report saved to {report_path}\n")
