"""
Benchmark 3000 diverse samples from evo_pca_full.jsonl (Ablation dataset).

Sampling strategy:
- 1200 benign (balanced sources)
- 1200 malicious_single (balanced sources)  
- 600 malicious_multistep (ALL complete sessions, not individual records)
  → ensures multi-step detection can actually be evaluated

Includes action_type inference (Fix 2 defensive logic).
"""
import json, sys, os, re, time, uuid, random
from collections import defaultdict, Counter

random.seed(42)
sys.path.insert(0, 'd:/DEMO_GROUP_1')

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
    if preflight_meta['ollama_status'] == 'OFFLINE':
        print("\n" + "!" * 80)
        print("  WARNING: Ollama is OFFLINE!")
        print("  V61 LLM Judge will timeout on all REVIEW samples.")
        print("  Results will NOT reflect true LLM Judge performance.")
        print("  Start Ollama and re-run for valid benchmark results.")
        print("!" * 80)
        response = input("\n  Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            sys.exit(1)
except ImportError:
    print("[WARN] scripts/preflight_check.py not found, skipping pre-flight")
    preflight_meta = None

# ─── Load full dataset ───
print("Loading evo_pca_full.jsonl...")
data = []
with open('d:/DEMO_GROUP_1/eval_dataset/output/evo_pca_full.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))
print(f"  Total records: {len(data)}")

# ─── Infer action_type for all records (Fix 2 logic) ───
def infer_action_type(action):
    """Infer action_type from action string if missing."""
    if re.match(r'^[a-zA-Z_]\w*\s*\(', action.strip()):
        return 'tool_call'
    return 'prompt'

for r in data:
    if 'action_type' not in r or not r['action_type']:
        r['action_type'] = infer_action_type(r['action'])

# ─── Group by category ───
benign = [r for r in data if r.get('attack_type') == 'benign']
single = [r for r in data if r.get('attack_type') == 'malicious_single']
multi  = [r for r in data if r.get('attack_type') == 'malicious_multistep']

print(f"  benign={len(benign)}, single={len(single)}, multi={len(multi)}")

# ─── Sample benign: stratified by source ───
benign_by_source = defaultdict(list)
for r in benign:
    benign_by_source[r.get('source_dataset', 'unknown')].append(r)

TARGET_BENIGN = 2000
sampled_benign = []
# Proportional sampling
for src, records in benign_by_source.items():
    proportion = len(records) / len(benign)
    n = max(1, round(proportion * TARGET_BENIGN))
    n = min(n, len(records))
    sampled = random.sample(records, n)
    sampled_benign.extend(sampled)
    print(f"  Benign from {src}: {n}/{len(records)}")

# Trim if over target
if len(sampled_benign) > TARGET_BENIGN:
    sampled_benign = random.sample(sampled_benign, TARGET_BENIGN)

# Assign unique sessions
for r in sampled_benign:
    r['session_id'] = f"bench_benign_{uuid.uuid4().hex[:8]}"
    r['step_num'] = 1

# ─── Sample malicious_single: stratified by source ───
single_by_source = defaultdict(list)
for r in single:
    single_by_source[r.get('source_dataset', 'unknown')].append(r)

TARGET_SINGLE = 2000
sampled_single = []
for src, records in single_by_source.items():
    proportion = len(records) / len(single)
    n = max(1, round(proportion * TARGET_SINGLE))
    n = min(n, len(records))
    sampled = random.sample(records, n)
    sampled_single.extend(sampled)
    print(f"  Single from {src}: {n}/{len(records)}")

if len(sampled_single) > TARGET_SINGLE:
    sampled_single = random.sample(sampled_single, TARGET_SINGLE)

for r in sampled_single:
    r['session_id'] = f"bench_single_{uuid.uuid4().hex[:8]}"
    r['step_num'] = 1

# ─── Sample malicious_multistep: COMPLETE sessions ───
multi_sessions = defaultdict(list)
for r in multi:
    multi_sessions[r['session_id']].append(r)

# Sort within each session by step_index
for sid in multi_sessions:
    multi_sessions[sid].sort(key=lambda x: x.get('step_index', 0))

# Take complete sessions
TARGET_MULTI_SESSIONS = len(multi_sessions)
session_ids = list(multi_sessions.keys())

sampled_multi_sessions = []
sampled_multi = []
used_sessions = 0
for sid in session_ids:
    session_records = multi_sessions[sid]
    # Re-ID sessions for benchmark isolation
    new_sid = f"bench_multi_{used_sessions:03d}_{uuid.uuid4().hex[:6]}"
    processed_records = []
    for step_num, r in enumerate(session_records, start=1):
        r = dict(r)  # copy
        r['session_id'] = new_sid
        r['step_num'] = step_num
        sampled_multi.append(r)
        processed_records.append(r)
    sampled_multi_sessions.append(processed_records)
    used_sessions += 1

print(f"  Multi-step: {len(sampled_multi)} records from {used_sessions} sessions")

# ─── Stats ───
dataset = sampled_benign + sampled_single + sampled_multi
print(f"\n=== Final Dataset: {len(dataset)} records ===")
print(f"  Benign: {len(sampled_benign)}")
print(f"  Single: {len(sampled_single)}")
print(f"  Multi:  {len(sampled_multi)} ({used_sessions} sessions)")

# Action type distribution
at_dist = Counter(r.get('action_type', 'MISSING') for r in dataset)
print(f"  action_type: {dict(at_dist)}")

# Source distribution
src_dist = Counter(r.get('source_dataset', '?') for r in dataset)
print(f"  Sources: {dict(src_dist)}")

# ─── Shuffle by session (preserve within-session order) ───
session_groups = defaultdict(list)
for r in dataset:
    session_groups[r['session_id']].append(r)
for grp in session_groups.values():
    grp.sort(key=lambda x: x.get('step_num', 1))

session_ids_final = list(session_groups.keys())
random.shuffle(session_ids_final)
dataset_shuffled = [r for sid in session_ids_final for r in session_groups[sid]]

# ─── Run Benchmark ───
print("\n" + "=" * 80)
print("  INITIALIZING FIREWALL PIPELINE...")
print("=" * 80)

from core.pipeline import UnifiedFirewallPipeline
from tests.run_benchmark import ActionTier

pipeline = UnifiedFirewallPipeline()

tp_single = fp = tn = fn_single = 0
tp_multi = fn_multi = 0
tier0_blocks = v61_blocks = lstm_blocks = heuristics_blocks = 0
llm_down_count = 0
latencies = []

multi_sessions_seen = set()
multi_session_blocked = defaultdict(bool)
multi_session_blocked_s1 = defaultdict(bool)

fp_records = []
fn_records = []

from tqdm.auto import tqdm
import threading, concurrent.futures, psutil

WARMUP_SESSIONS = 20
seen_sessions = set()
fp_ss = tn_ss = 0

def evaluate_record(record):
    t0 = time.perf_counter()
    action = record['action']
    session_id = record.get('session_id', 'default')
    action_type = record.get('action_type', 'prompt')
    actual_label = record.get('label') != 'benign'
    
    res = pipeline.scan(
        action=action, session_id=session_id,
        action_type=action_type
    )
    
    blocked = res['decision'] in ('BLOCK', 'QUARANTINE')
    return {
        'decision': 'DENY' if blocked else 'ALLOW',
        'latency_ms': (time.perf_counter() - t0) * 1000,
        'layer': res.get('layer', 'Unknown'),
        'reason': res.get('reason', 'Unknown'),
        'action_type': res.get('action_type', action_type),
        'tier0': 'BLOCK' if res.get('layer') == 'Tier0' and blocked else 'ALLOW',
    }

# Process sequentially within sessions, parallel across sessions
session_to_records = defaultdict(list)
for r in dataset_shuffled:
    session_to_records[r['session_id']].append(r)

def process_session(sid, records):
    results = []
    for rec in records:
        out = evaluate_record(rec)
        results.append((rec, out))
    return sid, results

print(f"\nEvaluating {len(dataset_shuffled)} records...")
process = psutil.Process()
pbar = tqdm(total=len(dataset_shuffled), desc="Evaluating")
pbar_lock = threading.Lock()

all_results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_session, sid, recs) 
               for sid, recs in session_to_records.items()]
    for future in concurrent.futures.as_completed(futures):
        sid, session_results = future.result()
        all_results.extend(session_results)
        with pbar_lock:
            pbar.update(len(session_results))
            ram_mb = process.memory_info().rss / (1024 * 1024)
            pbar.set_postfix({'RAM': f'{ram_mb:.0f}MB'})
pbar.close()

# Sort deterministically
all_results.sort(key=lambda x: (x[0].get('session_id', ''), x[0].get('step_num', 1)))

# ─── Count metrics ───
for record, out in all_results:
    sid = record.get('session_id', 'default')
    label = record['label']
    atype = record.get('attack_type', 'benign')
    step_num = record.get('step_num', 1)

    blocked = out['decision'] == 'DENY'
    latencies.append(out['latency_ms'])
    if '[LLM_DOWN]' in out.get('reason', ''):
        llm_down_count += 1

    if out.get('tier0') == 'BLOCK':
        tier0_blocks += 1
    if out.get('layer') == 'V61' and blocked:
        v61_blocks += 1
    if out.get('layer') == 'Heuristics' and blocked:
        heuristics_blocks += 1
    if out.get('layer') == 'Tier0.5-LSTM' and blocked:
        lstm_blocks += 1

    seen_sessions.add(sid)
    in_steady_state = len(seen_sessions) > WARMUP_SESSIONS

    if label == 'benign':
        if blocked:
            fp += 1
            fp_records.append(record)
            if in_steady_state: fp_ss += 1
        else:
            tn += 1
            if in_steady_state: tn_ss += 1
    elif atype == 'malicious_single':
        if blocked: tp_single += 1
        else:
            fn_single += 1
            fn_records.append(record)
    elif atype == 'malicious_multistep':
        multi_sessions_seen.add(sid)
        if blocked:
            tp_multi += 1
            multi_session_blocked[sid] = True
            if step_num == 1:
                multi_session_blocked_s1[sid] = True
        else:
            fn_multi += 1
            fn_records.append(record)

# ─── Compute metrics ───
import numpy as np

tot_b = fp + tn
tot_s = tp_single + fn_single
tot_m = tp_multi + fn_multi
tot_mal = tot_s + tot_m
tot_block = tp_single + tp_multi

fpr = fp / tot_b * 100 if tot_b else 0.0
tot_b_ss = fp_ss + tn_ss
fpr_ss = fp_ss / tot_b_ss * 100 if tot_b_ss else 0.0
absr_total = tot_block / tot_mal * 100 if tot_mal else 0.0
absr_single = tp_single / tot_s * 100 if tot_s else 0.0
absr_multi = tp_multi / tot_m * 100 if tot_m else 0.0

n_multi_sessions = len(multi_sessions_seen)
absr_multi_session = (
    sum(1 for s in multi_sessions_seen if multi_session_blocked[s]) /
    n_multi_sessions * 100 if n_multi_sessions else 0.0
)
absr_multi_session_early = (
    sum(1 for s in multi_sessions_seen if multi_session_blocked_s1[s]) /
    n_multi_sessions * 100 if n_multi_sessions else 0.0
)
avg_lat = np.mean(latencies) if latencies else 0.0

# ─── Print Results ───
print(f"\n================================================================================")
print(f"  BENCHMARK RESULTS ON {len(sampled_benign) + len(sampled_single) + sum(len(s) for s in sampled_multi_sessions)} SAMPLES (Eval Dataset)")
print(f"================================================================================\n")
print(f"  FPR={fpr:.2f}% (ss={fpr_ss:.2f}%)")
print(f"  ABSR Total={absr_total:.2f}%")
print(f"  ABSR Single-step={absr_single:.2f}%")
print(f"  ABSR Multi-step Action={absr_multi:.2f}%")
print(f"  ABSR Multi-step Session (any-step)={absr_multi_session:.2f}%")
print(f"  ABSR Multi-step Session (step1-only)={absr_multi_session_early:.2f}%")
print(f"  Avg Latency={avg_lat:.2f}ms")
print(f"  Tier0 Blocks={tier0_blocks}, V61 Blocks={v61_blocks}")
print(f"  LSTM Blocks={lstm_blocks}, Heuristics Blocks={heuristics_blocks}")
print(f"  LLM_DOWN Blocks={llm_down_count}")
if llm_down_count > 0:
    print(f"  >>> WARNING: {llm_down_count} samples hit LLM_DOWN timeout!")
    print(f"  >>> {llm_down_count}/{len(dataset_shuffled)} = {llm_down_count/len(dataset_shuffled)*100:.1f}% of all samples")
print("=" * 80)

print(f"\n  Breakdown:")
print(f"    Benign: {len(sampled_benign)} (FP={fp}, TN={tn})")
print(f"    Single: {len(sampled_single)} (TP={tp_single}, FN={fn_single})")
print(f"    Multi:  {len(sampled_multi)} from {used_sessions} sessions")
print(f"      TP_multi={tp_multi}, FN_multi={fn_multi}")
print(f"      Sessions blocked (any)={sum(1 for s in multi_sessions_seen if multi_session_blocked[s])}/{n_multi_sessions}")
print(f"      Sessions blocked (s1)={sum(1 for s in multi_sessions_seen if multi_session_blocked_s1[s])}/{n_multi_sessions}")

# ─── FP/FN Analysis ───
print(f"\n  Top FP sources:")
fp_src = Counter(r.get('source_dataset', '?') for r in fp_records)
for k, v in fp_src.most_common(5):
    print(f"    {k}: {v}")

print(f"\n  Top FN sources:")
fn_src = Counter(r.get('source_dataset', '?') for r in fn_records)
for k, v in fn_src.most_common(5):
    print(f"    {k}: {v}")

print(f"\n  FN by attack_type:")
fn_at = Counter(r.get('attack_type', '?') for r in fn_records)
for k, v in fn_at.most_common():
    print(f"    {k}: {v}")

# Save to CSV
import pandas as pd
results_dict = {
    'Metric': [
        'FPR (%) lower-better',
        'FPR Steady-State (%) lower-better',
        'ABSR Total (%) higher-better',
        'ABSR Single-step (%)',
        'ABSR Multi-step Action (%)',
        'ABSR Multi-step Session (any-step)%',
        'ABSR Multi-step Session (step1-only)%',
        'Avg Latency (ms) lower-better',
        'Tier 0 Blocks', 'V61 Blocks', 'Heuristics Blocks',
        'LSTM Blocks', 'Total Records', 'Multi Sessions',
    ],
    'Value': [
        round(fpr, 2), round(fpr_ss, 2), round(absr_total, 2),
        round(absr_single, 2), round(absr_multi, 2),
        round(absr_multi_session, 2), round(absr_multi_session_early, 2),
        round(avg_lat, 2), tier0_blocks, v61_blocks, heuristics_blocks,
        lstm_blocks, len(dataset_shuffled), n_multi_sessions,
    ]
}
df = pd.DataFrame(results_dict)
csv_path = 'd:/DEMO_GROUP_1/benchmark_5000_eval_results.csv'
df.to_csv(csv_path, index=False)
print(f"\n[OK] Results saved to {csv_path}")

# Save FP/FN details
report_path = 'd:/DEMO_GROUP_1/fp_fn_report_5000_eval.txt'
with open(report_path, "w", encoding='utf-8') as f:
    f.write(f"=== FP/FN Report — 5000 Eval Dataset Benchmark ===\n\n")
    f.write(f"FPR={fpr:.2f}% ABSR_total={absr_total:.2f}% ABSR_single={absr_single:.2f}% ABSR_multi={absr_multi:.2f}%\n\n")
    
    f.write(f"--- FALSE POSITIVES ({fp} records) ---\n\n")
    for i, r in enumerate(fp_records[:50]):
        f.write(f"FP-{i+1}: src={r.get('source_dataset','?')} action_type={r.get('action_type','?')} | {r['action'][:200]}\n\n")
    
    f.write(f"\n--- FALSE NEGATIVES ({len(fn_records)} records) ---\n\n")
    for i, r in enumerate(fn_records[:100]):
        f.write(f"FN-{i+1}: type={r.get('attack_type','?')} src={r.get('source_dataset','?')} action_type={r.get('action_type','?')} | {r['action'][:200]}\n\n")

print(f"[OK] FP/FN report saved to {report_path}")

# Save benchmark metadata
if 'preflight_meta' in locals() and preflight_meta:
    preflight_meta['llm_down_count'] = llm_down_count
    preflight_meta['total_records'] = len(dataset_shuffled)
    preflight_meta['v61_blocks'] = v61_blocks
    meta_path = write_benchmark_metadata(preflight_meta, csv_path)
    print(f"[OK] Benchmark metadata saved to {meta_path}")

