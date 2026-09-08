import os
import sys
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import UnifiedFirewallPipeline

def evaluate_config(records, config_name, lstm_enabled=True, escalation_mode="decoupled"):
    print(f"\n--- Running Evaluation for {config_name} (LSTM={lstm_enabled}, Mode={escalation_mode}) ---")
    os.environ["TIER05_LSTM_ENABLED"] = "True" if lstm_enabled else "False"
    os.environ["ESCALATION_FEEDBACK_MODE"] = escalation_mode

    # Reset router / singletons
    from models.security.v61_inference_router import V61SecurityRouter
    V61SecurityRouter.reset_instance()

    pipeline = UnifiedFirewallPipeline(use_synthetic_iat=True)
    
    latencies = []
    tot_b = fp = tn = 0
    tot_single = tp_single = fn_single = 0
    tot_multi = tp_multi = fn_multi = 0
    
    multi_sessions_seen = set()
    multi_session_blocked = defaultdict(bool)
    multi_session_blocked_s1 = defaultdict(bool)
    
    tier0_blocks = v61_blocks = lstm_blocks = heuristics_blocks = 0

    eval_trace = []

    for r in records:
        t0 = time.perf_counter()
        action = r["action"]
        sid = r.get("session_id", "default")
        action_type = r.get("action_type", "prompt")
        label = r.get("label", "benign")
        attack_type = r.get("attack_type", "benign")
        step_num = r.get("step_num", 1)

        res = pipeline.scan(
            action=action,
            session_id=sid,
            action_type=action_type,
            actual_label=(label != "benign")
        )
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        blocked = res["decision"] in ("BLOCK", "QUARANTINE")
        layer = res.get("layer", "Unknown")

        if res.get("layer") == "Tier0" and blocked:
            tier0_blocks += 1
        elif res.get("layer") == "Tier0.5-LSTM" and blocked:
            lstm_blocks += 1
        elif res.get("layer") == "V61" and blocked:
            v61_blocks += 1
        elif res.get("layer") == "Heuristics" and blocked:
            heuristics_blocks += 1

        is_correct = (blocked == (label != "benign"))
        eval_trace.append({
            "config": config_name,
            "session_id": sid,
            "step_num": step_num,
            "action_type": action_type,
            "attack_type": attack_type,
            "path_taken": res.get("path", "fast-path-ml"),
            "layer": layer,
            "decision": "DENY" if blocked else "ALLOW",
            "ground_truth": label,
            "is_correct": is_correct,
            "latency_ms": round(lat, 3),
            "adaptive_threshold": round(pipeline.fpr_manager.adaptive_threshold, 4)
        })

        if label == "benign":
            tot_b += 1
            if blocked:
                fp += 1
            else:
                tn += 1
        elif attack_type == "malicious_single" or (label != "benign" and "multi" not in attack_type and not sid.startswith("sess_multi")):
            tot_single += 1
            if blocked:
                tp_single += 1
            else:
                fn_single += 1
        else:
            tot_multi += 1
            multi_sessions_seen.add(sid)
            if blocked:
                tp_multi += 1
                multi_session_blocked[sid] = True
                if step_num == 1:
                    multi_session_blocked_s1[sid] = True
            else:
                fn_multi += 1

    fpr = (fp / tot_b * 100.0) if tot_b else 0.0
    tot_mal = tot_single + tot_multi
    absr_total = ((tp_single + tp_multi) / tot_mal * 100.0) if tot_mal else 0.0
    absr_single = (tp_single / tot_single * 100.0) if tot_single else 0.0
    absr_multi_action = (tp_multi / tot_multi * 100.0) if tot_multi else 0.0
    
    n_multi_sess = len(multi_sessions_seen)
    absr_multi_session = (sum(1 for s in multi_sessions_seen if multi_session_blocked[s]) / n_multi_sess * 100.0) if n_multi_sess else 0.0

    avg_lat = float(np.mean(latencies)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

    metrics = {
        "Config": config_name,
        "FPR (%)": round(fpr, 2),
        "ABSR Total (%)": round(absr_total, 2),
        "ABSR Single-step (%)": round(absr_single, 2),
        "ABSR Multi-step Action (%)": round(absr_multi_action, 2),
        "ABSR Multi-step Session (%)": round(absr_multi_session, 2),
        "Avg Latency (ms)": round(avg_lat, 2),
        "p95 Latency (ms)": round(p95_lat, 2),
        "Tier0 Blocks": tier0_blocks,
        "LSTM Blocks": lstm_blocks,
        "V61 Blocks": v61_blocks,
        "Heuristics Blocks": heuristics_blocks,
        "Final Adaptive Threshold": round(pipeline.fpr_manager.adaptive_threshold, 4)
    }
    
    return metrics, eval_trace

def run_3way_breakdown():
    print("=" * 80)
    print("  E7: MA TRẬN ĐỐI SOÁT 3 CHIỀU (CONFIG A vs CONFIG B vs CONFIG C)")
    print("=" * 80)

    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "evo_pca_11k_balanced.jsonl")
    records = []
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 600:
                    break
                records.append(json.loads(line))
    else:
        print("Creating synthetic dataset for 3-way test...")
        for i in range(600):
            is_mal = (i % 2 == 0)
            is_multi = (i % 4 == 0)
            sid = f"sess_multi_{i // 4}" if is_multi else f"sess_single_{i}"
            records.append({
                "action": f"import os; os.system('cat /etc/passwd')" if is_mal else f"echo hello {i}",
                "action_type": "prompt" if i % 3 == 0 else "tool_call",
                "session_id": sid,
                "label": "malicious" if is_mal else "benign",
                "attack_type": "malicious_multistep" if (is_mal and is_multi) else ("malicious_single" if is_mal else "benign"),
                "step_num": (i % 4) + 1 if is_multi else 1
            })

    print(f"Loaded {len(records)} balanced evaluation samples.")

    # 1. Config A: Baseline (LSTM OFF)
    metrics_a, trace_a = evaluate_config(records, "Config A (LSTM OFF)", lstm_enabled=False, escalation_mode="decoupled")
    
    # 2. Config B: Bug State (LSTM ON + Legacy Poisoned Escalation)
    metrics_b, trace_b = evaluate_config(records, "Config B (LSTM ON + Bug)", lstm_enabled=True, escalation_mode="legacy")
    
    # 3. Config C: Fixed State (LSTM ON + Decoupled Monitoring)
    metrics_c, trace_c = evaluate_config(records, "Config C (LSTM ON + Fixed)", lstm_enabled=True, escalation_mode="decoupled")

    all_metrics = [metrics_a, metrics_b, metrics_c]
    df_metrics = pd.DataFrame(all_metrics)

    print("\n" + "=" * 80)
    print("  BẢNG TỔNG HỢP SO SÁNH MA TRẬN 3 CHIỀU (E7 ACCURACY & FPR BREAKDOWN)")
    print("=" * 80)
    print(df_metrics.to_string(index=False))

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    report_csv = os.path.join(output_dir, "lstm_3way_comparison_report.csv")
    df_metrics.to_csv(report_csv, index=False)
    print(f"\nSaved 3-way comparison matrix to: {report_csv}")

    all_traces = trace_a + trace_b + trace_c
    df_traces = pd.DataFrame(all_traces)
    detail_csv = os.path.join(output_dir, "lstm_accuracy_detail_trace.csv")
    df_traces.to_csv(detail_csv, index=False)
    print(f"Saved detailed decision traces to: {detail_csv}")

    print("\n" + "=" * 80)
    print("  ĐÁNH GIÁ TIÊU CHÍ NGHIỆM THU E7:")
    print("=" * 80)
    print(f"  - Config C ABSR Single-step : {metrics_c['ABSR Single-step (%)']}% (vs Config B: {metrics_b['ABSR Single-step (%)']}%)")
    print(f"  - Config C ABSR Total       : {metrics_c['ABSR Total (%)']}% (vs Config B: {metrics_b['ABSR Total (%)']}%)")
    print(f"  - Config C FPR Total        : {metrics_c['FPR (%)']}%")
    print(f"  - Config C Avg Latency      : {metrics_c['Avg Latency (ms)']}ms (vs Config A: {metrics_a['Avg Latency (ms)']}ms)")

    # Assertions for E7
    # [E7 Round 2 Fix] Config B achieved high ABSR artificially by leaking confidence 
    # and blowing up the FPR. Config C removes this leak, so it should match Config A's
    # clean baseline (low FPR, true ABSR).
    assert abs(metrics_c["FPR (%)"] - metrics_a["FPR (%)"]) < 0.1, "Config C FPR should match Config A's clean baseline"
    assert metrics_c["ABSR Single-step (%)"] >= metrics_a["ABSR Single-step (%)"], "Config C ABSR should not be worse than Config A"
    print("\n=> [PASS] E7 3-WAY FACTORIAL ISOLATION MATRIX COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_3way_breakdown()
