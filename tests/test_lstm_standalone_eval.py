"""Standalone evaluation of LSTM probability to calibrate block_threshold.

Since sequence_risk (cosine distance) was found to have no discriminative power,
we disabled it. Now we need to rely purely on the neural network's `probability`
output. The block_threshold is configured in config/thresholds.json
(tier05_lstm.block_threshold), with fallback to LSTM_BLOCK_THRESHOLD constant.

This script runs the LSTM model over the balanced dataset and computes the
distribution of `probability` for benign vs malicious sessions, to recommend
a new block_threshold.
"""
import os, sys, json
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict

def evaluate_lstm_probability():
    print("=" * 80)
    print("  CALIBRATING LSTM BLOCK_THRESHOLD FROM EMPIRICAL PROBABILITY")
    print("=" * 80)
    
    # 1. Load dataset
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "output", "evo_pca_11k_balanced.jsonl")
    records = []
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    else:
        print(f"Dataset not found at {data_path}")
        return
    
    # 2. Group by session
    sessions = defaultdict(list)
    for r in records:
        sid = r.get("session_id", "default")
        sessions[sid].append(r)
    
    benign_sessions = {}
    malicious_sessions = {}
    for sid, recs in sessions.items():
        labels = set(r.get("label", "benign") for r in recs)
        if "malicious" in labels:
            malicious_sessions[sid] = recs
        else:
            benign_sessions[sid] = recs
            
    print(f"Total sessions: {len(sessions)}")
    print(f"Benign sessions: {len(benign_sessions)}")
    print(f"Malicious sessions: {len(malicious_sessions)}")
    
    # 3. Load LSTM Model
    try:
        from core.tier_lstm import SessionAwareLSTMRisk
        # Calibration mode: threshold=1.1 means LSTM never blocks here —
        # we only want to observe raw probability distribution, not test blocking.
        # The real block_threshold is set in config/thresholds.json (tier05_lstm.block_threshold).
        lstm = SessionAwareLSTMRisk(block_threshold=1.1)
    except Exception as e:
        print(f"Cannot load LSTM: {e}")
        return

    # Track maximum probability and sequence_risk reached in each session
    benign_max_probs = []
    malicious_max_probs = []
    benign_max_seq_risk = []
    malicious_max_seq_risk = []
    
    import tqdm
    
    print("\nEvaluating benign sessions...")
    for sid, recs in tqdm.tqdm(benign_sessions.items()):
        lstm.reset_session(sid)
        max_p = 0.0
        max_sr = 0.0
        for r in recs:
            try:
                res = lstm.update_and_score(r["action"], sid)
                p = res.get("probability", 0.0)
                sr = res.get("sequence_risk", 0.0)
                if p > max_p: max_p = p
                if sr > max_sr: max_sr = sr
            except Exception:
                pass
        benign_max_probs.append(max_p)
        benign_max_seq_risk.append(max_sr)
        
    print("\nEvaluating malicious sessions...")
    for sid, recs in tqdm.tqdm(malicious_sessions.items()):
        lstm.reset_session(sid)
        max_p = 0.0
        max_sr = 0.0
        for r in recs:
            try:
                res = lstm.update_and_score(r["action"], sid)
                p = res.get("probability", 0.0)
                sr = res.get("sequence_risk", 0.0)
                if p > max_p: max_p = p
                if sr > max_sr: max_sr = sr
            except Exception:
                pass
        malicious_max_probs.append(max_p)
        malicious_max_seq_risk.append(max_sr)
        
    benign_arr = np.array(benign_max_probs)
    mal_arr = np.array(malicious_max_probs)
    benign_sr_arr = np.array(benign_max_seq_risk)
    mal_sr_arr = np.array(malicious_max_seq_risk)
    
    print(f"\n{'='*80}")
    print(f"  BENIGN MAX PROBABILITY (N={len(benign_arr)})")
    print(f"{'='*80}")
    for pct in [50, 75, 90, 95, 97, 99, 99.5, 99.9]:
        print(f"  p{pct:>5}: {np.percentile(benign_arr, pct):.4f}")
        
    print(f"\n{'='*80}")
    print(f"  MALICIOUS MAX PROBABILITY (N={len(mal_arr)})")
    print(f"{'='*80}")
    for pct in [5, 10, 25, 50, 75, 90, 99]:
        print(f"  p{pct:>5}: {np.percentile(mal_arr, pct):.4f}")
        
    print(f"\n{'='*80}")
    print(f"  BENIGN MAX SEQUENCE_RISK (N={len(benign_sr_arr)})")
    print(f"{'='*80}")
    for pct in [50, 75, 90, 95, 97, 99, 99.5, 99.9]:
        print(f"  p{pct:>5}: {np.percentile(benign_sr_arr, pct):.4f}")
        
    print(f"\n{'='*80}")
    print(f"  MALICIOUS MAX SEQUENCE_RISK (N={len(mal_sr_arr)})")
    print(f"{'='*80}")
    for pct in [5, 10, 25, 50, 75, 90, 99]:
        print(f"  p{pct:>5}: {np.percentile(mal_sr_arr, pct):.4f}")
    
    # Recommend a threshold
    # Target: FPR <= 2% (so we look at benign p98)
    p98_benign = np.percentile(benign_arr, 98)
    p99_benign = np.percentile(benign_arr, 99)
    p999_benign = np.percentile(benign_arr, 99.9)
    
    print(f"\n{'='*80}")
    print(f"  RECOMMENDED THRESHOLDS")
    print(f"{'='*80}")
    print(f"  For ~2.0% FPR -> Threshold: {p98_benign:.4f} (Recall: {np.sum(mal_arr >= p98_benign)/len(mal_arr)*100:.1f}%)")
    print(f"  For ~1.0% FPR -> Threshold: {p99_benign:.4f} (Recall: {np.sum(mal_arr >= p99_benign)/len(mal_arr)*100:.1f}%)")
    print(f"  For ~0.1% FPR -> Threshold: {p999_benign:.4f} (Recall: {np.sum(mal_arr >= p999_benign)/len(mal_arr)*100:.1f}%)")
    print(f"  Current config  -> Threshold: see config/thresholds.json tier05_lstm.block_threshold")

if __name__ == "__main__":
    evaluate_lstm_probability()
