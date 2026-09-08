"""Calibrate SEMANTIC_DELTA_THRESHOLD from empirical benign cosine distance distribution.

Reads the balanced dataset, runs compute_sequence_risk on all benign sessions,
and outputs the p50/p90/p95/p99/max of cosine distances.
This tells us the correct threshold so that benign sessions do NOT trigger
is_delta_violation.
"""
import os, sys, json, warnings
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict

def calibrate_semantic_delta():
    print("=" * 80)
    print("  CALIBRATING SEMANTIC_DELTA_THRESHOLD FROM BENIGN DISTRIBUTION")
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
    
    # 2. Group by session and label
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
    
    # 3. Compute cosine distances for benign sessions
    try:
        from models.security.advanced_heuristics import SharedSemanticEncoder
        encoder = SharedSemanticEncoder.get()
        if encoder is None:
            print("SharedSemanticEncoder unavailable, cannot calibrate.")
            return
    except Exception as e:
        print(f"Cannot load encoder: {e}")
        return
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    benign_distances = []
    malicious_distances = []
    
    # Process benign sessions
    print("\nProcessing benign sessions...")
    for sid, recs in benign_sessions.items():
        actions = [r["action"] for r in recs]
        if len(actions) < 2:
            continue
        try:
            embeddings = encoder.encode(actions)
        except Exception:
            continue
        for i in range(1, len(actions)):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i-1].reshape(1, -1)
            )[0][0]
            dist = max(0.0, 1.0 - float(sim))
            benign_distances.append(dist)
    
    # Process malicious sessions
    print("Processing malicious sessions...")
    for sid, recs in malicious_sessions.items():
        actions = [r["action"] for r in recs]
        if len(actions) < 2:
            continue
        try:
            embeddings = encoder.encode(actions)
        except Exception:
            continue
        for i in range(1, len(actions)):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i-1].reshape(1, -1)
            )[0][0]
            dist = max(0.0, 1.0 - float(sim))
            malicious_distances.append(dist)
    
    benign_arr = np.array(benign_distances) if benign_distances else np.array([0.0])
    mal_arr = np.array(malicious_distances) if malicious_distances else np.array([0.0])
    
    # Apply the gain factor used in compute_sequence_risk
    gain = 1.2
    benign_gained = np.minimum(benign_arr * gain, 1.0)
    mal_gained = np.minimum(mal_arr * gain, 1.0)
    
    print(f"\n{'='*80}")
    print(f"  BENIGN COSINE DISTANCE DISTRIBUTION (raw, N={len(benign_arr)})")
    print(f"{'='*80}")
    for pct in [50, 75, 90, 95, 97, 99, 99.5, 99.9]:
        val = np.percentile(benign_arr, pct)
        print(f"  p{pct:>5}: {val:.4f}")
    print(f"  max  : {benign_arr.max():.4f}")
    print(f"  mean : {benign_arr.mean():.4f}")
    print(f"  std  : {benign_arr.std():.4f}")
    
    print(f"\n{'='*80}")
    print(f"  BENIGN COSINE DISTANCE DISTRIBUTION (after gain={gain}, N={len(benign_gained)})")
    print(f"{'='*80}")
    for pct in [50, 75, 90, 95, 97, 99, 99.5, 99.9]:
        val = np.percentile(benign_gained, pct)
        print(f"  p{pct:>5}: {val:.4f}")
    print(f"  max  : {benign_gained.max():.4f}")
    
    print(f"\n{'='*80}")
    print(f"  MALICIOUS COSINE DISTANCE DISTRIBUTION (after gain={gain}, N={len(mal_gained)})")
    print(f"{'='*80}")
    for pct in [50, 75, 90, 95, 97, 99]:
        val = np.percentile(mal_gained, pct)
        print(f"  p{pct:>5}: {val:.4f}")
    print(f"  max  : {mal_gained.max():.4f}")
    
    # Recommend threshold at benign p99.9 (after gain) + margin
    p99_benign = np.percentile(benign_gained, 99)
    p999_benign = np.percentile(benign_gained, 99.9)
    recommended = round(p999_benign + 0.01, 2)
    
    print(f"\n{'='*80}")
    print(f"  RECOMMENDATION")
    print(f"{'='*80}")
    print(f"  Benign p99  (after gain): {p99_benign:.4f}")
    print(f"  Benign p99.9 (after gain): {p999_benign:.4f}")
    print(f"  Current threshold: 0.65")
    print(f"  Recommended threshold (p99.9 + margin): {recommended}")
    
    # Check how many malicious would still be caught
    if len(mal_gained) > 0:
        mal_caught = np.sum(mal_gained >= recommended) / len(mal_gained) * 100
        print(f"  Malicious recall at recommended threshold: {mal_caught:.1f}%")
    
    # Check how many benign would be falsely flagged
    benign_fp = np.sum(benign_gained >= recommended) / len(benign_gained) * 100
    print(f"  Benign FP at recommended threshold: {benign_fp:.2f}%")
    
    # Save distributions for reference
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    import pandas as pd
    df = pd.DataFrame({
        "benign_raw": pd.Series(benign_arr),
        "benign_gained": pd.Series(benign_gained),
    })
    df.to_csv(os.path.join(output_dir, "semantic_delta_calibration_benign.csv"), index=False)
    
    df_mal = pd.DataFrame({
        "malicious_raw": pd.Series(mal_arr),
        "malicious_gained": pd.Series(mal_gained),
    })
    df_mal.to_csv(os.path.join(output_dir, "semantic_delta_calibration_malicious.csv"), index=False)
    
    print(f"\nSaved calibration data to {output_dir}/semantic_delta_calibration_*.csv")
    
    return recommended

if __name__ == "__main__":
    calibrate_semantic_delta()
