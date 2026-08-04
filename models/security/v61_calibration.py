import json
import logging
import os
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def calibrate_for_samples(samples, target_fpr):
    benign = [d for d in samples if d.get("label") == "benign" or (d.get("label") != "malicious" and d.get("label") != "unknown")]
    malicious = [d for d in samples if d.get("label") == "malicious"]
    
    if len(benign) < 10 or len(malicious) < 10:
        return None
        
    random.seed(42)
    random.shuffle(benign)
    random.shuffle(malicious)
    
    split_b = int(len(benign) * 0.7)
    split_m = int(len(malicious) * 0.7)
    
    val_set = benign[:split_b] + malicious[:split_m]
    test_set = benign[split_b:] + malicious[split_m:]
    
    y_true_val = [1 if d["label"] == "malicious" else 0 for d in val_set]
    y_scores_val = [d["ml_score"] for d in val_set]
    
    thresholds = sorted(list(set(y_scores_val)))
    
    total_benign_val = sum(1 for y in y_true_val if y == 0)
    total_malicious_val = sum(1 for y in y_true_val if y == 1)
    
    best_t = 1.0
    best_fnr_val = 1.0
    
    for t in thresholds:
        fp = sum(1 for y, s in zip(y_true_val, y_scores_val) if y == 0 and s >= t)
        fn = sum(1 for y, s in zip(y_true_val, y_scores_val) if y == 1 and s < t)
        fpr = fp / max(1, total_benign_val)
        fnr = fn / max(1, total_malicious_val)
        
        if fpr <= target_fpr:
            if fnr < best_fnr_val:
                best_fnr_val = fnr
                best_t = t
            elif fnr == best_fnr_val and t < best_t:
                best_t = t
                
    review_t = 0.35
    for t in reversed(thresholds):
        fn = sum(1 for y, s in zip(y_true_val, y_scores_val) if y == 1 and s < t)
        fnr = fn / max(1, total_malicious_val)
        if fnr <= 0.05:
            review_t = t
            break
            
    raw_block = best_t
    raw_review = review_t
    
    # If the model separates perfectly, raw_block might be lower than raw_review.
    # For a router, BLOCK must be >= REVIEW.
    final_block = max(raw_block, raw_review)
    final_review = min(raw_block, raw_review)
    
    final_review = max(0.20, final_review)
    return {"BLOCK": round(final_block, 4), "REVIEW": round(final_review, 4)}

def calibrate_thresholds(results_path, target_fpr=0.01):
    if not os.path.exists(results_path):
        logging.error(f"Khong tim thay file {results_path}.")
        return

    data = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try: data.append(json.loads(line))
            except: continue

    v61_samples = [d for d in data if d.get("ml_score") is not None and not d.get("llm_down")]
    for d in v61_samples:
        if "label" not in d:
            if d.get("label_source") == "self_reported_block": d["label"] = "malicious"
            elif d.get("label_source") == "self_reported_allow": d["label"] = "benign"
            else: d["label"] = "unknown"

    prompt_samples = [d for d in v61_samples if d.get("action_type") == "prompt"]
    action_samples = [d for d in v61_samples if d.get("action_type") == "tool_call"]
    
    out_dict = {}
    
    p_res = calibrate_for_samples(prompt_samples, target_fpr)
    if p_res:
        logging.info(f"Calibrated prompt_risk_model: {p_res}")
        out_dict["prompt_risk_model"] = p_res
        
    a_res = calibrate_for_samples(action_samples, target_fpr)
    if a_res:
        logging.info(f"Calibrated action_risk_model: {a_res}")
        out_dict["action_risk_model"] = a_res

    workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(workspace_dir, "config", "thresholds.json")
    
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            current_config = json.load(f)
    else:
        current_config = {}
        
    for k, v in out_dict.items():
        current_config[k] = v
        
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(current_config, f, indent=2)
        
    logging.info(f"Da cap nhat vao: {config_path}")

if __name__ == "__main__":
    workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    res_file = os.path.join(workspace_dir, "logs", "retrain_dataset.jsonl")
    if not os.path.exists(res_file):
        res_file = os.path.join(workspace_dir, "benchmark_baseline_results.json")
    calibrate_thresholds(res_file, target_fpr=0.04)
