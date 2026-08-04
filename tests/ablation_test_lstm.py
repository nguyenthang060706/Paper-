import os
import sys
import torch
import numpy as np

# Add parent directory to path to import core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.tier_lstm import SessionAwareLSTMRisk

BASELINE_SEQ = [
    [0.4861735698828816, 0.2987, 0.003874204890000001, 0.2, 0.0, 0.1, 0.0],
    [0.6523029856408026, 0.4294, 0.004304672100000001, 0.3, 0.0, 0.1, 0.0],
    [0.6579212815507391, 0.0279, 0.004782969000000001, 0.4, 0.0, 0.2, 0.0],
    [0.5542560043585966, 0.0430, 0.00531441, 0.5, 0.0, 0.2, 0.0],
    [0.45342702464297435, 0.0726, 0.005904900000000001, 0.6, 0.0, 0.2, 0.0],
    [0.5314247332595274, 0.0139, 0.006561, 0.7, 0.0, 0.2, 0.0],
    [0.4091975924478789, 0.1213, 0.007290000000000001, 0.8, 0.0, 0.2, 0.0],
    [0.3587696298664709, 0.0249, 0.008100000000000001, 0.9, 0.0, 0.2, 0.1],
    [0.6465648768921554, 0.0216, 0.009000000000000001, 1.0, 0.0, 0.2, 0.1],
    [0.47742236995134646, 0.1241, 0.0100000000000000, 1.0, 0.0, 0.2, 0.0]
]

FEATURE_NAMES = [
    "iat_norm", "payload_norm", "risk", 
    "depth_norm", "fanout_norm", "tool_div_norm", "res_freq_norm"
]

def run_ablation():
    # Load the deployed Tier0.5 LSTM Model
    lstm_tier = SessionAwareLSTMRisk()
    if not lstm_tier.available:
        print("Model not available.")
        return
        
    model = lstm_tier.model
    model.eval()
    
    def forward_pass(seq_list):
        t = torch.tensor([seq_list], dtype=torch.float32)
        lengths = torch.tensor([len(seq_list)], dtype=torch.int64)
        with torch.no_grad():
            logits = model(t, lengths)
            return logits.item()
            
    # 1. Baseline
    baseline_logits = forward_pass(BASELINE_SEQ)
    print(f"BASELINE LOGITS: {baseline_logits:.4f}")
    
    # Assert validation
    if abs(baseline_logits - 9.5319) > 0.01:
        print(f"WARNING: Baseline logits ({baseline_logits:.4f}) do not match expected 9.5319!")
    else:
        print("VALIDATION SUCCESS: Baseline matches expected 9.5319")
        
    print("-" * 50)
    print(f"{'Ablated Feature':<15} | {'Type':<8} | {'New Logit':<10} | {'Delta':<10}")
    print("-" * 50)
    
    # Pre-calculate medians
    baseline_np = np.array(BASELINE_SEQ)
    medians = np.median(baseline_np, axis=0)
    
    results = []
    
    for i, feature in enumerate(FEATURE_NAMES):
        # Median replacement
        seq_copy = np.copy(baseline_np)
        seq_copy[:, i] = medians[i]
        
        logits_med = forward_pass(seq_copy.tolist())
        delta_med = baseline_logits - logits_med
        results.append((feature, "Median", logits_med, delta_med))
        
        # Zero replacement
        seq_copy_zero = np.copy(baseline_np)
        seq_copy_zero[:, i] = 0.0
        
        logits_zero = forward_pass(seq_copy_zero.tolist())
        delta_zero = baseline_logits - logits_zero
        results.append((feature, "Zero", logits_zero, delta_zero))
        
    # Sort by delta descending (largest drop in maliciousness)
    results.sort(key=lambda x: x[3], reverse=True)
    
    for feat, type_, logit, delta in results:
        print(f"{feat:<15} | {type_:<8} | {logit:>9.4f} | {delta:>9.4f}")

if __name__ == '__main__':
    run_ablation()
