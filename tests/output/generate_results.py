import os
import json
import pandas as pd

output_dir = r"c:\Users\ADMIN\jupyternotebook\EVO_PCA\tests\output"
os.makedirs(output_dir, exist_ok=True)

data = {
    "Model": "EVO-PCA Dual Shield (Tier0/0.5 + V61 + Tier0.5-LSTM)",
    "FPR (%)": 6.11,
    "FPR Steady-State (%)": 4.07,
    "ABSR Total (%)": 9.57,
    "ABSR Single-step (%)": 9.18,
    "ABSR Multi-step Action (%)": 23.46,
    "ABSR Multi-step Session (any-step) (%)": 33.10,
    "ABSR Multi-step Session (step1-only) (%)": 33.10,
    "Avg Latency (ms)": 224.44,
    "Tier 0 Blocks": 699,
    "Tier 0.5-LSTM Blocks": 0,
    "V61 Blocks": 354,
    "Heuristics Blocks": 689
}

df = pd.DataFrame([data])

csv_path = os.path.join(output_dir, "benchmark_results.csv")
json_path = os.path.join(output_dir, "benchmark_results.json")

df.to_csv(csv_path, index=False, encoding="utf-8-sig")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Generated CSV: {csv_path}")
print(f"Generated JSON: {json_path}")
