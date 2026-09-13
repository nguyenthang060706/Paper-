import json, random
from collections import defaultdict

random.seed(42)
dataset_path = "ablation/output/evo_pca_full.jsonl"
data = []
with open(dataset_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

single = [r for r in data if r.get('attack_type') == 'malicious_single']
single_by_source = defaultdict(list)
for r in single:
    single_by_source[r.get('source_dataset', 'unknown')].append(r)

TARGET_SINGLE = 80
sampled_single = []
for src, records in single_by_source.items():
    proportion = len(records) / len(single)
    n = max(1, round(proportion * TARGET_SINGLE))
    n = min(n, len(records))
    sampled = random.sample(records, n)
    sampled_single.extend(sampled)

if len(sampled_single) > TARGET_SINGLE:
    sampled_single = random.sample(sampled_single, TARGET_SINGLE)

llmail_samples = [r for r in sampled_single if r.get('source_dataset') == 'llmail-inject']
print(f"Sampled llmail-inject records in smoke test: {len(llmail_samples)}")

with open("scratch/llmail_full_samples.json", "w", encoding="utf-8") as f:
    json.dump([{"idx": i+1, "action": r["action"]} for i, r in enumerate(llmail_samples)], f, indent=2, ensure_ascii=False)

print("Saved to scratch/llmail_full_samples.json")
