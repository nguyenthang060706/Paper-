import json
import random

random.seed(42)

input_file = "D:/DEMO_GROUP_1/Benchmark_Datasets/output/evo_pca_full_neuralchemy_backup_20260702.jsonl"
output_file = "D:/DEMO_GROUP_1/Benchmark_Datasets/output/evo_pca_7k_balanced.jsonl"

groups = {
    'benign_benign': [],
    'malicious_malicious_single': [],
    'benign_malicious_multistep': [],
    'malicious_malicious_multistep': []
}

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        key = str(r.get('label')) + '_' + str(r.get('attack_type'))
        if key in groups:
            groups[key].append(line)

# Stratified sampling
selected = []
# 1. Keep ALL multi-step (edge cases)
selected.extend(groups['benign_malicious_multistep'])
selected.extend(groups['malicious_malicious_multistep'])

# 2. Sample the rest to reach 7000 total
remaining_needed = 7000 - len(selected)
half = remaining_needed // 2

sampled_benign = random.sample(groups['benign_benign'], half)
sampled_single = random.sample(groups['malicious_malicious_single'], remaining_needed - half)

selected.extend(sampled_benign)
selected.extend(sampled_single)

# Shuffle the final dataset so it's not grouped
random.shuffle(selected)

with open(output_file, 'w', encoding='utf-8') as f:
    for line in selected:
        f.write(line)

print(f"Created {output_file} with {len(selected)} samples.")
print(f"- Benign (Single): {len(sampled_benign)}")
print(f"- Malicious (Single): {len(sampled_single)}")
print(f"- Benign (Multi-step): {len(groups['benign_malicious_multistep'])}")
print(f"- Malicious (Multi-step): {len(groups['malicious_malicious_multistep'])}")
