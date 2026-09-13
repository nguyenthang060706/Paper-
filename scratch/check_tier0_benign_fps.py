import sys; sys.path.insert(0, '.')
from core.tier0 import LlamaFirewallTier0
import json

t0 = LlamaFirewallTier0()
benign = []
with open('ablation/output/evo_pca_full.jsonl', encoding='utf-8') as f:
    for line in f:
        if '"attack_type": "benign"' in line:
            benign.append(json.loads(line))
            if len(benign) >= 1000:
                break

fps = [b for b in benign if t0.scan(b['action']).is_blocked]
print(f"Tier 0 benign FPs on 1000 samples: {len(fps)}")
for b in fps[:10]:
    print("FP:", b.get('source_dataset'), "|", repr(b['action'][:100]))
