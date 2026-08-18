"""
Deep probe LLMail-Inject:
1. Exact objectives JSON structure 
2. Scenario distribution (both splits)
3. Attack success rate per scenario
4. Team diversity per scenario
5. Estimate final sample size after smart sampling
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from datasets import load_dataset
from collections import Counter, defaultdict
import json

print("=" * 70)
print("DEEP PROBE: microsoft/LLMail-Inject")
print("=" * 70)

ds = load_dataset("microsoft/llmail-inject-challenge", trust_remote_code=True)

for split_name in ['Phase1', 'Phase2']:
    split = ds[split_name]
    print(f"\n{'='*70}")
    print(f"SPLIT: {split_name} — {len(split):,} rows")
    print(f"{'='*70}")
    
    # 1. Objectives JSON structure — check first 10 rows
    print(f"\n[1] Objectives JSON structure (first 5 rows):")
    obj_keys_all = Counter()
    obj_values_all = defaultdict(Counter)
    success_count = 0
    partial_count = 0
    
    for i, row in enumerate(split):
        try:
            obj = json.loads(row['objectives'])
            for k, v in obj.items():
                obj_keys_all[k] += 1
                obj_values_all[k][str(v)] += 1
            
            # Check attack success
            exfil_sent = obj.get('exfil.sent', False)
            exfil_dest = obj.get('exfil.destination', False)
            exfil_content = obj.get('exfil.content', False)
            
            if exfil_sent and exfil_dest and exfil_content:
                success_count += 1
            elif exfil_sent or exfil_dest or exfil_content:
                partial_count += 1
                
            if i < 5:
                print(f"  Row {i}: {json.dumps(obj, indent=2)}")
        except Exception as e:
            if i < 5:
                print(f"  Row {i}: PARSE ERROR — {e}")
    
    # 2. All objective keys and value distributions
    print(f"\n[2] Objective keys and value distributions:")
    for key in sorted(obj_keys_all.keys()):
        print(f"  '{key}' (appears in {obj_keys_all[key]:,} rows):")
        for val, cnt in obj_values_all[key].most_common():
            pct = 100 * cnt / len(split)
            print(f"    {val}: {cnt:,} ({pct:.1f}%)")
    
    # 3. Attack success summary
    total = len(split)
    fail_count = total - success_count - partial_count
    print(f"\n[3] Attack success summary:")
    print(f"  Full success (sent+dest+content):  {success_count:,} ({100*success_count/total:.1f}%)")
    print(f"  Partial (any exfil true):           {partial_count:,} ({100*partial_count/total:.1f}%)")
    print(f"  Failed (no exfil):                  {fail_count:,} ({100*fail_count/total:.1f}%)")
    
    # 4. Scenario distribution
    print(f"\n[4] Scenario distribution:")
    scenario_counter = Counter(row['scenario'] for row in split)
    for sc, cnt in scenario_counter.most_common():
        pct = 100 * cnt / total
        print(f"  '{sc}': {cnt:,} ({pct:.1f}%)")
    print(f"  Total unique scenarios: {len(scenario_counter)}")
    
    # 5. Team distribution
    print(f"\n[5] Team distribution:")
    team_counter = Counter(row['team_id'] for row in split)
    print(f"  Total unique teams: {len(team_counter)}")
    top5 = team_counter.most_common(5)
    for tid, cnt in top5:
        print(f"  Top team '{tid[:12]}...': {cnt:,} ({100*cnt/total:.1f}%)")
    bottom5 = team_counter.most_common()[-3:]
    for tid, cnt in bottom5:
        print(f"  Bottom team '{tid[:12]}...': {cnt:,}")
    
    # 6. Estimate smart sampling: top N per (scenario, team)
    print(f"\n[6] Smart sampling estimate (max 5 per scenario×team):")
    combo_counter = Counter((row['scenario'], row['team_id']) for row in split)
    estimated = sum(min(cnt, 5) for cnt in combo_counter.values())
    print(f"  Unique (scenario, team) combos: {len(combo_counter):,}")
    print(f"  Estimated rows after sample(5): {estimated:,}")
    estimated_10 = sum(min(cnt, 10) for cnt in combo_counter.values())
    print(f"  Estimated rows after sample(10): {estimated_10:,}")
    estimated_20 = sum(min(cnt, 20) for cnt in combo_counter.values())
    print(f"  Estimated rows after sample(20): {estimated_20:,}")

    # 7. Body field stats
    print(f"\n[7] Body field stats:")
    body_lengths = [len(row['body']) for row in split]
    print(f"  Min length: {min(body_lengths)}")
    print(f"  Max length: {max(body_lengths)}")
    print(f"  Mean length: {sum(body_lengths)/len(body_lengths):.0f}")
    empty_bodies = sum(1 for b in body_lengths if b == 0)
    print(f"  Empty bodies: {empty_bodies}")

print(f"\n{'='*70}")
print("DEEP PROBE COMPLETE")
print(f"{'='*70}")
