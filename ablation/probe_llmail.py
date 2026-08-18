"""
Probe LLMail-Inject dataset to check:
1. Total rows
2. Available columns
3. Unique task_type values and counts
4. Rows remaining after filter: task_type IN [tool_call, email_action, calendar_action]
5. If filter is too aggressive, check broader alternatives
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from datasets import load_dataset
import json

print("=" * 60)
print("PROBING: microsoft/LLMail-Inject")
print("=" * 60)

# Load dataset - try different configs
print("\n[1] Loading dataset...")
try:
    ds = load_dataset("microsoft/llmail-inject-challenge", trust_remote_code=True)
    print(f"  Configs/splits available: {ds}")
    
    # Get the main split
    for split_name in ds:
        split = ds[split_name]
        print(f"\n[2] Split '{split_name}':")
        print(f"  Total rows: {len(split):,}")
        print(f"  Columns: {split.column_names}")
        
        # Check each column for task_type-like fields
        print(f"\n[3] Checking columns for task_type info...")
        for col in split.column_names:
            try:
                unique_vals = set()
                for i, row in enumerate(split):
                    val = row[col]
                    if isinstance(val, str) and len(val) < 100:
                        unique_vals.add(val)
                    elif isinstance(val, (int, float, bool)):
                        unique_vals.add(str(val))
                    if i > 5000 or len(unique_vals) > 50:
                        break
                if 2 <= len(unique_vals) <= 30:
                    print(f"  Column '{col}' has {len(unique_vals)} unique values (sample of 5000 rows):")
                    for v in sorted(unique_vals):
                        count = sum(1 for row in split if str(row[col]) == v)
                        print(f"    '{v}': {count:,}")
            except Exception as e:
                pass
        
        # Try to find task_type column (exact or fuzzy)
        task_type_candidates = [c for c in split.column_names if 'task' in c.lower() or 'type' in c.lower() or 'category' in c.lower()]
        print(f"\n[4] Task-type candidate columns: {task_type_candidates}")
        
        for col in task_type_candidates:
            print(f"\n  Column '{col}' value distribution:")
            from collections import Counter
            vals = [str(row[col]) for row in split if row[col] is not None]
            counter = Counter(vals)
            for val, count in counter.most_common(20):
                truncated = val[:80] if len(val) > 80 else val
                print(f"    '{truncated}': {count:,}")
        
        # Apply the proposed filter
        TARGET_TYPES = {'tool_call', 'email_action', 'calendar_action'}
        print(f"\n[5] Applying filter: task_type IN {TARGET_TYPES}")
        
        for col in task_type_candidates:
            filtered = [row for row in split if str(row[col]) in TARGET_TYPES]
            print(f"  Filter on '{col}': {len(filtered):,} / {len(split):,} rows remain ({100*len(filtered)/len(split):.1f}%)")
        
        # Also check if there are other useful columns
        print(f"\n[6] Sample row (first 3):")
        for i in range(min(3, len(split))):
            row = split[i]
            for k, v in row.items():
                val_str = str(v)[:200]
                print(f"    {k}: {val_str}")
            print("  ---")
        
        break  # Only process first split for now

except Exception as e:
    print(f"  Error loading dataset: {e}")
    print(f"\n  Trying to list available configs...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.dataset_info("microsoft/llmail-inject-challenge")
        print(f"  Dataset card: {info.card_data}")
    except Exception as e2:
        print(f"  Also failed: {e2}")

print("\n" + "=" * 60)
print("PROBE COMPLETE")
print("=" * 60)
