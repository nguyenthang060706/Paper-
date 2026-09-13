import json, random, re
from collections import defaultdict

random.seed(42)
dataset_path = "ablation/output/evo_pca_full.jsonl"
data = []
with open(dataset_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

benign = [r for r in data if r.get('attack_type') == 'benign']
benign_by_source = defaultdict(list)
for r in benign:
    benign_by_source[r.get('source_dataset', 'unknown')].append(r)

TARGET_BENIGN = 80
sampled_benign = []
for src, records in benign_by_source.items():
    proportion = len(records) / len(benign)
    n = max(1, round(proportion * TARGET_BENIGN))
    n = min(n, len(records))
    sampled_benign.extend(random.sample(records, n))
if len(sampled_benign) > TARGET_BENIGN:
    sampled_benign = random.sample(sampled_benign, TARGET_BENIGN)

with open("scratch/llmail_full_samples.json", "r", encoding="utf-8") as f:
    llmail_samples = json.load(f)

print(f"Loaded {len(llmail_samples)} llmail samples, {len(sampled_benign)} benign samples.")

# Candidate patterns for Tier 0
candidate_patterns = [
    # 1. Delimiters (must run on raw or accept normalised form)
    ("chat_special_tokens", re.compile(r'(?i)(?:<\||<i)(?:im_end|im_start|system|user|assistant|end|endoftext|begin|tool)[\w\s|i]*\|?>')),
    ("llama_sys_tokens", re.compile(r'(?i)<\/?(?:s|INST|SYS)>(?:\[INST\]|<<SYS>>)?|\[INST\]|\[\/INST\]|<<SYS>>|<\/<<SYS>>')),
    
    # 2. Email injection hijacking patterns
    ("mail_summarize_exfil", re.compile(r'(?i)\b(?:when\s+(?:asked\s+to\s+)?summaris\w+|after\s+(?:the\s+)?summary|finish(?:ed)?\s+summariz\w+|before\s+you\s+summarize|summariz\w+\s+(?:it|the\s+email|those))\b.{0,120}?\b(?:send|forward|dispatch|mail)\s+(?:an?\s+)?(?:confirmation\s+)?email\b')),
    ("mail_exfil_target", re.compile(r'(?i)\b(?:send|forward|dispatch|mail)\s+(?:an?\s+)?(?:confirmation\s+)?email\s+(?:to\s+)?["\'`]?[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+["\'`]?')),
    ("mail_suppress_summary", re.compile(r'(?i)\b(?:do\s+not|not)\s+(?:output|perform)\s+summariz\w+.{0,80}?\b(?:send\s+email|json\s+format)\b')),
    ("mail_instruction_override", re.compile(r'(?i)\b(?:need\s+not|must\s+not)\s+follow\s+(?:the\s+)?instructions?\s+of\s+previous\b')),
    ("mail_api_call_hijack", re.compile(r'(?i)\b(?:issue\s+(?:an?\s+)?api_call\s+to\s+send|api_call\s*\(mail\)|api_ca\\?ll\s+with\s+send\s+email)\b')),
    ("mail_c2_address", re.compile(r'(?i)\b(?:contact@contact\.(?:com|6ov)|mari@contact\.com)\b')),
    ("mail_safe_list_phish", re.compile(r'(?i)\b(?:safe\s+email\s+list|blacklist)\s*:\s*[a-zA-Z0-9_.+-]+@')),
    ("mail_word_swap_attack", re.compile(r'(?i)\b(?:change|replace)\s+the\s+word\s+["\'`]\w+["\'`]\s+with\s+the\s+word\s+["\'`]\w+["\'`]')),
    ("mail_before_summarize_imperative", re.compile(r'(?i)\bdo\s+this\s+before\s+you\s+summarize\b')),
    ("mail_project_zenith_exfil", re.compile(r'(?i)\bProject\s+Zenith\b.{0,150}?\b(?:send\s+(?:an?\s+)?email|contact@)\b')),
]

print("\n--- Evaluating patterns on LLMail vs Benign ---")
caught_llmail = set()
for name, cp in candidate_patterns:
    ll_hits = [item["idx"] for item in llmail_samples if cp.search(item["action"])]
    bn_hits = [i for i, b in enumerate(sampled_benign) if cp.search(b["action"])]
    print(f"{name:32s}: LLMail={len(ll_hits):2d}/{len(llmail_samples)} | Benign FP={len(bn_hits)}")
    caught_llmail.update(ll_hits)

print(f"\nTotal unique LLMail caught: {len(caught_llmail)}/{len(llmail_samples)} ({len(caught_llmail)/len(llmail_samples)*100:.1f}%)")

uncaught = [item for item in llmail_samples if item["idx"] not in caught_llmail]
print(f"Uncaught count: {len(uncaught)}")
for u in uncaught[:10]:
    print(f"Uncaught [{u['idx']}]: {repr(u['action'][:140])}")
