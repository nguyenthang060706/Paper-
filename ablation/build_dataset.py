#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
"""
EVO-PCA v3.6 Benchmark Dataset Pipeline
=========================================
Tổng hợp 7 datasets công khai thành bộ benchmark 3 lớp:
  L1 — NLP single-action  (hackaprompt, neuralchemy, prodnull)
  L2 — Tool call injection (LLMail-Inject, BIPIA, prodnull-benign)
  L3 — Multi-step sequences (AgentHarm, agent-traces, synthetic)

Usage:
  python build_dataset.py [--hf-token TOKEN] [--skip-bipia] [--skip-agentharm]
                          [--output-dir ./output] [--seed 42]
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
SCHEMA_FIELDS = [
    "action", "label", "session_id", "attack_type",
    "step_index", "source_dataset", "augmented",
]
METADATA_FIELDS = [
    "exfil_success", "defense_caught", "scenario", "difficulty_level",
]
TARGET_RATIO = 0.6  # 60% malicious / 40% benign
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("evo-pca-pipeline")


# ══════════════════════════════════════════════════════════════
# LAYER 1 — NLP Single-Action
# ══════════════════════════════════════════════════════════════

def load_hackaprompt() -> pd.DataFrame:
    """
    hackaprompt/hackaprompt-dataset
    - Filter: difficulty_level >= 5
    - Dedup: exact-match on prompt (lowercase+strip)
    - Sample: max 15,000 malicious
    """
    from datasets import load_dataset

    log.info("Loading hackaprompt/hackaprompt-dataset...")
    ds = load_dataset("hackaprompt/hackaprompt-dataset", split="train")
    df = ds.to_pandas()
    log.info(f"  Raw rows: {len(df):,}")

    # Identify columns
    prompt_col = None
    diff_col = None
    for col in df.columns:
        if "prompt" in col.lower() or "user_input" in col.lower():
            prompt_col = prompt_col or col
        if "level" in col.lower() or "difficult" in col.lower():
            diff_col = diff_col or col

    if prompt_col is None:
        # Fallback: try common names
        for c in ["user_input", "prompt", "text"]:
            if c in df.columns:
                prompt_col = c
                break
    if diff_col is None:
        for c in ["level", "difficulty_level", "difficulty"]:
            if c in df.columns:
                diff_col = c
                break

    log.info(f"  Using prompt_col='{prompt_col}', diff_col='{diff_col}'")

    # Filter difficulty >= 5
    if diff_col and diff_col in df.columns:
        df[diff_col] = pd.to_numeric(df[diff_col], errors="coerce")
        before = len(df)
        df = df[df[diff_col] >= 5].copy()
        log.info(f"  After difficulty>=5 filter: {len(df):,} (dropped {before - len(df):,})")
    else:
        log.warning("  No difficulty column found — using all rows")

    # Dedup on lowercase+strip prompt
    if prompt_col and prompt_col in df.columns:
        df["_dedup_key"] = df[prompt_col].astype(str).str.lower().str.strip()
        before = len(df)
        df = df.drop_duplicates(subset="_dedup_key").copy()
        log.info(f"  After dedup: {len(df):,} (dropped {before - len(df):,})")
    else:
        log.warning("  No prompt column found — skipping dedup")
        prompt_col = df.columns[0]  # fallback

    # Sample max 15K
    if len(df) > 15_000:
        df = df.sample(n=15_000, random_state=SEED).copy()
        log.info(f"  Sampled down to 15,000")

    # Map to schema
    records = []
    for idx, row in df.iterrows():
        action = str(row.get(prompt_col, ""))
        if not action.strip():
            continue
        records.append({
            "action": action,
            "label": "malicious",
            "session_id": f"hack-{len(records)}",
            "attack_type": "malicious_single",
            "step_index": 0,
            "source_dataset": "hackaprompt",
            "augmented": False,
        })

    result = pd.DataFrame(records)
    log.info(f"  Final hackaprompt: {len(result):,} malicious samples")
    return result


def load_neuralchemy() -> pd.DataFrame:
    """
    neuralchemy/Prompt-injection-dataset
    - Config: 'core' (original, no augmented)
    - Rename: text → action
    - Map labels: 1 → malicious, 0 → benign
    """
    from datasets import load_dataset

    log.info("Loading neuralchemy/Prompt-injection-dataset...")
    try:
        ds = load_dataset("neuralchemy/Prompt-injection-dataset", "core", split="train")
    except Exception:
        log.warning("  'core' config not found, trying default...")
        ds = load_dataset("neuralchemy/Prompt-injection-dataset", split="train")

    df = ds.to_pandas()
    log.info(f"  Raw rows: {len(df):,}")
    log.info(f"  Columns: {list(df.columns)}")

    # Find text and label columns
    text_col = None
    label_col = None
    for col in df.columns:
        if col.lower() in ("text", "prompt", "action", "input"):
            text_col = text_col or col
        if col.lower() in ("label", "is_injection", "class", "target"):
            label_col = label_col or col

    if text_col is None:
        text_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[-1]

    log.info(f"  Using text_col='{text_col}', label_col='{label_col}'")

    records = []
    for idx, row in df.iterrows():
        action = str(row[text_col])
        if not action.strip():
            continue

        raw_label = row[label_col]
        if isinstance(raw_label, (int, float)):
            label = "malicious" if int(raw_label) == 1 else "benign"
        elif isinstance(raw_label, str):
            label = "malicious" if raw_label.lower() in ("1", "malicious", "injection", "true") else "benign"
        else:
            label = "benign"

        attack_type = "malicious_single" if label == "malicious" else "benign"

        records.append({
            "action": action,
            "label": label,
            "session_id": f"neur-{len(records)}",
            "attack_type": attack_type,
            "step_index": 0,
            "source_dataset": "neuralchemy",
            "augmented": False,
        })

    result = pd.DataFrame(records)
    mal = (result["label"] == "malicious").sum()
    ben = (result["label"] == "benign").sum()
    log.info(f"  Final neuralchemy: {len(result):,} total ({mal:,} malicious, {ben:,} benign)")
    return result


def load_prodnull(hf_token: str = None) -> pd.DataFrame:
    """
    prodnull/prompt-injection-repo-dataset
    - Gated: needs HuggingFace token
    - 24 NLP categories, obfuscation variants
    - Benign samples (label=0) will also be used for L2 benign pool
    """
    from datasets import load_dataset

    log.info("Loading prodnull/prompt-injection-repo-dataset...")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    try:
        ds = load_dataset("prodnull/prompt-injection-repo-dataset", split="train",
                          token=hf_token)
    except Exception as e:
        log.error(f"  Failed to load prodnull (gated dataset): {e}")
        log.error("  Provide --hf-token with accepted gating conditions")
        return pd.DataFrame(columns=SCHEMA_FIELDS)

    df = ds.to_pandas()
    log.info(f"  Raw rows: {len(df):,}")
    log.info(f"  Columns: {list(df.columns)}")

    # Find text and label columns
    text_col = None
    label_col = None
    for col in df.columns:
        if col.lower() in ("text", "prompt", "action", "input"):
            text_col = text_col or col
        if col.lower() in ("label", "is_injection", "class", "target"):
            label_col = label_col or col

    if text_col is None:
        text_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[-1]

    log.info(f"  Using text_col='{text_col}', label_col='{label_col}'")

    records = []
    for idx, row in df.iterrows():
        action = str(row[text_col])
        if not action.strip():
            continue

        raw_label = row[label_col]
        if isinstance(raw_label, (int, float)):
            label = "malicious" if int(raw_label) == 1 else "benign"
        elif isinstance(raw_label, str):
            label = "malicious" if raw_label.lower() in ("1", "malicious", "injection", "true") else "benign"
        else:
            label = "benign"

        attack_type = "malicious_single" if label == "malicious" else "benign"

        records.append({
            "action": action,
            "label": label,
            "session_id": f"prod-{len(records)}",
            "attack_type": attack_type,
            "step_index": 0,
            "source_dataset": "prodnull",
            "augmented": False,
        })

    result = pd.DataFrame(records)
    mal = (result["label"] == "malicious").sum()
    ben = (result["label"] == "benign").sum()
    log.info(f"  Final prodnull: {len(result):,} total ({mal:,} malicious, {ben:,} benign)")
    return result


# ══════════════════════════════════════════════════════════════
# LAYER 2 — Tool Call Injection
# ══════════════════════════════════════════════════════════════

def load_llmail_inject() -> pd.DataFrame:
    """
    microsoft/LLMail-Inject (Phase1 + Phase2)
    - ALL rows = malicious (injection attempts)
    - Label strategy: L1 (intent-based, all malicious)
    - Sampling: 10 per (scenario × team_id) combo
    - Metadata: exfil_success, defense_caught, scenario
    """
    from datasets import load_dataset

    log.info("Loading microsoft/llmail-inject-challenge...")
    ds = load_dataset("microsoft/llmail-inject-challenge")

    all_records = []

    for split_name in ["Phase1", "Phase2"]:
        if split_name not in ds:
            log.warning(f"  Split '{split_name}' not found, skipping")
            continue

        split = ds[split_name]
        log.info(f"  Processing {split_name}: {len(split):,} rows")

        # Convert to DataFrame for efficient sampling
        df = split.to_pandas()

        # Parse objectives JSON for metadata
        def parse_objectives(obj_str):
            try:
                obj = json.loads(obj_str) if isinstance(obj_str, str) else obj_str
                return {
                    "exfil_success": (
                        obj.get("exfil.sent", False) and
                        obj.get("exfil.destination", False) and
                        obj.get("exfil.content", False)
                    ),
                    "defense_caught": not obj.get("defense.undetected", True),
                }
            except (json.JSONDecodeError, TypeError):
                return {"exfil_success": False, "defense_caught": False}

        meta = df["objectives"].apply(parse_objectives).apply(pd.Series)
        df = pd.concat([df, meta], axis=1)

        # Smart sampling: max 10 per (scenario × team_id)
        df_sampled = (
            df.groupby(["scenario", "team_id"], group_keys=False)
            .apply(lambda x: x.sample(min(len(x), 10), random_state=SEED))
        )
        log.info(f"  After sampling (10 per combo): {len(df_sampled):,}")

        # Map to schema
        for idx, row in df_sampled.iterrows():
            body = str(row.get("body", ""))
            if not body.strip():
                continue

            all_records.append({
                "action": body,
                "label": "malicious",
                "session_id": f"llmail-{split_name.lower()}-{len(all_records)}",
                "attack_type": "malicious_single",
                "step_index": 0,
                "source_dataset": "llmail-inject",
                "augmented": False,
                # Metadata
                "exfil_success": row.get("exfil_success", False),
                "defense_caught": row.get("defense_caught", False),
                "scenario": row.get("scenario", "unknown"),
            })

    result = pd.DataFrame(all_records)
    log.info(f"  Final LLMail-Inject: {len(result):,} malicious samples")
    log.info(f"    exfil_success=True: {result['exfil_success'].sum():,}")
    log.info(f"    defense_caught=True: {result['defense_caught'].sum():,}")
    return result


def load_bipia() -> pd.DataFrame:
    """
    microsoft/BIPIA — Benchmark for Indirect Prompt Injection Attacks
    - Try HuggingFace first, fallback to processed versions
    - 5 tasks: email, web_qa, table_qa, summarization, code_qa
    - Extract 'attack' field as action
    - Create benign counterparts from context without attack
    """
    from datasets import load_dataset

    log.info("Loading microsoft/BIPIA...")

    # Try loading from HuggingFace (various known dataset IDs)
    bipia_sources = [
        "microsoft/BIPIA",
        "MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT",  # Pre-processed backup
        "wu981526092/BIPIA",  # Community mirror
        "rubend18/ChatGPT-Jailbreak-Prompts",  # Additional injection prompts
    ]

    df = None
    for source in bipia_sources:
        try:
            log.info(f"  Trying: {source}...")
            ds = load_dataset(source, split="train")
            df = ds.to_pandas()
            log.info(f"  Loaded from {source}: {len(df):,} rows")
            log.info(f"  Columns: {list(df.columns)}")
            break
        except Exception as e:
            log.warning(f"  Failed: {e}")
            continue

    if df is None or len(df) == 0:
        log.error("  Could not load BIPIA from any source")
        return pd.DataFrame(columns=SCHEMA_FIELDS)

    # Identify relevant columns
    attack_col = None
    context_col = None
    task_col = None

    for col in df.columns:
        cl = col.lower()
        if "attack" in cl or "injection" in cl or "injected" in cl:
            attack_col = attack_col or col
        if "context" in cl or "document" in cl or "content" in cl:
            context_col = context_col or col
        if "task" in cl or "domain" in cl or "type" in cl:
            task_col = task_col or col

    log.info(f"  attack_col='{attack_col}', context_col='{context_col}', task_col='{task_col}'")

    records = []

    # Malicious samples: rows with attack content
    if attack_col and attack_col in df.columns:
        for idx, row in df.iterrows():
            attack = str(row[attack_col])
            if not attack.strip() or attack.lower() == "nan":
                continue

            task = str(row.get(task_col, "unknown")) if task_col else "unknown"

            records.append({
                "action": attack,
                "label": "malicious",
                "session_id": f"bipia-{task}-{len(records)}",
                "attack_type": "malicious_single",
                "step_index": 0,
                "source_dataset": "bipia",
                "augmented": False,
            })
    else:
        # Fallback: use first text column
        text_col = df.select_dtypes(include=["object"]).columns[0]
        log.warning(f"  No attack column found, using '{text_col}' as fallback")
        for idx, row in df.iterrows():
            records.append({
                "action": str(row[text_col]),
                "label": "malicious",
                "session_id": f"bipia-{len(records)}",
                "attack_type": "malicious_single",
                "step_index": 0,
                "source_dataset": "bipia",
                "augmented": False,
            })

    # Benign counterparts: context without attack
    if context_col and context_col in df.columns:
        benign_df = df[df[context_col].notna()].copy()
        benign_sample = benign_df.sample(min(len(benign_df), len(records) // 2), random_state=SEED)
        for idx, row in benign_sample.iterrows():
            context = str(row[context_col])
            if not context.strip() or context.lower() == "nan":
                continue
            records.append({
                "action": context,
                "label": "benign",
                "session_id": f"bipia-benign-{len(records)}",
                "attack_type": "benign",
                "step_index": 0,
                "source_dataset": "bipia",
                "augmented": False,
            })

    result = pd.DataFrame(records)
    mal = (result["label"] == "malicious").sum()
    ben = (result["label"] == "benign").sum()
    log.info(f"  Final BIPIA: {len(result):,} total ({mal:,} malicious, {ben:,} benign)")
    return result


# ══════════════════════════════════════════════════════════════
# LAYER 3 — Multi-Step Sequences
# ══════════════════════════════════════════════════════════════

def load_agentharm() -> pd.DataFrame:
    """
    ai-safety-institute/AgentHarm
    - Extract tool trajectories as action sequences
    - Group by task → session_id
    - step_index per action
    - Only label LAST step as malicious (earlier steps look benign)
    """
    from datasets import load_dataset

    log.info("Loading ai-safety-institute/AgentHarm...")

    # AgentHarm has 3 configs: 'harmful', 'harmless_benign', 'chat'
    configs_to_load = [
        ("harmful", True),           # Harmful tasks → malicious
        ("harmless_benign", False),  # Benign counterparts → benign
    ]

    all_dfs = []
    for config_name, is_harmful in configs_to_load:
        loaded = False
        for split_name in ["train", "test_public", "validation", "test"]:
            try:
                log.info(f"  Loading config '{config_name}', split '{split_name}'...")
                ds = load_dataset("ai-safety-institute/AgentHarm", config_name, split=split_name)
                df = ds.to_pandas()
                df["_is_harmful"] = is_harmful
                df["_config"] = config_name
                all_dfs.append(df)
                log.info(f"    Config '{config_name}' split '{split_name}': {len(df):,} rows")
                loaded = True
                break
            except Exception as e:
                continue
        if not loaded:
            log.warning(f"  Could not load config '{config_name}' from any split")

    if not all_dfs:
        log.error("  Could not load any AgentHarm configs")
        return pd.DataFrame(columns=SCHEMA_FIELDS)

    df = pd.concat(all_dfs, ignore_index=True)

    log.info(f"  Total rows across configs: {len(df):,}")
    log.info(f"  Columns: {list(df.columns)}")

    records = []

    # Try to find trajectory/tool call columns
    tool_cols = [c for c in df.columns if any(
        k in c.lower() for k in ["tool", "function", "action", "trajectory", "call"]
    )]
    label_cols = [c for c in df.columns if any(
        k in c.lower() for k in ["label", "harm", "category", "type"]
    )]
    task_cols = [c for c in df.columns if any(
        k in c.lower() for k in ["task", "id", "name", "prompt"]
    )]

    log.info(f"  Tool-related columns: {tool_cols}")
    log.info(f"  Label-related columns: {label_cols}")
    log.info(f"  Task-related columns: {task_cols}")

    # Determine label column
    label_col = label_cols[0] if label_cols else None

    for idx, row in df.iterrows():
        task_id = str(row.get(task_cols[0], idx)) if task_cols else str(idx)
        session_id = f"agentharm-{task_id}"

        # Determine if harmful — use _is_harmful flag from config loading
        is_harmful = row.get("_is_harmful", True)
        if not isinstance(is_harmful, bool) and label_col:
            lval = str(row[label_col]).lower()
            is_harmful = any(k in lval for k in ["harm", "malicious", "unsafe", "1", "true"])

        # Extract tool calls / actions from the row
        actions = []
        for tc in tool_cols:
            val = row.get(tc)
            if val is None:
                continue
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        actions.extend([str(a) for a in parsed])
                    else:
                        actions.append(str(parsed))
                except json.JSONDecodeError:
                    actions.append(val)
            elif isinstance(val, list):
                actions.extend([str(a) for a in val])

        # If no tool columns found, use the task description as a single action
        if not actions:
            for tc in task_cols:
                val = row.get(tc)
                if val and isinstance(val, str) and len(val) > 10:
                    actions.append(val)
                    break

        if not actions:
            continue

        # Create records with step_index
        for step_idx, action_str in enumerate(actions):
            is_last = (step_idx == len(actions) - 1)

            records.append({
                "action": action_str,
                "label": "malicious" if (is_harmful and is_last) else "benign",
                "session_id": session_id,
                "attack_type": "malicious_multistep" if is_harmful else "benign",
                "step_index": step_idx,
                "source_dataset": "agentharm",
                "augmented": False,
            })

    result = pd.DataFrame(records)
    sessions = result["session_id"].nunique()
    mal_sessions = result[result["attack_type"] == "malicious_multistep"]["session_id"].nunique()
    log.info(f"  Final AgentHarm: {len(result):,} records across {sessions} sessions ({mal_sessions} malicious)")
    return result


def load_agent_traces() -> pd.DataFrame:
    """
    trace-commons/agent-traces
    - Extract tool calls from sessions
    - All = benign
    - Filter: >= 2 tool calls per session
    """
    from datasets import load_dataset

    log.info("Loading trace-commons/agent-traces...")

    try:
        ds = load_dataset("trace-commons/agent-traces")
    except Exception as e:
        log.error(f"  Failed to load agent-traces: {e}")
        return pd.DataFrame(columns=SCHEMA_FIELDS)

    records = []

    for split_name in ds:
        split = ds[split_name]
        log.info(f"  Processing split '{split_name}': {len(split):,} rows")

        for idx, row in enumerate(split):
            # Try to extract messages/tool calls
            messages = None
            session_id = None

            for col in ["messages", "conversation", "turns", "events"]:
                if col in row and row[col]:
                    messages = row[col]
                    break

            for col in ["session_id", "id", "conversation_id", "trace_id"]:
                if col in row and row[col]:
                    session_id = str(row[col])
                    break

            if session_id is None:
                session_id = f"trace-{split_name}-{idx}"

            if messages is None:
                continue

            # Extract tool calls from messages
            tool_calls = []
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict):
                        # Look for tool calls in various formats
                        if msg.get("role") == "tool" or msg.get("type") == "tool_call":
                            call_str = msg.get("content", "") or msg.get("function", "") or str(msg)
                            tool_calls.append(call_str)
                        elif "tool_calls" in msg and msg["tool_calls"]:
                            for tc in msg["tool_calls"]:
                                if isinstance(tc, dict):
                                    func = tc.get("function", {})
                                    name = func.get("name", "unknown") if isinstance(func, dict) else str(func)
                                    args = func.get("arguments", "") if isinstance(func, dict) else ""
                                    tool_calls.append(f"{name}({args})")
                                else:
                                    tool_calls.append(str(tc))
                    elif isinstance(msg, str):
                        tool_calls.append(msg)

            # Filter: >= 2 tool calls
            if len(tool_calls) < 2:
                continue

            for step_idx, tc in enumerate(tool_calls):
                records.append({
                    "action": tc,
                    "label": "benign",
                    "session_id": session_id,
                    "attack_type": "benign",
                    "step_index": step_idx,
                    "source_dataset": "agent-traces",
                    "augmented": False,
                })

            # Limit to prevent memory issues
            if len(records) > 50_000:
                log.warning("  Reached 50K records limit for agent-traces, stopping")
                break

    result = pd.DataFrame(records)
    if len(result) > 0:
        sessions = result["session_id"].nunique()
        log.info(f"  Final agent-traces: {len(result):,} records across {sessions} sessions (all benign)")
    else:
        log.warning("  No records extracted from agent-traces")
    return result


def load_synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load synthetic multi-step scenarios from synthetic_scenarios.py
    Returns: (main_scenarios_df, obfuscation_variants_df)
    """
    log.info("Loading synthetic multi-step scenarios...")

    try:
        from synthetic_scenarios import generate_main_scenarios, generate_obfuscation_variants

        main_records = generate_main_scenarios()
        obfuscation_records = generate_obfuscation_variants()

        main_df = pd.DataFrame(main_records)
        obfuscation_df = pd.DataFrame(obfuscation_records)

        mal_main = main_df[main_df["attack_type"] == "malicious_multistep"]["session_id"].nunique()
        ben_main = main_df[main_df["attack_type"] == "benign"]["session_id"].nunique()

        log.info(f"  Main scenarios: {len(main_df):,} records "
                 f"({mal_main} malicious sessions, {ben_main} benign sessions)")
        log.info(f"  Obfuscation variants: {len(obfuscation_df):,} records "
                 f"({obfuscation_df['session_id'].nunique()} sessions)")

        return main_df, obfuscation_df

    except ImportError:
        log.error("  synthetic_scenarios.py not found!")
        log.error("  Run: python synthetic_scenarios.py --test to generate")
        empty = pd.DataFrame(columns=SCHEMA_FIELDS)
        return empty, empty


# ══════════════════════════════════════════════════════════════
# DEDUPLICATION PIPELINE
# ══════════════════════════════════════════════════════════════

def dedup_exact(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: Exact-match dedup via SHA256 on action field."""
    log.info("Dedup Step 1: Exact-match (SHA256)...")
    before = len(df)

    df["_hash"] = df["action"].astype(str).str.lower().str.strip().apply(
        lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
    )
    df = df.drop_duplicates(subset="_hash").drop(columns="_hash").copy()

    log.info(f"  {before:,} → {len(df):,} (dropped {before - len(df):,})")
    return df


def dedup_minhash(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Step 2: Near-duplicate dedup via MinHash LSH."""
    log.info(f"Dedup Step 2: MinHash LSH (threshold={threshold})...")

    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        log.warning("  datasketch not installed, skipping MinHash dedup")
        return df

    before = len(df)
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    keep_indices = []
    seen = set()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  MinHash"):
        text = str(row["action"]).lower().strip()
        tokens = text.split()

        m = MinHash(num_perm=128)
        for token in tokens:
            m.update(token.encode("utf-8"))

        # Check for near-duplicates
        result = lsh.query(m)
        if not result:
            lsh.insert(str(idx), m)
            keep_indices.append(idx)
        # else: near-duplicate found, skip

    df = df.loc[keep_indices].copy()
    log.info(f"  {before:,} → {len(df):,} (dropped {before - len(df):,})")
    return df


def dedup_cross_dataset(df: pd.DataFrame, max_overlap_pct: float = 0.20) -> pd.DataFrame:
    """Step 3: Cross-dataset overlap check."""
    log.info("Dedup Step 3: Cross-dataset overlap check...")

    datasets = df["source_dataset"].unique()
    if len(datasets) < 2:
        log.info("  Only 1 dataset, skipping cross-check")
        return df

    # Check pairwise overlap
    hash_sets = {}
    for ds_name in datasets:
        mask = df["source_dataset"] == ds_name
        hashes = set(
            hashlib.md5(s.encode()).hexdigest()
            for s in df.loc[mask, "action"].astype(str).str.lower().str.strip()
        )
        hash_sets[ds_name] = hashes

    for i, ds1 in enumerate(datasets):
        for ds2 in datasets[i + 1:]:
            overlap = hash_sets[ds1] & hash_sets[ds2]
            pct1 = len(overlap) / max(len(hash_sets[ds1]), 1) * 100
            pct2 = len(overlap) / max(len(hash_sets[ds2]), 1) * 100
            log.info(f"  {ds1} ∩ {ds2}: {len(overlap)} overlapping "
                     f"({pct1:.1f}% of {ds1}, {pct2:.1f}% of {ds2})")

            if max(pct1, pct2) > max_overlap_pct * 100:
                # Remove overlapping samples from the smaller dataset
                smaller = ds1 if len(hash_sets[ds1]) < len(hash_sets[ds2]) else ds2
                overlap_actions = set(
                    s for s in df.loc[df["source_dataset"] == smaller, "action"]
                    .astype(str).str.lower().str.strip()
                    if hashlib.md5(s.encode()).hexdigest() in overlap
                )
                mask = ~(
                    (df["source_dataset"] == smaller) &
                    (df["action"].astype(str).str.lower().str.strip().isin(overlap_actions))
                )
                removed = (~mask).sum()
                df = df[mask].copy()
                log.info(f"  Removed {removed:,} overlapping samples from {smaller}")

    return df


# ══════════════════════════════════════════════════════════════
# BALANCING
# ══════════════════════════════════════════════════════════════

def balance_dataset(df: pd.DataFrame, target_mal_ratio: float = TARGET_RATIO,
                    layer_name: str = "") -> pd.DataFrame:
    """Balance malicious/benign ratio to target (default 60/40)."""
    log.info(f"Balancing {layer_name}...")

    mal_count = (df["label"] == "malicious").sum()
    ben_count = (df["label"] == "benign").sum()
    total = len(df)

    if total == 0:
        log.warning("  Empty dataframe, skipping balancing")
        return df

    if mal_count == 0 or ben_count == 0:
        log.warning(f"  Only one class present ({mal_count:,} mal / {ben_count:,} ben) — "
                    f"cannot balance. Returning as-is.")
        return df

    current_ratio = mal_count / total
    log.info(f"  Current: {mal_count:,} mal ({current_ratio:.1%}) / {ben_count:,} ben")

    if abs(current_ratio - target_mal_ratio) < 0.05:
        log.info(f"  Already within 5% of target ({target_mal_ratio:.0%}), no balancing needed")
        return df

    # Calculate target counts
    if current_ratio > target_mal_ratio:
        # Too many malicious → subsample malicious
        target_mal = int(ben_count * target_mal_ratio / (1 - target_mal_ratio))
        mal_df = df[df["label"] == "malicious"].sample(
            min(target_mal, mal_count), random_state=SEED
        )
        ben_df = df[df["label"] == "benign"]
    else:
        # Too many benign → subsample benign
        target_ben = int(mal_count * (1 - target_mal_ratio) / target_mal_ratio)
        ben_df = df[df["label"] == "benign"].sample(
            min(target_ben, ben_count), random_state=SEED
        )
        mal_df = df[df["label"] == "malicious"]

    result = pd.concat([mal_df, ben_df], ignore_index=True)
    new_ratio = (result["label"] == "malicious").sum() / len(result)
    log.info(f"  Balanced: {(result['label'] == 'malicious').sum():,} mal ({new_ratio:.1%}) / "
             f"{(result['label'] == 'benign').sum():,} ben")
    return result


# ══════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════

def validate_schema(df: pd.DataFrame, layer_name: str = "") -> dict:
    """Validate that all required fields are present and correctly typed."""
    log.info(f"Validating schema for {layer_name}...")
    report = {"layer": layer_name, "errors": [], "warnings": [], "stats": {}}

    # Check required fields
    for field in SCHEMA_FIELDS:
        if field not in df.columns:
            report["errors"].append(f"Missing required field: {field}")

    # Check types
    if "label" in df.columns:
        invalid_labels = df[~df["label"].isin(["malicious", "benign"])]
        if len(invalid_labels) > 0:
            report["errors"].append(
                f"{len(invalid_labels)} rows with invalid label values"
            )

    if "attack_type" in df.columns:
        valid_types = {"malicious_single", "malicious_multistep", "benign"}
        invalid_types = df[~df["attack_type"].isin(valid_types)]
        if len(invalid_types) > 0:
            report["errors"].append(
                f"{len(invalid_types)} rows with invalid attack_type values"
            )

    # Check session_id not null
    if "session_id" in df.columns:
        null_sessions = df["session_id"].isna().sum()
        if null_sessions > 0:
            report["errors"].append(f"{null_sessions} rows with null session_id")

    # Check step_index continuity for multi-step sessions
    if "step_index" in df.columns and "session_id" in df.columns:
        multistep = df[df["attack_type"] == "malicious_multistep"]
        if len(multistep) > 0:
            gap_sessions = []
            for sid, grp in multistep.groupby("session_id"):
                steps = sorted(grp["step_index"].tolist())
                expected = list(range(len(steps)))
                if steps != expected:
                    gap_sessions.append(sid)
            if gap_sessions:
                report["warnings"].append(
                    f"{len(gap_sessions)} sessions with non-continuous step_index"
                )

    # Stats
    report["stats"] = {
        "total_rows": len(df),
        "malicious_count": int((df["label"] == "malicious").sum()) if "label" in df.columns else 0,
        "benign_count": int((df["label"] == "benign").sum()) if "label" in df.columns else 0,
        "unique_sessions": int(df["session_id"].nunique()) if "session_id" in df.columns else 0,
        "source_datasets": dict(df["source_dataset"].value_counts()) if "source_dataset" in df.columns else {},
        "malicious_ratio": float(
            (df["label"] == "malicious").mean()
        ) if "label" in df.columns else 0,
    }

    # Log results
    if report["errors"]:
        for e in report["errors"]:
            log.error(f"  [FAIL] {e}")
    if report["warnings"]:
        for w in report["warnings"]:
            log.warning(f"  [WARN] {w}")
    log.info(f"  [OK] Stats: {json.dumps(report['stats'], indent=2, default=str)}")

    return report


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="EVO-PCA Dataset Pipeline")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token for gated datasets")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--skip-bipia", action="store_true",
                        help="Skip BIPIA dataset loading")
    parser.add_argument("--skip-agentharm", action="store_true",
                        help="Skip AgentHarm dataset loading")
    parser.add_argument("--skip-traces", action="store_true",
                        help="Skip agent-traces dataset loading")
    parser.add_argument("--skip-dedup", action="store_true",
                        help="Skip deduplication pipeline")
    parser.add_argument("--no-balance", action="store_true",
                        help="Skip balancing step")
    args = parser.parse_args()

    global SEED
    SEED = args.seed

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_reports = []

    # ── LAYER 1 ──────────────────────────────────────────────
    log.info("=" * 70)
    log.info("LAYER 1 — NLP Single-Action")
    log.info("=" * 70)

    l1_frames = []

    # hackaprompt
    try:
        df_hack = load_hackaprompt()
        l1_frames.append(df_hack)
    except Exception as e:
        log.error(f"Failed to load hackaprompt: {e}")

    # neuralchemy
    try:
        df_neur = load_neuralchemy()
        l1_frames.append(df_neur)
    except Exception as e:
        log.error(f"Failed to load neuralchemy: {e}")

    # prodnull
    try:
        df_prod = load_prodnull(hf_token=args.hf_token)
        l1_frames.append(df_prod)
    except Exception as e:
        log.error(f"Failed to load prodnull: {e}")

    if l1_frames:
        layer1 = pd.concat(l1_frames, ignore_index=True)
        log.info(f"\nLayer 1 raw total: {len(layer1):,}")

        # Dedup
        if not args.skip_dedup:
            layer1 = dedup_exact(layer1)
            if len(layer1) < 50_000:  # MinHash is slow on large datasets
                layer1 = dedup_minhash(layer1)
            layer1 = dedup_cross_dataset(layer1)

        # Balance
        if not args.no_balance:
            layer1 = balance_dataset(layer1, layer_name="Layer 1")

        # Validate & Save
        report = validate_schema(layer1, "Layer 1")
        validation_reports.append(report)

        # Ensure only schema fields for output (keep metadata separate)
        layer1_out = layer1[[c for c in SCHEMA_FIELDS if c in layer1.columns]].copy()
        layer1_out.to_parquet(output_dir / "evo_pca_layer1.parquet", index=False)
        layer1_out.to_json(output_dir / "evo_pca_layer1.jsonl", orient="records", lines=True,
                           force_ascii=False)
        log.info(f"Saved Layer 1: {len(layer1_out):,} rows")
    else:
        layer1 = pd.DataFrame(columns=SCHEMA_FIELDS)
        log.warning("No Layer 1 data loaded!")

    # ── LAYER 2 ──────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("LAYER 2 — Tool Call Injection")
    log.info("=" * 70)

    l2_frames = []

    # LLMail-Inject
    try:
        df_llmail = load_llmail_inject()
        l2_frames.append(df_llmail)
    except Exception as e:
        log.error(f"Failed to load LLMail-Inject: {e}")

    # BIPIA
    if not args.skip_bipia:
        try:
            df_bipia = load_bipia()
            l2_frames.append(df_bipia)
        except Exception as e:
            log.error(f"Failed to load BIPIA: {e}")

    # Benign pool from prodnull (if available)
    if "df_prod" in dir() and df_prod is not None and len(df_prod) > 0:
        benign_from_prod = df_prod[df_prod["label"] == "benign"].copy()
        benign_from_prod["source_dataset"] = "prodnull-benign-l2"
        if len(benign_from_prod) > 0:
            l2_frames.append(benign_from_prod)
            log.info(f"Added {len(benign_from_prod):,} benign samples from prodnull to L2")

    if l2_frames:
        layer2 = pd.concat(l2_frames, ignore_index=True)
        log.info(f"\nLayer 2 raw total: {len(layer2):,}")

        # Dedup
        if not args.skip_dedup:
            layer2 = dedup_exact(layer2)

        # Balance
        if not args.no_balance:
            layer2 = balance_dataset(layer2, layer_name="Layer 2")

        # Validate & Save
        report = validate_schema(layer2, "Layer 2")
        validation_reports.append(report)

        # Save with metadata for LLMail
        layer2.to_parquet(output_dir / "evo_pca_layer2.parquet", index=False)
        layer2_out = layer2[[c for c in SCHEMA_FIELDS if c in layer2.columns]].copy()
        layer2_out.to_json(output_dir / "evo_pca_layer2.jsonl", orient="records", lines=True,
                           force_ascii=False)
        log.info(f"Saved Layer 2: {len(layer2):,} rows (parquet with metadata)")
    else:
        layer2 = pd.DataFrame(columns=SCHEMA_FIELDS)
        log.warning("No Layer 2 data loaded!")

    # ── LAYER 3 ──────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("LAYER 3 — Multi-Step Sequences")
    log.info("=" * 70)

    l3_frames = []

    # AgentHarm
    if not args.skip_agentharm:
        try:
            df_ah = load_agentharm()
            l3_frames.append(df_ah)
        except Exception as e:
            log.error(f"Failed to load AgentHarm: {e}")

    # agent-traces
    if not args.skip_traces:
        try:
            df_traces = load_agent_traces()
            l3_frames.append(df_traces)
        except Exception as e:
            log.error(f"Failed to load agent-traces: {e}")

    # Synthetic scenarios
    try:
        df_synth_main, df_synth_obfuscation = load_synthetic()
        l3_frames.append(df_synth_main)

        # Save obfuscation separately
        if len(df_synth_obfuscation) > 0:
            df_synth_obfuscation.to_parquet(
                output_dir / "evo_pca_layer3_obfuscation.parquet", index=False
            )
            df_synth_obfuscation.to_json(
                output_dir / "evo_pca_layer3_obfuscation.jsonl",
                orient="records", lines=True, force_ascii=False,
            )
            log.info(f"Saved L3 obfuscation: {len(df_synth_obfuscation):,} rows (separate file)")
    except Exception as e:
        log.error(f"Failed to load synthetic scenarios: {e}")

    if l3_frames:
        layer3 = pd.concat(l3_frames, ignore_index=True)
        log.info(f"\nLayer 3 raw total: {len(layer3):,}")

        # Dedup for L3: deduplicate SESSIONS, not individual actions
        # Individual action dedup would break step_index continuity within sessions
        if not args.skip_dedup:
            before = len(layer3)
            # Create a session fingerprint by concatenating all actions in order
            session_hashes = (
                layer3.sort_values(["session_id", "step_index"])
                .groupby("session_id")["action"]
                .apply(lambda actions: hashlib.sha256(
                    "|".join(actions.astype(str)).lower().strip().encode("utf-8")
                ).hexdigest())
            )
            # Find duplicate sessions
            dup_mask = session_hashes.duplicated(keep="first")
            dup_sessions = set(dup_mask[dup_mask].index)
            if dup_sessions:
                layer3 = layer3[~layer3["session_id"].isin(dup_sessions)].copy()
                log.info(f"  L3 session-level dedup: {before:,} -> {len(layer3):,} "
                         f"(removed {len(dup_sessions)} duplicate sessions)")
            else:
                log.info(f"  L3 session-level dedup: no duplicate sessions found")

        # Note: no balancing for L3 — sessions are complete units
        # Balancing would break session integrity

        # Validate & Save
        report = validate_schema(layer3, "Layer 3")
        validation_reports.append(report)

        layer3.to_parquet(output_dir / "evo_pca_layer3.parquet", index=False)
        layer3_out = layer3[[c for c in SCHEMA_FIELDS if c in layer3.columns]].copy()
        layer3_out.to_json(output_dir / "evo_pca_layer3.jsonl", orient="records", lines=True,
                           force_ascii=False)
        log.info(f"Saved Layer 3: {len(layer3):,} rows")
    else:
        layer3 = pd.DataFrame(columns=SCHEMA_FIELDS)
        log.warning("No Layer 3 data loaded!")

    # ── COMBINED ─────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("COMBINED DATASET")
    log.info("=" * 70)

    # Combine using only schema fields
    combined_frames = []
    for ldf, lname in [(layer1, "L1"), (layer2, "L2"), (layer3, "L3")]:
        if len(ldf) > 0:
            schema_cols = [c for c in SCHEMA_FIELDS if c in ldf.columns]
            combined_frames.append(ldf[schema_cols].copy())

    if combined_frames:
        full = pd.concat(combined_frames, ignore_index=True)
        full.to_parquet(output_dir / "evo_pca_full.parquet", index=False)
        full.to_json(output_dir / "evo_pca_full.jsonl", orient="records", lines=True,
                     force_ascii=False)

        report = validate_schema(full, "Full Combined")
        validation_reports.append(report)

        log.info(f"\n{'='*70}")
        log.info("FINAL SUMMARY")
        log.info(f"{'='*70}")
        log.info(f"Total records: {len(full):,}")
        log.info(f"  Malicious: {(full['label'] == 'malicious').sum():,}")
        log.info(f"  Benign:    {(full['label'] == 'benign').sum():,}")
        log.info(f"  Ratio:     {(full['label'] == 'malicious').mean():.1%} malicious")
        log.info(f"\nBy source_dataset:")
        for src, cnt in full["source_dataset"].value_counts().items():
            log.info(f"  {src}: {cnt:,}")
        log.info(f"\nBy attack_type:")
        for at, cnt in full["attack_type"].value_counts().items():
            log.info(f"  {at}: {cnt:,}")

    # ── SAVE VALIDATION REPORT ───────────────────────────────
    report_path = output_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(validation_reports, f, indent=2, default=str, ensure_ascii=False)
    log.info(f"\nValidation report saved: {report_path}")

    log.info("\n[DONE] Pipeline complete!")


if __name__ == "__main__":
    main()
