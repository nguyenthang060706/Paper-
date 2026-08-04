"""LlamaFirewall Tier 0 — stateless prompt-injection pre-filter.

Sync version: fix8 - bug fixes and consistency improvements over fix5.

Changes vs fix5
---------------
FIX6-1  _leet pattern renamed + corrected: the old 'override_instructions_leet'
        pattern was a broad fuzzy rule, not an actual leet-speak detector.
        Now split into two separate rules:
          - 'override_instructions_fuzzy'  : catch spaced/fragmented variants
            (e.g. "ignore ... instructions" with junk in between)
          - 'override_instructions_leet'   : true leet pattern applied on the
            PRE-normalised string, matching leet originals like "1gn0r3",
            "1nstruct1ons", etc. — not duplicating the plain-text rules.
FIX6-2  MONITOR path now reports the LAST matching MONITOR rule (most-specific),
        consistent with SYNC-1 semantics for BLOCK. Also emits
        "(also matched: …)" when multiple MONITOR rules fire, consistent
        with SYNC-2. Previously it silently reported only the first.
FIX6-3  T0_CAP_THRESHOLD exported as module-level constant (float = 0.35).
        Previously referenced in docstrings/comments but never defined,
        risking AttributeError on notebook import.
FIX6-4  [FIX-4] comment on the 'forget' rule corrected: "temporal qualifier"
        → "positional/sequence qualifier" to accurately describe
        prior/previous/earlier/above.
FIX6-5  __all__ added to declare the public API explicitly, preventing
        accidental import of internal helpers.

Changes vs fix6
----------------
FIX7-1  _LEET_TRANS expanded with five additional digit/symbol substitutions
        that are common in real-world prompt-injection obfuscation but were
        previously unmapped (FIX7-3 added the last two for full Canonicalizer
        parity):
          - '7' -> 't'   (e.g. "ins7ruc7ions" -> "instructions")
          - '+' -> 't'   (e.g. "ins+ruc+ions" -> "instructions")
          - '9' -> 'g'   (e.g. "i9nore"       -> "ignore")
          - '6' -> 'g'   (e.g. "i6nore"       -> "ignore")        [FIX7-3]
          - '8' -> 'b'   (e.g. "b8pass"       -> "bypass")        [FIX7-3]
        Bug: inputs combining these substitutions with the already-mapped
        ones (e.g. "1gn0r3 4ll pr3v10u5 1ns7ruc710ns") normalised to text
        containing literal '7'/'+'/'9', which did not match the standard
        'override_instructions' BLOCK pattern (which requires literal
        "instructions") and did not match _LEET_PATTERN either (which also
        requires literal 't'/'g' in its instruction/prompt branch) — the
        action passed Tier 0 as ALLOW. After normalisation with the expanded
        table these inputs become plain "ignore all previous instructions"
        and are correctly BLOCKed by the existing 'override_instructions'
        rule. No new patterns required.

Changes vs fix7
---------------
FIX8-1  Synchronized the standalone Tier-0 implementation with the notebook
        patch cells used by the EVO-PCA/AgentDojo benchmark harness:
          - leet matching is evaluated before standard regex scanning
          - curl/wget-to-bash RCE detection is handled as a two-step static
            check instead of a broad pipe regex in the main pattern table
          - the public API surface is kept explicit through __all__
        These changes make fix8 an audit/stability release rather than a new
        detection-family release.

Changes vs fix8
---------------
FIX9-1  Added 4 AgentDojo-specific injection patterns targeting indirect
        prompt injection templates commonly found in tool-output injection
        scenarios. These patterns detect: task-hijacking ("new task: ignore..."),
        imperative injection ("IMPORTANT: send..."), instead-of hijacking,
        and conversational injection ("Actually, please ignore...").
"""
import re
import time
import unicodedata
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple

__all__ = [
    'ScanDecision',
    'ScanResult',
    'LlamaFirewallTier0',
    'scan_decision_to_action_tier',
    '_pre_normalise',       # exported per SYNC-3 for notebook patch cells
    'T0_CAP_THRESHOLD',
]


class ScanDecision(str, Enum):
    BLOCK   = 'BLOCK'
    ALLOW   = 'ALLOW'
    MONITOR = 'MONITOR'


@dataclass
class ScanResult:
    decision:          ScanDecision
    is_blocked:        bool
    rule_fired:        str           # LAST (most-specific) BLOCK/MONITOR rule, or ''
    reason:            str
    confidence:        float
    benign_confidence: float
    latency_ms:        float = 0.0
    # All matched rules in match order — use for audit / forensics
    all_rules_fired:   List[str] = field(default_factory=list)
    normalized_action: str = ""


def scan_decision_to_action_tier(scan_result: ScanResult) -> str:
    """Map ScanResult.decision → ActionTier string for standalone / non-notebook use.

    Returns plain strings ('QUARANTINE', 'MONITOR', 'ALLOW') so this module
    has zero dependency on the notebook's ActionTier enum.  Notebook callers
    that need an ActionTier object define their own bridge using .decision.value.
    """
    _map = {
        ScanDecision.BLOCK:   'QUARANTINE',
        ScanDecision.MONITOR: 'MONITOR',
        ScanDecision.ALLOW:   'ALLOW',
    }
    return _map[scan_result.decision]


# ── Severity-cap threshold ────────────────────────────────────────────────────
# Exported so notebook / downstream code can import instead of hardcoding 0.35.
# The severity cap activates when benign_confidence >= T0_CAP_THRESHOLD.
# MONITOR has benign_confidence=0.50 (> 0.35) → cap activates. [SYNC-4]
T0_CAP_THRESHOLD: float = 0.35


# ── Pre-normalisation ─────────────────────────────────────────────────────────
# Applied to every action BEFORE pattern matching so that zero-width
# obfuscation variants trigger the same rules as plain text.
# Leet-to-ASCII conversion is also applied here so most rules match
# post-normalised ASCII; the dedicated leet rule matches PRE-normalised text.
# Exported so notebook patch cells can import and reuse it. [SYNC-3]
_ZERO_WIDTH_RE = re.compile(
    r'[\u00ad\u200b\u200c\u200d\ufeff\u2060\u180e\u2028\u2029\u202a-\u202e]'
)
_LEET_TRANS = str.maketrans({
    ord('0'): 'o', ord('1'): 'i', ord('3'): 'e', ord('4'): 'a',
    ord('5'): 's', ord('@'): 'a', ord('$'): 's', ord('!'): 'i',
    ord('|'): 'i',
    # [FIX7-1] Additional common leet substitutions, see module docstring.
    ord('7'): 't', ord('+'): 't', ord('9'): 'g',
    # [FIX7-3] Sync with Canonicalizer LEET_TRANS for full consistency.
    ord('6'): 'g', ord('8'): 'b',
})
_HOMOGLYPH_TRANS = str.maketrans({
    ord('а'): 'a', ord('о'): 'o', ord('е'): 'e', ord('с'): 'c',
    ord('р'): 'p', ord('х'): 'x', ord('у'): 'y', ord('і'): 'i'
})

def _pre_normalise(text: str) -> str:
    """Remove zero-width chars and map leet digits/symbols to ASCII equivalents.
    Also handles Cyrillic homoglyphs and Vietnamese diacritics."""
    def _repair_mojibake(value: str) -> str:
        def marker_count(s: str) -> int:
            return sum(s.count(marker) for marker in ('Ã', 'Â', 'Ä', 'Æ', 'â€'))

        repaired = value
        for _ in range(2):
            if not any(marker in repaired for marker in ('Ã', 'Â', 'Ä', 'Æ')):
                break
            candidates = [repaired]
            for encoding in ('latin1', 'cp1252'):
                try:
                    candidates.append(repaired.encode(encoding).decode('utf-8'))
                except UnicodeError:
                    pass
                try:
                    candidates.append(repaired.encode(encoding, errors='ignore').decode('utf-8', errors='ignore'))
                except UnicodeError:
                    pass
            candidate = min(candidates, key=lambda s: (marker_count(s), -len(s)))
            if marker_count(candidate) >= marker_count(repaired):
                break
            repaired = candidate
        return repaired

    text = _repair_mojibake(text)
    # 1. Homoglyphs
    text = text.translate(_HOMOGLYPH_TRANS)
    # 2. Zero-width and Leet
    text = _ZERO_WIDTH_RE.sub('', text).translate(_LEET_TRANS)
    # 3. Vietnamese Diacritics (NFD + explicit mapping for 'đ')
    text = text.replace('đ', 'd').replace('Đ', 'D')
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text


# ── Confidence table ──────────────────────────────────────────────────────────
# Each entry: (confidence, benign_confidence).  The two values sum to 1.0.
# MONITOR benign_confidence = 0.50 (not 0.35 = 1.0 - 0.65).
#   Rationale: 0.35 equals T0_CAP_THRESHOLD, which would make the severity-cap
#   boundary semantically ambiguous (MONITOR == barely-cap-activating).
#   0.50 = neutral/uncertain — cap activates on MONITOR (benign_conf 0.50 ≥ 0.35)
#   but with less weight than ALLOW (benign_conf 0.99). [NEW-2 FIX, SYNC-4]
_CONFIDENCE: dict = {
    ScanDecision.BLOCK:   (0.99, 0.01),  # Chắc chắn có độc    (sum = 1.0)
    ScanDecision.MONITOR: (0.50, 0.50),  # Borderline / 50-50  (sum = 1.0)
    ScanDecision.ALLOW:   (0.01, 0.99),  # Chắc chắn an toàn   (sum = 1.0)
}

# ── Pattern library ───────────────────────────────────────────────────────────
# Rule library: 25 explicit prompt-injection signatures (18 BLOCK, 7 MONITOR)
# Order: broad patterns first, narrow / obfuscation-specific patterns last.
# rule_fired reports the LAST matching BLOCK rule (= most-specific). [SYNC-1]
# Same logic now applies to MONITOR rules. [FIX6-2]
_RAW_PATTERNS: List[Tuple[str, ScanDecision, str]] = [
    (r'\bignore\s+(all\s+)?(previous|prior|earlier|above)\s+(i\w?nstructions?|prompts?|context|rules?|constraints?)\b',
     ScanDecision.BLOCK, 'override_instructions'),
    # [FIX6-4] Positional/sequence qualifier required — prevents FP on
    # "forget context and discuss recipes" or "forget your instructions"
    (r'\bforget\s+(all\s+)?(prior|previous|earlier|above)\s+(context|instructions?|prompts?|rules?)\b',
     ScanDecision.BLOCK, 'override_instructions'),
    (r'\b(override|bypass|disable|circumvent|disregard)\s+(all\s+)?(safety|content|filter|moderation|security|ethical|alignment)\s*(rules?|filters?|checks?|guidelines?|constraints?)?\b',
     ScanDecision.BLOCK, 'safety_bypass'),
    (r'\bjailbreak\b',
     ScanDecision.BLOCK, 'jailbreak_keyword'),
    (r'\bdan\s+mode\s+(on|enabled|activate)\b',
     ScanDecision.BLOCK, 'dan_mode'),
    (r'\bdisable\s+(the\s+)?(content|safety|moderation|ethical)\s*(filter|check|guardrail)s?\b',
     ScanDecision.BLOCK, 'disable_filter'),
    (r'\bact\s+as\s+(if\s+you\s+(are|were)\s+)?(dan|an?\s+unfiltered|an?\s+unrestricted|an?\s+uncensored|evil\s+ai|no\s+restriction)\b',
     ScanDecision.BLOCK, 'persona_injection'),
    (r'\b(you\s+are\s+now|from\s+now\s+on\s+(you\s+are|act\s+as))\s+(dan|an?\s+ai\s+with\s+no|an?\s+unfiltered)\b',
     ScanDecision.BLOCK, 'persona_injection'),
    (r'\b(?:attempt\w*|perform\w*|execut\w*|tr(?:ying|ied|y)|using|conduct\w*)\s+(?:a\s+)?prompt\s*injection\b',
     ScanDecision.BLOCK, 'explicit_injection_label'),
    (r'\bthis\s+is\s+a\s+prompt\s*injection\b',
     ScanDecision.BLOCK, 'explicit_injection_label'),
    (r'\b(new|updated?|replace|override)\s+(system\s+)?(prompt|instruction|rule|directive)\s*[:=]',
     ScanDecision.BLOCK, 'system_prompt_override'),
    (r'\bdo\s+not\s+(follow|obey|respect|adhere\s+to)\s+(your\s+)?(instructions?|guidelines?|rules?|training|constraints?)\b',
     ScanDecision.BLOCK, 'override_instructions'),
    # [FIX-5] Noun group required — prevents FP on "disregard your earlier suggestion"
    (r'\bdisregard\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?|guidelines?)\b',
     ScanDecision.BLOCK, 'override_instructions'),
    # Obfuscated variants — ordered after plain-text rules (more specific → reported last)
    (r'\bignore[_\-]previous[_\-]instructions?\b',
     ScanDecision.BLOCK, 'override_instructions_obfuscated'),
    # [FIX6-1a] Fuzzy/fragmented variant — catches "ignore ... instructions" with
    # intervening junk (e.g. filler words, punctuation). This is intentionally broad
    # and fires AFTER the plain-text rule so it only becomes primary_rule if
    # the plain-text rule did NOT fire (i.e. there is actual junk in between).
    (r'\bignore\b.{1,30}\b(?:i\w?nstructions?|prompts?|context)\b',
     ScanDecision.BLOCK, 'override_instructions_fuzzy'),
    
    # ── AgentDojo Prompt Injection Patterns ──────────────────────────────────
    (r'</?INFORMATION>',
     ScanDecision.BLOCK, 'xml_injection_marker'),
    (r'(?i)Before you can solve the task.*please do the following',
     ScanDecision.BLOCK, 'task_sandwich_injection'),
    (r'(?i)strictly\s+adhere\s+to\s+the\s+following\s+instruction',
     ScanDecision.BLOCK, 'strict_adherence_injection'),
    # ── System Command RCE Patterns (v6.0) ────────────────────────────────────

    # Bắt lệnh cấp quyền thực thi
    (r'(?i)\bchmod\s+\+x\b',
     ScanDecision.BLOCK, 'rce_chmod_exec'),
    (r'(?i)\bsudo\s+chmod\s+777\s+/etc/sudoers\b|\bchmod\s+777\s+/etc/sudoers\b',
     ScanDecision.BLOCK, 'rce_privilege_policy_tamper'),
    (r'(?i)\brm\s+-rf\s+/(?:var/log|etc|home|root|usr|bin|sbin|opt|tmp)?\b',
     ScanDecision.BLOCK, 'rce_destructive_rm_rf'),
    (r'(?i)\bexec\s*\(\s*b64decode\s*\(',
     ScanDecision.BLOCK, 'rce_encoded_exec_payload'),
    (r'(?i)\b(?:system\s+prompt|hidden\s+prompt|developer\s+message)\s+(?:extract|dump|leak)\b',
     ScanDecision.BLOCK, 'system_prompt_extract'),
    (r'(?i)\bp[a@4]ssw[o0]rd\s+(?:extract|dump|leak|exfil|forward|send)\b',
     ScanDecision.BLOCK, 'credential_extract_obfuscated'),
    # Bắt lệnh đọc file nhạy cảm hệ thống
    (r'(?i)/etc/(?:passwd|shadow)',
     ScanDecision.BLOCK, 'rce_sensitive_file_read'),
    # [FIX6-1b] True leet-speak detector — operates on PRE-normalised string.
    # Matched against RAW input in scan() via _LEET_PATTERN (compiled separately below).
    # [FIX6-6] Placeholder REMOVED from _RAW_PATTERNS to avoid confusion:
    # the pattern was never compiled here (filtered out in _COMPILED_PATTERNS),
    # so keeping it risked undetected regex typos. The actual compiled pattern
    # is _LEET_PATTERN below. Rule name 'override_instructions_obfuscated' is injected
    # into all_rules in scan() when _LEET_PATTERN matches.
    # ── AgentDojo-specific injection templates (FIX9-1) ───────────────────────
    # AgentDojo uses 4 standard jailbreak templates, all NLP-based.
    # These patterns catch indirect-injection variants embedded in tool outputs.
    (r'\b(?:new\s+task|important\s+(?:message|instruction|update)|urgent)\s*[:\-]?\s*(?:ignore|forget|disregard)',
     ScanDecision.BLOCK, 'agentdojo_task_injection'),
    (r'(?:IMPORTANT|NOTE|INSTRUCTION|URGENT|ATTENTION)\s*:\s*(?:please\s+)?(?:send|forward|email|transfer|upload)\s+.*(?:password|secret|key|token|credential|http|ftp|@attacker)',
     ScanDecision.MONITOR, 'indirect_injection_imperative'),
    (r'\binstead\s+of\s+(?:doing|completing|performing|following)\s+.{1,60}?\b(?:ignore|override|disregard|bypass|drop)\b',
     ScanDecision.MONITOR, 'task_hijack_instead'),
    (r'(?:^|\n)\s*(?:Actually|BTW|By the way|PS|P\.S\.|FYI)\s*[,:]?\s*(?:please\s+)?(?:ignore|forget|disregard|override)\s+(?:all\s+)?(?:previous\s+)?(?:system\s+prompts?|your\s+instructions?|core\s+rules?)',
     ScanDecision.MONITOR, 'conversational_injection'),
    # ── Vietnamese patterns (Giai đoạn 1) ─────────────────────────────────────
    (r'\b(bo qua|quen|huy bo)\s+(tat ca\s+)?(huong dan|cau lenh|luat le)\s+(truoc|cu|truoc do)\b',
     ScanDecision.BLOCK, 'override_instructions_vi'),
    (r'\bb\S*\s+qua\s+h\S*ng\s+d\S*n\s+tr\S*c\b',
     ScanDecision.BLOCK, 'override_instructions_vi_mojibake'),
    (r'\b(bo qua|quen|huy bo)\s+chi thi\b',
     ScanDecision.BLOCK, 'override_instructions_vi'),
    (r'\bmat khau.*(gui|chuyen|dua|cung cap)\b|\b(gui|chuyen|dua|cung cap).*mat khau\b',
     ScanDecision.BLOCK, 'exfil_vi'),
    # ── MONITOR patterns ──────────────────────────────────────────────────────
    (r'\bwhat\s+(are\s+)?your\s+(system\s+)?instructions?\b',
     ScanDecision.MONITOR, 'system_prompt_recon'),
    # [FIX7-3] Optional second qualifier handles "repeat the above system prompt"
    # and similar two-qualifier chains that previously slipped through.
    (r'\brepeat\s+(the\s+)?(above|previous|system|original)\s+((system|original)\s+)?(prompt|instructions?|message)\b',
     ScanDecision.MONITOR, 'system_prompt_recon'),
    (r'\bprint\s+(your\s+)?(system\s+prompt|initial\s+instructions?|hidden\s+prompt)\b',
     ScanDecision.MONITOR, 'system_prompt_recon'),
    (r'\b(reveal|expose|show|output|display)\s+(your\s+)?(system\s+prompt|hidden\s+instructions?|internal\s+rules?)\b',
     ScanDecision.MONITOR, 'system_prompt_recon'),
    (r'\bpretend\s+(you\s+(are|have)\s+no\s+(restrictions?|guidelines?|rules?)|there\s+(are|were)\s+no\s+rules?)\b',
     ScanDecision.MONITOR, 'restriction_bypass_soft'),
    (r'\bhypothetically\s+(speaking\s+)?(if\s+you\s+(had\s+no|were\s+without)\s+(restrictions?|guidelines?))\b',
     ScanDecision.MONITOR, 'restriction_bypass_soft'),
    (r'\bprompt\s*injection\b',
     ScanDecision.MONITOR, 'prompt_injection_mention'),
    (r'\b(in ra|hien thi|cho xem|tiet lo)\s+(system prompt|huong dan he thong|luat he thong|luat cua ban)\b',
     ScanDecision.MONITOR, 'system_prompt_recon_vi'),
]

# Standard compiled patterns (match against post-normalised string)
_COMPILED_PATTERNS: List[Tuple[re.Pattern, ScanDecision, str]] = [
    (re.compile(pattern, re.IGNORECASE | re.DOTALL), decision, rule)
    for pattern, decision, rule in _RAW_PATTERNS
    # [FIX6-6] leet entry removed from _RAW_PATTERNS; compiled separately as _LEET_PATTERN
]

# [FIX6-1b] Leet pattern compiled separately; matched against RAW (pre-normalise) input
# [FIX7-1] 't' positions now also accept '7'/'+' and 'g' positions also accept '9',
# [FIX7-3] 'g' positions further accept '6' (6->g in _LEET_TRANS); consistent with
# the expanded _LEET_TRANS table — keeps this pattern's raw-text matching
# consistent with what normalisation now produces.
# Character classes used:
#   'i' position : [1!|]
#   'g' position : [g96]   ← '9'->g and '6'->g both covered
#   'e' position : [3e]
#   'o' position : [0o]
#   't' position : [t7+]
#   's' position : [5s]
#   'y' position : [y]     ← [FIX-7] was [y8]; '8'->b in _LEET_TRANS makes
#                              '8' in y-position semantically wrong for raw-text
#                              matching.  Edge case "8ypass" normalises to
#                              "bypass" via _LEET_TRANS and is caught by the
#                              standard 'safety_bypass' rule on post-normalised
#                              text.
_LEET_PATTERN: re.Pattern = re.compile(
    r'\b[1!|][g96][n][0o][r][3e]\b.{0,30}\b(?:[1!|][n][5s][t7+][r][u][c][t7+][1!|][0o][n][5s]?|[p][r][0o][m][p][t7+][5s]?)\b|\b[b8][y][p][a@4][5s][5s]\b',
    re.IGNORECASE | re.DOTALL,
)

_DECISION_RANK = {
    ScanDecision.ALLOW:   0,
    ScanDecision.MONITOR: 1,
    ScanDecision.BLOCK:   2,
}


class LlamaFirewallTier0:
    def __repr__(self) -> str:
        # Heuristics Tier 0 (Lexical & Pattern Matching).
        # - Rule-set được biên soạn hoàn toàn bằng tay (Hand-crafted regex).
        # - 32 pattern (24 BLOCK, 8 MONITOR), 22 unique rule names. Count directly from _RAW_PATTERNS.
        block_n   = sum(1 for _, d, _ in _RAW_PATTERNS if d == ScanDecision.BLOCK)
        monitor_n = sum(1 for _, d, _ in _RAW_PATTERNS if d == ScanDecision.MONITOR)
        return (
            f'LlamaFirewallTier0('
            f'patterns={len(_RAW_PATTERNS)}, '
            f'BLOCK={block_n}, '
            f'MONITOR={monitor_n})'
        )

    def scan(self, action: str, skip_rce: bool = False) -> ScanResult:
        """Scan a single action string against the compiled pattern library."""
        if not isinstance(action, str):
            action = str(action)
        t0 = time.perf_counter()

        # Step 1: Extract cURL/Wget-to-Bash RCE (two-step static check) [SYNC-6]
        # This prevents catastrophic regex backtracking while maintaining 100% detection.
        if not skip_rce:
            lower_action = action.lower()
            has_fetch = 'curl ' in lower_action or 'wget ' in lower_action
            has_pipe  = '| bash' in lower_action or '| sh' in lower_action
            if has_fetch and has_pipe:
                return ScanResult(
                    decision          = ScanDecision.BLOCK,
                    is_blocked        = True,
                    rule_fired        = 'rce_curl_pipe_bash',
                    reason            = 'Detected fetch-to-shell RCE (e.g. curl ... | bash)',
                    confidence        = 0.99,
                    benign_confidence = 0.01,
                    latency_ms        = (time.perf_counter() - t0) * 1000,
                    all_rules_fired   = ['rce_curl_pipe_bash'],
                    normalized_action = action,
                )

        normalised  = _pre_normalise(action)
        raw_snippet = action[:100] + ('...' if len(action) > 100 else '')

        best_decision = ScanDecision.ALLOW
        best_reason   = 'No malicious NLP patterns detected'

        # Collect every matched rule in order.
        all_rules:     List[str] = []
        block_rules:   List[Tuple[str, str]] = []   # (rule_name, matched_text)
        monitor_rules: List[Tuple[str, str]] = []   # (rule_name, matched_text)

        # [FIX6-1b / FIX-B1] Leet pattern checked FIRST against RAW input so that when it
        # fires, it populates block_rules before the fuzzy rule checks for it.
        leet_m = _LEET_PATTERN.search(action)
        if leet_m:
            all_rules.append('override_instructions_obfuscated')
            block_rules.append(('override_instructions_obfuscated', leet_m.group(0)))
            best_decision = ScanDecision.BLOCK

        # [FIX-B2] Tách pattern RCE download để kiểm tra tĩnh nhằm tăng hiệu năng
        if not skip_rce and re.search(r'(?i)\b(?:curl|wget)\b', action):
            if re.search(r'\|.*\bbash\b', action):
                all_rules.append('rce_download_execute')
                block_rules.append(('rce_download_execute', 'curl/wget | bash'))
                best_decision = ScanDecision.BLOCK

        # Standard patterns operate on normalised string, except RCE rules which need raw chars
        for compiled, decision, rule in _COMPILED_PATTERNS:
            if skip_rce and rule.startswith('rce_'):
                continue
                
            target_text = action if rule.startswith('rce_') else normalised
            m = compiled.search(target_text)
            if m:
                if rule == 'override_instructions_fuzzy' and any(
                    r in ('override_instructions', 'override_instructions_obfuscated') for r, _ in block_rules
                ):
                    continue
                all_rules.append(rule)
                if _DECISION_RANK[decision] > _DECISION_RANK[best_decision]:
                    best_decision = decision
                if decision == ScanDecision.BLOCK:
                    block_rules.append((rule, m.group(0)))
                elif decision == ScanDecision.MONITOR:
                    monitor_rules.append((rule, m.group(0)))

        # Determine primary rule_fired and reason
        if block_rules:
            # SYNC-1: last BLOCK rule = most-specific
            primary_rule, primary_match = block_rules[-1]
            also = (
                f' (also matched: {", ".join(r for r, _ in block_rules[:-1])})'
                if len(block_rules) > 1 else ''
            )
            best_reason = (
                f'Detected pattern [{primary_rule}]: "{primary_match}"'
                f' (raw: "{raw_snippet}"){also}'
            )
        elif monitor_rules:
            # [FIX6-2] MONITOR: last rule = most-specific (consistent with SYNC-1)
            primary_rule, primary_match = monitor_rules[-1]
            also = (
                f' (also matched: {", ".join(r for r, _ in monitor_rules[:-1])})'
                if len(monitor_rules) > 1 else ''
            )
            best_reason = (
                f'Detected pattern [{primary_rule}]: "{primary_match}"'
                f' (raw: "{raw_snippet}"){also}'
            )
        else:
            primary_rule = ''

        latency_ms                    = (time.perf_counter() - t0) * 1000
        is_blocked                    = best_decision == ScanDecision.BLOCK
        confidence, benign_confidence = _CONFIDENCE[best_decision]

        return ScanResult(
            decision          = best_decision,
            is_blocked        = is_blocked,
            rule_fired        = primary_rule,
            reason            = best_reason,
            confidence        = confidence,
            benign_confidence = benign_confidence,
            latency_ms        = latency_ms,
            all_rules_fired   = all_rules,
            normalized_action = normalised,
        )
