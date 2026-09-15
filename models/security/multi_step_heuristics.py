"""Multi-Step Attack Detection Heuristic State-Machine.

Tracks per-session security state transitions to detect multi-stage kill-chains
(e.g., Recon -> Exfiltration, Auth Access -> Privilege Escalation, Evasion -> Execution)
with strict false-positive suppression via TTL and benign context whitelisting.

Pattern groups and TTL values calibrated via Giai Doan 1 audit:
  - tests/audit_killchain_coverage.py (coverage gain + FPR per group)
  - tests/audit_ttl_calibration.py (ABSR vs FPR at each TTL)
"""

import re
import time
import threading
import json
import os
from typing import Dict, List, Optional, Tuple, Set
from types import SimpleNamespace

# Regex indicators for action intent tagging
_AUTH_CRED_PATTERNS = re.compile(
    r'(/etc/shadow|/etc/passwd|\.ssh/id_\w+|\.aws/credentials|\.env\b|secrets?\.json|'
    r'api_key|access_token|password_dump|vault_secret|\.kube/config)',
    re.IGNORECASE
)

_PRIVESC_PATTERNS = re.compile(
    r'\b(sudo\s+|chmod\s+[0-7]*[4-7][0-7]{2}\b|chmod\s+\+s\b|usermod\s+-aG\s+(sudo|wheel|root)|'
    r'/etc/sudoers|setuid|setgid|pkexec|doas)\b',
    re.IGNORECASE
)

_RECON_PATTERNS = re.compile(
    r'\b(find\s+/.*-(name|iname)\s+.*(key|secret|cred|pass|token)|'
    r'grep\s+-[rn].*(password|secret|key|token)\s+/|'
    r'ls\s+-la\s+(/root|/var/secrets|/etc/security|/secrets))\b',
    re.IGNORECASE
)

_EXFIL_COMMAND_PATTERNS = re.compile(
    r'\b(curl|wget|nc|netcat|socat|scp|rsync|ftp|tftp)\b',
    re.IGNORECASE
)

_EXTERNAL_URL_PATTERNS = re.compile(
    r'https?://(?!localhost|127\.0\.0\.1|0\.0\.0\.0|::1|test\.local|internal\.corp)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    re.IGNORECASE
)

_DEFENSE_EVASION_PATTERNS = re.compile(
    r'\b(set\s+\+o\s+history|history\s+-c|export\s+HISTFILE=/dev/null|'
    r'rm\s+(-rf|\s+)*(/var/log/|.*\.bash_history)|'
    r'killall\s+-9\s+(syslogd|auditd|journald)|iptables\s+-F)\b',
    re.IGNORECASE
)

_REMOTE_EXEC_PATTERNS = re.compile(
    r'\b(eval\(|exec\(|bash\s+<\(|sh\s+<\(|python\s+-c\s+["\'].*import\s+(socket|os|subprocess)|'
    r'system\(|popen\()\b',
    re.IGNORECASE
)

# Benign / DevOps tool patterns that should not be tagged as standalone exfiltration
_BENIGN_LOCAL_TOOLS = re.compile(
    r'\b(npm\s+(test|run|install|start)|pytest|cargo\s+(build|test)|git\s+(status|diff|checkout|pull|push)|'
    r'curl\s+.*(localhost|127\.0\.0\.1|0\.0\.0\.0)|python\s+-m\s+unittest|docker\s+ps)\b',
    re.IGNORECASE
)

# ── NEW PATTERN GROUPS (Giai Doan 1 audit: coverage gain + FPR measured) ──
# Each group was tested independently against 83 benign DevOps actions.

# INFRA_CRED_ACCESS: K8s secrets, Vault reads, Terraform state, AWS/GCP IAM
# Audit result: +2.5% coverage, 0% FPR (SAFE TO ADD)
# Key distinction: kubectl get SECRET (credential access) vs kubectl get PODS (routine)
_INFRA_CRED_PATTERNS = re.compile(
    r'\b('
    r'kubectl\s+get\s+secret|kubectl\s+describe\s+secret|'
    r'vault\s+(read|kv\s+(get|put))\b|'
    r'terraform\s+state\s+(pull|show)\b|'
    r'aws\s+(secretsmanager\s+get|sts\s+get-caller|iam\s+create-access-key)|'
    r'gcloud\s+(auth\s+print-access-token|secrets\s+versions\s+access)|'
    r'etcdctl\s+get\s+.*/secrets?|'
    r'consul\s+kv\s+get\s+.*(?:password|secret|credential|token)'
    r')\b',
    re.IGNORECASE
)

# DATA_DUMP: Database/cache bulk extraction commands
# Audit result: +4.2% coverage, 1.2% FPR (ADD w/ CAUTION)
# FP fix: pg_dump alone is benign (backup); only tag when combined with
# external URL or suspicious path (/tmp). Applied via _tag_action context check.
_DATA_DUMP_PATTERNS = re.compile(
    r'\b('
    r'mongoexport\s+--host|mongodump\s+--|'
    r'COPY\s*\(.*\)\s*TO\b|'
    r'redis-cli\s+.*\b(KEYS\s+["\']?\*|MGET\s|DUMP\s)|'
    r'kafka-console-consumer\b.*--from-beginning|'
    r'sqlite3\s+.*\.(db|sqlite)\b|'
    r'SELECT\s+\*\s+FROM\s+\w+|'
    r'information_schema\.tables'
    r')\b',
    re.IGNORECASE
)

# pg_dump is benign for backups but suspicious when combined with exfil indicators
_PG_DUMP_PATTERN = re.compile(r'\bpg_dump\b', re.IGNORECASE)

# NETWORK_RECON_EXPANDED: Active scanning and enumeration
# Audit result: +4.2% coverage, 0% FPR (SAFE TO ADD)
_NETWORK_RECON_PATTERNS = re.compile(
    r'\b('
    r'nmap\s+-[sSpPOAT]|nmap\s+.*--open|'
    r'tcpdump\s+-[iwrn]|'
    r'arpspoof\b|'
    r'ldapsearch\s+-[xHWD]|'
    r'BloodHound|SharpHound|'
    r'masscan\b'
    r')\b',
    re.IGNORECASE
)
# Ping sweep pattern (separate due to regex complexity)
_PING_SWEEP_PATTERN = re.compile(
    r'for\s+.*ping\s+-c\s*1\s+|ping\s+.*\$\{?i\}?',
    re.IGNORECASE
)

# PERSISTENCE: Backdoor installation and persistence mechanisms
# Audit result: +1.7% coverage, 0% FPR (SAFE TO ADD)
_PERSISTENCE_PATTERNS = re.compile(
    r'\b('
    r'crontab\s+-[re]|\(crontab\s+-l|echo\s+.*crontab|'
    r'reg\s+add\s+.*\\\\Run|'
    r'systemctl\s+enable\s+|'
    r'insmod\s+|modprobe\s+.*\.ko|'
    r'gitlab-runner\s+register|'
    r'docker\s+run\s+.*--privileged'
    r')\b',
    re.IGNORECASE
)
# authorized_keys WRITE (append/overwrite) — separate due to >> operator
_AUTH_KEYS_WRITE_PATTERN = re.compile(
    r'>>\s*.*authorized_keys|echo\s+.*ssh-rsa\s+.*authorized_keys',
    re.IGNORECASE
)

# ── AGENTIC TOOL CALL PATTERNS (AgentHarm & Synthetic API Functions) ──
# Audit result on 7,086 benign sessions: FPR = 0.395% (well within budget)

# AGENTIC_DATA_PATTERNS: Tools accessing sensitive personal, medical, financial, or system data
_AGENTIC_DATA_PATTERNS = re.compile(
    r'\b('
    r'access_(customer|payment|medical|user|financial|photo|system)_'
    r'|query_(health|patients|db|records)\b'
    r'|extract_(card_numbers|contacts)\b'
    r'|get_(api_keys|ssh_keys|wifi_passwords|process_list|contacts|browser_history|clipboard|inbox)\b'
    r'|read_database\b'
    r'|read_file\b.*(secret|cred|pass|token|\.env|id_rsa|shadow|config|key|session)'
    r')',
    re.IGNORECASE
)

# AGENTIC_EXFIL_PATTERNS: Tools transmitting data to third-party or adversary endpoints
_AGENTIC_EXFIL_PATTERNS = re.compile(
    r'\b('
    r'send_(email|bulk_sms)\b'
    r'|http_(post|put)\b'
    r'|ftp_upload\b'
    r'|websocket_send\b'
    r'|dropbox_upload\b'
    r'|slack_webhook\b'
    r'|webhook_post\b'
    r'|telegram_send\b'
    r'|irc_send\b'
    r'|pastebin_post\b'
    r'|upload_to_(forum|social)\b'
    r'|bulk_post\b'
    r'|upload_s3\b'
    r'|submit_prescriptions\b'
    r')',
    re.IGNORECASE
)

# AGENTIC_EVASION_PATTERNS: Tools that tamper with logs, timestamps, or audit trails
_AGENTIC_EVASION_PATTERNS = re.compile(
    r'\b('
    r'clear_logs\b'
    r'|cover_tracks\b'
    r'|modify_timestamp\b'
    r'|remove_cron_entry\b'
    r'|rm_file\b'
    r')',
    re.IGNORECASE
)

# Safe readonly commands: do NOT increment benign_counter when session holds
# critical flags (AUTH_CRED_ACCESS, INFRA_CRED_ACCESS, DEFENSE_EVASION, AGENTIC_DATA_ACCESS)
_SAFE_READONLY_CMDS = re.compile(
    r'^\s*(ls|pwd|date|whoami|echo|cat\s+(README|CHANGELOG))',
    re.IGNORECASE
)

# Critical flags that should NOT be cleared by interleaved safe-readonly commands
_CRITICAL_FLAGS = {
    'AUTH_CRED_ACCESS', 'INFRA_CRED_ACCESS', 'DEFENSE_EVASION',
    'DATA_DUMP', 'AGENTIC_DATA_ACCESS'
}


class SessionState:
    """Represents the temporal security context of a single session."""
    __slots__ = ('session_id', 'stage_flags', 'action_history', 'benign_counter', 'last_updated', 'forced_review_remaining')

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.stage_flags: Set[str] = set()
        self.action_history: List[Tuple[float, str, str]] = []  # (timestamp, action_tag, raw_action_snippet)
        self.benign_counter: int = 0
        self.last_updated: float = time.time()
        self.forced_review_remaining: int = 0

    def touch(self):
        self.last_updated = time.time()

    def reset_if_expired(self, ttl_seconds: float = 30.0, max_benign_steps: int = 5) -> bool:
        """Resets suspicious state if inactive for > TTL or after max_benign_steps consecutive benign actions."""
        now = time.time()
        if (now - self.last_updated) > ttl_seconds or self.benign_counter >= max_benign_steps:
            self.stage_flags.clear()
            self.benign_counter = 0
            self.touch()
            return True
        return False

    def has_critical_flags(self) -> bool:
        """Returns True if session holds any critical security flags."""
        return bool(self.stage_flags & _CRITICAL_FLAGS)


class MultiStepResult:
    """Standard result structure returned by HeuristicStateTracker."""
    def __init__(
        self,
        is_blocked: bool = False,
        risk_level: str = "CLEAN",
        reason: str = "",
        rule_fired: str = "",
        confidence: float = 0.0,
        stage_flags: Optional[Set[str]] = None
    ):
        self.is_blocked = is_blocked
        self.risk_level = risk_level  # "CLEAN", "WARNING_LEVEL_1", "BLOCK"
        self.reason = reason
        self.rule_fired = rule_fired
        self.confidence = confidence
        self.stage_flags = stage_flags or set()
        self.decision = SimpleNamespace(value="BLOCK" if is_blocked else "ALLOW")

    def to_dict(self) -> dict:
        return {
            "is_blocked": self.is_blocked,
            "decision": "BLOCK" if self.is_blocked else "ALLOW",
            "risk_level": self.risk_level,
            "reason": self.reason,
            "rule_fired": self.rule_fired,
            "confidence": self.confidence,
            "stage_flags": list(self.stage_flags)
        }


class HeuristicStateTracker:
    """
    Singleton / Thread-safe State Tracker evaluating multi-stage kill-chains.
    Zero inference latency, deterministic transitions, automated TTL cleanup.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, ttl_seconds: float = None, max_benign_steps: int = None):
        if getattr(self, '_initialized', False):
            return
        # Load calibrated values from config if not explicitly provided
        cfg = self._load_config()
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else cfg.get('heuristic_ttl_seconds', 120.0)
        self.max_benign_steps = max_benign_steps if max_benign_steps is not None else cfg.get('max_benign_steps', 40)
        self.sessions: Dict[str, SessionState] = {}
        self.session_lock = threading.Lock()
        self._initialized = True

    @staticmethod
    def _load_config() -> dict:
        """Load multi_step_defense config from thresholds.json."""
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'thresholds.json')
        try:
            with open(config_path, 'r') as f:
                return json.load(f).get('multi_step_defense', {})
        except Exception:
            return {}

    @classmethod
    def reset_instance(cls):
        """Resets the singleton state for testing."""
        with cls._lock:
            if cls._instance is not None:
                with cls._instance.session_lock:
                    cls._instance.sessions.clear()
            cls._instance = None

    def reset_session(self, session_id: str):
        """Explicitly clear tracking state for a session."""
        with self.session_lock:
            if session_id in self.sessions:
                del self.sessions[session_id]

    def inject_early_warning(self, session_id: str, intent_result: dict, max_forced_actions: int = 2):
        """
        Gatekeeping: inject_early_warning acquires session_lock.
        Injects early warning intent flags and arms the forced review countdown.
        """
        with self.session_lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = SessionState(session_id)
            state = self.sessions[session_id]
            state.stage_flags.add("EARLY_INTENT_WARNING")
            state.forced_review_remaining = max_forced_actions
            state.touch()
            risk = intent_result.get('risk_score', 0.0) if isinstance(intent_result, dict) else 0.0
            state.action_history.append(
                (time.time(), "EARLY_INTENT_WARNING", f"risk={risk:.2f}")
            )

    def consume_forced_review(self, session_id: str) -> bool:
        """
        Consume 1 forced-review credit thread-safely.
        Returns True if review is still forced for this session, False otherwise.
        """
        with self.session_lock:
            state = self.sessions.get(session_id)
            if state and state.forced_review_remaining > 0:
                state.forced_review_remaining -= 1
                return True
            return False

    def _tag_action(self, action: str, action_type: str) -> Set[str]:
        """Tags action with security intent labels.
        
        Pattern groups and FPR cost (measured by tests/audit_killchain_coverage.py):
          - AUTH_CRED_ACCESS:        original, 0% FPR
          - PRIVILEGE_ESCALATION:    original, 0% FPR
          - RECON_DISCOVERY:         original, 0% FPR
          - DEFENSE_EVASION:         original, 0% FPR
          - CODE_EXECUTION:          original, 0% FPR
          - DATA_EXFILTRATION:       original, 0% FPR
          - INFRA_CRED_ACCESS:       +2.5% coverage, 0% FPR (SAFE)
          - DATA_DUMP:               +4.2% coverage, 1.2% FPR (CAUTION, pg_dump FP fixed)
          - NETWORK_RECON:           +4.2% coverage, 0% FPR (SAFE)
          - PERSISTENCE:             +1.7% coverage, 0% FPR (SAFE)
        """
        tags = set()
        
        # Check Benign Local Tools first to prevent false tagging of curl localhost etc.
        is_benign_local = bool(_BENIGN_LOCAL_TOOLS.search(action))

        # ── Original patterns ──
        if _AUTH_CRED_PATTERNS.search(action):
            tags.add("AUTH_CRED_ACCESS")

        if _PRIVESC_PATTERNS.search(action):
            tags.add("PRIVILEGE_ESCALATION")

        if _RECON_PATTERNS.search(action):
            tags.add("RECON_DISCOVERY")

        if _DEFENSE_EVASION_PATTERNS.search(action):
            tags.add("DEFENSE_EVASION")

        if _REMOTE_EXEC_PATTERNS.search(action):
            tags.add("CODE_EXECUTION")

        # Exfiltration requires command indicator + external destination URL (not localhost)
        if _EXFIL_COMMAND_PATTERNS.search(action) and not is_benign_local:
            if _EXTERNAL_URL_PATTERNS.search(action) or "POST" in action.upper():
                tags.add("DATA_EXFILTRATION")

        # ── New pattern groups (Giai Doan 1 audit verified) ──

        # INFRA_CRED_ACCESS: K8s secrets, Vault, Terraform state, cloud IAM
        if _INFRA_CRED_PATTERNS.search(action):
            tags.add("INFRA_CRED_ACCESS")

        # DATA_DUMP: Database/cache bulk extraction
        if _DATA_DUMP_PATTERNS.search(action):
            tags.add("DATA_DUMP")
        elif _PG_DUMP_PATTERN.search(action):
            # pg_dump alone is benign (backup). Only tag when combined with
            # external URL or suspicious temp path — avoids the measured FP.
            has_external = bool(_EXTERNAL_URL_PATTERNS.search(action))
            has_suspicious_path = bool(re.search(r'/tmp/', action))
            if has_external or has_suspicious_path:
                tags.add("DATA_DUMP")

        # NETWORK_RECON: Active scanning and enumeration
        if _NETWORK_RECON_PATTERNS.search(action) or _PING_SWEEP_PATTERN.search(action):
            tags.add("NETWORK_RECON")

        # PERSISTENCE: Backdoor, cron persistence, authorized_keys write
        if _PERSISTENCE_PATTERNS.search(action) or _AUTH_KEYS_WRITE_PATTERN.search(action):
            tags.add("PERSISTENCE")

        # ── AGENTIC TOOL CALL PATTERNS (AgentHarm & Synthetic API functions) ──
        if _AGENTIC_DATA_PATTERNS.search(action):
            tags.add("AGENTIC_DATA_ACCESS")
            tags.add("RECON_DISCOVERY")

        if _AGENTIC_EXFIL_PATTERNS.search(action):
            tags.add("AGENTIC_EXFILTRATION")
            tags.add("DATA_EXFILTRATION")

        if _AGENTIC_EVASION_PATTERNS.search(action):
            tags.add("DEFENSE_EVASION")

        return tags

    def evaluate(self, action: str, session_id: str, action_type: str = "prompt") -> MultiStepResult:
        """
        Evaluates an action against the session state and returns detection verdict.

        Thread-safety: the entire method body runs under session_lock to prevent
        race conditions when multiple threads process actions for the same session_id
        concurrently. The critical section is pure CPU (dict/set state transitions, no I/O),
        so lock contention impact is negligible (<10% p95 latency impact).

        CRITICAL SECTION GATEKEEPING RULES:
        1. NO I/O: Never perform file reads/writes, network calls, sockets, or sleep inside lock.
        2. NO HEAVY INFERENCE: Never invoke LLMs, ONNX/PyTorch models, embeddings, or ML scorers.
        3. STATELESS PRE-PROCESSING: All stateless parsing/tagging (_tag_action) MUST execute OUTSIDE lock.
        4. CPU BUDGET: Operations inside the lock must complete in <= 100 microseconds.
        Violations are automatically caught by tests/test_lock_scope_regression.py.
        """
        if not session_id:
            session_id = "default_session"

        # _tag_action is stateless (pure regex matching) — safe to call outside lock
        action_tags = self._tag_action(action, action_type)
        snippet = (action[:80] + '...') if len(action) > 80 else action

        with self.session_lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = SessionState(session_id)
            state = self.sessions[session_id]

            state.reset_if_expired(self.ttl_seconds, self.max_benign_steps)

            # If no security tags fired, handle benign counter with critical-flag protection
            if not action_tags:
                # Benign-reset protection: safe readonly commands (ls, pwd, date, whoami)
                # do NOT increment benign_counter when session holds critical flags.
                # This prevents TTL-reset from clearing attack state due to interleaved
                # innocuous commands. Measured: FPR impact = 0% (audit_ttl_calibration.py).
                is_safe_readonly = bool(_SAFE_READONLY_CMDS.match(action))
                if is_safe_readonly and state.has_critical_flags():
                    # Don't count towards benign_counter, preserve attack state
                    state.touch()
                    state.action_history.append((time.time(), "BENIGN_PROTECTED", snippet))
                    return MultiStepResult(
                        is_blocked=False,
                        risk_level="CLEAN",
                        reason="Benign readonly step (critical flags preserved)",
                        stage_flags=set(state.stage_flags)
                    )

                state.benign_counter += 1
                if state.benign_counter >= self.max_benign_steps:
                    state.stage_flags.clear()
                state.touch()
                state.action_history.append((time.time(), "BENIGN", snippet))
                return MultiStepResult(
                    is_blocked=False,
                    risk_level="CLEAN",
                    reason="Benign step executed",
                    stage_flags=set(state.stage_flags)
                )

            # Action fired security tags: record and check kill-chain state transitions
            state.benign_counter = 0  # Reset benign counter on suspicious action
            state.touch()

            # -------------------------------------------------------------
            # KILL-CHAIN TRANSITION RULES
            # Original 3 chains + 3 new chains (Giai Doan 2)
            # -------------------------------------------------------------
            
            # Kill-chain 1: Auth/Credential Access followed by Privilege Escalation
            if "PRIVILEGE_ESCALATION" in action_tags:
                if "AUTH_CRED_ACCESS" in state.stage_flags or "INFRA_CRED_ACCESS" in state.stage_flags:
                    state.stage_flags.add("KILLCHAIN_AUTH_PRIVESC")
                    state.action_history.append((time.time(), "KILLCHAIN_AUTH_PRIVESC", snippet))
                    return MultiStepResult(
                        is_blocked=True,
                        risk_level="BLOCK",
                        reason="[KILLCHAIN_AUTH_PRIVESC] Privilege escalation attempted following credential access.",
                        rule_fired="MULTI_STEP_KILLCHAIN_AUTH_PRIVESC",
                        confidence=0.98,
                        stage_flags=set(state.stage_flags)
                    )

            # Kill-chain 2: Recon / Discovery followed by Data Exfiltration
            if "DATA_EXFILTRATION" in action_tags:
                prior_recon = state.stage_flags & {
                    "RECON_DISCOVERY", "AUTH_CRED_ACCESS", "INFRA_CRED_ACCESS",
                    "DATA_DUMP", "NETWORK_RECON"
                }
                if prior_recon:
                    state.stage_flags.add("KILLCHAIN_RECON_EXFIL")
                    state.action_history.append((time.time(), "KILLCHAIN_RECON_EXFIL", snippet))
                    return MultiStepResult(
                        is_blocked=True,
                        risk_level="BLOCK",
                        reason=f"[KILLCHAIN_RECON_EXFIL] Outbound data exfiltration after {','.join(prior_recon)}.",
                        rule_fired="MULTI_STEP_KILLCHAIN_RECON_EXFIL",
                        confidence=0.99,
                        stage_flags=set(state.stage_flags)
                    )

            # Kill-chain 3: Defense Evasion followed by Code Execution
            if "CODE_EXECUTION" in action_tags:
                if "DEFENSE_EVASION" in state.stage_flags:
                    state.stage_flags.add("KILLCHAIN_EVASION_EXEC")
                    state.action_history.append((time.time(), "KILLCHAIN_EVASION_EXEC", snippet))
                    return MultiStepResult(
                        is_blocked=True,
                        risk_level="BLOCK",
                        reason="[KILLCHAIN_EVASION_EXEC] Dynamic code execution attempted after defense evasion / log tampering.",
                        rule_fired="MULTI_STEP_KILLCHAIN_EVASION_EXEC",
                        confidence=0.97,
                        stage_flags=set(state.stage_flags)
                    )

            # ── NEW Kill-chain 4: Infra/DB Credential + Exfiltration ──
            if "DATA_EXFILTRATION" in action_tags:
                if "INFRA_CRED_ACCESS" in state.stage_flags or "DATA_DUMP" in state.stage_flags:
                    state.stage_flags.add("KILLCHAIN_INFRA_EXFIL")
                    state.action_history.append((time.time(), "KILLCHAIN_INFRA_EXFIL", snippet))
                    return MultiStepResult(
                        is_blocked=True,
                        risk_level="BLOCK",
                        reason="[KILLCHAIN_INFRA_EXFIL] Data exfiltration following infrastructure credential access or database dump.",
                        rule_fired="MULTI_STEP_KILLCHAIN_INFRA_EXFIL",
                        confidence=0.97,
                        stage_flags=set(state.stage_flags)
                    )

            # ── NEW Kill-chain 5: Network Recon + Persistence ──
            if "PERSISTENCE" in action_tags:
                if "NETWORK_RECON" in state.stage_flags or "RECON_DISCOVERY" in state.stage_flags:
                    state.stage_flags.add("KILLCHAIN_RECON_PERSISTENCE")
                    state.action_history.append((time.time(), "KILLCHAIN_RECON_PERSISTENCE", snippet))
                    return MultiStepResult(
                        is_blocked=True,
                        risk_level="BLOCK",
                        reason="[KILLCHAIN_RECON_PERSISTENCE] Persistence/backdoor installation following network reconnaissance.",
                        rule_fired="MULTI_STEP_KILLCHAIN_RECON_PERSISTENCE",
                        confidence=0.96,
                        stage_flags=set(state.stage_flags)
                    )

            # ── NEW Kill-chain 6: Data Dump + Exfiltration ──
            if "DATA_EXFILTRATION" in action_tags:
                if "DATA_DUMP" in state.stage_flags:
                    state.stage_flags.add("KILLCHAIN_DUMP_EXFIL")
                    state.action_history.append((time.time(), "KILLCHAIN_DUMP_EXFIL", snippet))
                    return MultiStepResult(
                        is_blocked=True,
                        risk_level="BLOCK",
                        reason="[KILLCHAIN_DUMP_EXFIL] Database/cache dump followed by outbound data exfiltration.",
                        rule_fired="MULTI_STEP_KILLCHAIN_DUMP_EXFIL",
                        confidence=0.98,
                        stage_flags=set(state.stage_flags)
                    )

            # Update active state flags with current action tags
            state.stage_flags.update(action_tags)
            state.action_history.append((time.time(), ",".join(action_tags), snippet))

            # Not yet a complete kill-chain, but mark as Level-1 Warning for LLM Session Judge
            return MultiStepResult(
                is_blocked=False,
                risk_level="WARNING_LEVEL_1",
                reason=f"Suspicious state stage flagged: {','.join(action_tags)}",
                rule_fired="MULTI_STEP_STAGE_FLAGGED",
                confidence=0.60,
                stage_flags=set(state.stage_flags)
            )
