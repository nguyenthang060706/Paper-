import time
import re
import math
import threading
import hashlib
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import base64
import logging
import json
import unicodedata
import codecs
from pathlib import Path
logger = logging.getLogger(__name__)
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from models.security.shared_utils import ScoreEvo

CONFIG_PATH = Path(__file__).parent.parent.parent / 'config' / 'thresholds.json'

def get_heuristics_config() -> dict:
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f).get('voting_aggregator', {})
    except Exception as e:
        logger.warning(f"Failed to load thresholds config: {e}")
        return {}

@dataclass
class AdaptiveEscalationManager:
    """
    Label-free adaptive threshold manager with Hysteresis Deadband for production inference.
    """
    target_escalation_rate: float = 0.10
    initial_threshold: float = 0.70
    min_threshold: float = 0.58
    max_threshold: float = 0.95
    loosen_step: float = 0.03
    tighten_step: float = 0.06
    window_size: int = 100
    warmup_decisions: int = 30

    _threshold: float = field(init=False)
    _decisions: list = field(default_factory=list)
    _total_decisions: int = field(default=0, init=False)
    _consecutive_violation: int = field(default=0, init=False)

    def __post_init__(self):
        self.min_threshold = float(self.min_threshold)
        self.max_threshold = float(self.max_threshold)
        self._threshold = self.initial_threshold
        from collections import deque
        self._decisions = deque(maxlen=self.window_size)
        self._consecutive_violation = 0

    @property
    def adaptive_threshold(self) -> float:
        return self._threshold

    def record_decision(self, is_escalated: bool) -> None:
        self._decisions.append(is_escalated)
        self._total_decisions += 1
        if self._total_decisions >= self.warmup_decisions and self._total_decisions % 10 == 0:
            self._adjust_threshold()

    def _adjust_threshold(self) -> None:
        window = list(self._decisions)
        if not window:
            return
        escalation_rate = sum(window) / len(window)
        ratio = escalation_rate / max(self.target_escalation_rate, 1e-9)
        
        # Hysteresis Deadband (>= 2 consecutive check cycles)
        if ratio > 1.2:
            if self._consecutive_violation > 0:
                self._consecutive_violation += 1
            else:
                self._consecutive_violation = 1
                
            if self._consecutive_violation >= 2:
                self._threshold = min(self._threshold + self.tighten_step, self.max_threshold)
                self._consecutive_violation = 0
        elif ratio < 0.5:
            if self._consecutive_violation < 0:
                self._consecutive_violation -= 1
            else:
                self._consecutive_violation = -1
                
            if abs(self._consecutive_violation) >= 2:
                self._threshold = max(self._threshold - self.loosen_step, self.min_threshold)
                self._consecutive_violation = 0
        else:
            self._consecutive_violation = 0

    def get_metrics(self) -> Dict[str, float]:
        window = list(self._decisions)
        esc_rate = sum(window) / len(window) if window else 0.0
        return {
            'adaptive_threshold': round(self._threshold, 4),
            'escalation_rate': round(esc_rate, 4),
            'target_escalation_rate': self.target_escalation_rate,
            'total_decisions': self._total_decisions,
            'window_size': len(window),
            'consecutive_violation': float(self._consecutive_violation)
        }

    def __repr__(self) -> str:
        m = self.get_metrics()
        return (f"AdaptiveEscalationManager(threshold={m['adaptive_threshold']:.3f}, "
                f"esc_rate={m['escalation_rate']:.3f}/{self.target_escalation_rate}, "
                f"decisions={m['total_decisions']}")

class ActionTier(str, Enum):
    ALLOW = 'ALLOW'
    MONITOR = 'MONITOR'
    REVIEW = 'REVIEW'
    DENY = 'DENY'
    QUARANTINE = 'QUARANTINE'

class RiskSignal(BaseModel):
    """Individual risk indicator with validation."""
    name: str = Field(..., description='Signal identifier')
    severity: int = Field(..., ge=0, le=100, description='0-100 severity scale')
    confidence: float = Field(..., ge=0.0, le=1.0, description='0-1 confidence score')
    is_critical: bool = Field(False, description='Overrides other signals if True')
    source: str = Field('unknown', description='Detection source')
    evidence: List[str] = Field(default_factory=list, description='Supporting evidence')

    def __str__(self) -> str:
        return f'RiskSignal({self.name}, severity={self.severity}, confidence={self.confidence:.1%}, critical={self.is_critical})'

    def weighted_score(self) -> float:
        """Calculate weighted risk score."""
        return self.severity * self.confidence

class Canonicalizer:
    """Normalize and decode obfuscated inputs."""
    ZERO_WIDTH_RE = re.compile('[\\u00ad\\u200b\\u200c\\u200d\\ufeff\\u2060\\u180e\\u2028\\u2029\\u202a-\\u202e]')
    MULTI_SPACE_RE = re.compile('\\s+')
    BASE64_RE = re.compile('(?<![A-Za-z0-9+/])(?:[A-Za-z0-9+/]{20,}={0,2})(?![A-Za-z0-9+/])')
    HEX_ESCAPE_RE = re.compile('\\\\x([0-9a-fA-F]{2})')
    UNICODE_ESCAPE_RE = re.compile('\\\\u([0-9a-fA-F]{4})')
    LEET_TRANS = str.maketrans({'0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '6': 'g', '7': 't', '8': 'b', '9': 'g', '@': 'a', '$': 's', '!': 'i', '+': 't', '|': 'i'})
    HOMOGLYPHS = str.maketrans({'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'х': 'x', 'α': 'a', 'β': 'b'})

    @classmethod
    def canonicalize(cls, action: str) -> str:
        """
        Execute full canonicalization pipeline.
        
        Steps:
        1. Remove zero-width characters
        2. Decode base64
        3. Normalize whitespace
        4. Resolve escape sequences
        5. Normalize leet speak
        6. Normalize homoglyphs

        [SYNC v44 / FIX 3A] Base64 decode now runs BEFORE whitespace
        normalization. Previously MULTI_SPACE_RE collapsed internal
        whitespace inside base64-looking tokens before decoding,
        corrupting the encoded payload and letting obfuscated attacks
        through canonicalization undetected. Matches v43 / IEEE order.
        """
        if not isinstance(action, str):
            raise ValueError(f'Expected str, got {type(action).__name__}')
        if not action.strip():
            return ''
        try:
            action = unicodedata.normalize('NFKC', action)
            
            _rot13_re = re.compile(r'\b(?:vafgehpgvbaf|vtaber|sbetrg|qvfertneq|olcnff|birreevqr)\b', re.IGNORECASE)
            if _rot13_re.search(action):
                try:
                    action = codecs.decode(action, 'rot_13')
                except Exception:
                    pass

            action = cls._remove_zero_width(action)
            action = cls._decode_base64_inline(action)
            action = cls._normalize_whitespace(action)
            action = cls._resolve_escape_sequences(action)
            action = cls._normalize_leet(action)
            action = cls._normalize_homoglyphs(action)
            return action
        except Exception as e:
            logger.warning(f'Canonicalization error: {e}')
            return action

    @classmethod
    def _remove_zero_width(cls, text: str) -> str:
        return cls.ZERO_WIDTH_RE.sub('', text)

    @classmethod
    def _normalize_whitespace(cls, text: str) -> str:
        return cls.MULTI_SPACE_RE.sub(' ', text).strip()

    @classmethod
    def _decode_base64_inline(cls, text: str) -> str:

        def decode_match(m):
            try:
                decoded = base64.b64decode(m.group()).decode('utf-8', errors='strict')
                return decoded if decoded.isprintable() else m.group()
            except Exception:
                return m.group()
        return cls.BASE64_RE.sub(decode_match, text)

    @classmethod
    def _resolve_escape_sequences(cls, text: str) -> str:
        text = cls.HEX_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), text)
        text = cls.UNICODE_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), text)
        return text
    _SHELL_CMD_PREFIXES = re.compile('^\\s*(?:cat|ls|rm|cp|mv|chmod|chown|find|grep|awk|sed|python[23]?|pip[23]?|bash|sh|zsh|fish|curl|wget|nc|netcat|nmap|tcpdump|ss|ps|top|kill|sudo|su|exec|export|echo|printf|base64|xxd|od|hexdump|dd|truncate|shred|tar|gzip|zip|unzip|openssl|ssh|scp|sftp|rsync|git|apt|yum|systemctl|service|cron|env|set|export|source|eval)\\b', re.IGNORECASE)

    @classmethod
    def _normalize_leet(cls, text: str) -> str:
        if cls._SHELL_CMD_PREFIXES.match(text):
            return text
        return text.translate(cls.LEET_TRANS)

    @classmethod
    def _normalize_homoglyphs(cls, text: str) -> str:
        return text.translate(cls.HOMOGLYPHS)

class PermissionGate:
    """Map actions to required capabilities and flag missing permissions.
    
    v3.6b fixes:
    - shell.exec: now catches `exec <binary>` and bare `exec` command prefix  
    - fs.delete: added rm, del without flags
    - network.external: added nc/netcat/openssl as exfil channels
    - HIGH_RISK_PAIRS: deny combos that are dangerous regardless of permission list
    """
    CAPABILITY_PATTERNS = {
        'secrets.read': [
            r'\b(?:api[_ -]?key|secret|token|credential|password|private key|browser cookies?)\b',
            r'(?<!\w)(?:~?/)?(?:\.npmrc|\.pypirc|\.netrc|\.aws/credentials)(?!\w)',
            r'(?:^|\s|/)(?:\.env|config\.json|settings\.py)\b',
            r'(?:^|\s|/|\b)(?:\.ssh/authorized_keys|\.ssh/id_rsa|/etc/passwd|/etc/shadow)\b',
            r'\b(?:Login\s?Data|keychain|vault\s+login)\b'
        ],
        'network.external': [
            r'\b(?:curl|wget|scp|sftp|ftp|webhook)\b',
            r'\bhttps?://\b|\b(?:external|remote|public|pastebin|gist|s3|bucket|discord|slack)\b',
            r'\b(?:nc|netcat|ncat|openssl\s+s_client|socat)\b'
        ],
        'shell.exec': [
            r'\b(?:powershell|pwsh|cmd\s*/c|bash\s+-c|sh\s+-c|python\s+-c|subprocess|os\.system|eval|exec)\b',
            r'\b(?:subprocess\.(?:run|Popen|call|check_output)|os\.(?:system|popen|spawn\w*)|pty\.spawn|importlib\.import_module)\b|(?:getattr|__import__)\s*\(',
            r'^\s*exec\s+(?!\()',
            r'\b(?:execvp?|execle?|posix_spawn|CreateProcess|ShellExecute)\b'
        ],
        'shell.recon': [
            r'\bls\s+-l[a-z]*\s+~',
            r'\bls\s+~',
            r'\b(?:net\s+group|net\s+user|net\s+localgroup)\b',
            r'\b(?:cat|head|tail)\s+/var/log/',
            r'\b(?:syslog|auth\.log|cron\s+jobs?|crontab\s+-l)\b',
            r'\b(?:docker\s+ps|docker\s+run|docker\s+exec|terraform\s+workspace|terraform\s+state)\b',
            r'\b(?:SharpHound|nmap|mongosh|ip_forward|bloodhound|http\.server|SimpleHTTPServer)\b'
        ],
        'database.access': [
            r'\b(?:psql|mysql|sqlite3)\b\s+.*(?:-U|-d|--user|--database)\b',
            r'\b(?:information_schema\.tables|table_schema)\b'
        ],
        'fs.delete': [
            r'\b(?:rm\s+-[rRf]+|rm\s+--recursive|del\s+/[fqs]|delete|remove|shred|format\s+c:|truncate\s+-s\s+0)\b',
            r'\brm\s+(?:-[rRf]*\s+)?/'
        ],
        'policy.override': [
            r'\b(?:ignore previous|forget your rules|bypass|override|disable safety|developer mode)\b'
        ],
        'audit.cover_tracks': [
            r'(?:\bhistory\s+-c\b|\bunset\s+HISTFILE\b|\bexport\s+HISTFILE=/dev/null\b|\bClear-History\b|\bshred\b|\btruncate\s+-s\s+0\b|\brm\s+.*\.bash_history\b|\bwevtutil\s+cl\b|\bauditpol\b)'
        ]
    }
    HIGH_RISK_COMBOS = [({'secrets.read', 'network.external'}, 95, 'Data exfiltration combo: reading secrets + sending externally'), ({'shell.exec', 'network.external'}, 90, 'RCE + network: arbitrary code execution with exfil capability'), ({'shell.exec', 'fs.delete'}, 85, 'Destructive RCE: execute code and delete files'), ({'secrets.read', 'audit.cover_tracks'}, 88, 'Secrets access followed by evidence removal'), ({'network.external', 'audit.cover_tracks'}, 90, 'External communication followed by evidence removal')]
    # Hướng A: Mở rộng quyền mặc định để bao gồm các thao tác phổ biến của AgentDojo
    # Giữ lại 'secrets.read', 'fs.delete', 'policy.override', 'audit.cover_tracks' là các quyền nhạy cảm cần xin phép.
    DEFAULT_PERMISSIONS = {'fs.read', 'fs.write', 'network.external', 'shell.exec'}

    def __init__(self, default_permissions=None):
        self.default_permissions = set(default_permissions or self.DEFAULT_PERMISSIONS)
        self.capability_patterns = {cap: [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns] for cap, patterns in self.CAPABILITY_PATTERNS.items()}

    def required_permissions(self, canonicalized_action: str):
        return {cap for cap, patterns in self.capability_patterns.items() if any((p.search(canonicalized_action) for p in patterns))}

    @staticmethod
    def check_taint_overlap(action_text: str, session) -> Optional[str]:
        if session is None:
            return None
        with session.lock:
            for t in session.tainted_values:
                if t.value in action_text:
                    return t.value
        return None
        
    @staticmethod
    def has_explicit_exfil_target(action_text: str, user_goal_text: str) -> Optional[str]:
        # Quick scan for explicit URLs or emails in the action text
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')

        lower_action = action_text.lower()
        collaboration_tools = (
            'invite_user_to_slack', 'create_calendar_event',
            'send_direct_message', 'send_message'
        )
        sensitive_or_hijack = re.search(
            r'\b(?:password|secret|token|credential|api[_ -]?key|private[_ -]?key|'
            r'\.env|bank|account|profit|forecast|confidential|attachment|file_id|'
            r'ignore previous|system prompt|override|bypass)\b',
            lower_action,
            re.IGNORECASE,
        )
        if any(tool in lower_action for tool in collaboration_tools) and not sensitive_or_hijack:
            return None
        
        for pattern in (email_pattern, url_pattern):
            for m in pattern.finditer(action_text):
                token = m.group(0)
                if not user_goal_text or token not in user_goal_text:
                    return token
        return None

    def detect(self, canonicalized_action: str, granted_permissions=None, session=None, enable_provenance=False, skip_rce=False, tool_name=None) -> List[RiskSignal]:
        required = self.required_permissions(canonicalized_action)
        if skip_rce:
            required.discard('shell.exec')
            required.discard('fs.delete')
            
        granted = set(granted_permissions or self.default_permissions)
        missing = sorted(required - granted)
        signals = []
        
        if tool_name:
            try:
                from models.security.function_risk_registry import check_function_signature
                sig_signal = check_function_signature(tool_name)
                if sig_signal:
                    signals.append(sig_signal)
            except Exception as e:
                logger.error(f"Error checking function signature for {tool_name}: {e}", exc_info=True)
            
        critical_combo = {'secrets.read', 'network.external', 'policy.override'}
        if missing:
            if critical_combo.issubset(required):
                sev, conf = (92, 0.93)
            elif 'secrets.read' in missing:
                sev, conf = (90, 0.95)
            elif 'database.access' in missing:
                sev, conf = (90, 0.95)
            elif 'shell.recon' in missing:
                sev, conf = (90, 0.95)
            elif len(missing) >= 2:
                sev, conf = (60, 0.85)
            else:
                sev, conf = (40, 0.8)
            is_crit = critical_combo.issubset(required) or sev >= 90
            signals.append(RiskSignal(name='missing_required_permission', severity=sev, confidence=conf, is_critical=is_crit, source='permission_gate', evidence=[f'required_permissions={sorted(required)}', f'granted_permissions={sorted(granted)}', f'missing_permissions={missing}']))
        for combo, severity, reason in self.HIGH_RISK_COMBOS:
            if combo.issubset(required):
                is_crit = (severity >= 90)
                
                if enable_provenance and session is not None:
                    overlap_value = self.check_taint_overlap(canonicalized_action, session)
                    if overlap_value:
                        is_crit = True
                        reason = f"{reason} — matched tainted value: {overlap_value}"
                    else:
                        exfil_target = self.has_explicit_exfil_target(canonicalized_action, getattr(session, 'user_goal_text', ''))
                        if exfil_target:
                            is_crit = True
                            reason = f"{reason} — explicit suspicious target: {exfil_target}"
                        else:
                            continue
                            
                signals.append(RiskSignal(name='high_risk_capability_combo', severity=severity, confidence=0.95, is_critical=is_crit, source='permission_gate', evidence=[f'Dangerous combination detected: {sorted(combo)}', reason, f'action={repr(canonicalized_action[:80])}']))
        return signals


@dataclass
class ActionChainRule:
    name: str
    preceding_pattern: str
    following_pattern: str
    time_window_seconds: int
    risk_score: float
    description: str


class SharedSemanticEncoder:
    """Thread-safe Singleton wrapper for all-MiniLM-L6-v2 to enable zero-cost dense embeddings reuse across modules."""
    __instance = None
    __lock = threading.Lock()
    _encoder = None
    _load_error = False

    @classmethod
    def get(cls):
        if cls._encoder is None and not cls._load_error:
            with cls.__lock:
                if cls._encoder is None and not cls._load_error:
                    try:
                        from sentence_transformers import SentenceTransformer
                        cls._encoder = SentenceTransformer("all-MiniLM-L6-v2")
                        logger.info("[SharedSemanticEncoder] Successfully loaded all-MiniLM-L6-v2.")
                    except Exception as e:
                        logger.error(f"[SharedSemanticEncoder] Failed to load SentenceTransformer: {e}")
                        cls._load_error = True
        return cls._encoder

    @classmethod
    def encode(cls, text_or_list):
        enc = cls.get()
        if enc is None:
            return None
        return enc.encode(text_or_list)


class SemanticCamouflageDetector:
    __instance = None
    __instance_lock = threading.Lock()
    
    def __new__(cls, **kwargs):
        if cls.__instance is None:
            with cls.__instance_lock:
                if cls.__instance is None:
                    cls.__instance = super().__new__(cls)
                    cls.__instance._initialize()
        return cls.__instance
        
    def _initialize(self):
        self.inference_lock = threading.Lock()
        self.dangerous_patterns = [
            "rm -rf /",
            "dd if=/dev/zero of=",
            "fork() while true",
            "exec /bin/bash",
            ":(){$|&};:",
            "issue api_call to send an email with arguments",
            "transfer funds to account",
            "ignore previous instructions and execute",
            "print system prompt"
        ]
        patterns_str = json.dumps(self.dangerous_patterns, sort_keys=True)
        self.current_hash = hashlib.sha256(patterns_str.encode('utf-8')).hexdigest()
        
        self.threshold = 1.0
        self.is_valid = False
        self.encoder = None
        self.pattern_embeddings = None
        
        try:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'semantic_config.json'
            with open(config_path, 'r') as f:
                config = json.load(f)
                semantic_config = config.get('semantic_camouflage', {})
                saved_hash = semantic_config.get('anchor_hash')
                if saved_hash != self.current_hash:
                    logger.error("[SemanticCamouflageDetector] Hash mismatch! You added new anchors but forgot to recalibrate. Fail-Open to TF-IDF.")
                else:
                    self.threshold = semantic_config.get('threshold', 1.0)
                    self.is_valid = True
        except Exception as e:
            logger.error(f"[SemanticCamouflageDetector] Failed to load config: {e}. Fail-Open to TF-IDF.")
        
        self.encoder = SharedSemanticEncoder.get()
        if self.encoder is None:
            logger.error("[SemanticCamouflageDetector] SharedSemanticEncoder unavailable. Fail-open.")
            self.is_valid = False
        elif self.is_valid:
            try:
                self.pattern_embeddings = self.encoder.encode(self.dangerous_patterns)
            except Exception as e:
                logger.error(f"[SemanticCamouflageDetector] Failed pattern embedding: {e}. Fail-open.")
                self.is_valid = False
        
    def is_camouflaged(self, action: str) -> Tuple[bool, float]:
        if not self.is_valid or get_heuristics_config().get('disable_semantic', False):
            return False, 0.0
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            with self.inference_lock:
                action_embedding = self.encoder.encode([action])
            similarities = cosine_similarity(action_embedding, self.pattern_embeddings)[0]
            max_sim = float(similarities.max())
            
            return max_sim >= self.threshold, max_sim
        except Exception as e:
            logger.warning(f"[max_sim] Error during semantic check: {e}")
            return False, 0.0

class VotingAggregator:
    """Aggregate multiple risk signals into final decision.

    FIX v3.6:
    - weighted_score: MAX ? weighted mean (dòi h?i d?ng thu?n nhi?u detector)
    - Ngưỡng DENY: 75 ? 82, REVIEW: 60 ? 68
    - Critical signals v?n ? QUARANTINE ngay
    """

    @staticmethod
    def weighted_score(signals: List[RiskSignal]) -> float:
        """Compute aggregated risk score.

        Non-critical path: true weighted mean = S(sev×conf) / S(conf),
        matching the formula in the paper (Section III.F).
        Critical path: max of critical weighted scores + 10% bonus from
        non-critical signals (preserved from v3.6b for hard-gate cases).

        [SYNC FIX — BUG-B] Non-critical formula was max-anchored
        (max + 0.15*rest), not a weighted mean. Replaced with
        S(sev*conf)/S(conf) to match paper and v43. Now both files
        produce identical scores for identical inputs.
        """
        if not signals:
            return 0.0
        if any((s.is_critical for s in signals)):
            critical_scores = [s.weighted_score() for s in signals if s.is_critical]
            other_scores = [s.weighted_score() for s in signals if not s.is_critical]
            base = max(critical_scores)
            bonus = sum(other_scores) * 0.1
            return min(100.0, base + bonus)
        total_conf = sum((s.confidence for s in signals))
        if total_conf == 0.0:
            return 0.0
        return min(100.0, sum((s.severity * s.confidence for s in signals)) / total_conf)

    @staticmethod
    def max_severity(signals: List[RiskSignal]) -> float:
        return max((s.severity for s in signals), default=0.0)
    @classmethod
    def get_threshold(cls, name: str, default: int) -> int:
        config = get_heuristics_config()
        return config.get(name, default)

    @classmethod
    def DENY_THRESHOLD(cls) -> int:
        return cls.get_threshold("DENY", 82)
        
    @classmethod
    def QUARANTINE_THRESHOLD(cls) -> int:
        return cls.get_threshold("QUARANTINE", 90)
        
    @classmethod
    def REVIEW_THRESHOLD(cls) -> int:
        return cls.get_threshold("REVIEW", 68)
        
    @classmethod
    def MONITOR_THRESHOLD(cls) -> int:
        return cls.get_threshold("MONITOR", 40)

    @classmethod
    def vote(cls, signals: List[RiskSignal]) -> Tuple[ActionTier, float]:
        """
        Aggregate signals into decision tier.

        FIX v3.6c (BUG-1 FIX — VotingAggregator inconsistency):
        Thêm score-based QUARANTINE path (>= 90) d? d?ng b? với v3.8.
        Critical hard-gate gi? nguyên: is_critical ? QUARANTINE ngay.
        Thresholds:
        1. is_critical=True                  ? QUARANTINE (hard gate)
        2. weighted_score >= 90              ? QUARANTINE (score path, m?i)
        3. weighted_score >= 82 AND max > 85 ? DENY
        4. weighted_score >= 68 OR  max > 75 ? REVIEW
        5. weighted_score >= 40              ? MONITOR
        6. Otherwise                         ? ALLOW
        """
        if not signals:
            return (ActionTier.ALLOW, 0.0)
        if any((s.is_critical for s in signals)):
            return (ActionTier.QUARANTINE, 1.0)
        
        weighted = ScoreEvo(cls.weighted_score(signals))
        max_sev = ScoreEvo(cls.max_severity(signals))
        q_thresh = ScoreEvo(cls.QUARANTINE_THRESHOLD())
        d_thresh = ScoreEvo(cls.DENY_THRESHOLD())
        r_thresh = ScoreEvo(cls.REVIEW_THRESHOLD())
        m_thresh = ScoreEvo(cls.MONITOR_THRESHOLD())
        
        if weighted >= q_thresh:
            return (ActionTier.QUARANTINE, min(float(weighted) / 100, 1.0))
        if weighted >= d_thresh and max_sev > ScoreEvo(85):
            return (ActionTier.DENY, min(float(weighted) / 100, 1.0))
        if weighted >= r_thresh or max_sev > ScoreEvo(75):
            return (ActionTier.REVIEW, min(max(float(weighted) / 75, 0.5), 1.0))
        if weighted >= m_thresh:
            return (ActionTier.MONITOR, min(float(weighted) / float(r_thresh), 0.99))
        return (ActionTier.ALLOW, max(0.0, 1.0 - float(weighted) / 100))

    @classmethod
    def explain(cls, tier: ActionTier, signals: List[RiskSignal]) -> str:
        lines = [f"{'=' * 60}", f'Decision: {tier.value}', f"{'=' * 60}", f'Signals ({len(signals)}):']
        for signal in sorted(signals, key=lambda s: s.severity, reverse=True):
            crit = '[CRITICAL]' if signal.is_critical else ''
            lines.append(f'  • {signal.name:20s} severity={signal.severity:3d} confidence={signal.confidence:5.1%} {crit}')
        weighted = cls.weighted_score(signals)
        max_sev = cls.max_severity(signals)
        lines.extend(['', f'Weighted Score: {weighted:.1f}/100', f'Max Severity: {max_sev:.0f}/100'])
        return '\n'.join(lines)
