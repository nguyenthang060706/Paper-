"""LlamaFirewall Tier 0.5 — Session-aware lightweight injection detector.

Sits between Tier 0 (stateless pattern matching) and Tier 1-5 (EVO-PCA pipeline).
Adds cross-step correlation detection without requiring ML or LLM.

Key capabilities:
- Detects split-injection attacks (malicious intent distributed across steps)
- Tracks session-level behavioral flags for cross-step escalation
- Detects intent-shift patterns (benign setup -> malicious execution)
- Zero external dependencies (no ML model, no LLM calls)

Note on Thread-Safety: 
    This module uses collections.OrderedDict for session tracking which is NOT thread-safe.
    If running in a highly concurrent environment (e.g., parallel benchmarking), 
    external locking (like threading.Lock) should be applied when calling scan().

Usage:
    from llamafirewall_tier05_session import SessionAwareTier05
    tier05 = SessionAwareTier05()
    result = tier05.scan(action, session_id="user_123")
"""
import re
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

try:
    from core.tier0 import (
        LlamaFirewallTier0, ScanDecision, ScanResult, _pre_normalise
    )
except ImportError:
    # BUG 2 FIXED: Cung cấp Fallback Class thực sự để tránh NameError khi import thiếu
    class ScanDecision(str, Enum):
        ALLOW = 'ALLOW'
        BLOCK = 'BLOCK'
        MONITOR = 'MONITOR'
        
    @dataclass
    class ScanResult:
        decision: ScanDecision
        is_blocked: bool
        rule_fired: str
        reason: str
        confidence: float
        benign_confidence: float
        latency_ms: float
        all_rules_fired: List[str] = field(default_factory=list)
        normalized_action: str = ""
        
    LlamaFirewallTier0 = type("DummyTier0", (), {"scan": lambda self, x: None})
    def _pre_normalise(x): return x


__all__ = ['SessionAwareTier05', 'SessionFlags', 'Tier05Config']


class Tier05Config:
    FLAG_EXPIRY_SECONDS = 3600  # 1 giờ (cho Production)
    FLAG_EXPIRY_STEPS = 20      # 20 bước (cho Benchmark AgentDojo)
    ESCALATION_THRESHOLD = 1    # Nâng cấp lên BLOCK ngay lần kích hoạt đầu tiên (Hướng A)


class SessionFlags(str, Enum):
    """Behavioral flags tracked per session."""
    PERSISTENT_INSTRUCTION = 'persistent_instruction'  # "remember", "always", "from now on"
    SENSITIVE_DATA_MENTION = 'sensitive_data_mention'   # passwords, keys, credentials
    EXFIL_VERB            = 'exfil_verb'               # send, forward, upload, transfer
    ROLE_OVERRIDE         = 'role_override'             # "you are now", "act as"
    SYSTEM_PROMPT_PROBE   = 'system_prompt_probe'       # "what are your instructions"
    DESTRUCTIVE_INTENT    = 'destructive_intent'        # delete, remove, drop, format
    RECON_ACTIVITY        = 'recon_activity'            # ls, cat, find, whoami
    COVER_TRACKS          = 'cover_tracks'              # history -c, rm logs
    SENSITIVE_DATA_EXT    = 'sensitive_data_ext'        # passport, bank account, id number
    SUSPICIOUS_TARGET     = 'suspicious_target'         # explicit unapproved domain/email
    INDIRECT_INJECTION_INDICATOR = 'indirect_injection_indicator'  # hidden instruction in tool output


# Flag detection patterns (compiled once)
_FLAG_PATTERNS: Dict[SessionFlags, List[re.Pattern]] = {
    SessionFlags.PERSISTENT_INSTRUCTION: [
        re.compile(r'\b(?:remember|always|from\s+now\s+on|going\s+forward|henceforth|ignore(?: your)? previous|important message)\b', re.I),
    ],
    SessionFlags.SENSITIVE_DATA_MENTION: [
        re.compile(r'\b(?:(?:\w*(?:password|secret|api[_\s-]?key|credential|private[_\s-]?key|id_rsa|ssh[_\s-]?key|\.env|mat khau|security code))|token|access_token|auth_token)\b', re.I),
    ],
    SessionFlags.EXFIL_VERB: [
        re.compile(r'\b(?:send|forward|email|upload|transfer|post|submit|exfil|gui|chuyen|dua|cung cap)\b', re.I),
    ],
    SessionFlags.ROLE_OVERRIDE: [
        re.compile(r'\b(?:you\s+are\s+now|act\s+as|pretend\s+(?:to\s+be|you\s+are)|your\s+new\s+role)\b', re.I),
    ],
    SessionFlags.SYSTEM_PROMPT_PROBE: [
        re.compile(r'\b(?:what\s+are\s+your\s+instructions|system\s+prompt|show\s+(?:me\s+)?your\s+(?:rules|prompt)|huong dan he thong|luat he thong|cho xem)\b', re.I),
    ],
    SessionFlags.DESTRUCTIVE_INTENT: [
        re.compile(r'\b(?:delete|remove|drop|format|destroy|wipe|purge|truncate)\b', re.I),
    ],
    SessionFlags.RECON_ACTIVITY: [
        # BUG FIXED: Viết lại từng cụm với ranh giới từ riêng để lệnh kết thúc bằng ký tự đặc biệt (find /)
        # hoặc khoảng trắng (id) không bị gãy bởi \b chung ở cuối mảng.
        re.compile(r'(?:\bls\s+-la\b|\bcat\s+/etc\b|\bfind\s+/(?:\s|$)|\bwhoami\b|\buname\b|\bid\b|\benv\b|\bprintenv\b)', re.I),
    ],
    SessionFlags.COVER_TRACKS: [
        re.compile(r'\b(?:history\s+-c|clear[_\s-]?history|rm\s+.*log|shred|unset\s+HISTFILE)\b', re.I),
    ],
    SessionFlags.SENSITIVE_DATA_EXT: [
        re.compile(r'\b(?:passport|bank account|id(?:_|-|\s)?number|user information)\b', re.I),
    ],
    SessionFlags.INDIRECT_INJECTION_INDICATOR: [
        # Pattern 1: "send email to X with body Y" embedded in longer text
        re.compile(r'send\s+(?:an?\s+)?(?:email|message|mail)\s+(?:to\s+)?\S+@\S+.*(?:body|content|text)\s*[=:]\s*["\']?\w+', re.I | re.DOTALL),
        # Pattern 2: Explicit API call instruction
        re.compile(r'(?:issue|make|call)\s+(?:an?\s+)?api[_\s]?call\s+to\s+send', re.I),
        # Pattern 3: "confirmation" + email address in same action (>200 chars)
        re.compile(r'confirmation.*?(?:\S+@\S+\.\w+)|(?:\S+@\S+\.\w+).*?confirmation', re.I | re.DOTALL),
        # Pattern 4: Chat role injection in tool output
        re.compile(r'<\|(?:user|assistant|system)\|>.*?(?:send|email|forward|transfer)', re.I | re.DOTALL),
        # Pattern 5: Backtick-wrapped instruction within larger text
        re.compile(r'[`]{3,}.*?(?:send|email|forward|transfer).*?[`]{3,}', re.I | re.DOTALL),
    ],
}

# Dangerous cross-step flag combinations
# Heuristics Parameters - Documented Decision
# Các ngưỡng này được đúc kết từ phân tích manual các mẫu tấn công AgentDojo.
# Khác với các mô hình Machine Learning thuần túy, đây là hệ thống Rule-based/Heuristics
# nhằm bọc lót các case rõ ràng, do đó CHỦ Ý giữ tĩnh và KHÔNG dùng cơ chế auto-calibration.
# Chỉ thay đổi khi có Security Engineer review và bổ sung luận cứ mới.
_DANGEROUS_COMBOS: List[Tuple[Set[SessionFlags], float, str, int]] = [
    # (required_flags, risk_score, description, escalation_threshold)
    ({SessionFlags.RECON_ACTIVITY, SessionFlags.SENSITIVE_DATA_MENTION, SessionFlags.EXFIL_VERB},
     0.92, 'Recon -> credential read -> exfil chain detected', 1),
    ({SessionFlags.DESTRUCTIVE_INTENT, SessionFlags.COVER_TRACKS},
     0.88, 'Destructive action + evidence removal', 1),
    ({SessionFlags.SYSTEM_PROMPT_PROBE, SessionFlags.ROLE_OVERRIDE},
     0.82, 'System prompt probe + role override attempt', 2),
    ({SessionFlags.ROLE_OVERRIDE, SessionFlags.SENSITIVE_DATA_MENTION},
     0.80, 'Role override followed by sensitive data access', 2),
    ({SessionFlags.PERSISTENT_INSTRUCTION, SessionFlags.EXFIL_VERB},
     0.75, 'Persistent instruction planting + exfiltration attempt', 2),
    ({SessionFlags.RECON_ACTIVITY, SessionFlags.DESTRUCTIVE_INTENT},
     0.78, 'Reconnaissance followed by destructive action', 2),
    # Weak combos (broad exfil verb) get higher thresholds and require an explicit target
    ({SessionFlags.SENSITIVE_DATA_MENTION, SessionFlags.EXFIL_VERB, SessionFlags.SUSPICIOUS_TARGET},
     0.80, 'Credential access + exfiltration verb across session steps (with target)', 2),
    ({SessionFlags.DESTRUCTIVE_INTENT, SessionFlags.EXFIL_VERB, SessionFlags.SUSPICIOUS_TARGET},
     0.80, 'Destructive + Exfil (with target)', 2),
    ({SessionFlags.SENSITIVE_DATA_EXT, SessionFlags.EXFIL_VERB, SessionFlags.SUSPICIOUS_TARGET},
     0.75, 'ID Data Exfiltration (with target)', 2),
    ({SessionFlags.INDIRECT_INJECTION_INDICATOR},
     0.95, 'Indirect prompt injection detected in tool output', 1),
]

@dataclass
class FlagEvent:
    flag: SessionFlags
    timestamp: float
    step: int

@dataclass
class TaintRecord:
    value: str
    source_tool: str
    step_registered: int
    origin_decision: str
    timestamp: float

@dataclass
class TrustedLookup:
    """Value returned by a trusted internal tool (get_contacts, search_directory, etc.).
    Used by _has_explicit_exfil_target to whitelist emails/URLs that the agent
    looked up legitimately within the same session."""
    value: str
    source_tool: str
    step_registered: int
    timestamp: float

@dataclass
class SessionState:
    """Per-session tracking state."""
    session_id: str
    flags: Dict[SessionFlags, FlagEvent] = field(default_factory=dict) # Lưu flag kèm meta (timestamp/step)
    flag_history: List[Tuple[str, SessionFlags, int]] = field(default_factory=list)
    action_count: int = 0
    block_count: int = 0
    combo_trigger_counts: Dict[str, int] = field(default_factory=dict) # Đếm số lần trigger combo (cho logic Escalation)
    last_combo_fired: str = ''
    lock: threading.Lock = field(default_factory=threading.Lock)
    tainted_values: List[TaintRecord] = field(default_factory=list)
    trusted_values: List[TrustedLookup] = field(default_factory=list)
    user_goal_text: str = ""
    semantic_taints: List[str] = field(default_factory=list) # Base64 or semantic hashes (up to K)


class SessionAwareTier05:
    """Tier 0.5: Cross-step correlation detector.

    Wraps LlamaFirewallTier0 and adds session-level behavioral analysis.
    Zero external dependencies — uses only regex pattern matching.
    """

    def __init__(self, tier0: Optional['LlamaFirewallTier0'] = None,
                 max_sessions: int = 1000):
        self._tier0 = tier0
        # BUG 3 FIXED: Dùng OrderedDict để implement LRU Cache
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._sessions_lock = threading.Lock()
        self._max_sessions = max_sessions

    def _get_session(self, session_id: str) -> SessionState:
        with self._sessions_lock:
            if session_id in self._sessions:
                # LRU Cache: Di chuyển session đang dùng lên cuối dict (mới nhất)
                self._sessions.move_to_end(session_id)
            else:
                if len(self._sessions) >= self._max_sessions:
                    # Xóa session đầu tiên (cũ nhất)
                    self._sessions.popitem(last=False)
                self._sessions[session_id] = SessionState(session_id=session_id)
            return self._sessions[session_id]
            
    def get_session_ref(self, session_id: str) -> SessionState:
        """Get a reference to a session, creating it if it doesn't exist."""
        return self._get_session(session_id)
        
    def register_taint(self, session: SessionState, raw_tool_output: str, decision: str, tool_name: str) -> None:
        """Trích xuất và đăng ký taint từ output của tool."""
        if not raw_tool_output:
            return
            
        import time
        import re
        current_time = time.time()
        
        EXTRACTABLE_PATTERNS = [
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), # Email
            re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'),              # URL
            re.compile(r'\b(?:[0-9]{4}[- ]?){3}[0-9]{4}\b')                       # Credit card / Account like
        ]
        
        with session.lock:
            for pattern in EXTRACTABLE_PATTERNS:
                for match in pattern.finditer(raw_tool_output):
                    token = match.group(0)
                    # Check if already exists
                    if not any(t.value == token for t in session.tainted_values):
                        session.tainted_values.append(TaintRecord(
                            value=token,
                            source_tool=tool_name,
                            step_registered=session.action_count,
                            origin_decision=decision,
                            timestamp=current_time
                        ))
            
            # Phase 2 Semantic Taint: Store the entire raw output for O(K) fingerprinting
            if len(raw_tool_output) > 10:
                session.semantic_taints.append(raw_tool_output)
                # Keep only last K (e.g. 3) to maintain O(K)
                if len(session.semantic_taints) > 3:
                    session.semantic_taints.pop(0)

    # Tools whose output values are considered trusted internal lookups.
    # Values extracted from these tools' outputs will NOT trigger
    # _has_explicit_exfil_target even if they don't appear in user_goal_text.
    TRUSTED_LOOKUP_TOOLS = frozenset({
        'get_contacts', 'search_contacts', 'get_user_info',
        'search_directory', 'list_contacts', 'get_contact_info',
        'get_recipients', 'lookup_user', 'get_channel_members',
        'get_all_spam', 'get_inbox', 'get_unread_emails',
        'search_emails', 'get_email_by_id',
    })

    def register_trusted_lookup(self, session: SessionState, tool_output: str, tool_name: str) -> None:
        """Register values from trusted internal tools as legitimate lookup results.
        These values will be whitelisted by _has_explicit_exfil_target."""
        if not tool_output or tool_name not in self.TRUSTED_LOOKUP_TOOLS:
            return

        import time
        import re
        current_time = time.time()

        EXTRACTABLE_PATTERNS = [
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'),
        ]

        with session.lock:
            for pattern in EXTRACTABLE_PATTERNS:
                for match in pattern.finditer(tool_output):
                    token = match.group(0)
                    if not any(t.value == token for t in session.trusted_values):
                        session.trusted_values.append(TrustedLookup(
                            value=token,
                            source_tool=tool_name,
                            step_registered=session.action_count,
                            timestamp=current_time
                        ))

    def _cleanup_expired_flags(self, session: SessionState, current_time: float):
        """BUG 4 FIXED: Dọn dẹp flags dựa trên Time và Step Decay"""
        expired_flags = []
        for flag, event in session.flags.items():
            time_elapsed = current_time - event.timestamp
            steps_elapsed = session.action_count - event.step
            
            if (time_elapsed > Tier05Config.FLAG_EXPIRY_SECONDS or 
                steps_elapsed > Tier05Config.FLAG_EXPIRY_STEPS):
                expired_flags.append(flag)
                
        for flag in expired_flags:
            del session.flags[flag]
            
        # Cleanup tainted_values
        alive_taints = []
        for t in session.tainted_values:
            if (current_time - t.timestamp <= Tier05Config.FLAG_EXPIRY_SECONDS and
                session.action_count - t.step_registered <= Tier05Config.FLAG_EXPIRY_STEPS):
                alive_taints.append(t)
        session.tainted_values = alive_taints

    def _extract_flags(self, action: str, skip_rce: bool = False) -> Set[SessionFlags]:
        """Detect behavioral flags from a single action."""
        flags = set()
        for flag_type, patterns in _FLAG_PATTERNS.items():
            if skip_rce and flag_type in {SessionFlags.DESTRUCTIVE_INTENT, SessionFlags.RECON_ACTIVITY, SessionFlags.COVER_TRACKS}:
                continue
            if any(p.search(action) for p in patterns):
                flags.add(flag_type)
        return flags

    def _check_combos(self, session: SessionState, new_flags: Set[SessionFlags] = None) -> Optional[Tuple[float, str, int]]:
        """Check if accumulated session flags form a dangerous combination."""
        best_match = None
        # Chỉ check với các flags đang active (chưa hết hạn)
        active_flags = set(session.flags.keys())
        
        for required_flags, risk_score, description, threshold in _DANGEROUS_COMBOS:
            if required_flags.issubset(active_flags):
                # DOMINO EFFECT FIX: Only trigger if the current action actively contributed to this combo
                if new_flags is not None and not required_flags.intersection(new_flags):
                    continue
                
                if best_match is None or risk_score > best_match[0]:
                    best_match = (risk_score, description, threshold)
        return best_match

    def scan(self, action: str, session_id: str = "default", action_type: str = "prompt", skip_rce: bool = False) -> ScanResult:
        """Scan action with session context."""
        t0 = time.perf_counter()
        current_time_wall = time.time()
        
        session = self._get_session(session_id)
        
        with session.lock:
            if action_type == "prompt" and not session.user_goal_text:
                session.user_goal_text = action
                
            session.action_count += 1
            
            # Cleanup các flag rác trước khi đánh giá
            self._cleanup_expired_flags(session, current_time_wall)

            # Phase 1: Tier 0 scan (if available)
            tier0_result = None
            if self._tier0 is not None:
                # [Phase 1.2] Cô lập biến số: Tier 0 chỉ quét prompt (NLP), bỏ qua tool_call (JSON/API)
                if action_type in ("prompt", "tool_output", "tool_call"):
                    tier0_result = self._tier0.scan(action, skip_rce=skip_rce)
                    if tier0_result and getattr(tier0_result, 'is_blocked', False):
                        session.block_count += 1
                        return tier0_result
                else:
                    # Dummy ALLOW cho tool_call để qua cửa Tier 0, nhường cho Tier 0.5 & V61
                    tier0_result = ScanResult(
                        decision=ScanDecision.ALLOW,
                        is_blocked=False,
                        rule_fired='',
                        reason='',
                        confidence=0.01,
                        benign_confidence=0.99,
                        latency_ms=0.0
                    )

            # Phase 2: Extract and accumulate behavioral flags
            normalised_action = _pre_normalise(action)
            new_flags = self._extract_flags(normalised_action, skip_rce=skip_rce)
            
            # --- PHASE 2.1: Context-Aware Target Verification ---
            # Require explicit unapproved target for exfiltration combos to fire
            from models.security.advanced_heuristics import PermissionGate
            target = PermissionGate.has_explicit_exfil_target(action, session.user_goal_text)
            if target and not any(t.value == target for t in session.trusted_values):
                new_flags.add(SessionFlags.SUSPICIOUS_TARGET)

            for flag in new_flags:
                # Ghi nhận flag mới hoặc cập nhật timestamp/step nếu flag đã tồn tại
                session.flags[flag] = FlagEvent(flag=flag, timestamp=current_time_wall, step=session.action_count)
                # Chỉ lưu history lần đầu
                if flag not in [f for _, f, _ in session.flag_history]:
                    session.flag_history.append((action[:80], flag, session.action_count))

            # Phase 3: Check dangerous combinations
            combo = self._check_combos(session, new_flags=new_flags)
            latency_ms = (time.perf_counter() - t0) * 1000

            if combo is not None:
                risk_score, description, threshold = combo
                
                # Cập nhật số lần kích hoạt của combo này
                session.combo_trigger_counts[description] = session.combo_trigger_counts.get(description, 0) + 1
                trigger_count = session.combo_trigger_counts[description]
                
                # ESCALATION POLICY: Nâng cấp nếu MONITOR lặp lại nhiều lần
                if risk_score < 0.85 and trigger_count >= threshold:
                    risk_score = 0.85  # Ép lên BLOCK level
                    description = f"ESCALATED (Repeated {trigger_count}x): {description}"

                is_repeat = (description == session.last_combo_fired)
                session.last_combo_fired = description
                
                # BUG 1 FIXED: Logic quyết định BLOCK/MONITOR tách biệt hoàn toàn với logic REPEAT
                # Repeat chỉ dùng để đổi rule_fired, KHÔNG dùng để hạ is_blocked.
                is_blocked = risk_score >= 0.85
                decision = ScanDecision.BLOCK if is_blocked else ScanDecision.MONITOR
                
                rule_fired = 'cross_step_combo'
                if not is_blocked:
                    rule_fired = 'cross_step_suspicious'
                if is_repeat:
                    rule_fired = 'cross_step_repeat'

                if is_blocked:
                    session.block_count += 1
                    
                return ScanResult(
                    decision=decision,
                    is_blocked=is_blocked, # Nếu nguy hiểm, VẪN BLOCK dù là repeat
                    rule_fired=rule_fired,
                    reason=f'{"Repeat: " if is_repeat else ""}{description}',
                    confidence=risk_score, # Giữ nguyên độ tự tin thật, không hardcode 0.50
                    benign_confidence=1.0 - risk_score,
                    latency_ms=latency_ms,
                    all_rules_fired=[rule_fired],
                    normalized_action=normalised_action,
                )

            # No combo triggered — return Tier 0 result or ALLOW
            if tier0_result is not None:
                return tier0_result

            return ScanResult(
                decision=ScanDecision.ALLOW,
                is_blocked=False,
                rule_fired='',
                reason='No malicious patterns detected (Tier 0 + Tier 0.5)',
                confidence=0.01,
                benign_confidence=0.99,
                latency_ms=latency_ms,
                all_rules_fired=[],
                normalized_action=normalised_action,
            )

    def reset_session(self, session_id: str) -> None:
        with self._sessions_lock:
            self._sessions.pop(session_id, None)

    def reset_all(self) -> None:
        with self._sessions_lock:
            self._sessions.clear()

    def get_session_report(self, session_id: str) -> Dict:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
        if session is None:
            return {'error': f'Session {session_id} not found'}
            
        with session.lock:
            return {
                'session_id': session.session_id,
                'action_count': session.action_count,
                'block_count': session.block_count,
                'active_flags': sorted(f.value for f in session.flags.keys()),
                'flag_history': [
                    {'action': a, 'flag': f.value, 'step': s}
                    for a, f, s in session.flag_history
                ],
                'combo_trigger_counts': dict(session.combo_trigger_counts),
                'last_combo_fired': session.last_combo_fired,
            }

