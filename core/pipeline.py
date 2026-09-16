import os
import sys
import logging
import json
import uuid
import random
import re

# Fix imports since we moved files
from core.tier0 import LlamaFirewallTier0
from core.tier05 import SessionAwareTier05
from models.security.multi_step_heuristics import HeuristicStateTracker
from models.security.llm_session_judge import LLMSessionJudge
from models.security.v61_inference_router import V61SecurityRouter
from models.security.feedback_logger import FeedbackLogger
from models.security.advanced_heuristics import VotingAggregator, RiskSignal, Canonicalizer, PermissionGate, AdaptiveEscalationManager
from models.security.boundary_detector import InstructionBoundaryDetector
from models.security.global_threat_tracker import GlobalThreatTracker
from models.security.signal_registry import SignalRegistry, EscalationManagerType
from models.security.function_risk_registry import check_function_signature, HIGH_RISK_FUNCTIONS
from models.security.shared_utils import is_benign_dev_shell
from core.early_intent_classifier import EarlyIntentClassifier
from models.security.email_vector_detector import EmailVectorDetector

_config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'thresholds.json')
try:
    with open(_config_path) as f:
        _thresholds = json.load(f)
    # Dual architectural role (single source of truth from thresholds.json):
    #   (a) v61_action_review_threshold: V61 action model internal ALLOW→REVIEW boundary
    #   (b) session_judge_band_ceil: scores >= this bypass Session Judge, go to V61 LLM Action Judge
    _action_review_threshold = _thresholds.get('action_risk_model', {}).get('REVIEW', 0.5577)
    _action_block_threshold = _thresholds.get('action_risk_model', {}).get('BLOCK', 0.6429)
    _suspicious_band_min = _thresholds.get('multi_step_defense', {}).get('suspicious_band_min', 0.40)
except Exception:
    _action_review_threshold = 0.5577
    _action_block_threshold = 0.6429
    _suspicious_band_min = 0.40

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

READ_ONLY_TOOLS = {"Read", "Glob", "Grep", "ls", "cat", "find", "stat", "read_file", "search_dir"}

_COMMAND_INDICATORS = re.compile(
    r'\b(rm|curl|wget|chmod|chown|exec|eval|base64|sudo|bash|sh|nc|netcat|'
    r'dd|mkfs|kill|systemctl|crontab)\b'
    r'|[|;&]{1,2}'          # pipe, chain, background operators
    r'|https?://',          # URL — dấu hiệu exfiltration/download
    re.IGNORECASE
)


def _is_quoted_tool_result(action: str) -> bool:
    stripped = action.strip()
    if re.match(r'^Token\(|^PowerShellResult\(|^\s*\{.*"(?:stdout|output|returncode)":', stripped):
        return True

    lines = stripped.splitlines()
    
    if len(lines) >= 2:
        numbered_lines = sum(1 for line in lines if re.match(r'^\s*\d+[\t :.-]', line))
        if numbered_lines / len(lines) > 0.3:
            return True

    # Lấy tất cả các indicators
    indicators = _COMMAND_INDICATORS.findall(action)
    
    is_command = False
    if indicators:
        # Nếu bắt đầu trực tiếp bằng một lệnh nguy hiểm -> chắc chắn là lệnh
        starts_with_cmd = bool(re.match(r'^\s*(rm|curl|wget|chmod|chown|exec|eval|base64|sudo|bash|sh|nc|netcat|dd|mkfs|kill|systemctl|crontab)\b', action, re.IGNORECASE))
        
        if starts_with_cmd:
            is_command = True
        elif len(lines) <= 3 and bool(re.match(r'^\s*[\w./-]+\s', action)):
            # Lệnh ngắn (<=3 dòng) và có format của một command execution
            is_command = True
        elif len(indicators) >= 3:
            # Script dài nhưng chứa quá nhiều dấu hiệu command/pipe
            is_command = True
            
    if is_command:
        return False

    if len(lines) < 2:
        # File paths like `drivers\video\efifb.c`
        if len(lines) == 1 and ('/' in lines[0] or '\\' in lines[0]):
            return True
        return False
        
    return False


def _extract_instruction_segments(text: str) -> list[str]:
    """Extract likely instruction segments from long text for separate scanning."""
    if len(text) < 300:
        return []
    
    segments = []
    # Extract sentences with imperative verbs
    imperative_re = re.compile(
        r'(?:^|\.\s+|\n)([^.]*?\b(?:send|forward|email|transfer|upload|execute|run|'
        r'please\s+(?:send|forward|do)|you\s+(?:must|should|need\s+to))\b[^.]*\.?)',
        re.IGNORECASE | re.MULTILINE
    )
    for m in imperative_re.finditer(text):
        seg = m.group(1).strip()
        if 20 < len(seg) < 500:
            segments.append(seg)
    
    return segments


class UnifiedFirewallPipeline:
    """
    Orchestrator điều phối các tầng phòng thủ:
    Tier 0 (Fast Regex) -> Tier 0.5 (Heuristic State-Machine) -> V61 Router -> LLM Session Judge.
    """
    def __init__(self, use_synthetic_iat: bool = False):
        self.use_synthetic_iat = use_synthetic_iat
        logger.info("Initializing Multi-Step Hardened Pipeline: Tier 0 -> Heuristic State-Machine -> V61 -> LLM Session Judge")
        self.tier0 = LlamaFirewallTier0()
        self.tier05_base = SessionAwareTier05(tier0=self.tier0)
        try:
            from core.lstm_tier05 import LSTMTier05Wrapper
            self.tier05 = LSTMTier05Wrapper(self.tier05_base, use_synthetic_iat=self.use_synthetic_iat)
        except Exception as e:
            logger.error(f"Failed to load LSTM Tier 0.5 Wrapper: {e}")
            self.tier05 = self.tier05_base
        self.state_tracker = HeuristicStateTracker()
        self.session_judge = LLMSessionJudge()
        self.v61 = V61SecurityRouter()
        self.feedback_logger = FeedbackLogger()
        self.permission_gate = PermissionGate()
        self.fpr_manager = AdaptiveEscalationManager(
            target_escalation_rate=0.05, 
            initial_threshold=0.72, 
            min_threshold=0.58, 
            warmup_decisions=30
        )
        self.boundary_detector = InstructionBoundaryDetector()
        self.global_tracker = GlobalThreatTracker()
        self.signal_registry = SignalRegistry.get_instance()
        self.early_intent = EarlyIntentClassifier()

    def _can_fast_pass_tool_call(self, action: str, tool_name: str, 
                                  session_ref, active_flags: list) -> bool:
        """
        Fast-pass cho tool_call KHI VÀ CHỈ KHI tất cả 4 gates đều pass:
        1. Không có session-level threat flags (active_flags trống)
        2. tool_name KHÔNG nằm trong HIGH_RISK_FUNCTIONS (hoặc signature severity >= 60)
        3. PermissionGate KHÔNG detect capability nguy hiểm nào
        4. is_benign_dev_shell() trả True
        """
        if active_flags:
            return False
        sig = check_function_signature(tool_name)
        if sig is not None and sig.severity >= 60:
            return False
        action_str = json.dumps(action, ensure_ascii=False) if isinstance(action, dict) else str(action)
        canonicalized = Canonicalizer.canonicalize(action_str)
        gate_signals = self.permission_gate.detect(
            canonicalized, session=session_ref, 
            enable_provenance=False, skip_rce=False,
            tool_name=tool_name, is_suspicious_dangerous_tool=False
        )
        if gate_signals:
            return False
        if not is_benign_dev_shell(action_str):
            return False
        return True
            
    def scan(self, action: str, session_id: str, action_type: str = "prompt", actual_label: bool = None, user_id: str = None) -> dict:
        """
        Quét hành động đầu vào (Ingress) qua toàn bộ pipeline.
        Hỗ trợ SHADOW_MODE qua biến môi trường (mặc định False).
        """
        is_shadow_mode = os.environ.get("SHADOW_MODE", "False").lower() == "true"
        effective_user_id = user_id or session_id
        
        # Defensive auto-inference of action_type & safe dict handling
        tool_name = None
        if isinstance(action, dict):
            action_type = "tool_call"
            tool_name = action.get("name") or action.get("tool")
            action = json.dumps(action, ensure_ascii=False)
        elif action_type in (None, "prompt"):
            if isinstance(action, str) and re.match(r'^[a-zA-Z_]\w*\s*\(', action.strip()):
                action_type = "tool_call"

        result = {
            "decision": "ALLOW",
            "layer": None,
            "reason": "",
            "llm_down": False,
            "ml_score": 0.0,
            "was_shadow_blocked": False,
            "action_type": action_type
        }
        
        if action_type == "tool_call" and not tool_name:
            m = re.match(r'^(\w+)\(', action.strip())
            if m:
                tool_name = m.group(1)
            
        skip_rce = False
        if tool_name in READ_ONLY_TOOLS:
            skip_rce = True
        elif not tool_name and _is_quoted_tool_result(action):
            skip_rce = True
        
        # --- Layer 1A: Tier 0.5 Base Rule Scanning ---
        t05_res = self.tier05.scan(action, session_id=session_id, action_type=action_type, skip_rce=skip_rce)
        rule_fired = getattr(t05_res, 'rule_fired', '')
        actual_layer = "Tier0.5-LSTM" if "LSTM" in rule_fired else ("Tier0.5" if "cross_step" in rule_fired else "Tier0")
        
        decision_val = getattr(t05_res, 'decision', 'ALLOW')
        t05_decision_str = (decision_val.value if hasattr(decision_val, 'value') else str(decision_val)).upper()
        is_t05_blocked = getattr(t05_res, 'is_blocked', False)

        # --- Layer 1B: Heuristic State-Machine (Multi-Step Kill-Chain Detection) ---
        sess_state = self.state_tracker.sessions.get(session_id)
        step_count = len(getattr(sess_state, 'action_history', [])) if sess_state else 0
        multi_step_res = self.state_tracker.evaluate(action, session_id=session_id, action_type=action_type)

        force_review = False
        session_ref = self.tier05.get_session_ref(session_id)
        user_goal = getattr(session_ref, 'user_goal_text', '') if session_ref else ''
        report = self.tier05.get_session_report(session_id)
        active_flags = report.get('active_flags', [])
        
        is_suspicious_dangerous_tool = False
        if action_type == "tool_call" and tool_name:
            if tool_name in HIGH_RISK_FUNCTIONS:
                has_taint = PermissionGate.check_taint_overlap(action, session_ref)
                has_exfil = PermissionGate.has_explicit_exfil_target(action, user_goal)
                sig = check_function_signature(tool_name)
                
                # [MỚI] Điều kiện First-Step: severity cao ngay bước đầu, không cần chờ active_flags tích lũy
                is_first_step_high_risk = (step_count == 0 and sig is not None and sig.severity >= 80)
                
                if len(active_flags) > 0 or has_taint or has_exfil or is_first_step_high_risk:
                    is_suspicious_dangerous_tool = True
                    force_review = True

        # Early-Intent trigger on prompt & session cooldown integration
        if action_type == "prompt":
            early_cfg = _thresholds.get("early_intent", {})
            risk_th = float(early_cfg.get("risk_threshold", 0.50))
            max_forced = int(early_cfg.get("max_forced_actions", 2))
            intent_res = self.early_intent.classify(action, session_id)
            if intent_res.get("risk_score", 0.0) >= risk_th:
                self.state_tracker.inject_early_warning(session_id, intent_res, max_forced_actions=max_forced)
                force_review = True

        # [CRITICAL ZERO-REGRESSION]: Merge bằng OR, không ghi đè cờ True hiện tại
        force_review = force_review or self.state_tracker.consume_forced_review(session_id)

        # Check if Tier0.5 Base blocked
        if is_t05_blocked and actual_layer == "Tier0" and action_type != "prompt":
            if not is_suspicious_dangerous_tool:
                is_t05_blocked = False
                t05_decision_str = "ALLOW"
                t05_res.confidence = 0.0

        if is_t05_blocked:
            result["decision"] = "BLOCK"
            result["layer"] = actual_layer
            result["reason"] = getattr(t05_res, 'reason', '')
        elif multi_step_res.is_blocked:
            # Multi-Step Kill-chain directly triggered
            result["decision"] = "BLOCK"
            result["layer"] = "MultiStep-Heuristics"
            result["reason"] = multi_step_res.reason
        elif action_type == "tool_call" and self._can_fast_pass_tool_call(action, tool_name, session_ref, active_flags):
            # Fast-pass: Defensive negative gate for benign tools, bypass V61 ML / LLM Judge
            result["decision"] = "ALLOW"
            result["layer"] = "FastPass-BenignTool"
            result["reason"] = "Fast-pass for verified benign development tool call"
            if self.signal_registry.can_feed_escalation("FastPass-BenignTool", EscalationManagerType.V61_ADAPTIVE):
                self.fpr_manager.record_decision(is_escalated=False, layer="FastPass-BenignTool")
            # FALL THROUGH to Advanced Heuristics (Production Gate) below!
        else:
            # --- Layer 2: V61 Router (ML Model + LLM Action Judge) ---
            tier05_risk_score = getattr(t05_res, 'confidence', 0.0)
            all_rules_fired = getattr(t05_res, 'all_rules_fired', [])
            v61_action_type = "prompt" if skip_rce else action_type
            
            # Extract instruction segments for long text heuristic
            segments = []
            if len(action) > 500:
                segments = _extract_instruction_segments(action)
            
            v61_res = self.v61.check_action(
                action,
                tier05_decision=t05_decision_str,
                tier05_risk_score=tier05_risk_score,
                tier05_rules=all_rules_fired,
                action_type=v61_action_type,
                session_flags=active_flags,
                adaptive_threshold=self.fpr_manager.adaptive_threshold,
                force_review=force_review,
                tool_name=tool_name,
                user_goal_text=user_goal
            )

            # Check segments if original action passed
            if v61_res.get("decision", "ALLOW") != "BLOCK" and segments:
                for seg in segments:
                    seg_res = self.v61.check_action(
                        seg,
                        tier05_decision=t05_decision_str,
                        tier05_risk_score=tier05_risk_score,
                        tier05_rules=all_rules_fired,
                        action_type="prompt",
                        session_flags=active_flags,
                        adaptive_threshold=self.fpr_manager.adaptive_threshold,
                        force_review=force_review,
                        tool_name=tool_name,
                        user_goal_text=user_goal
                    )
                    if seg_res.get("decision") == "BLOCK":
                        v61_res = seg_res
                        v61_res["judge_reason"] = "[LONG_TEXT_HEURISTIC] Blocked on extracted instruction: " + seg_res.get("judge_reason", "")
                        break

            ml_score = float(v61_res.get("score", 0.0))
            result["ml_score"] = ml_score
            result["path"] = v61_res.get("path")
            result["llm_down"] = v61_res.get("llm_down", False)

            # --- Layer 2B: LLM Session Judge Trigger Routing ---
            # Triggered if in suspicious score band [0.40, 0.5577) OR if State-Machine flagged Level-1 Warning
            is_suspicious_band = (_suspicious_band_min <= ml_score < _action_review_threshold)
            is_state_warning = (multi_step_res.risk_level == "WARNING_LEVEL_1")
            
            if (is_suspicious_band or is_state_warning) and v61_res.get("decision") != "BLOCK":
                session_history = [item[2] for item in getattr(self.state_tracker.sessions.get(session_id), 'action_history', [])]
                session_decision, session_reason, _ = self.session_judge.judge_session(
                    current_action=action,
                    session_history=session_history,
                    ml_score=ml_score,
                    state_flags=list(multi_step_res.stage_flags),
                    session_id=session_id
                )
                if session_decision == "BLOCK":
                    result["decision"] = "BLOCK"
                    result["layer"] = "LLM-Session-Judge"
                    result["reason"] = f"[LLM_SESSION_JUDGE] {session_reason}"
                else:
                    result["decision"] = "ALLOW"
                    result["layer"] = "V61"
                    result["reason"] = v61_res.get("judge_reason", "")
            else:
                result["decision"] = v61_res.get("decision", "ALLOW")
                result["layer"] = "V61"
                result["reason"] = v61_res.get("judge_reason", "")

        # --- Advanced Heuristics (Production Gate) ---
        try:
            enable_provenance = self._get_provenance_enabled()
            session = self.tier05.get_session_ref(session_id)
            
            signals = []
            action_str = json.dumps(action, ensure_ascii=False) if isinstance(action, dict) else str(action)
            canonicalized_action_str = Canonicalizer.canonicalize(action_str)
            
            gate_signals = self.permission_gate.detect(
                canonicalized_action_str, 
                session=session, 
                enable_provenance=enable_provenance,
                skip_rce=skip_rce,
                tool_name=tool_name,
                is_suspicious_dangerous_tool=is_suspicious_dangerous_tool
            )
            signals.extend(gate_signals)
            
            is_violated, boundary_conf, violations = self.boundary_detector.detect(canonicalized_action_str)
            if is_violated:
                signals.append(RiskSignal(
                    name='instruction_boundary_violation',
                    severity=int(boundary_conf * 85),
                    confidence=boundary_conf,
                    is_critical=boundary_conf > 0.85,
                    source='boundary_detector',
                    evidence=[f"Embedded instruction (conf={v.confidence:.2f}) at pos {v.position_ratio:.1%} in {v.context_type}" 
                              for v in violations[:3]]
                ))

            # [MỚI] First-Step check cho prompt chứa instruction độc hại ngay bước 1
            if step_count == 0 and action_type == "prompt" and is_violated and boundary_conf > 0.85:
                signals.append(RiskSignal(
                    name='first_step_embedded_instruction',
                    severity=90,
                    confidence=boundary_conf,
                    is_critical=True,
                    source='first_step_check',
                    evidence=[f"First-step embedded instruction violation conf={boundary_conf:.2f}"]
                ))

            email_signal = EmailVectorDetector.detect(canonicalized_action_str)
            if email_signal:
                signals.append(email_signal)
            
            # Multi-step state signal
            if multi_step_res.risk_level == "WARNING_LEVEL_1":
                signals.append(RiskSignal(
                    name='multistep_stage_warning',
                    severity=60,
                    confidence=0.75,
                    source='multi_step_heuristics'
                ))
            
            # Layer 3: GlobalThreatTracker (Cross-Session APT Correlation)
            self.global_tracker.register_stages_from_signals(effective_user_id, session_id, signals)
            cross_signal = self.global_tracker.check_cross_session_correlation(effective_user_id)
            if cross_signal:
                signals.append(cross_signal)

            if result.get("ml_score", 0) > _action_review_threshold:
                signals.append(RiskSignal(
                    name='v61_ml_score',
                    severity=int(result["ml_score"] * 100),
                    confidence=0.85,
                    source='v61_ml'
                ))
                
            if result.get("path") == "slow-path-llm" and not result.get("llm_down"):
                sev = 95 if result.get("decision") == "BLOCK" else 10
                signals.append(RiskSignal(
                    name='v61_llm_judge',
                    severity=sev,
                    confidence=0.95,
                    is_critical=(sev > 90),
                    source='v61_llm'
                ))
                
            heuristics_tier, heuristics_score, contributing_signals = VotingAggregator.vote(signals)
            result["heuristics_decision"] = heuristics_tier.value
            result["heuristics_score"] = float(heuristics_score)
            result["contributing_signals"] = contributing_signals

            # Hard gate decision override
            is_escalation = (result["decision"] != "BLOCK")
            if heuristics_tier.value in ["DENY", "QUARANTINE"]:
                is_shell_tool = tool_name and tool_name.lower() in ['bash', 'cmd', 'exec', 'eval', 'python', 'powershell', 'shell']
                has_critical_signal = any(getattr(s, 'is_critical', False) for s in contributing_signals) if contributing_signals else any(getattr(s, 'is_critical', False) for s in signals)
                if is_shell_tool and is_benign_dev_shell(action) and not has_critical_signal:
                    result["heuristics_decision"] = heuristics_tier.value
                    result["heuristics_downgraded"] = True
                elif action_type == "prompt" or is_suspicious_dangerous_tool or is_shell_tool or has_critical_signal or heuristics_tier.value == "QUARANTINE":
                    result["decision"] = "BLOCK"
                    if is_escalation:
                        result["layer"] = "Heuristics"
                    result["reason"] = f"[HEURISTICS {heuristics_tier.value}] Aggregated score: {heuristics_score:.2f} - " + result["reason"]
                else:
                    result["heuristics_decision"] = heuristics_tier.value
                    result["heuristics_downgraded"] = True

        except Exception as e:
            result["heuristics_error"] = str(e)
            logger.error(f"Heuristics Error: {e}")
            
        # --- Shadow Mode Logic ---
        if result["decision"] == "BLOCK" and is_shadow_mode:
            result["was_shadow_blocked"] = True
            result["shadow_blocked_layer"] = result["layer"]
            result["decision"] = "ALLOW"
            result["reason"] = f"[SHADOW BLOCK] Action would have been blocked by {result['layer']}. " + result["reason"]
            
        # --- Telemetry & FPR Budget Recording ---
        is_real_block = result["decision"] == "BLOCK" or result.get("was_shadow_blocked")
        decision_layer = str(result.get("layer") or "unknown")
        
        # Enforce Signal Registry Escalation Policy:
        # ONLY authorized sources (V61 ML) are permitted to feed V61 adaptive escalation.
        # Other layers (Heuristics, MultiStep-Heuristics, Session-Judge) are STRICTLY BLOCKED
        # from contaminating the V61 adaptive threshold feedback loop (anti-E7 protection).
        if self.signal_registry.can_feed_escalation(decision_layer, EscalationManagerType.V61_ADAPTIVE):
            self.fpr_manager.record_decision(is_escalated=bool(is_real_block), layer=decision_layer)
            
        if result.get("llm_down"):
            result["event_type"] = "LLM_DOWN_INCIDENT"
            result["label_source"] = "system_error"
            self.feedback_logger.log(result, action)
        elif is_real_block:
            result["label_source"] = "self_reported_block"
            self.feedback_logger.log(result, action)
        elif random.random() < 0.10:
            result["label_source"] = "self_reported_allow"
            self.feedback_logger.log(result, action)
            
        result["action_type"] = action_type
        return result

    def _get_provenance_enabled(self) -> bool:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'thresholds.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                    return cfg.get('provenance_tagging', {}).get('enabled', False)
            except Exception:
                pass
        return False

    def sanitize(self, raw_data: str, tool_name: str = "unknown", session_id: str = "default") -> dict:
        enable_provenance = self._get_provenance_enabled()
        session = self.tier05.get_session_ref(session_id)
        return self.v61.sanitize_data(
            raw_data, 
            tool_name=tool_name, 
            session=session, 
            enable_provenance=enable_provenance, 
            tier05_instance=self.tier05
        )

    def sanitize_batch(self, raw_data_list: list[str], tool_name: str = "unknown", session_id: str = "default") -> list[dict]:
        enable_provenance = self._get_provenance_enabled()
        session = self.tier05.get_session_ref(session_id)
        return self.v61.sanitize_data_batch(
            raw_data_list, 
            tool_name=tool_name, 
            session=session, 
            enable_provenance=enable_provenance, 
            tier05_instance=self.tier05
        )
