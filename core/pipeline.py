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
from models.security.v61_inference_router import V61SecurityRouter
from models.security.feedback_logger import FeedbackLogger
from models.security.advanced_heuristics import VotingAggregator, RiskSignal, Canonicalizer, PermissionGate, AdaptiveEscalationManager

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
    lines = action.strip().splitlines()
    
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
    Orchestrator chung điều phối cả 3 tầng bảo mật:
    Tier 0 -> Tier 0.5 -> V61 (ML + LLM Judge).
    Cung cấp các cổng scan ingress và sanitize egress.
    """
    def __init__(self, use_synthetic_iat: bool = False):
        self.use_synthetic_iat = use_synthetic_iat
        logger.info("Initializing Production Pipeline: Tier 0 -> Tier 0.5 -> V61")
        self.tier0 = LlamaFirewallTier0()
        self.tier05_base = SessionAwareTier05(tier0=self.tier0)
        
        # Inject the Neural Network LSTM as Tier 0.5 Wrapper (Phase 12)
        try:
            from core.lstm_tier05 import LSTMTier05Wrapper
            self.tier05 = LSTMTier05Wrapper(self.tier05_base, use_synthetic_iat=self.use_synthetic_iat)
        except Exception as e:
            logger.error(f"Failed to load LSTM Tier 0.5 Wrapper: {e}")
            self.tier05 = self.tier05_base
        # Khởi tạo không tham số để dùng Environment Variables
        self.v61 = V61SecurityRouter()
        self.feedback_logger = FeedbackLogger()
        self.permission_gate = PermissionGate()
        self.fpr_manager = AdaptiveEscalationManager(
            target_escalation_rate=0.10, 
            initial_threshold=0.72, 
            min_threshold=0.58, 
            warmup_decisions=30
        )
            
    def scan(self, action: str, session_id: str, action_type: str = "prompt", actual_label: bool = None) -> dict:
        """
        Quét hành động đầu vào (Ingress) qua toàn bộ pipeline.
        Hỗ trợ SHADOW_MODE qua biến môi trường (mặc định False).
        """
        is_shadow_mode = os.environ.get("SHADOW_MODE", "False").lower() == "true"
        
        result = {
            "decision": "ALLOW",
            "layer": None,
            "reason": "",
            "llm_down": False,
            "ml_score": 0.0,
            "was_shadow_blocked": False,
            "action_type": action_type
        }
        
        try:
            from core.tier_lstm import _extract_tool_and_resource
            tool_name, _ = _extract_tool_and_resource(action, action_type)
        except Exception:
            tool_name = None
            
        skip_rce = False
        if tool_name in READ_ONLY_TOOLS:
            skip_rce = True
        elif not tool_name and _is_quoted_tool_result(action):
            skip_rce = True
        
        # --- Layer 1: Tier 0.5 (Gọi ĐÚNG 1 LẦN) ---
        t05_res = self.tier05.scan(action, session_id=session_id, action_type=action_type, skip_rce=skip_rce)
        if session_id.startswith('sess_multi'):
            print(f"[DEBUG-TIER0.5] session_id={session_id} is_blocked={getattr(t05_res, 'is_blocked', False)}")
        
        rule_fired = getattr(t05_res, 'rule_fired', '')
        if rule_fired == "LSTM_TEMPORAL_MODEL":
            actual_layer = "Tier0.5-LSTM"
        else:
            actual_layer = "Tier0.5" if "cross_step" in rule_fired else "Tier0"
        
        decision_val = getattr(t05_res, 'decision', 'ALLOW')
        t05_decision_str = (decision_val.value if hasattr(decision_val, 'value') else str(decision_val)).upper()
        
        # --- Phase 1.2: Consequence-based Gate (Phase 4) ---
        is_t05_blocked = getattr(t05_res, 'is_blocked', False)
        force_review = False
        
        # Extract early for provenance and signal checks
        session_ref = self.tier05.get_session_ref(session_id)
        report = self.tier05.get_session_report(session_id)
        active_flags = report.get('active_flags', [])
        
        is_suspicious_dangerous_tool = False
        if action_type == "tool_call" and tool_name:
            from models.security.function_risk_registry import HIGH_RISK_FUNCTIONS
            if tool_name in HIGH_RISK_FUNCTIONS:
                from models.security.advanced_heuristics import PermissionGate
                has_taint = PermissionGate.check_taint_overlap(action, session_ref)
                user_goal = getattr(session_ref, 'user_goal_text', '') if session_ref else ''
                has_exfil = PermissionGate.has_explicit_exfil_target(action, user_goal)
                
                if len(active_flags) > 0 or has_taint or has_exfil:
                    is_suspicious_dangerous_tool = True
                    force_review = True

        # Heuristic Gate Optimization: Ngăn Tier0 block nhầm (FPR) trên payloads không phải prompt thuần túy.
        # Tuy nhiên, nếu là tool nguy hiểm + có dấu hiệu khả nghi (Gap 6), vô hiệu hóa bypass này!
        if is_t05_blocked and actual_layer == "Tier0" and action_type != "prompt":
            if not is_suspicious_dangerous_tool:
                is_t05_blocked = False
                t05_decision_str = "ALLOW"
                t05_res.confidence = 0.0
            
        if is_t05_blocked:
            result["decision"] = "BLOCK"
            result["layer"] = actual_layer
            result["reason"] = getattr(t05_res, 'reason', '')
        else:
            # --- Layer 2: V61 (ML Model + LLM Judge) ---
            normalized_action = getattr(t05_res, 'normalized_action', action)
            tier05_risk_score = getattr(t05_res, 'confidence', 0.0)
            all_rules_fired = getattr(t05_res, 'all_rules_fired', [])
            
            # V61 always evaluates (to catch Secrets/Exfiltration), regardless of skip_rce
            
            # If it's a read-only tool output or quoted text, the action model (trained on commands)
            # will flag it as an anomalous command. We evaluate it using the prompt model (text) instead.
            v61_action_type = "prompt" if skip_rce else action_type
            
            # --- LONG TEXT HEURISTIC ---
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
                force_review=force_review
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
                        force_review=force_review
                    )
                    if seg_res.get("decision") == "BLOCK":
                        v61_res = seg_res
                        v61_res["judge_reason"] = "[LONG_TEXT_HEURISTIC] Blocked on extracted instruction: " + seg_res.get("judge_reason", "")
                        break

            result["decision"] = v61_res.get("decision", "ALLOW")
            result["layer"] = "V61"
            
            base_reason = v61_res.get("judge_reason", "")
            if t05_decision_str == "MONITOR":
                monitor_reason = getattr(t05_res, 'reason', 'Suspicious activity detected')
                result["reason"] = f"[{actual_layer} MONITOR: {monitor_reason}] " + base_reason
            else:
                result["reason"] = base_reason
                
            result["llm_down"] = v61_res.get("llm_down", False)
            result["ml_score"] = v61_res.get("score", 0.0)
            result["path"] = v61_res.get("path")

        # --- Advanced Heuristics (Production Gate) ---
        try:
            enable_provenance = self._get_provenance_enabled()
            session = self.tier05.get_session_ref(session_id)
            
            signals = []
            
            action_str = json.dumps(action, ensure_ascii=False) if isinstance(action, dict) else str(action)
            canonicalized_action_str = Canonicalizer.canonicalize(action_str)
            # Detect using PermissionGate (Provenance Tagging & High Risk Combos)
            gate_signals = self.permission_gate.detect(
                canonicalized_action_str, 
                session=session, 
                enable_provenance=enable_provenance,
                skip_rce=skip_rce,
                tool_name=tool_name
            )
            signals.extend(gate_signals)
            
            if getattr(t05_res, 'confidence', 0) > 0.1:
                signals.append(RiskSignal(
                    name=getattr(t05_res, 'rule_fired', 'tier05_flag') or 'tier05_flag',
                    severity=int(getattr(t05_res, 'confidence', 0) * 100),
                    confidence=0.8,
                    is_critical=False,
                    source='tier05'
                ))
                
            if result.get("ml_score", 0) > 0:
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
                
            heuristics_tier, heuristics_score = VotingAggregator.vote(signals)
            result["heuristics_decision"] = heuristics_tier.value
            result["heuristics_score"] = float(heuristics_score)

            # Hard gate decision override
            is_escalation = (result["decision"] != "BLOCK")
            if heuristics_tier.value in ["DENY", "QUARANTINE"]:
                result["decision"] = "BLOCK"
                # Only overwrite layer if Heuristics escalated an ALLOW to a BLOCK
                if is_escalation:
                    result["layer"] = "Heuristics"
                result["reason"] = f"[HEURISTICS {heuristics_tier.value}] Aggregated score: {heuristics_score:.2f} - " + result["reason"]

        except Exception as e:
            result["heuristics_error"] = str(e)
            logger.error(f"Heuristics Error: {e}")
            
        # --- Áp dụng Shadow Mode logic nếu được bật ---
        if result["decision"] == "BLOCK" and is_shadow_mode:
            result["was_shadow_blocked"] = True
            result["shadow_blocked_layer"] = result["layer"]
            result["decision"] = "ALLOW"
            result["reason"] = f"[SHADOW BLOCK] Hành động lẽ ra đã bị chặn bởi {result['layer']}. " + result["reason"]
            
        # --- Ghi nhận log ---
        is_real_block = result["decision"] == "BLOCK" or result.get("was_shadow_blocked")
        
        # --- Cập nhật FPR Budget Manager (Phase 1.1) ---
        if result.get("layer") != "Tier0":
            self.fpr_manager.record_decision(is_escalated=bool(is_real_block))
            
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
            
        # Ensure action_type and reason are always accurately populated in the result
        result["action_type"] = action_type
        
        return result

    def _get_provenance_enabled(self) -> bool:
        import json
        import os
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
        """
        Làm sạch dữ liệu đầu ra từ công cụ (Egress Sanitize).
        """
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
        """
        Làm sạch dữ liệu đầu ra từ công cụ theo batch (Egress Sanitize).
        """
        enable_provenance = self._get_provenance_enabled()
        session = self.tier05.get_session_ref(session_id)
        return self.v61.sanitize_data_batch(
            raw_data_list, 
            tool_name=tool_name, 
            session=session, 
            enable_provenance=enable_provenance, 
            tier05_instance=self.tier05
        )
