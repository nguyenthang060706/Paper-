import os
import time
from types import SimpleNamespace
import json
from core.tier_lstm import SessionAwareLSTMRisk, LSTM_BLOCK_THRESHOLD

def _load_lstm_config():
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "thresholds.json")
        with open(config_path, "r") as f:
            data = json.load(f)
            return data.get("tier05_lstm", {})
    except Exception:
        return {}

class LSTMTier05Wrapper:
    """
    Wraps the fallback Tier0.5 (Regex) with the robust LSTM Temporal Model from tier_lstm.py.
    This eliminates hardcoded absolute paths and prevents the silent fallback bug.
    """
    def __init__(self, fallback_tier05, use_synthetic_iat: bool = False):
        self.fallback = fallback_tier05
        
        # Production safety: synthetic IAT is for offline benchmarks ONLY.
        # In production, wall-clock inter-arrival time is the ground-truth temporal signal.
        if use_synthetic_iat:
            import warnings
            warnings.warn(
                "[PRODUCTION SAFETY] use_synthetic_iat=True is for OFFLINE BENCHMARK ONLY. "
                "In production, set use_synthetic_iat=False to use real wall-clock IAT.",
                UserWarning, stacklevel=2
            )
        
        # Load configs
        lstm_cfg = _load_lstm_config()
        self.block_threshold = float(lstm_cfg.get("block_threshold", LSTM_BLOCK_THRESHOLD))
        self.confidence_floor = float(lstm_cfg.get("confidence_propagation_floor", 0.50))
        
        # Resolve path robustly relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tier05_path = os.path.join(base_dir, "models", "artifacts", "v90_tier_0_5_session_risk.pth")
        
        print(f"[LSTM-Tier0.5] Initializing Neural Network Shield from tier_lstm.py (block_threshold={self.block_threshold})...")
        self.lstm_risk = SessionAwareLSTMRisk(
            model_path=tier05_path,
            block_threshold=self.block_threshold,
            use_synthetic_iat=use_synthetic_iat
        )
        
    def scan(self, action: str, session_id: str, action_type: str = "prompt", skip_rce: bool = False) -> 'ScanResult':
        """
        Intercept the scan call. We first let the fallback (SessionAwareTier05) process it,
        so it updates the session state (flags, counts, etc.).
        Then we run the LSTM on the updated window and potentially override the decision.
        """
        fallback_res = self.fallback.scan(action, session_id=session_id, action_type=action_type, skip_rce=skip_rce)
        
        # If regex already blocked it, just return
        if getattr(fallback_res, 'is_blocked', False):
            return fallback_res
            
        # Bypass LSTM completely for read-only tools and file dumps
        if skip_rce:
            return fallback_res
            
        # Check feature flag (Default: False)
        if os.environ.get("TIER05_LSTM_ENABLED", "False").lower() not in ("true", "1", "yes"):
            return fallback_res
            
        # 2. Extract upstream score
        upstream_score = getattr(fallback_res, 'confidence', 0.0)
        
        # 3. Predict using the real SessionAwareLSTMRisk
        lstm_res = self.lstm_risk.update_and_score(
            session_id=session_id,
            action=action,
            action_type=action_type, # Propagated properly
            upstream_risk_score=upstream_score
        )
        
        # 4. Merge results
        if lstm_res.get("is_blocked", False):
            confidence = lstm_res.get("probability", 0.99)
            print(f"[LSTM-Tier0.5] [BLOCKED] Session {session_id}! Confidence: {confidence*100:.2f}%")
            return SimpleNamespace(
                is_blocked=True,
                decision=SimpleNamespace(value="BLOCK"),
                confidence=confidence,
                reason=lstm_res.get("reason", "LSTM Sequence Risk Threshold Exceeded"),
                rule_fired="LSTM_TEMPORAL_MODEL",
                all_rules_fired=["LSTM_TEMPORAL_MODEL"],
                normalized_action=action
            )
            
        # [E7 Round 2 FIX] Confidence Contamination Leak
        # In decoupled mode, LSTM MUST NOT inject its confidence into the fallback result.
        # Doing so contaminates Tier0/V61's RiskSignals and causes Heuristics to block,
        # elevating the FPR even if the adaptive threshold isn't escalated.
        escalation_mode = os.environ.get("ESCALATION_FEEDBACK_MODE", "decoupled").lower()
        if escalation_mode == "legacy":
            lstm_conf = lstm_res.get("probability", 0.0)
            if lstm_conf >= self.confidence_floor:
                fallback_res.confidence = max(getattr(fallback_res, 'confidence', 0.0), lstm_conf)
                
        return fallback_res

    def get_session_report(self, session_id: str):
        return self.fallback.get_session_report(session_id)
        
    def get_session_ref(self, session_id: str):
        return self.fallback.get_session_ref(session_id)

    def register_taint(self, session, raw_data, decision, tool_name="unknown"):
        if hasattr(self.fallback, 'register_taint'):
            return self.fallback.register_taint(session, raw_data, decision, tool_name)

    def register_trusted_lookup(self, session, raw_data, tool_name="unknown"):
        if hasattr(self.fallback, 'register_trusted_lookup'):
            return self.fallback.register_trusted_lookup(session, raw_data, tool_name)

    def __getattr__(self, name):
        return getattr(self.fallback, name)
