"""
GlobalThreatTracker — Cross-session APT correlation engine.
Tracks tactical-stage flags (sensitive_read, external_send, cover_tracks)
per user_id across DISTINCT session_id boundaries, detecting low-and-slow
multi-session attack campaigns.

IMPORTANT — measurement caveat:
The AgentDojo/EVO-PCA benchmark dataset (evo_pca_full.jsonl) does not carry
a real user_id field distinguishing which sessions belong to the same
end-user. In production, user_id would come from the calling application's
authentication context (unavailable in a replayed benchmark trace).

When scan() is called without an explicit user_id, it defaults to session_id
itself, making GlobalThreatTracker a safe no-op on this benchmark (each
session is treated as its own isolated user, so no cross-session signal can
ever fire) — this avoids false correlations, but also means the benchmark's
reported FPR/ABSR numbers do NOT reflect GlobalThreatTracker's real-world
cross-session detection capability. See tests/test_global_threat_tracker.py
for functional verification of the correlation logic in isolation.
"""

from collections import defaultdict
import threading
from typing import Optional, List
from models.security.advanced_heuristics import RiskSignal

class GlobalThreatTracker:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_events_per_user: int = 200):
        with self.__class__._instance_lock:
            if not getattr(self, "_initialized", False):
                self.max_events_per_user = max_events_per_user
                self.user_history = defaultdict(list)  # user_id -> event dicts
                self._emitted = set()
                self._initialized = True

    STAGE_FROM_SIGNAL = {
        "intent_flow_exfiltration": {"sensitive_read", "external_send"},
        "multi_step_exfiltration_chain": set(),
        "missing_required_permission": set(),
        "high_risk_capability_combo": set(),
    }

    CORRELATION_RULES = [
        ({"sensitive_read", "external_send", "cover_tracks"}, 96, True, "Cross-session exfiltration with cover tracks", 0.93),
        ({"sensitive_read", "external_send"},                 91, True, "Cross-session sensitive read plus external send", 0.89),
        ({"sensitive_read", "cover_tracks"},                  86, False, "Cross-session sensitive read plus evidence removal", 0.86),
        ({"external_send", "cover_tracks"},                   88, False, "Cross-session external send plus evidence removal", 0.87),
    ]

    @classmethod
    def reset_instance(cls):
        """Force re-creation and clear memory on next construction. Call between Ablation Study configs."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.user_history.clear()
                cls._instance._emitted.clear()
                cls._instance = None

    @staticmethod
    def _signal_stages(signal) -> set:
        stages = set(GlobalThreatTracker.STAGE_FROM_SIGNAL.get(getattr(signal, "name", ""), set()))
        evidence = getattr(signal, "evidence", []) or []
        joined = " | ".join(str(x) for x in evidence)
        lowered = joined.lower()

        if "secrets.read" in lowered or "sensitive_read" in lowered:
            stages.add("sensitive_read")
        if "network.external" in lowered or "external_send" in lowered or "http" in lowered:
            stages.add("external_send")
        if "audit.cover_tracks" in lowered or "cover_tracks" in lowered or "history -c" in lowered:
            stages.add("cover_tracks")
        if "shell.exec" in lowered or "privileged_exec" in lowered:
            stages.add("privileged_exec")
        return stages

    def register_stages_from_signals(self, user_id: str, session_id: str, signals: list):
        if not signals:
            return

        event_stages = set()
        max_severity = 0
        sources = []
        for signal in signals:
            event_stages |= self._signal_stages(signal)
            max_severity = max(max_severity, int(getattr(signal, "severity", 0)))
            sources.append(getattr(signal, "source", getattr(signal, "name", "unknown")))

        if not event_stages:
            return

        event = {
            "session_id": session_id,
            "stages": frozenset(event_stages),
            "max_severity": max_severity,
            "sources": tuple(sorted(set(sources))),
        }
        history = self.user_history.get(user_id)
        if history is None:
            history = []
            self.user_history[user_id] = history
            
        history.append(event)
        if len(history) > self.max_events_per_user:
            del history[:-self.max_events_per_user]

    def register_stage(self, user_id: str, session_id: str, stage: str, severity: int = 50, source: str = "manual"):
        """Utility to register a specific stage directly without a RiskSignal object."""
        event = {
            "session_id": session_id,
            "stages": frozenset([stage]),
            "max_severity": severity,
            "sources": (source,),
        }
        history = self.user_history.get(user_id)
        if history is None:
            history = []
            self.user_history[user_id] = history
            
        history.append(event)
        if len(history) > self.max_events_per_user:
            del history[:-self.max_events_per_user]

    def check_cross_session_correlation(self, user_id: str) -> Optional[RiskSignal]:
        history = self.user_history.get(user_id)
        if not history:
            return None
            
        sessions = {e["session_id"] for e in history}
        if len(sessions) < 2:
            return None

        combined = set().union(*(set(e["stages"]) for e in history))
        for rule_idx, (required, severity, is_critical, description, confidence) in enumerate(self.CORRELATION_RULES):
            if required.issubset(combined):
                emit_key = (user_id, rule_idx)
                if emit_key in self._emitted:
                    continue
                self._emitted.add(emit_key)
                return RiskSignal(
                    name="cross_session_correlation",
                    severity=severity,
                    confidence=confidence,
                    is_critical=is_critical,
                    source="global_threat_tracker",
                    evidence=[
                        f"user_id={user_id}",
                        f"sessions={sorted(sessions)}",
                        f"combined_stages={sorted(combined)}",
                        f"rule={description}",
                    ],
                )
        return None

    def reset(self, user_id: str = None) -> None:
        if user_id is None:
            self.user_history.clear()
            self._emitted.clear()
        else:
            self.user_history.pop(user_id, None)
            self._emitted = {k for k in self._emitted if k[0] != user_id}
