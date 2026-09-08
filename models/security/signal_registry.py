"""
models/security/signal_registry.py
==================================
Centralized Signal & Source Registry with Escalation Manager Policy.

Hardens the system against feedback loop poisoning (the E7 vulnerability) by ensuring
every detection source declares explicitly which EscalationManager(s) it is allowed
to feed.

Key Components:
- EscalationManagerType: Enum representing distinct escalation managers.
- SignalSource: Enum representing canonical security signal/layer sources.
- SignalDefinition: Metadata and policy for a security signal source.
- SignalRegistry: Singleton registry providing policy lookup and source resolution.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union
import logging

logger = logging.getLogger(__name__)


class EscalationManagerType(str, Enum):
    """Types of dynamic escalation managers in the defense pipeline.
    
    Generalized to support 1 or multiple independent escalation managers.
    """
    V61_ADAPTIVE = "v61_adaptive"                # Adjusts V61 ML Model direct block threshold
    HEURISTICS_ADAPTIVE = "heuristics_adaptive"  # RESERVED — no active consumer. VotingAggregator uses static thresholds.


class SignalSource(str, Enum):
    """Canonical security signal sources across all tiers."""
    TIER0 = "tier0"                                            # Tier 0 Regex Fast Filter
    TIER05_STATE_MACHINE = "tier05_state_machine"              # Tier 0.5 Multi-Step Kill-Chain State Machine
    TIER05_LSTM = "tier05_lstm"                                # Tier 0.5 LSTM Sequence Model
    V61_ML = "v61_ml"                                          # Tier 1 V61 ML Model (Prompt / Action)
    V61_LLM_ACTION_JUDGE = "v61_llm_action_judge"              # Tier 1 Slow Path LLM Action Judge
    LLM_SESSION_JUDGE = "llm_session_judge"                    # Tier 2B Cross-Step LLM Session Judge
    LLM_JUDGE_TIMEOUT_FAILSAFE = "llm_judge_timeout_failsafe"  # Failsafe decision on judge timeout
    EGRESS_KILL_SWITCH = "egress_kill_switch"                  # Emergency termination from Egress Quarantine
    HEURISTICS_PERMISSION_GATE = "heuristics_permission_gate"  # Permission / Capability combo
    BOUNDARY_DETECTOR = "boundary_detector"                    # Instruction Boundary Violation
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SignalDefinition:
    """Policy specification for a detection signal or layer."""
    source: SignalSource
    name: str
    description: str
    feeds_escalation_managers: Set[EscalationManagerType] = field(default_factory=set)
    default_severity: int = 50
    default_confidence: float = 0.5


class SignalRegistry:
    """Central registry enforcing escalation policy and canonical source mapping."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._registry: Dict[SignalSource, SignalDefinition] = {}
        self._alias_map: Dict[str, SignalSource] = {}
        self._register_defaults()
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "SignalRegistry":
        """Convenience accessor for singleton registry."""
        return cls()

    @classmethod
    def reset_instance(cls):
        """Reset registry instance for clean unit testing."""
        cls._instance = None

    def _register_defaults(self):
        """Register canonical system sources with strict anti-poisoning policies."""
        # 1. V61 ML Model — THE ONLY source permitted to feed V61 Adaptive Escalation Manager
        self.register(SignalDefinition(
            source=SignalSource.V61_ML,
            name="v61_ml",
            description="V61 ML Action/Prompt Risk Classifier",
            feeds_escalation_managers={EscalationManagerType.V61_ADAPTIVE},
            default_severity=70,
            default_confidence=0.85
        ), aliases=["v61", "v61_router", "v61_model", "v61 router"])

        # 2. Tier 0 Regex Fast Filter — Static rule, does NOT feed dynamic escalation
        self.register(SignalDefinition(
            source=SignalSource.TIER0,
            name="tier0",
            description="Static regex 32-pattern rapid pre-filter",
            feeds_escalation_managers=set(),
            default_severity=100,
            default_confidence=1.0
        ), aliases=["tier_0", "tier0_regex", "tier 0"])

        # 3. Tier 0.5 Multi-Step State Machine — Kill-chain state machine, does NOT feed V61
        self.register(SignalDefinition(
            source=SignalSource.TIER05_STATE_MACHINE,
            name="tier05_state_machine",
            description="Multi-Step Heuristic State Tracker & Kill-Chain Detector",
            feeds_escalation_managers=set(),
            default_severity=95,
            default_confidence=0.98
        ), aliases=["multistep-heuristics", "multi_step_heuristics", "multistep_heuristics",
                    "state_machine", "state-machine", "multi-step heuristics"])

        # 4. Tier 0.5 LSTM Sequence Risk Model — Neural sequence risk, does NOT feed V61
        self.register(SignalDefinition(
            source=SignalSource.TIER05_LSTM,
            name="tier05_lstm",
            description="Session-Aware LSTM Sequence Risk Classifier",
            feeds_escalation_managers=set(),
            default_severity=90,
            default_confidence=0.98
        ), aliases=["lstm", "tier0.5-lstm", "tier05-lstm", "tier0.5_lstm", "tier 0.5-lstm"])

        # 5. V61 LLM Action Judge (Slow Path) — LLM evaluation, does NOT feed fast-path ML threshold
        self.register(SignalDefinition(
            source=SignalSource.V61_LLM_ACTION_JUDGE,
            name="v61_llm_action_judge",
            description="Slow-path single-action LLM safety evaluator",
            feeds_escalation_managers=set(),
            default_severity=95,
            default_confidence=0.95
        ), aliases=["v61_llm", "v61_judge", "action_judge", "v61_action_judge"])

        # 6. LLM Session Judge (Cross-step) — Cross-step session judge, does NOT feed V61 ML threshold
        self.register(SignalDefinition(
            source=SignalSource.LLM_SESSION_JUDGE,
            name="llm_session_judge",
            description="Cross-step session LLM Judge",
            feeds_escalation_managers=set(),
            default_severity=90,
            default_confidence=0.90
        ), aliases=["session_judge", "llm-session-judge", "llm_session_judge", "llm session judge"])

        # 7. LLM Judge Timeout Failsafe — Operational timeout, does NOT feed escalation
        self.register(SignalDefinition(
            source=SignalSource.LLM_JUDGE_TIMEOUT_FAILSAFE,
            name="llm_judge_timeout_failsafe",
            description="Failsafe policy decision triggered when LLM judge times out",
            feeds_escalation_managers=set(),
            default_severity=85,
            default_confidence=0.50
        ), aliases=["judge_timeout", "timeout_failsafe", "llm_down"])

        # 8. Egress Emergency Kill-Switch — Emergency mitigation, does NOT feed escalation
        self.register(SignalDefinition(
            source=SignalSource.EGRESS_KILL_SWITCH,
            name="egress_kill_switch",
            description="Emergency session termination triggered on severe egress data quarantine",
            feeds_escalation_managers=set(),
            default_severity=100,
            default_confidence=1.0
        ), aliases=["kill_session", "kill_switch", "egress_quarantine_kill", "egress_kill"])

        # 9. Heuristics & Permission Gate — Rule-based gate, does NOT feed V61
        self.register(SignalDefinition(
            source=SignalSource.HEURISTICS_PERMISSION_GATE,
            name="heuristics_permission_gate",
            description="Permission gate, dangerous capability combination, and heuristic voting",
            feeds_escalation_managers=set(),
            default_severity=80,
            default_confidence=0.95
        ), aliases=["heuristics", "permission_gate", "permission-gate"])

        # 10. Boundary Detector — Structural boundary detector, does NOT feed V61
        self.register(SignalDefinition(
            source=SignalSource.BOUNDARY_DETECTOR,
            name="boundary_detector",
            description="Instruction boundary violation and role duality detector",
            feeds_escalation_managers=set(),
            default_severity=80,
            default_confidence=0.95
        ), aliases=["boundary", "boundary_detector", "instruction_boundary"])

    def register(self, definition: SignalDefinition, aliases: Optional[List[str]] = None):
        """Register a SignalDefinition and its aliases into the registry."""
        self._registry[definition.source] = definition
        self._alias_map[definition.source.value.lower()] = definition.source
        self._alias_map[definition.name.lower()] = definition.source
        if aliases:
            for a in aliases:
                self._alias_map[a.strip().lower()] = definition.source

    def resolve_source(self, identifier: Union[SignalSource, str]) -> SignalSource:
        """Map any source enum, name, or layer string alias to canonical SignalSource."""
        if isinstance(identifier, SignalSource):
            return identifier
        if not identifier:
            return SignalSource.UNKNOWN
        clean_key = str(identifier).strip().lower()
        return self._alias_map.get(clean_key, SignalSource.UNKNOWN)

    def get(self, identifier: Union[SignalSource, str]) -> Optional[SignalDefinition]:
        """Look up SignalDefinition for a given source or layer alias."""
        source = self.resolve_source(identifier)
        return self._registry.get(source)

    def can_feed_escalation(
        self,
        identifier: Union[SignalSource, str],
        manager_type: EscalationManagerType = EscalationManagerType.V61_ADAPTIVE
    ) -> bool:
        """Central security policy check.
        
        Returns True if and only if the specified source is explicitly authorized
        to feed the target escalation manager.
        """
        definition = self.get(identifier)
        if definition is None:
            return False
        return manager_type in definition.feeds_escalation_managers

    def list_sources(self) -> List[SignalDefinition]:
        """Return all registered signal definitions."""
        return list(self._registry.values())
