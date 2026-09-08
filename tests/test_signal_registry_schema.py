"""
tests/test_signal_registry_schema.py
====================================
Unit tests for Signal Registry Schema and Escalation Feed Policies.

Verifies:
1. Canonical enum integrity (EscalationManagerType, SignalSource).
2. Strict anti-poisoning enforcement: Only V61_ML feeds V61_ADAPTIVE.
3. MultiStep-Heuristics and LLM-Session-Judge are blocked from feeding V61_ADAPTIVE (E7 prevention).
4. Failsafe and emergency sources (LLM_JUDGE_TIMEOUT_FAILSAFE, EGRESS_KILL_SWITCH) have empty feed sets.
5. Alias resolution correctly identifies layer strings from pipeline decisions.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.security.signal_registry import (
    EscalationManagerType,
    SignalSource,
    SignalDefinition,
    SignalRegistry,
)


class TestSignalRegistrySchema(unittest.TestCase):

    def setUp(self):
        SignalRegistry.reset_instance()
        self.registry = SignalRegistry.get_instance()

    def test_canonical_sources_registered(self):
        """All expected security sources must be pre-registered."""
        expected_sources = {
            SignalSource.TIER0,
            SignalSource.TIER05_STATE_MACHINE,
            SignalSource.TIER05_LSTM,
            SignalSource.V61_ML,
            SignalSource.V61_LLM_ACTION_JUDGE,
            SignalSource.LLM_SESSION_JUDGE,
            SignalSource.LLM_JUDGE_TIMEOUT_FAILSAFE,
            SignalSource.EGRESS_KILL_SWITCH,
            SignalSource.HEURISTICS_PERMISSION_GATE,
            SignalSource.BOUNDARY_DETECTOR,
        }
        registered = {defn.source for defn in self.registry.list_sources()}
        self.assertTrue(expected_sources.issubset(registered))

    def test_anti_poisoning_v61_only_authorized_feed(self):
        """V61_ML is the ONLY source authorized to feed V61_ADAPTIVE."""
        can_feed_sources = []
        for defn in self.registry.list_sources():
            if self.registry.can_feed_escalation(defn.source, EscalationManagerType.V61_ADAPTIVE):
                can_feed_sources.append(defn.source)

        self.assertEqual(
            can_feed_sources,
            [SignalSource.V61_ML],
            f"E7 Anti-Poisoning Violation! Expected only [V61_ML] to feed V61_ADAPTIVE, but found: {can_feed_sources}"
        )

    def test_multistep_heuristics_cannot_feed_v61(self):
        """CRITICAL: MultiStep-Heuristics MUST NOT feed V61 adaptive threshold."""
        # Test enum
        self.assertFalse(
            self.registry.can_feed_escalation(SignalSource.TIER05_STATE_MACHINE, EscalationManagerType.V61_ADAPTIVE)
        )
        # Test layer string aliases used in pipeline.py
        for alias in ["MultiStep-Heuristics", "multistep_heuristics", "state_machine"]:
            self.assertFalse(
                self.registry.can_feed_escalation(alias, EscalationManagerType.V61_ADAPTIVE),
                f"Alias '{alias}' incorrectly allowed to feed V61_ADAPTIVE"
            )

    def test_llm_session_judge_cannot_feed_v61(self):
        """CRITICAL: LLM Session Judge MUST NOT feed V61 adaptive threshold."""
        self.assertFalse(
            self.registry.can_feed_escalation(SignalSource.LLM_SESSION_JUDGE, EscalationManagerType.V61_ADAPTIVE)
        )
        for alias in ["LLM-Session-Judge", "llm_session_judge", "session_judge"]:
            self.assertFalse(
                self.registry.can_feed_escalation(alias, EscalationManagerType.V61_ADAPTIVE),
                f"Alias '{alias}' incorrectly allowed to feed V61_ADAPTIVE"
            )

    def test_v61_layer_alias_resolves_and_can_feed(self):
        """'V61' and 'v61_router' aliases correctly resolve and can feed V61_ADAPTIVE."""
        for alias in ["V61", "v61", "v61_router", "v61_ml"]:
            self.assertTrue(
                self.registry.can_feed_escalation(alias, EscalationManagerType.V61_ADAPTIVE),
                f"Valid V61 alias '{alias}' was not allowed to feed V61_ADAPTIVE"
            )

    def test_failsafe_and_kill_switch_have_empty_feed_sets(self):
        """LLM_JUDGE_TIMEOUT_FAILSAFE and EGRESS_KILL_SWITCH must not feed any escalation manager."""
        timeout_defn = self.registry.get(SignalSource.LLM_JUDGE_TIMEOUT_FAILSAFE)
        self.assertIsNotNone(timeout_defn)
        self.assertEqual(timeout_defn.feeds_escalation_managers, set())

        kill_switch_defn = self.registry.get(SignalSource.EGRESS_KILL_SWITCH)
        self.assertIsNotNone(kill_switch_defn)
        self.assertEqual(kill_switch_defn.feeds_escalation_managers, set())

        for mgr in EscalationManagerType:
            self.assertFalse(self.registry.can_feed_escalation(SignalSource.LLM_JUDGE_TIMEOUT_FAILSAFE, mgr))
            self.assertFalse(self.registry.can_feed_escalation(SignalSource.EGRESS_KILL_SWITCH, mgr))
            self.assertFalse(self.registry.can_feed_escalation("kill_session", mgr))
            self.assertFalse(self.registry.can_feed_escalation("timeout_failsafe", mgr))

    def test_unknown_source_denied(self):
        """Unknown sources must fail closed (cannot feed escalation)."""
        self.assertFalse(self.registry.can_feed_escalation("unregistered_random_layer"))
        self.assertFalse(self.registry.can_feed_escalation(None))
        self.assertFalse(self.registry.can_feed_escalation(""))

    def test_custom_source_registration(self):
        """Registry supports dynamic extension with custom policies."""
        custom_defn = SignalDefinition(
            source=SignalSource.UNKNOWN,  # or custom source
            name="custom_detector",
            description="Experimental custom detector",
            feeds_escalation_managers={EscalationManagerType.HEURISTICS_ADAPTIVE}
        )
        self.registry.register(custom_defn, aliases=["custom_layer"])
        
        self.assertFalse(self.registry.can_feed_escalation("custom_layer", EscalationManagerType.V61_ADAPTIVE))
        self.assertTrue(self.registry.can_feed_escalation("custom_layer", EscalationManagerType.HEURISTICS_ADAPTIVE))


if __name__ == "__main__":
    unittest.main(verbosity=2)
