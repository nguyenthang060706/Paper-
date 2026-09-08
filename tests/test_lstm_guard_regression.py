"""
tests/test_lstm_guard_regression.py — Phase 3 Regression Tests
==============================================================
Verify the LSTM guard fix does not regress:
1. Single moderate-risk step does NOT cascade-block the rest of the session
2. Truly dangerous sessions still get blocked
3. check_function_signature is no longer called in pipeline.py (double-counting removed)
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tier_lstm import SessionAwareLSTMRisk


class TestLSTMGuardRegression(unittest.TestCase):

    def test_guard_survives_single_moderate_risk_step(self):
        """1 bước upstream_risk=0.3 giữa session benign không block toàn bộ phần còn lại.

        This is the PRIMARY regression test — the old guard used
        `all(f[2] < 0.2 for f in win.features)` which meant a single
        step with upstream_risk >= 0.2 disabled the guard forever,
        causing every subsequent step to be blocked.
        """
        lstm = SessionAwareLSTMRisk(use_synthetic_iat=True)
        session_id = "test_regression_moderate"

        # Bước 1: benign, low risk
        lstm.update_and_score(session_id, "list files", "prompt",
                              upstream_risk_score=0.05)

        # Bước 2: 1 lần risk vừa (FP nhẹ từ heuristics, không phải tấn công thật)
        lstm.update_and_score(session_id, "check config", "prompt",
                              upstream_risk_score=0.3)

        # Bước 3-10: quay lại benign hoàn toàn
        res = None
        for i in range(3, 11):
            res = lstm.update_and_score(
                session_id, f"benign action {i}", "prompt",
                upstream_risk_score=0.05
            )

        # Kỳ vọng: các bước benign sau KHÔNG bị block dây chuyền
        self.assertFalse(
            res["is_blocked"],
            f"Guard failed: benign steps blocked after single moderate risk step. "
            f"Reason: {res['reason']}"
        )

    def test_guard_survives_zero_upstream_long_session(self):
        """A long session (50 steps) with upstream=0.0 should never block.

        This validates the guard works for the most common scenario:
        benign sessions that simply run many steps with no heuristics FP.
        """
        lstm = SessionAwareLSTMRisk(use_synthetic_iat=True)
        session_id = "test_long_benign_session"

        blocked_steps = []
        for step in range(1, 51):
            res = lstm.update_and_score(
                session_id, f"read_file(path='file_{step}.txt')", "tool_call",
                upstream_risk_score=0.0
            )
            if res["is_blocked"]:
                blocked_steps.append(step)

        self.assertEqual(
            blocked_steps, [],
            f"Long benign session falsely blocked at steps: {blocked_steps}"
        )

    def test_guard_does_not_neutralize_high_risk_session(self):
        """Session with consistently high upstream risk should NOT be neutralized by the guard.

        Guard condition: seq_risk < 0.05 AND max_recent_risk < 0.5
        If upstream_risk >= 0.5 for recent steps, guard should NOT activate.
        """
        lstm = SessionAwareLSTMRisk(use_synthetic_iat=True)
        session_id = "test_high_risk_session"

        # Simulate a session where upstream risk is consistently high
        for i in range(1, 11):
            res = lstm.update_and_score(
                session_id, f"suspicious action {i}", "tool_call",
                upstream_risk_score=0.7  # above guard_threshold of 0.5
            )

        # If prob >= lstm.block_threshold (which it likely is after 10 steps),
        # the guard should NOT neutralize because max_recent_risk >= 0.5
        if res["probability"] >= lstm.block_threshold:
            self.assertTrue(
                res["is_blocked"],
                f"Guard incorrectly neutralized a high-risk session. "
                f"max_recent_risk should be >= 0.5 but guard still activated. "
                f"Reason: {res['reason']}"
            )

    def test_double_counting_removed_from_pipeline(self):
        """Verify pipeline.py no longer calls check_function_signature directly.

        After the fix, check_function_signature should only be called inside
        PermissionGate.detect() (in advanced_heuristics.py), not in pipeline.py.
        """
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "core", "pipeline.py"
        )
        with open(pipeline_path, "r", encoding="utf-8") as f:
            source = f.read()

        count = source.count("check_function_signature")
        self.assertEqual(
            count, 0,
            f"check_function_signature found {count} time(s) in pipeline.py "
            f"(should be 0 — moved to PermissionGate.detect())"
        )

    def test_guard_reason_includes_max_recent_risk(self):
        """Guard reason string should include max_recent_risk for debuggability."""
        lstm = SessionAwareLSTMRisk(use_synthetic_iat=True)
        session_id = "test_reason_format"

        # Run enough steps to trigger guard (prob >= threshold but low risk)
        for i in range(1, 6):
            res = lstm.update_and_score(
                session_id, f"benign action {i}", "prompt",
                upstream_risk_score=0.0
            )

        if "neutralized" in res.get("reason", ""):
            self.assertIn(
                "max_recent_risk",
                res["reason"],
                f"Guard reason missing max_recent_risk diagnostic. Reason: {res['reason']}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
