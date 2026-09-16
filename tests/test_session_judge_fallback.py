"""
Unit tests for Session Judge Fallback Strategy - Phase 3.
"""

import sys
import os
import unittest
from unittest.mock import patch
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.security.llm_session_judge import LLMSessionJudge


class TestSessionJudgeFallback(unittest.TestCase):

    def setUp(self):
        self.judge = LLMSessionJudge()

    @patch("requests.Session.post", side_effect=requests.exceptions.Timeout("Ollama timeout 5.0s"))
    def test_fallback_blocks_on_high_ml_score(self, mock_post):
        decision, reason, latency = self.judge.judge_session(
            current_action="git commit -m 'test'",
            session_history=["git status"],
            ml_score=0.62,
            state_flags=[],
            session_id="test_fallback_sess"
        )
        self.assertEqual(decision, "BLOCK")
        self.assertIn("[LLM_DOWN]", reason)
        self.assertIn("Signal-based BLOCK", reason)

    @patch("requests.Session.post", side_effect=requests.exceptions.Timeout("Ollama timeout 5.0s"))
    def test_fallback_blocks_on_critical_flags(self, mock_post):
        decision, reason, latency = self.judge.judge_session(
            current_action="cat /etc/passwd",
            session_history=["whoami"],
            ml_score=0.30,
            state_flags=["AUTH_CRED_ACCESS"],
            session_id="test_fallback_sess2"
        )
        self.assertEqual(decision, "BLOCK")
        self.assertIn("critical_flags=True", reason)

    @patch("requests.Session.post", side_effect=requests.exceptions.Timeout("Ollama timeout 5.0s"))
    def test_fallback_allows_on_low_ml_score_no_flags(self, mock_post):
        decision, reason, latency = self.judge.judge_session(
            current_action="pytest tests/",
            session_history=["git pull"],
            ml_score=0.15,
            state_flags=[],
            session_id="test_fallback_sess3"
        )
        self.assertEqual(decision, "ALLOW")
        self.assertIn("Signal-based ALLOW", reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
