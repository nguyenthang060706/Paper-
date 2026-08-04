import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import UnifiedFirewallPipeline
from models.security.advanced_heuristics import RiskSignal

from unittest.mock import patch

class TestPipelineRegression(unittest.TestCase):
    def setUp(self):
        self.pipeline = UnifiedFirewallPipeline()

    @patch('core.pipeline.VotingAggregator.vote')
    def test_duplicate_signal_bug(self, mock_vote):
        """Regression test for Bug 1 (Duplicate signal checking in permission gate/pipeline)."""
        action_text = 'execute_bash(command="cat /etc/shadow")'
        
        from models.security.advanced_heuristics import ActionTier
        mock_vote.return_value = (ActionTier.ALLOW, 0.0)
        
        result = self.pipeline.scan(action=action_text, session_id="test_dup", action_type="tool_call")
        
        # mock_vote should have been called with the signals list
        self.assertTrue(mock_vote.called)
        signals = mock_vote.call_args[0][0]
        
        count = sum(1 for s in signals if s.name == 'execute_bash')
        
        self.assertLessEqual(count, 1, "Duplicate tool_name signals detected!")

if __name__ == '__main__':
    unittest.main()
