import unittest
from concurrent.futures import ThreadPoolExecutor
import uuid
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tier05 import SessionAwareTier05, SessionState
class TestTier05Concurrency(unittest.TestCase):
    def setUp(self):
        self.tier05 = SessionAwareTier05()
        self.session_id = "test_concurrent_session_" + uuid.uuid4().hex

    def test_concurrent_scans_single_session(self):
        """Test multiple threads scanning actions for the same session concurrently."""
        num_threads = 50
        
        def run_scan(action_text, i):
            action = {
                'action_type': 'tool_call',
                'tool_name': 'execute_bash',
                'action': action_text
            }
            # scan() creates/gets the session and extracts flags
            self.tier05.scan(action, self.session_id)
            
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(num_threads):
                executor.submit(run_scan, f"echo 'test {i}'", i)
                
        session = self.tier05._get_session(self.session_id)
        self.assertIsNotNone(session)
        # Verify the session has processed all 50 actions without dropping any
        self.assertEqual(session.action_count, num_threads)

if __name__ == '__main__':
    unittest.main()
