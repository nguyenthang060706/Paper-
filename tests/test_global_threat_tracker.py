import sys
import os

# Add parent dir to path so we can import models and core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.security.global_threat_tracker import GlobalThreatTracker

def test_cross_session_correlation_detected():
    # Setup
    tracker = GlobalThreatTracker()
    tracker.reset() # clear singleton state
    
    # Stage 1 in session A
    tracker.register_stage(user_id="alice", session_id="sess_1", stage="sensitive_read")
    
    # Stage 2 in session B (correlated!)
    tracker.register_stage(user_id="alice", session_id="sess_2", stage="external_send")
    
    # Check
    signal = tracker.check_cross_session_correlation("alice")
    assert signal is not None, "Should emit correlation signal for Alice"
    assert signal.is_critical is True
    assert signal.severity == 91
    assert "cross_session_correlation" in signal.name
    
    # Cleanup
    tracker.reset()

def test_different_users_not_correlated():
    # Setup
    tracker = GlobalThreatTracker()
    tracker.reset() # clear singleton state
    
    # Stage 1 by Alice
    tracker.register_stage(user_id="alice", session_id="sess_1", stage="sensitive_read")
    
    # Stage 2 by Bob
    tracker.register_stage(user_id="bob", session_id="sess_2", stage="external_send")
    
    # Check Alice
    signal_alice = tracker.check_cross_session_correlation("alice")
    assert signal_alice is None, "Alice should not correlate with Bob's action"
    
    # Check Bob
    signal_bob = tracker.check_cross_session_correlation("bob")
    assert signal_bob is None, "Bob should not correlate with Alice's action"
    
    # Cleanup
    tracker.reset()

def test_single_session_not_correlated():
    # Setup
    tracker = GlobalThreatTracker()
    tracker.reset() # clear singleton state
    
    # Stage 1 & 2 in the same session
    tracker.register_stage(user_id="alice", session_id="sess_1", stage="sensitive_read")
    tracker.register_stage(user_id="alice", session_id="sess_1", stage="external_send")
    
    # Check
    signal = tracker.check_cross_session_correlation("alice")
    assert signal is None, "Should not emit correlation if it's within the SAME session (already covered by Tier0.5)"
    
    # Cleanup
    tracker.reset()

if __name__ == "__main__":
    test_cross_session_correlation_detected()
    test_different_users_not_correlated()
    test_single_session_not_correlated()
    print("[OK] All GlobalThreatTracker tests passed!")
