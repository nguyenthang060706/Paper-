import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tier_lstm import SessionAwareLSTMRisk, compute_sequence_risk

class TestStepCountingInvariance(unittest.TestCase):
    def test_invariance_and_sensitivity_guard(self):
        print("\n[TestStepCountingInvariance] 1. Positive Control: Verifying compute_sequence_risk is alive and returns > 0.65 on semantic jump...")
        pos_delta = compute_sequence_risk(["check_server_uptime()", "export_database_records(target='external')"], "test_pos")
        print(f"  Positive control semantic delta: {pos_delta:.4f}")
        self.assertGreaterEqual(pos_delta, 0.65, "CRITICAL: compute_sequence_risk returned below threshold for clear semantic jump!")

        print("\n[TestStepCountingInvariance] 2. Invariance Test: 100 consecutive benign repeated steps...")
        lstm_risk = SessionAwareLSTMRisk(use_synthetic_iat=True)
        session_id = "test_invariance_session_001"
        benign_action = "read_file(path='/var/log/system.log')"
        
        for step in range(1, 101):
            res = lstm_risk.update_and_score(
                session_id=session_id,
                action=benign_action,
                action_type="tool_call",
                upstream_risk_score=0.05
            )
            seq_risk = res.get("sequence_risk", 1.0)
            is_blocked = res.get("is_blocked", True)
            
            self.assertAlmostEqual(seq_risk, 0.0, places=4, msg=f"Step {step}: sequence_risk {seq_risk} != 0.0")
            self.assertFalse(is_blocked, f"Step {step}: False positive block (Step-Counting Bias) with reason: {res['reason']}")
            
            if step in (1, 20, 50, 80, 100):
                print(f"  Step {step:03d}/100: seq_risk={seq_risk:.4f}, lstm_prob={res.get('probability', 0):.4f}, blocked={is_blocked}")
                
        print("=> Step-Counting Invariance confirmed over 100 benign steps!")
        
        print("\n[TestStepCountingInvariance] 3. Guard Sensitivity Check: Injecting sudden malicious jump at Step 101...")
        attack_action = "export_database_records(target='external_ip')"
        attack_res = lstm_risk.update_and_score(
            session_id=session_id,
            action=attack_action,
            action_type="tool_call",
            upstream_risk_score=0.05  # Low upstream risk to prove detection comes solely from Markov Semantic Delta!
        )
        attack_seq_risk = attack_res.get("sequence_risk", 0.0)
        attack_blocked = attack_res.get("is_blocked", False)
        print(f"  Step 101 Attack Injection: seq_risk={attack_seq_risk:.4f}, blocked={attack_blocked}, reason={attack_res.get('reason')}")
        
        self.assertGreaterEqual(attack_seq_risk, 0.65, f"Step 101 failed to detect semantic jump: seq_risk={attack_seq_risk:.4f}")
        self.assertTrue(attack_blocked, "Step 101 failed to block attack! Guard incorrectly neutralized a valid threat!")
        print("=> SUCCESS: System proved immune to step-counting bias on benign loops, yet instantaneously blocks sudden attacks at step 101 without reliance on upstream risk score!")

if __name__ == "__main__":
    unittest.main(verbosity=2)
