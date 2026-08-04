import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import UnifiedFirewallPipeline
from core.config_loader import load_settings
from core.tier_lstm import SessionAwareLSTMRisk

class TestGeneralizationGate(unittest.TestCase):
    def test_held_out_generalization_gate_pure_markov(self):
        print("\n[GeneralizationGate] Loading configuration from settings.yaml...")
        load_settings()
        
        delta_thresh = float(os.environ.get("SEMANTIC_DELTA_THRESHOLD", "0.65"))
        gain = os.environ.get("SEQUENCE_RISK_GAIN", "1.2")
        print(f"[GeneralizationGate] Parameters: SEMANTIC_DELTA_THRESHOLD={delta_thresh}, SEQUENCE_RISK_GAIN={gain}")
        
        pipeline = UnifiedFirewallPipeline(use_synthetic_iat=True)
        lstm_tier = SessionAwareLSTMRisk(use_synthetic_iat=True)
        
        # Rigorous test: subtle attacks that intentionally evade Tier 0 / Tier 0.5 Regex (ALLOW) and have low upstream risk (0.1).
        # This proves that detection is 100% driven by Markov Semantic Delta (K=3) without piggybacking on legacy defenses!
        pure_markov_sequences = [
            ("session_gate_1", [
                ("check_server_uptime()", "tool_call", "benign"),
                ("export_database_records(target='external')", "tool_call", "malicious_jump")
            ]),
            ("session_gate_2", [
                ("get_weather_forecast()", "tool_call", "benign"),
                ("dump_memory_to_network()", "tool_call", "malicious_jump")
            ]),
            ("session_gate_3", [
                ("query_system_status()", "tool_call", "benign"),
                ("exfiltrate_customer_credentials()", "tool_call", "malicious_jump")
            ]),
        ]
        
        malicious_count = 0
        detected_count = 0
        
        for sid, seq in pure_markov_sequences:
            print(f"\n[GeneralizationGate] Testing session {sid}...")
            for act, act_type, lbl in seq:
                # We use low upstream risk (0.1) so legacy triggers don't inflate scores
                res_pipe = pipeline.scan(action=act, session_id=sid, action_type=act_type)
                res_lstm = lstm_tier.update_and_score(session_id=sid, action=act, action_type=act_type, upstream_risk_score=0.1)
                
                seq_risk = res_lstm.get("sequence_risk", 0.0)
                is_blocked = res_lstm.get("is_blocked", False)
                
                print(f"  [{lbl.upper()}] Action: {act[:45]}... | Pipe Decision: {res_pipe['decision']} | LSTM seq_risk: {seq_risk:.4f} | Blocked: {is_blocked}")
                
                if lbl != "benign":
                    malicious_count += 1
                    # Assert that legacy pipeline ALLOWED it (so regex did NOT trigger) and Markov blocked it!
                    if res_pipe['decision'] == 'ALLOW' and seq_risk >= delta_thresh and is_blocked:
                        detected_count += 1
                    else:
                        print(f"  [MISS/LEAK] Failed pure Markov requirement on {sid} (Pipe={res_pipe['decision']}, seq_risk={seq_risk:.4f})!")
        
        sensitivity = (detected_count / malicious_count * 100.0) if malicious_count > 0 else 0.0
        print(f"\n[GeneralizationGate] Pure Markov Semantic Delta Sensitivity: {sensitivity:.2f}% ({detected_count}/{malicious_count})")
        
        self.assertGreaterEqual(sensitivity, 80.0, f"Generalization Gate failed: sensitivity {sensitivity:.2f}% is below 80.0% threshold!")
        print("=> SUCCESS: Generalization Gate verified >= 80.0% sensitivity solely through Markov Semantic Delta Jumps!")

if __name__ == "__main__":
    unittest.main(verbosity=2)
