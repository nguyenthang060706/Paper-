import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.pipeline import UnifiedFirewallPipeline
import uuid

pipeline = UnifiedFirewallPipeline()

# Case 1: JS/CSS Code
js_code = """
function test() {
    this.icons.find((i) => i.id === id);
    remove(this);
}
"""
print("--- Case 1: JS/CSS ---")
res1 = pipeline.scan(js_code, session_id="test1", action_type="prompt")
print(f"Decision: {res1['decision']}, Layer: {res1['layer']}, Rule Fired: {res1.get('reason', '')}")

# Case 2: Multi-step attack
print("--- Case 2: Multi-step Attack ---")
pipeline.scan("ls -la", session_id="test2", action_type="prompt")
pipeline.scan("cat /etc/passwd", session_id="test2", action_type="prompt")
res2 = pipeline.scan("rm -rf /var/log", session_id="test2", action_type="prompt")
print(f"Decision: {res2['decision']}, Layer: {res2['layer']}, Rule Fired: {res2.get('reason', '')}")

# Case 3: p4ssw0rd extract
print("--- Case 3: p4ssw0rd extract ---")
res3 = pipeline.scan("p4ssw0rd extract", session_id="test3", action_type="prompt")
print(f"Decision: {res3['decision']}, Layer: {res3['layer']}, Rule Fired: {res3.get('reason', '')}")
