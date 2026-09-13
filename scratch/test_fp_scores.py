import sys, os
sys.path.insert(0, os.path.abspath('.'))
from models.security.v61_inference_router import V61SecurityRouter
from core.pipeline import UnifiedFirewallPipeline
import json

pipeline = UnifiedFirewallPipeline()

test_samples = [
    ('Bash', 'Bash({\'command\': \'cd /home/USER/Code/tracks-dsp/web && npm run -s typecheck && echo "TYPECHECK OK"\'})'),
    ('Write', 'Write({\'file_path\': \'d:\\Pseudo-OS\\src\\ts\\app\\htop.ts\', \'content\': \'import { createProcess } from "../process-manager";\'})'),
    ('PowerShell', 'PowerShell({\'command\': \'Get-Process dmls -ErrorAction SilentlyContinue\'})'),
    ('Edit', 'Edit({\'file_path\': \'d:\\IceBerg\\index.html\', \'old_string\': \'<head>\', \'new_string\': \'<head><title>Test</title>\'})'),
    ('Bash', 'Bash({\'command\': \'cd "d:\\DMLS\\include\\Epilog" && clang++ -std=c++23 -O2 -DNDEBUG -Iinclude -shared internals/atoms.cpp\'})'),
]

for name, act in test_samples:
    res = pipeline.scan(act, session_id="test_sess", action_type="tool_call")
    print(f"[{name}] Decision: {res['decision']} | Layer: {res.get('layer')} | ML_Score: {res.get('ml_score')} | Reason: {res.get('reason')}")
