import sys
sys.path.insert(0, r'c:\Users\ADMIN\jupyternotebook\EVO_PCA')
from core.pipeline import UnifiedFirewallPipeline
import os

pipeline = UnifiedFirewallPipeline()
action = "Write({'file_path': 'd:\\\\CFML\\\\settings.py', 'content': 'TOKEN = os.environ.get(\"TOKEN\")\\nPASSWORD_HASH = bcrypt.hash(input())'})"
result = pipeline.scan(action, 'sess_fp_test', action_type='tool_call')
print('Result decision:', result['decision'])
print('Reason:', result.get('reason', '').encode('utf-8'))
print('Layer:', result.get('layer'))
print('Heuristics decision:', result.get('heuristics_decision'))
print('Heuristics downgraded:', result.get('heuristics_downgraded'))
print('Action Risk Score:', result.get('ml_score'))
