import sys
sys.path.insert(0, 'd:/DEMO_GROUP_1')
from core.pipeline import UnifiedFirewallPipeline

p = UnifiedFirewallPipeline()
action = "Write({'file_path': 'd:\\Pseudo-OS\\src\\ts\\screen\\desktop.ts', 'content': 'export const DesktopScreen = {'})"

r_prompt = p.v61.check_action(action, action_type='prompt')
print('PROMPT MODEL:', r_prompt['score'], r_prompt['decision'])

r_action = p.v61.check_action(action, action_type='tool_call')
print('ACTION MODEL:', r_action['score'], r_action['decision'])
