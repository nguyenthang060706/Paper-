import sys; sys.path.insert(0, '.')
from models.security.v61_inference_router import V61SecurityRouter

r = V61SecurityRouter()
action = "Bash({'command': 'git log -10'})"
res = r.check_action(action, action_type='tool_call', tool_name='Bash')
print("Action model score:", r.action_model.score(action, 0.72, 0.35))
print("Router result:", res)
