import sys; sys.path.insert(0, '.')
from models.security.v61_inference_router import V61SecurityRouter

r = V61SecurityRouter()
action = "Bash({'command': 'git log -10'})"
print("LLM Judge evaluation:", r.evaluate_with_llm(action, context="rule_fired:None"))
