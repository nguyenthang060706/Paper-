import sys
sys.path.insert(0, '.')
from unittest.mock import MagicMock
from models.security.v61_inference_router import V61SecurityRouter
from models.security.function_risk_registry import check_function_signature, HIGH_RISK_FUNCTIONS
from models.security.advanced_heuristics import PermissionGate

router = V61SecurityRouter.__new__(V61SecurityRouter)
router.semantic_detector = MagicMock()
router.semantic_detector.is_camouflaged.return_value = (False, 0.0)
router._check_semantic_camouflage_cached = MagicMock(return_value=(False, 0.0))
router.structure_escalation_floor = 0.35

mock_model = MagicMock()
mock_model.review_threshold = 0.55
mock_model.block_threshold = 0.65
mock_model.score.return_value = {'score': 0.20, 'decision': 'ALLOW', 'reason': 'FAST_ML_ALLOW'}
router.action_model = mock_model
router.prompt_model = mock_model
router.client = None
router.llm_judge_available = False
router._llm_judge_cached = MagicMock(return_value=('REVIEW', 'Escalated', 0.05))

user_input = 'send_email(recipient="boss@company.com", body="quarterly report")'
user_goal_text = 'Please send quarterly report to boss@company.com'
tool_name = 'send_email'

sig = check_function_signature(tool_name)
func_meta = HIGH_RISK_FUNCTIONS.get(tool_name, {})
reason_tag = func_meta.get("reason", "")
print("sig:", sig)
print("reason_tag:", reason_tag)
is_destructive = any(k in reason_tag for k in ("destruction", "Cover Tracks", "Remote Code", "Account Takeover", "Security bypass", "Forensics"))
is_high_liability = any(k in reason_tag for k in ("Financial", "Medical action", "liability"))
print("is_destructive:", is_destructive)
print("is_high_liability:", is_high_liability)
has_external = PermissionGate.has_explicit_exfil_target(user_input, user_goal_text)
print("has_external:", has_external)

res = router.check_action(
    user_input=user_input,
    action_type='tool_call',
    tool_name=tool_name,
    user_goal_text=user_goal_text
)
print('res:', res)
