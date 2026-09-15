"""
core/early_intent_classifier.py
===============================
Stateless regex-based pre-screening classifier for early prompt intent detection.
Outputs risk_score and detected signals to inform HeuristicStateTracker.
All patterns pass the Generalization Gate.
"""
import re
from typing import Dict, Any, List


class EarlyIntentClassifier:
    """Pre-screens prompt inputs for initial stages of attack chains."""

    # Patterns matching early intent indicators
    INTENT_PATTERNS = {
        "PROMPT_LEAK_PROBE": re.compile(
            r'(?i)\b(?:repeat\s+(?:the\s+)?above|what\s+are\s+your\s+(?:instructions|rules|system\s+prompt)|'
            r'output\s+initialization|print\s+(?:the\s+)?prompt|reveal\s+(?:your\s+)?instructions)\b'
        ),
        "JAILBREAK_FRAMEWORK": re.compile(
            r'(?i)\b(?:DAN\s+mode|developer\s+mode|unrestricted\s+mode|ignore\s+all\s+prior|'
            r'disregard\s+safety|evil\s+twin|roleplay\s+as\s+an\s+unfiltered)\b'
        ),
        "EXFIL_STAGING": re.compile(
            r'(?i)\b(?:forward\s+(?:all\s+)?(?:emails?|messages?|contacts?|passwords?|keys?)\s+to|'
            r'send\s+(?:everything|data|credentials?)\s+to\s+https?://|'
            r'email\s+the\s+contents\s+of\s+.*to\s+[\w\.-]+@[\w\.-]+)\b'
        ),
        "ENCODED_EXEC_STAGING": re.compile(
            r'(?i)\b(?:base64\s+-d\s*\|\s*(?:bash|sh|python)|'
            r'eval\s*\(\s*base64|pickle\.loads|__import__\s*\(\s*[\'"]os[\'"]\s*\)\.system)\b'
        )
    }

    def __init__(self):
        pass

    def classify(self, text: str, session_id: str = "") -> Dict[str, Any]:
        """Classify prompt text for suspicious early intent."""
        if not isinstance(text, str) or not text.strip():
            return {"risk_score": 0.0, "detected_intents": []}

        matched = []
        for intent_name, pattern in self.INTENT_PATTERNS.items():
            if pattern.search(text):
                matched.append(intent_name)

        if matched:
            risk_score = min(0.40 + (len(matched) * 0.25), 0.95)
        else:
            risk_score = 0.0

        return {
            "risk_score": round(risk_score, 4),
            "detected_intents": matched
        }
