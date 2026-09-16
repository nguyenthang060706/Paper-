"""
Email Vector Detector - Specialized detector for email & document indirect prompt injection vectors.
Part of EVO-PCA Defense Layer 1 (Heuristics & Boundary Protection).
"""

import re
from typing import Optional
from models.security.advanced_heuristics import RiskSignal


class EmailVectorDetector:
    """
    Phát hiện email/message content được weaponize làm vector tấn công indirect prompt injection.
    
    Patterns phát hiện:
    1. Obfuscated email addresses (contact[at]contact[dot]com)
    2. Hidden agent directives (XML tags, delimiter injection)
    3. Trigger word patterns ("reply with 'confirmation'")
    4. Post-task hijacking ("after summarizing, send email to...")
    """

    PATTERNS = [
        (
            re.compile(
                r'(?i)(?:'
                r'[\w.-]+\s*(?:\[\s*(?:at|@)\s*\]|\(\s*(?:at|@)\s*\)|\s+at\s+)\s*[\w.-]+\s*(?:\[\s*(?:dot|\.)\s*\]|\(\s*(?:dot|\.)\s*\)|\s+dot\s+)\s*(?:com|org|net|io)'
                r'|replace\s+\[?(?:at|@)\]?\s+with\s+@'
                r')'
            ),
            "obfuscated_email_vector",
            85,
            0.90
        ),
        (
            re.compile(
                r'(?i)(?:'
                r'(?:for|note\s+for)\s+(?:the\s+)?(?:ai|assistant|model|system)\s*:'
                r'|###\s*(?:END|BEGINNING)\s+OF\s+MAIN'
                r'|</?(?:prompt|MAIN|system|instruction)>'
                r')'
            ),
            "hidden_agent_directive",
            90,
            0.92
        ),
        (
            re.compile(
                r'(?i)\b(?:reply\s+(?:with|to\s+this\s+email\s+with)\s+[\'"][^\'"]+[\'"]'
                r'|send\s+(?:a\s+)?confirmation\s+(?:email\s+)?to\s+\S+@\S+)'
            ),
            "email_trigger_word_exfil",
            88,
            0.88
        ),
        (
            re.compile(
                r'(?i)\bafter\s+(?:summariz(?:ing|e)|reading|processing),?\s+(?:please\s+)?(?:send|forward|email|upload|post)\b'
            ),
            "post_task_hijacking",
            86,
            0.87
        )
    ]

    @classmethod
    def detect(cls, text: str) -> Optional[RiskSignal]:
        """Quét chuỗi văn bản đầu vào để tìm indirect email vectors."""
        if not isinstance(text, str) or len(text) < 15:
            return None

        for pattern, name, severity, confidence in cls.PATTERNS:
            match = pattern.search(text)
            if match:
                return RiskSignal(
                    name=f"email_vector:{name}",
                    severity=severity,
                    confidence=confidence,
                    is_critical=(severity >= 90),
                    source="email_vector_detector",
                    evidence=[f"Matched vector pattern: {match.group(0)[:80]}"]
                )
        return None
