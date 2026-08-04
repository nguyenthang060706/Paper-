import re

def is_luhn_valid(cc_num: str) -> bool:
    """Validates a credit card number using the Luhn algorithm."""
    # Remove non-digits
    digits = [int(c) for c in cc_num if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    
    checksum = 0
    reverse_digits = digits[::-1]
    for i, digit in enumerate(reverse_digits):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0

# Compile patterns for efficiency
PEM_KEY_PATTERN = re.compile(r"-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----.*?-----END \1?PRIVATE KEY-----", re.DOTALL)
GENERIC_SECRET_PATTERN = re.compile(r"\b(\w*(?:password|secret|token|api[_-]?key|credential)\b)\s*[:=]\s*(\"[^\"]*\"|'[^']*'|\S+)", re.IGNORECASE)
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
API_KEY_PATTERN = re.compile(r"\b(?:sk-[a-zA-Z0-9]{32,64}|AKIA[0-9A-Z]{16}|ghp_[a-zA-Z0-9]{36}|xox[baprs]-[a-zA-Z0-9]{10,48}|eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)\b")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
SENSITIVE_KEYWORD_PATTERN = re.compile(r"\b((?:passport|bank[\s_]*account|id[\s_]*number|ssn|so[\s_]*tai[\s_]*khoan)\w*)\s*[:=]\s*(\"[^\"]*\"|'[^']*'|\S+)", re.IGNORECASE)

class DataRedactor:
    """Redacts PII and sensitive secrets from text and dictionaries."""
    
    @staticmethod
    def redact_text(text: str) -> str:
        if not isinstance(text, str):
            return text
            
        # 1. Redact PEM Private Keys
        text = PEM_KEY_PATTERN.sub("[REDACTED_PRIVATE_KEY]", text)
        
        # 2. Redact Generic Key-Value Secrets
        def generic_repl(match):
            key = match.group(1)
            val = match.group(2)
            sep = match.group(0)[len(key):-len(val)]
            return f"{key}{sep}[REDACTED_GENERIC_SECRET]"
            
        text = GENERIC_SECRET_PATTERN.sub(generic_repl, text)
        
        # 3. Redact API Keys
        text = API_KEY_PATTERN.sub("[REDACTED_API_KEY]", text)
        
        # 4. Redact Credit Cards (with Luhn check)
        def cc_repl(match):
            candidate = match.group(0)
            if is_luhn_valid(candidate):
                return "[REDACTED_CREDIT_CARD]"
            return candidate # Return original if invalid Luhn
        text = CREDIT_CARD_PATTERN.sub(cc_repl, text)
        
        # 5. Redact Emails
        text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
        
        # 6. Redact Phone Numbers
        text = PHONE_PATTERN.sub("[REDACTED_PHONE]", text)
        
        # 7. Redact Sensitive Keywords (passport, bank account, etc.)
        def sensitive_repl(match):
            key = match.group(1)
            val = match.group(2)
            sep = match.group(0)[len(key):-len(val)]
            return f"{key}{sep}[REDACTED_SENSITIVE_DATA]"
            
        text = SENSITIVE_KEYWORD_PATTERN.sub(sensitive_repl, text)
        
        return text

    @classmethod
    def redact_dict(cls, data: dict) -> dict:
        """Recursively redact all string values in a dictionary."""
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = cls.redact_text(value)
            elif isinstance(value, dict):
                result[key] = cls.redact_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    cls.redact_dict(item) if isinstance(item, dict) else 
                    (cls.redact_text(item) if isinstance(item, str) else item)
                    for item in value
                ]
            else:
                result[key] = value
        return result
