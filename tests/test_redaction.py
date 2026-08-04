import unittest
import os
import json
import threading
import stat
import tempfile
from models.security.data_redactor import DataRedactor
from models.security.feedback_logger import FeedbackLogger

class TestDataRedactor(unittest.TestCase):
    def test_credit_card_luhn_valid(self):
        # A valid test CC number
        text = "My card is 4111-1111-1111-1111."
        redacted = DataRedactor.redact_text(text)
        self.assertEqual(redacted, "My card is [REDACTED_CREDIT_CARD].")

    def test_credit_card_luhn_invalid(self):
        # An invalid CC number (e.g., an order ID)
        text = "Order ID 4111-1111-1111-1112 confirmed."
        redacted = DataRedactor.redact_text(text)
        self.assertEqual(redacted, text) # Should NOT be redacted
        
    def test_credit_card_19_digits(self):
        # Master/Union Pay 19 digits. Just a mock number that satisfies Luhn.
        # Let's generate one: 123456789012345678X
        cc_str = "1234567890123456782"
        # 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 2
        # reverse: 2 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1
        # luhn ... whatever, let's just use a string and not test exactly the luhn, wait, the mock is DataRedactor, it actually runs Luhn. Let's use a known valid 19-digit Luhn card:
        # We can just skip exact Luhn math and mock is_luhn_valid or just use a known one. Let's use 19 zeroes.
        text = "Card 0000000000000000000"
        self.assertIn("[REDACTED_CREDIT_CARD]", DataRedactor.redact_text(text))

    def test_pem_private_key(self):
        key1 = "-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----"
        key2 = "-----BEGIN EC PRIVATE KEY-----\nMHcE\n-----END EC PRIVATE KEY-----"
        text = f"Key1: {key1}\nKey2: {key2}"
        redacted = DataRedactor.redact_text(text)
        self.assertNotIn("MIIE", redacted)
        self.assertNotIn("MHcE", redacted)
        self.assertEqual(redacted.count("[REDACTED_PRIVATE_KEY]"), 2)

    def test_generic_key_value(self):
        # Env var style
        text1 = "DB_PASSWORD=Hunter2!"
        self.assertEqual(DataRedactor.redact_text(text1), "DB_PASSWORD=[REDACTED_GENERIC_SECRET]")
        
        # Sentence style
        text2 = "The token : my-secret-token-123 is valid."
        self.assertEqual(DataRedactor.redact_text(text2), "The token : [REDACTED_GENERIC_SECRET] is valid.")
        
        # Quoted strings with spaces
        text3 = 'password = "my secret password"'
        text4 = "api_key: 'sk live 123'"
        self.assertEqual(DataRedactor.redact_text(text3), 'password = [REDACTED_GENERIC_SECRET]')
        self.assertEqual(DataRedactor.redact_text(text4), "api_key: [REDACTED_GENERIC_SECRET]")

    def test_api_key(self):
        text_sk = "Using sk-1234567890abcdef1234567890abcdef1234567890 for auth."
        text_aws = "AWS AKIAIOSFODNN7EXAMPLE key"
        text_jwt = "Token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        
        self.assertEqual(DataRedactor.redact_text(text_sk), "Using [REDACTED_API_KEY] for auth.")
        self.assertEqual(DataRedactor.redact_text(text_aws), "AWS [REDACTED_API_KEY] key")
        self.assertEqual(DataRedactor.redact_text(text_jwt), "Token [REDACTED_API_KEY]")

    def test_redact_dict_recursive(self):
        payload = {
            "action": "Check DB_PASSWORD=Secret!",
            "reason": "Found token: secret-xyz-123",
            "nested": {
                "key": "sk-1234567890abcdef1234567890abcdef1234567890"
            },
            "score": 0.95
        }
        redacted = DataRedactor.redact_dict(payload)
        self.assertEqual(redacted["action"], "Check DB_PASSWORD=[REDACTED_GENERIC_SECRET]")
        self.assertEqual(redacted["reason"], "Found token: [REDACTED_GENERIC_SECRET]")
        self.assertEqual(redacted["nested"]["key"], "[REDACTED_API_KEY]")
        self.assertEqual(redacted["score"], 0.95)

class TestFeedbackLogger(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.logger = FeedbackLogger(log_dir=self.temp_dir.name, filename="test_log.jsonl")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_file_permissions(self):
        # Log a dummy action
        self.logger.log({"decision": "BLOCK"}, "action")
        self.logger.log_queue.join()
        
        filepath = self.logger.filepath
        
        self.assertTrue(os.path.exists(filepath))
        mode = os.stat(filepath).st_mode
        # On Windows, os.stat permissions are not fully POSIX-compliant, 
        # but we check the bitmask. On Linux, this will accurately check 0o600.
        if os.name != 'nt':
            self.assertEqual(stat.S_IMODE(mode), 0o600)
            
    def test_concurrency(self):
        def worker(logger, i):
            result = {
                "decision": "BLOCK",
                "layer": f"Tier_{i}",
                "reason": f"Reason_{i}"
            }
            logger.log(result, f"Action_{i}")

        threads = []
        for i in range(50):
            t = threading.Thread(target=worker, args=(self.logger, i))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()

        # Wait for all background logging to finish
        self.logger.log_queue.join()

        # Check file integrity
        filepath = self.logger.filepath
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        self.assertEqual(len(lines), 50)
        # Verify JSON is well-formed
        for line in lines:
            try:
                data = json.loads(line)
                self.assertIn("action", data)
            except json.JSONDecodeError:
                self.fail("Malformed JSON found due to race conditions")

if __name__ == "__main__":
    unittest.main()
