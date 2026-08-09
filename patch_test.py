with open('tests/test_pipeline.py', 'r') as f:
    content = f.read()

test_func = '''
    def test_write_code_content_not_quarantined(self):
        """Write() containing code with 'password'/'token' should NOT be QUARANTINE."""
        action = "Write({'file_path': 'd:\\\\\\\\CFML\\\\\\\\settings.py', 'content': 'TOKEN = os.environ.get(\\"TOKEN\\")\\\\nPASSWORD_HASH = bcrypt.hash(input())'})"
        result = self.pipeline.scan(action, "sess_fp_test", action_type="tool_call")
        self.assertNotIn(result.get("heuristics_decision"), ["DENY", "QUARANTINE"])
'''

if "test_write_code_content_not_quarantined" not in content:
    content = content.replace("if __name__ == '__main__':", test_func + "\nif __name__ == '__main__':")
    with open('tests/test_pipeline.py', 'w') as f:
        f.write(content)
