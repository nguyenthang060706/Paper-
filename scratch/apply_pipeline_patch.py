with open("core/pipeline.py", "r", encoding="utf-8") as f:
    content = f.read()

helper_def = '''def is_benign_dev_shell(action: str) -> bool:
    """
    Identify benign developer / build shell commands (git, make, uv, python, ls, etc.)
    and distinguish them from destructive / malicious shell commands.
    """
    if not isinstance(action, str):
        return False
    lower = action.lower()
    
    # Absolute destructive / malicious signatures - MUST NEVER be whitelisted
    malicious_sigs = [
        'curl ', 'wget ', '| bash', '| sh', '| python', '| perl',
        'rm -rf /', 'rm -rf ~', 'mkfs', 'dd if=', ':(){ :|:& };:',
        'chmod 777 /', 'chmod +x /etc', 'chmod 777 /etc',
        '/etc/shadow', '/etc/sudoers',
        '/dev/tcp/', 'nc -e', 'ncat -e', 'bash -i',
        'cat /etc/passwd', 'cat /etc/shadow',
        'pkill -9', 'kill -9 1',
        'pay 1 btc', 'ransom', 'harassment_site'
    ]
    for sig in malicious_sigs:
        if sig in lower:
            return False
            
    # Benign dev command prefixes / keywords
    benign_patterns = [
        r'\\b(?:git\\s+(?:status|log|diff|branch|checkout|clone|commit|push|pull|fetch|add|reset))\\b',
        r'\\b(?:uv\\s+run|npm\\s+(?:run|test|install|ci)|cargo\\s+(?:build|test|run)|make(?:\\s+[\\w\\-]+)?|mvn|gradle)\\b',
        r'\\b(?:python(?:3)?\\s+[\\w\\./\\-]+|pytest)\\b',
        r'\\b(?:ls|dir|cat|head|tail|grep|find|pwd|mkdir|cd|echo)\\b',
        r'\\b(?:glab\\s+api|gh\\s+repo|adb\\s+|wsl\\.exe)\\b',
        r'\\b(?:Select-Object|Get-Process|Stop-Process)\\b',
        r'\\b(?:node_modules|build_models|compile)\\b',
    ]
    for bp in benign_patterns:
        if re.search(bp, action, re.IGNORECASE):
            return True
            
    return False


def _extract_instruction_segments(text: str) -> list[str]:'''

target_helper = 'def _extract_instruction_segments(text: str) -> list[str]:'

target_ingress = '''        # Defensive auto-inference of action_type (Fix 2 logic)
        if action_type in (None, "prompt"):
            if isinstance(action, dict):
                action_type = "tool_call"
            elif isinstance(action, str) and re.match(r'^[a-zA-Z_]\w*\s*\(', action.strip()):
                action_type = "tool_call"'''

replacement_ingress = '''        # Defensive auto-inference of action_type & safe dict handling
        tool_name = None
        if isinstance(action, dict):
            action_type = "tool_call"
            tool_name = action.get("name") or action.get("tool")
            action = json.dumps(action, ensure_ascii=False)
        elif action_type in (None, "prompt"):
            if isinstance(action, str) and re.match(r'^[a-zA-Z_]\w*\s*\(', action.strip()):
                action_type = "tool_call"'''

target_tool_extract = '''        tool_name = None
        if action_type == "tool_call":
            # Extract tool name from dictionary or string
            if isinstance(action, dict):
                tool_name = action.get("name") or action.get("tool")
            else:
                m = re.match(r'^(\w+)\(', action.strip())
                if m:
                    tool_name = m.group(1)'''

replacement_tool_extract = '''        if action_type == "tool_call" and not tool_name:
            m = re.match(r'^(\w+)\(', action.strip())
            if m:
                tool_name = m.group(1)'''

target_heuristics = '''            # Hard gate decision override
            is_escalation = (result["decision"] != "BLOCK")
            if heuristics_tier.value in ["DENY", "QUARANTINE"]:
                is_shell_tool = tool_name and tool_name.lower() in ['bash', 'cmd', 'exec', 'eval', 'python', 'powershell', 'shell']
                if action_type == "prompt" or is_suspicious_dangerous_tool or is_shell_tool:
                    result["decision"] = "BLOCK"
                    if is_escalation:
                        result["layer"] = "Heuristics"
                    result["reason"] = f"[HEURISTICS {heuristics_tier.value}] Aggregated score: {heuristics_score:.2f} - " + result["reason"]
                else:
                    result["heuristics_decision"] = heuristics_tier.value
                    result["heuristics_downgraded"] = True'''

replacement_heuristics = '''            # Hard gate decision override
            is_escalation = (result["decision"] != "BLOCK")
            if heuristics_tier.value in ["DENY", "QUARANTINE"]:
                is_shell_tool = tool_name and tool_name.lower() in ['bash', 'cmd', 'exec', 'eval', 'python', 'powershell', 'shell']
                if is_shell_tool and is_benign_dev_shell(action):
                    result["heuristics_decision"] = heuristics_tier.value
                    result["heuristics_downgraded"] = True
                elif action_type == "prompt" or is_suspicious_dangerous_tool or is_shell_tool:
                    result["decision"] = "BLOCK"
                    if is_escalation:
                        result["layer"] = "Heuristics"
                    result["reason"] = f"[HEURISTICS {heuristics_tier.value}] Aggregated score: {heuristics_score:.2f} - " + result["reason"]
                else:
                    result["heuristics_decision"] = heuristics_tier.value
                    result["heuristics_downgraded"] = True'''

assert target_helper in content, "target_helper not found"
assert target_ingress in content, "target_ingress not found"
assert target_tool_extract in content, "target_tool_extract not found"
assert target_heuristics in content, "target_heuristics not found"

content = content.replace(target_helper, helper_def, 1)
content = content.replace(target_ingress, replacement_ingress, 1)
content = content.replace(target_tool_extract, replacement_tool_extract, 1)
content = content.replace(target_heuristics, replacement_heuristics, 1)

with open("core/pipeline.py", "w", encoding="utf-8") as f:
    f.write(content)

print("SUCCESS: Patched core/pipeline.py")
