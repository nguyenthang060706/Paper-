import re
import ast
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ParsedAction:
    tool_name: str
    arguments: Dict[str, Any]
    raw_action: str
    format: str # 'dict_literal', 'kwargs', 'positional_args', 'json', 'xml', 'plain_text'
    
    @property
    def envelope(self) -> str:
        """Returns the structure of the action without the large data content.
        This is useful for scanning behavioral flags (like 'you are now...') without
        flagging the actual file content or email bodies being operated on.
        """
        if self.format in ('plain_text', 'json', 'xml'):
            return self.raw_action
            
        # Keys that typically contain user content, not behavioral instructions
        DATA_KEYS = {'content', 'body', 'new_string', 'old_string', 'text', 'message', 'data', 'description', 'comment', 'note', 'html', 'markdown', 'code', 'script', 'template'}
        
        parts = [f"{self.tool_name}("]
        if self.format == 'dict_literal':
            args_parts = []
            for k, v in self.arguments.items():
                if isinstance(k, str) and k.lower() in DATA_KEYS and isinstance(v, str):
                    # We strip without length constraint because data keys should not contain behavioral instructions
                    args_parts.append(f"'{k}': '[DATA]'")
                else:
                    args_parts.append(f"'{k}': {repr(v)}")
            parts.append("{" + ", ".join(args_parts) + "}")
            
        elif self.format == 'kwargs':
            args_parts = []
            for k, v in self.arguments.items():
                if isinstance(k, str) and k.lower() in DATA_KEYS and isinstance(v, str):
                    args_parts.append(f"{k}='[DATA]'")
                else:
                    args_parts.append(f"{k}={repr(v)}")
            parts.append(", ".join(args_parts))
            
        elif self.format == 'positional_args':
            args_parts = []
            for k, v in self.arguments.items():
                # For positional args, we don't know the key name, so we guess by length.
                if isinstance(v, str) and len(v) > 50:
                    args_parts.append("'[DATA]'")
                else:
                    args_parts.append(repr(v))
            parts.append(", ".join(args_parts))
            
        parts.append(")")
        return "".join(parts)

class ActionParser:
    """Unified parser that handles multiple tool call formats."""
    PG_RE = re.compile(r'^(\w+)\((\{.*\})\)\s*$', re.DOTALL)
    LSTM_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$', re.DOTALL)
    
    @classmethod
    def parse(cls, action: str) -> ParsedAction:
        if not isinstance(action, str):
            return ParsedAction("", {}, str(action), 'plain_text')
            
        action = action.strip()
        
        # 1. Try dict_literal (e.g., Write({'content': '...', 'file_path': '...'}))
        m = cls.PG_RE.match(action)
        if m:
            tool_name = m.group(1)
            dict_str = m.group(2)
            try:
                parsed = ast.literal_eval(dict_str)
                if isinstance(parsed, dict):
                    return ParsedAction(tool_name, parsed, action, 'dict_literal')
            except (ValueError, SyntaxError):
                pass
                
        # 2. Try kwargs / positional (e.g., send_email(to='x', body='y'))
        m = cls.LSTM_RE.match(action)
        if m:
            tool_name = m.group(1)
            args_str = m.group(2).strip()
            
            try:
                # Use Python's AST to safely parse the arguments
                tree = ast.parse(f"dummy({args_str})", mode='eval')
                call_node = tree.body
                if isinstance(call_node, ast.Call):
                    args_dict = {}
                    is_kwargs = False
                    for kw in call_node.keywords:
                        is_kwargs = True
                        if kw.arg is None:
                            continue 
                        try:
                            val = ast.literal_eval(kw.value)
                        except Exception:
                            val = ast.unparse(kw.value)
                        args_dict[kw.arg] = val
                        
                    for i, arg in enumerate(call_node.args):
                        try:
                            val = ast.literal_eval(arg)
                        except Exception:
                            val = ast.unparse(arg)
                        args_dict[f'arg{i}'] = val
                    
                    if is_kwargs:
                        return ParsedAction(tool_name, args_dict, action, 'kwargs')
                    elif args_dict:
                        return ParsedAction(tool_name, args_dict, action, 'positional_args')
                    else:
                        return ParsedAction(tool_name, {}, action, 'positional_args')
            except Exception:
                pass

        # 3. Simple JSON / XML checks
        if action.startswith('{') or action.startswith('['):
            return ParsedAction("", {}, action, 'json')
        if action.startswith('<'):
            return ParsedAction("", {}, action, 'xml')
            
        return ParsedAction("", {}, action, 'plain_text')
