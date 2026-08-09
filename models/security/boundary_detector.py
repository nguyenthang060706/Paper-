import re
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

@dataclass
class BoundaryViolation:
    position_ratio: float      # 0.0 = start, 1.0 = end
    instruction_text: str      # The extracted instruction
    context_type: str          # 'json_data', 'markdown_body', 'email_body', 'free_text'
    confidence: float          # Detection confidence
    violation_type: str        # 'embedded_imperative', 'role_shift'

class StructuredParser:
    """Layer A: Parses structured formats to isolate data from metadata."""
    
    @staticmethod
    def extract_text_nodes(text: str) -> List[Tuple[str, str, int, int]]:
        """Extract text nodes from structured formats.
        Returns: list of (node_type, content, start_idx, end_idx)
        """
        nodes = []
        
        # 1. Try Tool Call format e.g., tool_name(arg1="value", arg2='long string here')
        # Only apply this if the text actually looks like a tool call to avoid false positives on HTML attributes
        if re.match(r'^\s*\w+\s*\(.*\)\s*$', text, re.DOTALL):
            tool_call_pattern = re.compile(r'(\w+)=([\'"])(.*?)\2', re.DOTALL)
            matched_tool_args = False
            for match in tool_call_pattern.finditer(text):
                arg_name = match.group(1)
                content = match.group(3).strip()
                if len(content) > 20:
                    nodes.append((f'tool_arg_{arg_name}', content, match.start(3), match.end(3)))
                    matched_tool_args = True
            
            if matched_tool_args:
                return nodes

        # 2. Try JSON
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                # Extract string values from JSON with tracked offsets to fix duplicate bug
                last_search_idx = [0]
                def _walk_json(obj: Any, path: str):
                    if isinstance(obj, str) and len(obj) > 20:
                        idx = text.find(obj[:20], last_search_idx[0]) 
                        if idx != -1:
                            nodes.append(('json_data', obj, idx, idx + len(obj)))
                            last_search_idx[0] = idx + len(obj)
                    elif isinstance(obj, dict):
                        for k, v in obj.items():
                            _walk_json(v, f"{path}.{k}")
                    elif isinstance(obj, list):
                        for i, v in enumerate(obj):
                            _walk_json(v, f"{path}[{i}]")
                _walk_json(parsed, "$")
                if nodes: return nodes
        except json.JSONDecodeError:
            pass
            
        # 3. Try XML/HTML-like Tool Tags (e.g. <tool_output>...</tool_output>)
        tag_pattern = re.compile(r'<([a-zA-Z0-9_]+)>(.*?)</\1>', re.DOTALL)
        for match in tag_pattern.finditer(text):
            tag_name = match.group(1).lower()
            content = match.group(2).strip()
            if len(content) > 20:
                nodes.append((f'xml_{tag_name}', content, match.start(2), match.end(2)))
        
        if nodes: return nodes
        
        # 3. Email-like structure
        email_body_match = re.search(r'(?:Subject|To|From):\s*.*?\n\s*\n(.*?)(?:\Z|---+)', text, re.IGNORECASE | re.DOTALL)
        if email_body_match:
            content = email_body_match.group(1).strip()
            if len(content) > 20:
                nodes.append(('email_body', content, email_body_match.start(1), email_body_match.end(1)))
                return nodes
                
        # 4. Fallback: Free Text (Treat whole thing as one data node if long enough)
        if len(text) > 50:
            nodes.append(('free_text', text, 0, len(text)))
            
        return nodes

class SemanticRoleClassifier:
    """Layer B: Small NLP Classifier to distinguish DATA vs INSTRUCTION.
    
    This is a proxy wrapper. In production, this would load a ONNX-quantized 
    small model (e.g., DeBERTa-v3-tiny) fine-tuned on Alpaca vs Wiki text.
    For this prototype, if the model isn't available, we use an advanced heuristic
    fallback that is strictly more robust than simple regex.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._model_loaded = False
        # Try loading actual ONNX model here in the future
        
    def classify_sentence(self, sentence: str) -> Tuple[str, float]:
        """Classify a single sentence. 
        Returns: ('INSTRUCTION' or 'DATA', confidence_score)
        """
        # Prototype Fallback Heuristics (simulating model output)
        # We look for linguistic markers of control flow (imperatives, conditions)
        
        sentence = sentence.lower().strip()
        
        # Strong instruction indicators
        if re.match(r'^(?:you\s+must|please\s+|ignore|execute|send|forward|call)\b', sentence):
            return ('INSTRUCTION', 0.90)
            
        if re.search(r'\b(?:it\s+is\s+(?:essential|important|crucial)|make\s+sure)\s+(?:to|that)\b', sentence):
            return ('INSTRUCTION', 0.85)
            
        # Action verbs + targets
        action_verbs = len(re.findall(r'\b(?:send|email|forward|transfer|upload|execute|run)\b', sentence))
        targets = len(re.findall(r'(?:@[\w.]+\.\w+|https?://)', sentence))
        if action_verbs > 0 and targets > 0:
            return ('INSTRUCTION', 0.80)
            
        return ('DATA', 0.80)
        
class InstructionBoundaryDetector:
    def __init__(self):
        self.parser = StructuredParser()
        self.classifier = SemanticRoleClassifier()
        
    def _chunk_text(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentence-like chunks with offsets."""
        chunks = []
        for match in re.finditer(r'[^.!?\n]+[.!?\n]*', text):
            chunk = match.group(0).strip()
            if len(chunk) > 10:
                chunks.append((chunk, match.start(), match.end()))
        return chunks

    def detect(self, text: str) -> Tuple[bool, float, List[BoundaryViolation]]:
        violations = []
        
        # 1. Parse structured data
        nodes = self.parser.extract_text_nodes(text)
        
        for context_type, content, start_idx, end_idx in nodes:
            # 2. Chunk unstructured content within the node
            chunks = self._chunk_text(content)
            
            for chunk_text, c_start, c_end in chunks:
                # 3. Classify each chunk
                role, conf = self.classifier.classify_sentence(chunk_text)
                
                if role == 'INSTRUCTION' and conf > 0.7:
                    # An instruction inside a data node is a boundary violation
                    abs_start = start_idx + c_start
                    pos_ratio = abs_start / max(len(text), 1)
                    
                    violations.append(BoundaryViolation(
                        position_ratio=pos_ratio,
                        instruction_text=chunk_text[:100],
                        context_type=context_type,
                        confidence=conf,
                        violation_type='embedded_imperative'
                    ))
                    
        if not violations:
            return False, 0.0, []
            
        max_conf = max(v.confidence for v in violations)
        return True, max_conf, violations
