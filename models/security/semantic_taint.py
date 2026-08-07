import re
import base64
import binascii
import math
import urllib.parse
from typing import List, Tuple, Optional

class ObfuscationDetector:
    """Detects and decodes common obfuscation techniques used in exfiltration."""
    
    @staticmethod
    def shannon_entropy(data: str) -> float:
        """Calculates Shannon entropy of a string to detect encrypted/encoded blobs."""
        if not data:
            return 0.0
        entropy = 0
        for x in set(data):
            p_x = float(data.count(x)) / len(data)
            entropy += - p_x * math.log2(p_x)
        return entropy

    @staticmethod
    def try_decode(text: str) -> Tuple[bool, str, str]:
        """Try to decode text if it's obfuscated.
        Returns (is_decoded, decoded_text, method_used).
        """
        text = text.strip()
        
        # 1. Check URL Encoding
        if '%' in text:
            decoded_url = urllib.parse.unquote(text)
            if decoded_url != text:
                return True, decoded_url, 'url_encoded'
                
        # 2. Check Base64
        # Look for a string that matches base64 pattern and is long enough
        b64_pattern = re.compile(r'^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$')
        if len(text) >= 16 and b64_pattern.match(text):
            try:
                decoded_b64 = base64.b64decode(text).decode('utf-8')
                # Ensure it decodes to mostly printable characters
                if len(decoded_b64) > 5 and all(32 <= ord(c) < 127 or c in '\n\r\t' for c in decoded_b64):
                    return True, decoded_b64, 'base64'
            except (binascii.Error, UnicodeDecodeError):
                pass
                
        # 3. Check Hex Encoding
        hex_pattern = re.compile(r'^[0-9a-fA-F]+$')
        if len(text) >= 16 and len(text) % 2 == 0 and hex_pattern.match(text):
            try:
                decoded_hex = bytes.fromhex(text).decode('utf-8')
                if len(decoded_hex) > 5 and all(32 <= ord(c) < 127 or c in '\n\r\t' for c in decoded_hex):
                    return True, decoded_hex, 'hex'
            except (ValueError, UnicodeDecodeError):
                pass
                
        return False, text, 'none'

class BoundedSemanticTaintTracker:
    """Manages O(K) semantic fingerprints to track data exfiltration."""
    
    def __init__(self, max_fingerprints: int = 3, similarity_threshold: float = 0.75):
        self.max_k = max_fingerprints
        self.threshold = similarity_threshold
        
    def _compute_pseudo_embedding(self, text: str) -> set:
        """Mock for actual ONNX embedding: uses character trigrams for fast semantic approximation.
        In production, this would be `model.encode(text)`.
        """
        text = text.lower()
        # Create character trigrams
        return set(text[i:i+3] for i in range(len(text)-2))
        
    def _cosine_similarity_mock(self, set1: set, set2: set) -> float:
        """Jaccard similarity as a proxy for cosine similarity on embeddings."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union

    def analyze_taint(self, session_taints: List[str], outgoing_data: str) -> Tuple[bool, float, str]:
        """
        Check if outgoing data matches any of the stored semantic taints.
        Returns: (is_tainted, max_similarity_score, detection_reason)
        """
        if not session_taints or not outgoing_data:
            return False, 0.0, ""
            
        # 1. Pre-check: Obfuscation Decode
        is_decoded, data_to_check, method = ObfuscationDetector.try_decode(outgoing_data)
        
        # 2. Heuristic: If it's a very long unknown blob with high entropy, flag it speculatively
        entropy = ObfuscationDetector.shannon_entropy(outgoing_data)
        if entropy > 5.5 and len(outgoing_data) > 100 and not is_decoded:
             return True, 0.60, f"high_entropy_blob_suspected_exfil_{entropy:.2f}"
             
        # 3. Check against up to K stored taints
        outgoing_embed = self._compute_pseudo_embedding(data_to_check)
        
        # O(K) complexity because len(session_taints) <= max_k
        max_sim = 0.0
        for taint_text in session_taints[-self.max_k:]:
            taint_embed = self._compute_pseudo_embedding(taint_text)
            sim = self._cosine_similarity_mock(outgoing_embed, taint_embed)
            if sim > max_sim:
                max_sim = sim
                
        if max_sim >= self.threshold:
            reason = "Semantic match found"
            if is_decoded:
                reason += f" (after decoding {method})"
            return True, max_sim, reason
            
        return False, max_sim, ""
