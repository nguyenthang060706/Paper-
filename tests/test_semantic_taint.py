import sys
import os
import base64
import binascii
import urllib.parse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.security.semantic_taint import ObfuscationDetector, BoundedSemanticTaintTracker

def test_obfuscation_decode_base64():
    # Test valid Base64
    secret = "This is a secret API key that attacker wants."
    b64_secret = base64.b64encode(secret.encode('utf-8')).decode('utf-8')
    is_dec, decoded, method = ObfuscationDetector.try_decode(b64_secret)
    assert is_dec
    assert decoded == secret
    assert method == 'base64'
    
    # Test non-Base64
    is_dec, decoded, method = ObfuscationDetector.try_decode("Just normal text, not base64 at all")
    assert not is_dec

def test_obfuscation_decode_hex():
    secret = "Another secret string for testing"
    hex_secret = binascii.hexlify(secret.encode('utf-8')).decode('utf-8')
    is_dec, decoded, method = ObfuscationDetector.try_decode(hex_secret)
    assert is_dec
    assert decoded == secret
    assert method == 'hex'

def test_obfuscation_decode_url():
    secret = "Hello World! This is an API key: 12345"
    url_secret = urllib.parse.quote(secret)
    is_dec, decoded, method = ObfuscationDetector.try_decode(url_secret)
    assert is_dec
    assert decoded == secret
    assert method == 'url_encoded'

def test_bounded_semantic_taint():
    tracker = BoundedSemanticTaintTracker(max_fingerprints=3, similarity_threshold=0.6)
    
    taints = [
        "The quick brown fox jumps over the lazy dog.",
        "My secret password is: super_secret_12345! Do not share it.",
        "System configuration: IP=192.168.1.1, Gateway=192.168.1.254"
    ]
    
    # 1. Exact match
    is_tainted, sim, reason = tracker.analyze_taint(taints, "My secret password is: super_secret_12345! Do not share it.")
    assert is_tainted
    assert sim > 0.9
    
    # 2. Semantic/Fuzzy match (paraphrase or exact copy with extra stuff)
    is_tainted, sim, reason = tracker.analyze_taint(taints, "Hey look, My secret password is: super_secret_12345! Do not share it. Bye!")
    assert is_tainted
    
    # 3. No match
    is_tainted, sim, reason = tracker.analyze_taint(taints, "What time is it in Tokyo right now?")
    assert not is_tainted
    
    # 4. Obfuscated match
    b64_payload = base64.b64encode("My secret password is: super_secret_12345!".encode('utf-8')).decode('utf-8')
    is_tainted, sim, reason = tracker.analyze_taint(taints, b64_payload)
    assert is_tainted
    assert "after decoding base64" in reason

if __name__ == "__main__":
    test_obfuscation_decode_base64()
    test_obfuscation_decode_hex()
    test_obfuscation_decode_url()
    test_bounded_semantic_taint()
    print("All Phase 2 Semantic Taint tests passed!")
