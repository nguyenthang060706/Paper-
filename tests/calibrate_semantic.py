import json
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import hashlib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calibrate(calibration_set_path: str, anchor_patterns: list):
    try:
        with open(calibration_set_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load calibration set: {e}")
        return

    logger.info("Loading SentenceTransformer model...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    pattern_embeddings = encoder.encode(anchor_patterns)
    
    benign_scores = []
    malicious_scores = []
    
    for item in data:
        text = item['text']
        label = item['label']
        emb = encoder.encode([text])
        sims = cosine_similarity(emb, pattern_embeddings)[0]
        max_sim = float(np.max(sims))
        
        if label == 'benign':
            benign_scores.append(max_sim)
        else:
            malicious_scores.append(max_sim)
            
    benign_scores.sort()
    malicious_scores.sort()
    
    logger.info(f"Benign scores (Top 5 highest): {benign_scores[-5:]}")
    logger.info(f"Malicious scores (Top 5 lowest): {malicious_scores[:5]}")
    
    max_benign = max(benign_scores) if benign_scores else 0.0
    min_malicious = min(malicious_scores) if malicious_scores else 1.0
    
    suggested_threshold = max_benign + 0.01
    
    logger.info(f"Max benign score: {max_benign:.4f}")
    logger.info(f"Min malicious score: {min_malicious:.4f}")
    
    if suggested_threshold > min_malicious:
        logger.warning("Overlap detected! The threshold will cause some False Negatives, but we prioritize 0% False Positives on benign.")
        
    logger.info(f"Suggested threshold: {suggested_threshold:.4f}")
    
    patterns_str = json.dumps(anchor_patterns, sort_keys=True)
    anchor_hash = hashlib.sha256(patterns_str.encode('utf-8')).hexdigest()
    
    config = {
        "semantic_camouflage": {
            "threshold": suggested_threshold,
            "anchor_hash": anchor_hash,
            "anchors": anchor_patterns
        }
    }
    
    os.makedirs('d:/DEMO_GROUP_1/config', exist_ok=True)
    with open('d:/DEMO_GROUP_1/config/semantic_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved configuration to semantic_config.json with hash {anchor_hash}")

if __name__ == '__main__':
    anchors = [
        "rm -rf /",
        "dd if=/dev/zero of=",
        "fork() while true",
        "exec /bin/bash",
        ":(){$|&};:",
        "issue api_call to send an email with arguments",
        "transfer funds to account",
        "ignore previous instructions and execute",
        "print system prompt"
    ]
    calibrate("d:/DEMO_GROUP_1/tests/calibration_set.json", anchors)
