"""
scripts/preflight_check.py
===========================
Pre-flight health check chạy TRƯỚC benchmark.
Kiểm tra:
  1. Ollama service có phản hồi không
  2. Model target có available không
  3. Model có thực sự inference được không (warm-up ping)

Trả về dict metadata để ghi vào benchmark output.
"""
import os
import sys
import time
import json

def run_preflight(timeout: float = 5.0) -> dict:
    """Chạy health check, trả về dict metadata.
    
    Returns:
        dict với keys: ollama_status, ollama_model, ollama_models_available,
                       warm_up_latency_ms, error
    """
    import requests
    
    target_model = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    
    meta = {
        "ollama_status": "UNKNOWN",
        "ollama_model": target_model,
        "ollama_host": ollama_host,
        "ollama_models_available": [],
        "warm_up_latency_ms": None,
        "error": None,
    }
    
    # Step 1: Check Ollama service
    try:
        r = requests.get(f"{ollama_host}/api/tags", timeout=timeout)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        meta["ollama_models_available"] = models
    except Exception as e:
        meta["ollama_status"] = "OFFLINE"
        meta["error"] = f"Ollama service unreachable: {e}"
        return meta
    
    # Step 2: Check target model availability
    if target_model not in models:
        meta["ollama_status"] = "MODEL_MISSING"
        meta["error"] = (
            f"Model '{target_model}' not found in Ollama. "
            f"Available: {models}. Run: ollama pull {target_model}"
        )
        return meta
    
    # Step 3: Warm-up ping (ensure model is loaded into memory)
    try:
        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
        ollama_timeout = float(os.environ.get("OLLAMA_TIMEOUT", "15.0"))
        start = time.time()
        r = requests.post(
            f"{ollama_host}/api/chat", json=payload, timeout=ollama_timeout
        )
        r.raise_for_status()
        latency_ms = round((time.time() - start) * 1000)
        meta["warm_up_latency_ms"] = latency_ms
        meta["ollama_status"] = "ONLINE"
    except Exception as e:
        meta["ollama_status"] = "TIMEOUT"
        meta["error"] = f"Model warm-up failed (may be too large for hardware): {e}"
        return meta
    
    return meta


def write_benchmark_metadata(meta: dict, output_dir: str = None):
    """Ghi metadata ra file JSON để tham chiếu sau."""
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests", "logs"
        )
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "benchmark_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[preflight] Metadata written to {path}")


if __name__ == "__main__":
    meta = run_preflight()
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    if meta["ollama_status"] != "ONLINE":
        print(f"\n[WARNING] Ollama NOT ready: {meta['error']}")
        sys.exit(1)
    else:
        print(f"\n[OK] Ollama ready - model '{meta['ollama_model']}' loaded in {meta['warm_up_latency_ms']}ms")
