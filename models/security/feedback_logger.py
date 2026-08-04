import os
import json
import threading
import queue
import logging
import atexit
from .data_redactor import DataRedactor

class FeedbackLogger:
    def __init__(self, log_dir: str = "logs", filename: str = "retrain_dataset.jsonl"):
        self.log_dir = log_dir
        self.filename = filename
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
            
        self.filepath = os.path.join(self.log_dir, self.filename)
        
        # Đảm bảo file nếu có sẵn thì cũng phải siết quyền
        if os.path.exists(self.filepath) and os.name != 'nt':
            os.chmod(self.filepath, 0o600)
            
        # Asynchronous Logging Queue
        self.log_queue = queue.Queue(maxsize=1000)
        self.dropped_count = 0
        self.dropped_lock = threading.Lock()
        
        # Start worker thread
        self._worker_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._worker_thread.start()
        
        # Đảm bảo flush dữ liệu khi process exit
        atexit.register(self.shutdown)

    def shutdown(self):
        """Chạy lúc exit để đảm bảo ghi nốt queue."""
        if self._worker_thread.is_alive():
            try:
                self.log_queue.put(None, timeout=2.0) # Sentinel to stop thread
            except queue.Full:
                pass # Queue full and thread stuck, avoid hanging process
            self._worker_thread.join(timeout=5.0) # Wait for thread to finish

    def log(self, result: dict, action: str):
        payload = {
            "event_type": result.get("event_type", "FIREWALL_BLOCK"),
            "action": action,
            "action_type": result.get("action_type", "unknown"),
            "decision": result.get("decision"),
            "layer": result.get("layer"),
            "reason": result.get("reason"),
            "ml_score": result.get("ml_score"),
            "was_shadow_blocked": result.get("was_shadow_blocked"),
            "shadow_blocked_layer": result.get("shadow_blocked_layer"),
            "label_source": result.get("label_source")
        }
        
        try:
            self.log_queue.put_nowait(payload)
        except queue.Full:
            with self.dropped_lock:
                self.dropped_count += 1
                if self.dropped_count % 100 == 0:
                    logging.warning(f"FeedbackLogger queue full. Dropped {self.dropped_count} samples so far.")

    def _writer_loop(self):
        while True:
            try:
                payload = self.log_queue.get()
                if payload is None: # Sentinel value for shutdown
                    break
                    
                try:
                    # Redact the entire payload IN THE BACKGROUND (tránh CPU hot-path)
                    redacted_payload = DataRedactor.redact_dict(payload)
                    json_line = json.dumps(redacted_payload, ensure_ascii=False) + "\n"
                    
                    # Atomic file writing with secure permissions (single thread, no lock needed)
                    fd = os.open(self.filepath, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
                    try:
                        with os.fdopen(fd, 'a', encoding='utf-8') as f:
                            f.write(json_line)
                    except Exception as e:
                        logging.error(f"Failed to write log: {e}")
                except Exception as e:
                    logging.error(f"Error processing payload in FeedbackLogger: {e}")
                finally:
                    self.log_queue.task_done()
            except Exception as e:
                # Catch-all để đảm bảo thread không bao giờ chết
                logging.error(f"Fatal error in FeedbackLogger worker thread: {e}")
