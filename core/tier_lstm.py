"""core/tier_lstm.py — Tier 0.5 Temporal LSTM (Phase 9 / v90) session-risk scorer.

Wraps the trained `v90_tier_0_5_session_risk.pth` checkpoint (Phase9LSTMModel,
input_dim=7, hidden_dim=64, num_layers=2 — see models/12_phase9_feature_eng.py)
and exposes a stateful `.update_and_score()` call that a benchmark or
production loop can invoke once per agent step, alongside the existing
Tier 0 / Tier 0.5-regex / V61 pipeline in core/pipeline.py.

IMPORTANT — feature-parity caveat (read before trusting ABSR numbers):
The LSTM was trained on synthetic multi-agent sessions
(tier_0_5_synthetic_sessions_*.jsonl) that carry *real* topological signals:
  iat_ms_since_prev_in_session, payload_bytes, content_risk_score,
  max_delegation_depth, max_fanout_degree, tool diversity, resource frequency.

The AgentDojo/EVO-PCA benchmark dataset built in run_benchmark.py's
build_dataset() is a *linear* single-agent session (session_id -> ordered
step_num), replayed in batch — it does NOT carry real wall-clock timestamps,
delegation depth, or fan-out branching. Concretely, in this benchmark:
  - fanout_norm is always 0.0 (no branching exists in the dataset — there is
    nothing to measure).
  - iat_norm is derived from the *wall-clock time between successive
    `update_and_score()` calls for the same session_id during benchmark
    execution*. This is a real timing signal, but it reflects benchmark
    processing latency, not the original attacker's real inter-action
    cadence, since the dataset is replayed rather than executed live.
  - depth_norm is derived from step_num (position in the linear session),
    which is a reasonable proxy for delegation depth in a single-agent
    setting but is not the same measurement the model was trained on.
  - content_risk_score is populated from the upstream Tier0.5-regex /
    V61 ML score already computed earlier in the pipeline for this action
    (this mirrors the training data's stated intent: "content_risk_score
    (from V65 model conceptually)").

In short: this integration is a best-effort port of a sequence model trained
on richer session telemetry onto a flatter benchmark dataset. Treat any
Tier 0.5-LSTM block/ABSR contribution as directional, not as a faithful
reproduction of the model's held-out eval numbers.
"""
import os
import re
import time
import warnings
import threading
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CORE_DIR)
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "models", "artifacts")
DEFAULT_LSTM_PATH = os.environ.get(
    "EVO_PCA_LSTM_MODEL_PATH",
    os.path.join(ARTIFACTS_DIR, "v90_tier_0_5_session_risk.pth"),
)

# Matches format_tool_call()'s output: "tool_name(arg=val, ...)"
_TOOL_CALL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)
_FIRST_ARG_VALUE_RE = re.compile(r"^[^=,]+=\s*(.+?)(?:,|$)")

# Max steps kept in the rolling per-session window (bounds memory + LSTM
# forward-pass cost on very long sessions; mirrors Tier05Config.FLAG_EXPIRY_STEPS).
MAX_SESSION_WINDOW = 50

# Per plan: block when the LSTM's malicious-probability confidence >= 0.98.
LSTM_BLOCK_THRESHOLD = float(os.environ.get("EVO_PCA_LSTM_BLOCK_THRESHOLD", "0.98"))


def _extract_tool_and_resource(action: str, action_type: str):
    """Best-effort extraction of (tool_name, resource_key) from an action string.

    For action_type == 'tool_call', action is expected to be
    format_tool_call()'s output: "tool_name(arg1=val1, arg2=val2)".
    For prompts, there is no tool/resource to extract.
    """
    if not isinstance(action, str):
        return None, None
    m = _TOOL_CALL_RE.match(action.strip())
    if not m:
        return None, None
    tool_name = m.group(1)
    args_str = m.group(2)
    resource_key = None
    arg_m = _FIRST_ARG_VALUE_RE.match(args_str)
    if arg_m:
        resource_key = arg_m.group(1).strip()[:120]  # cap length, avoid huge keys
    return tool_name, resource_key


class _BoundedSessionCache:
    def __init__(self, maxsize=50, ttl=3600):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = OrderedDict()  # key: text_hash, val: (timestamp, embedding)
        self.last_access = time.time()

    def get_or_compute(self, text, shared_encoder):
        now = time.time()
        self.last_access = now
        h = hash(text)
        if h in self.cache:
            ts, emb = self.cache[h]
            if now - ts < self.ttl:
                self.cache.move_to_end(h)
                return emb
            else:
                del self.cache[h]
        if shared_encoder is None:
            return None
        emb = shared_encoder.encode([text])[0]
        self.cache[h] = (now, emb)
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
        return emb


class SessionEmbeddingCacheManager:
    __instance = None
    __lock = threading.Lock()

    def __init__(self):
        self.sessions = OrderedDict()
        self.max_sessions = 5000

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            with cls.__lock:
                if cls.__instance is None:
                    cls.__instance = cls()
        return cls.__instance

    def get_bounded_cache(self, session_id: str, max_size=50, ttl=3600) -> _BoundedSessionCache:
        now = time.time()
        with self.__lock:
            expired = [sid for sid, c in self.sessions.items() if now - c.last_access > ttl]
            for sid in expired:
                del self.sessions[sid]
            if session_id not in self.sessions:
                self.sessions[session_id] = _BoundedSessionCache(maxsize=max_size, ttl=ttl)
                if len(self.sessions) > self.max_sessions:
                    self.sessions.popitem(last=False)
            else:
                self.sessions.move_to_end(session_id)
            return self.sessions[session_id]


def compute_sequence_risk(actions_window: List[str], session_id: str, max_window: int = 3) -> float:
    window = actions_window[-max_window:]
    if len(window) < 2:
        return 0.0
    try:
        from models.security.advanced_heuristics import SharedSemanticEncoder
        from sklearn.metrics.pairwise import cosine_similarity
        encoder = SharedSemanticEncoder.get()
        if encoder is None:
            return 0.0  # Degraded mode, avoid fail-open TF-IDF contamination
        cache_mgr = SessionEmbeddingCacheManager.get_instance()
        cache = cache_mgr.get_bounded_cache(session_id, max_size=50, ttl=3600)
        deltas = []
        for i in range(1, len(window)):
            emb_curr = cache.get_or_compute(window[i], shared_encoder=SharedSemanticEncoder)
            emb_prev = cache.get_or_compute(window[i-1], shared_encoder=SharedSemanticEncoder)
            if emb_curr is not None and emb_prev is not None:
                sim = cosine_similarity(emb_curr.reshape(1, -1), emb_prev.reshape(1, -1))[0][0]
                dist = max(0.0, 1.0 - float(sim))
                deltas.append(dist)
        if not deltas:
            return 0.0
        gain = float(os.environ.get("SEQUENCE_RISK_GAIN", "1.2"))
        try:
            from config import settings
            gain = float(getattr(settings, "SEQUENCE_RISK_GAIN", gain))
        except Exception:
            pass
        return min(1.0, max(deltas) * gain)
    except Exception as e:
        warnings.warn(f"[compute_sequence_risk] Error: {e}", RuntimeWarning)
        return 0.0


class _SessionWindow:
    """Rolling per-session state used to build the 7-dim LSTM feature sequence."""

    __slots__ = (
        "features", "seen_tools", "resource_counts",
        "step_count", "last_scan_wall_time", "actions_window",
    )

    def __init__(self):
        self.features: List[List[float]] = []
        self.seen_tools = set()
        self.resource_counts: Dict[str, int] = defaultdict(int)
        self.step_count = 0
        self.last_scan_wall_time: Optional[float] = None
        self.actions_window: List[str] = []


class SessionAwareLSTMRisk:
    """Stateful Tier 0.5 Temporal LSTM scorer (Phase 9 / v90 checkpoint).

    Usage (per agent step, before the tool actually executes):
        lstm_tier = SessionAwareLSTMRisk()
        result = lstm_tier.update_and_score(
            session_id=session_id,
            action=action_text,
            action_type=action_type,       # 'prompt' | 'tool_call'
            upstream_risk_score=upstream_score,  # e.g. V61 ml_score or Tier0.5 confidence, in [0, 1]
        )
        if result["is_blocked"]:
            # intercept — mimic firewall "Access Denied" behavior
            ...

    Thread-safety: like core/tier05.py, session state is kept in a plain dict
    and is NOT thread-safe. Apply external locking if running concurrently.
    """

    def __init__(self, model_path: str = None, block_threshold: float = None,
                 max_window: int = MAX_SESSION_WINDOW, max_sessions: int = 20000,
                 use_synthetic_iat: bool = False):
        self.model_path = model_path or DEFAULT_LSTM_PATH
        self.block_threshold = (
            block_threshold if block_threshold is not None else LSTM_BLOCK_THRESHOLD
        )
        self.max_window = max_window
        self.max_sessions = max_sessions
        self.use_synthetic_iat = use_synthetic_iat
        # LRU-bounded: a full AgentDojo run creates thousands of unique
        # session_ids (one per user/injection task x jailbreak template) that
        # are each touched only once, in sequence, then never revisited.
        # Without a cap this dict grows unbounded for the life of the process.
        self._sessions: "OrderedDict[str, _SessionWindow]" = OrderedDict()
        self._lock = threading.RLock()
        self.model = None
        self.available = False
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self):
        try:
            import torch
            import torch.nn as nn
            torch.set_num_threads(1)
        except ImportError as e:
            warnings.warn(
                f"[Tier0.5-LSTM] PyTorch not installed ({e}). "
                "LSTM session-risk scoring is DISABLED; this tier will fail "
                "open (never blocks) until torch is available.",
                RuntimeWarning,
            )
            return

        if not os.path.exists(self.model_path):
            warnings.warn(
                f"[Tier0.5-LSTM] Checkpoint not found at {self.model_path!r}. "
                "LSTM session-risk scoring is DISABLED; this tier will fail open.",
                RuntimeWarning,
            )
            return

        class Phase9LSTMModel(nn.Module):
            """Must match models/12_phase9_feature_eng.py exactly (input_dim=7)."""

            def __init__(self, input_dim=7, hidden_dim=64, output_dim=1, num_layers=2):
                super().__init__()
                self.layer_norm = nn.LayerNorm(input_dim)
                self.lstm = nn.LSTM(
                    input_dim, hidden_dim, num_layers=num_layers,
                    batch_first=True, bidirectional=False,
                )
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x, lengths):
                x = self.layer_norm(x)
                packed_x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False,
                )
                packed_out, (hn, cn) = self.lstm(packed_x)
                last_hidden = hn[-1]
                return self.fc(last_hidden)

        try:
            state_dict = torch.load(self.model_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model = Phase9LSTMModel(input_dim=7, hidden_dim=64, num_layers=2)
            model.load_state_dict(state_dict)
            model.eval()
            self._torch = torch
            self.model = model
            self.available = True
            print(f"[Tier0.5-LSTM] Loaded {os.path.basename(self.model_path)} "
                  f"(block_threshold={self.block_threshold}).")
        except Exception as e:
            warnings.warn(
                f"[Tier0.5-LSTM] Failed to load checkpoint {self.model_path!r}: "
                f"{type(e).__name__}: {e}. LSTM session-risk scoring is DISABLED "
                "(fail-open).",
                RuntimeWarning,
            )
            self.model = None
            self.available = False

    # ------------------------------------------------------------------
    # Session bookkeeping
    # ------------------------------------------------------------------
    def reset_session(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)

    def _get_window(self, session_id: str) -> _SessionWindow:
        win = self._sessions.get(session_id)
        if win is None:
            win = _SessionWindow()
            self._sessions[session_id] = win
            if len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)  # evict oldest (LRU)
        else:
            self._sessions.move_to_end(session_id)
        return win

    # ------------------------------------------------------------------
    # Feature extraction (7-dim)
    # ------------------------------------------------------------------
    def _build_feature_vector(self, win: _SessionWindow, action: str,
                               action_type: str, upstream_risk_score: float) -> List[float]:
        now = time.perf_counter()
        
        if win.last_scan_wall_time is None:
            iat_ms = 0.0
        else:
            if self.use_synthetic_iat:
                import numpy as np
                iat_ms = max(0.0, np.random.normal(5000.0, 1000.0))
            else:
                iat_ms = max(0.0, (now - win.last_scan_wall_time) * 1000.0)
                
        win.last_scan_wall_time = now

        payload_bytes = float(len((action or "")))
        
        # Apply exponential decay to the risk score history (0.9^distance)
        current_risk = float(max(0.0, min(1.0, upstream_risk_score or 0.0)))
        decayed_risk = current_risk
        
        # If we have history, apply decay factor to previous steps' risk context
        if win.features:
            # We don't alter history, but we can compute a smoothed risk for this step
            # OR better yet, just let the LSTM handle sequence dependencies, but since
            # upstream_risk_score is often highly correlated with maliciousness,
            # capping it via decay helps prevent a single false positive from ruining the session.
            # Actually, per the instructions, we can decay the risk scores already in win.features
            # Wait, modifying win.features directly alters the sequence. The instructions say:
            # "nhân điểm risk của các bước cũ với 0.9 ** (step hiện tại - step đó) trước khi đưa vào LSTM"
            pass # We will apply this when feeding into the LSTM in update_and_score

        win.step_count += 1
        risk = current_risk
        # Cap depth to 10.0 to prevent out-of-distribution depth_norm in long sessions
        depth = min(float(win.step_count), 10.0)
        max_fanout = 0.0  # see module docstring: no branching in this dataset

        tool_name, resource_key = _extract_tool_and_resource(action, action_type)
        if tool_name:
            win.seen_tools.add(tool_name)
        if resource_key:
            win.resource_counts[resource_key] += 1
            res_freq = float(win.resource_counts[resource_key])
        else:
            res_freq = 0.0
        tool_div = float(len(win.seen_tools))

        iat_norm = min(iat_ms / 10000.0, 10.0)
        payload_norm = min(payload_bytes / 10000.0, 10.0)
        depth_norm = depth / 10.0
        fanout_norm = max_fanout / 10.0
        tool_div_norm = tool_div / 10.0
        res_freq_norm = res_freq / 10.0

        return [
            iat_norm, payload_norm, risk,
            depth_norm, fanout_norm,
            tool_div_norm, res_freq_norm,
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_and_score(self, session_id: str, action: str, action_type: str = "prompt",
                          upstream_risk_score: float = 0.0) -> dict:
        """Append this step's features to the session window and score the
        cumulative sequence with the LSTM.

        Returns a dict:
          {
            "available": bool,          # False => model disabled, fail-open
            "probability": float,       # sigmoid(logit) in [0, 1], or 0.0 if unavailable
            "is_blocked": bool,         # probability >= block_threshold
            "layer": "Tier0.5-LSTM",
            "reason": str,
          }
        """
        result = {
            "available": self.available,
            "probability": 0.0,
            "is_blocked": False,
            "layer": "Tier0.5-LSTM",
            "reason": "",
        }
        if not self.available:
            result["reason"] = "LSTM unavailable (fail-open) — see startup warning."
            return result

        with self._lock:
            win = self._get_window(session_id)
            win.actions_window.append(action or "")
            if len(win.actions_window) > self.max_window:
                win.actions_window = win.actions_window[-self.max_window:]
            seq_risk = compute_sequence_risk(win.actions_window, session_id, max_window=3)

            feat = self._build_feature_vector(win, action, action_type, upstream_risk_score)
            win.features.append(feat)
            if len(win.features) > self.max_window:
                win.features = win.features[-self.max_window:]
            
            # Apply time-decay to past upstream_risk_score (index 2)
            seq = []
            current_step = len(win.features)
            for i, f in enumerate(win.features):
                f_copy = list(f)
                distance = current_step - (i + 1)
                f_copy[2] = f_copy[2] * (0.9 ** distance)
                seq.append(f_copy)

        try:
            torch = self._torch
            
            # The PyTorch model was only trained on short sequences.
            # Feeding T=50 causes hidden state accumulation (OOD), saturating at 0.9999.
            # We must cap the sequence fed to the LSTM to the last 10 steps.
            lstm_seq = seq[-10:]
            
            x = torch.tensor([lstm_seq], dtype=torch.float32)  # (1, min(T,10), 7)
            lengths = torch.tensor([len(lstm_seq)], dtype=torch.long)
            with torch.no_grad():
                logits = self.model(x, lengths)
                prob = float(torch.sigmoid(logits).item())
        except Exception as e:
            warnings.warn(
                f"[Tier0.5-LSTM] Inference error for session {session_id!r}: "
                f"{type(e).__name__}: {e}. Failing open for this step.",
                RuntimeWarning,
            )
            result["reason"] = f"Inference error (fail-open): {e}"
            return result

        delta_thresh = float(os.environ.get("SEMANTIC_DELTA_THRESHOLD", "0.65"))
        try:
            from config import settings
            delta_thresh = float(getattr(settings, "SEMANTIC_DELTA_THRESHOLD", delta_thresh))
        except Exception:
            pass

        result["probability"] = prob
        result["sequence_risk"] = seq_risk
        if prob >= self.block_threshold or seq_risk >= delta_thresh:
            # Guard against step-counting bias in pure repeating benign actions
            if seq_risk < 0.05 and all(f[2] < 0.2 for f in win.features):
                result["is_blocked"] = False
                result["reason"] = (
                    f"Tier0.5-LSTM: probability={prob:.4f}, neutralized by Step-Counting Invariance Guard (seq_risk={seq_risk:.3f})"
                )
            else:
                result["is_blocked"] = True
                result["reason"] = (
                    f"Tier0.5-LSTM: risk triggered (prob={prob:.4f} >= {self.block_threshold} or seq_risk={seq_risk:.3f} >= {delta_thresh})"
                )
        else:
            result["reason"] = f"Tier0.5-LSTM: probability={prob:.4f} (below threshold, seq_risk={seq_risk:.3f})"
        return result


__all__ = ["SessionAwareLSTMRisk", "LSTM_BLOCK_THRESHOLD", "DEFAULT_LSTM_PATH", "compute_sequence_risk", "SessionEmbeddingCacheManager"]
