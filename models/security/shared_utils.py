"""shared_utils.py — Shared utilities for EVO-PCA v61 security models.

Single source of truth for functions used by both v61_inference_router
and v61_context_sanitizer, eliminating DRY violations.

[FIX-1] ensemble_predict_proba extracted from both files.
[FIX-4] v61_prob_to_evo_weighted / evo_weighted_to_v61_prob enforced here
        instead of being comment-only in the notebook.
"""

import joblib
import numpy as np


def _predict_proba_single(estimator, X):
    """Call predict_proba with joblib forced to single-threaded execution.

    This avoids nested thread pools when the surrounding process is already
    using threads, particularly during benchmark scoring or concurrent
    evaluation workloads.
    """
    if not hasattr(estimator, "predict_proba"):
        raise AttributeError(f"Estimator {estimator!r} has no predict_proba method")

    try:
        with joblib.parallel_backend("threading", n_jobs=1):
            return estimator.predict_proba(X, n_jobs=1)
    except TypeError:
        return estimator.predict_proba(X)
    except Exception:
        try:
            return estimator.predict_proba(X, n_jobs=1)
        except TypeError:
            return estimator.predict_proba(X)


def ensemble_predict_proba(X, rf_cal, lr_cal, w_rf=0.4, w_lr=0.6):
    """Weighted ensemble of RF and LR calibrated probabilities.

    Trọng số: LR calibration tốt hơn RF trên sparse text features.
    Điều chỉnh sau khi có calibration curve thực tế.

    Args:
        X: Feature matrix (output of feature_union.transform).
        rf_cal: Calibrated Random Forest classifier.
        lr_cal: Calibrated Logistic Regression classifier.
        w_rf: Weight for RF (default 0.4).
        w_lr: Weight for LR (default 0.6).

    Returns:
        np.ndarray of blended probabilities for the positive class.
    """
    rf_prob = _predict_proba_single(rf_cal, X)[:, 1]
    lr_prob = _predict_proba_single(lr_cal, X)[:, 1]
    return w_rf * rf_prob + w_lr * lr_prob


# ═══════════════════════════════════════════════════════════════════════════
# [FIX-4] SCALE BOUNDARY: V61 Router [0.0, 1.0] vs EVO-PCA [0, 100]
# ───────────────────────────────────────────────────────────────────────────
# V61SecurityRouter (v61_inference_router.py):
#   ActionRiskModel.score()  → returns float in [0.0, 1.0]
#   Thresholds: BLOCK ≥ 0.72 | REVIEW 0.35–0.72 | ALLOW < 0.35
#
# EVO-PCA VotingAggregator (notebook):
#   vote(signals) → returns weighted_score in [0, 100]
#   Thresholds: QUARANTINE >= 90 | DENY >= 82 | REVIEW >= 68 | MONITOR >= 40
#
# The two scales are INCOMPATIBLE.  Never compare or threshold-check a V61
# probability directly against an EVO-PCA weighted score, or vice versa.
# ═══════════════════════════════════════════════════════════════════════════

# Scale identifier constants — attach to score dicts for clarity
SCALE_V61 = "v61_0_1"
SCALE_EVO_PCA = "evo_pca_0_100"

class BaseScore:
    def __init__(self, value: float):
        self.value = float(value)

    def _check_type(self, other):
        if not isinstance(other, type(self)):
            raise TypeError(f"Thang đo không tương thích! Không thể so sánh {type(self).__name__} với {type(other).__name__}. Vui lòng kiểm tra lại logic.")

    def __lt__(self, other):
        self._check_type(other)
        return self.value < other.value

    def __le__(self, other):
        self._check_type(other)
        return self.value <= other.value

    def __eq__(self, other):
        self._check_type(other)
        return self.value == other.value

    def __ge__(self, other):
        self._check_type(other)
        return self.value >= other.value

    def __gt__(self, other):
        self._check_type(other)
        return self.value > other.value
        
    def __float__(self):
        return self.value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


class ScoreV61(BaseScore):
    """Thang đo xác suất cho V61 Router [0.0, 1.0]."""
    def __init__(self, value: float):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"ScoreV61 phải nằm trong khoảng [0.0, 1.0], nhận được {value}")
        super().__init__(value)


class ScoreEvo(BaseScore):
    """Thang đo trọng số cho EVO-PCA [0, 100]."""
    def __init__(self, value: float):
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"ScoreEvo phải nằm trong khoảng [0, 100], nhận được {value}")
        super().__init__(value)


def v61_prob_to_evo_weighted(v61_prob: float) -> float:
    """Rough display-only conversion from V61 router probability (0–1)
    to an approximate EVO-PCA weighted score (0–100).

    NOT calibrated — use only for logging and cross-system dashboards.
    """
    if not 0.0 <= v61_prob <= 1.0:
        raise ValueError(f"Expected V61 prob in [0, 1], got {v61_prob}")
    return round(v61_prob * 100, 2)


def evo_weighted_to_v61_prob(evo_score: float) -> float:
    """Rough display-only conversion from EVO-PCA weighted score (0–100)
    to an approximate V61 probability (0–1).

    NOT calibrated — use only for logging and cross-system dashboards.
    """
    if not 0.0 <= evo_score <= 100.0:
        raise ValueError(f"Expected EVO-PCA score in [0, 100], got {evo_score}")
    return round(evo_score / 100, 4)
