import pytest
import os
import sys

# Setup path so we can import from models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.security.advanced_heuristics import VotingAggregator, RiskSignal, ActionTier

def test_action_tier_enum():
    """Verify ActionTier has all 5 required values."""
    expected = {"ALLOW", "MONITOR", "REVIEW", "DENY", "QUARANTINE"}
    actual = {t.name for t in ActionTier}
    assert expected.issubset(actual), f"Missing tiers! Found: {actual}"

def test_voting_aggregator_branches():
    """Test VotingAggregator with different signals to hit all 5 decision branches."""
    
    # 1. ALLOW Branch (Empty or low score)
    tier, score = VotingAggregator.vote([])
    assert tier == ActionTier.ALLOW
    
    tier, score = VotingAggregator.vote([
        RiskSignal(name="low_risk", severity=20, confidence=0.5)
    ])
    assert tier == ActionTier.ALLOW

    # 2. MONITOR Branch
    tier, score = VotingAggregator.vote([
        RiskSignal(name="med_risk", severity=45, confidence=0.8)
    ])
    assert tier == ActionTier.MONITOR

    # 3. REVIEW Branch (Weighted >= 68)
    tier, score = VotingAggregator.vote([
        RiskSignal(name="high_risk", severity=70, confidence=0.9)
    ])
    assert tier == ActionTier.REVIEW

    # 4. DENY Branch (Weighted >= 82 AND max > 85)
    tier, score = VotingAggregator.vote([
        RiskSignal(name="very_high", severity=86, confidence=1.0)
    ])
    assert tier == ActionTier.DENY

    # 5. QUARANTINE Branch (Weighted >= 90 or is_critical)
    # 5a. Score based QUARANTINE
    tier, score = VotingAggregator.vote([
        RiskSignal(name="extreme", severity=95, confidence=1.0)
    ])
    assert tier == ActionTier.QUARANTINE

    # 5b. Critical flag based QUARANTINE
    tier, score = VotingAggregator.vote([
        RiskSignal(name="critical_flag", severity=10, confidence=0.5, is_critical=True)
    ])
    assert tier == ActionTier.QUARANTINE
