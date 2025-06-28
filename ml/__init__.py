"""Machine Learning package for DART trading platform.

Provides AI trading models, feature extraction, risk management,
and automated trading capabilities.
"""

from ml.auto_trader import AutoTrader
from ml.trading_ai import TradingAI

# Optional imports that may fail if dependencies are missing
try:
    from ml.deep_rl_agent import SoftActorCriticV2
    from ml.feature_extractor import MultiModalFeatureExtractor
    from ml.risk_manager import AdvancedRiskManager
except ImportError:
    SoftActorCriticV2 = None
    MultiModalFeatureExtractor = None
    AdvancedRiskManager = None

__all__ = [
    "AutoTrader",
    "TradingAI",
    "SoftActorCriticV2",
    "MultiModalFeatureExtractor",
    "AdvancedRiskManager",
]
