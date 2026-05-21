"""Machine Learning package for DART trading platform.

Provides AI trading models, feature extraction, risk management,
and automated trading capabilities.
"""

try:
    from ml.auto_trader import AutoTrader
except ImportError:
    AutoTrader = None

try:
    from ml.trading_ai_v3 import TradingAIv3
    TradingAI = TradingAIv3  # Map legacy name to modern implementation
except ImportError:
    TradingAIv3 = None
    TradingAI = None

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
    "TradingAIv3",
    "TradingAI",
    "SoftActorCriticV2",
    "MultiModalFeatureExtractor",
    "AdvancedRiskManager",
]
