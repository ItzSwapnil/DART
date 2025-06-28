"""Test that all DART modules can be imported successfully."""



class TestCoreImports:
    """Test core module imports."""

    def test_import_deriv_client(self):
        """Test DerivClient can be imported."""
        from api.deriv_client import DerivClient

        assert DerivClient is not None

    def test_import_trading_ai(self):
        """Test TradingAI can be imported."""
        from ml.trading_ai import TradingAI

        assert TradingAI is not None

    def test_import_auto_trader(self):
        """Test AutoTrader can be imported."""
        from ml.auto_trader import AutoTrader

        assert AutoTrader is not None

    def test_import_settings(self):
        """Test settings can be imported."""
        from config.settings import DERIV_APP_ID, TRADE_AMOUNT

        assert DERIV_APP_ID is not None
        assert TRADE_AMOUNT is not None


class TestMLImports:
    """Test ML module imports."""

    def test_import_feature_extractor(self):
        """Test feature extractor can be imported."""
        from ml.feature_extractor import MultiModalFeatureExtractor

        assert MultiModalFeatureExtractor is not None

    def test_import_risk_manager(self):
        """Test risk manager can be imported."""
        from ml.risk_manager import AdvancedRiskManager

        assert AdvancedRiskManager is not None

    def test_import_deep_rl_agent(self):
        """Test deep RL agent can be imported."""
        from ml.deep_rl_agent import SoftActorCriticV2

        assert SoftActorCriticV2 is not None


class TestUtilsImports:
    """Test utility module imports."""

    def test_import_api_utils(self):
        """Test API utilities can be imported."""
        from utils.api_utils import CircuitBreaker, RetryConfig

        assert CircuitBreaker is not None
        assert RetryConfig is not None

    def test_import_timeframe(self):
        """Test timeframe utility can be imported."""
        from utils.timeframe import TIMEFRAME_MAPPING

        assert TIMEFRAME_MAPPING is not None
