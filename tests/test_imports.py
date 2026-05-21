"""Test that all DART modules can be imported successfully."""



class TestCoreImports:
    """Test core module imports."""

    def test_import_deriv_client(self):
        """Test DerivClient can be imported."""
        from api.deriv_client import DerivClient

        assert DerivClient is not None

    def test_import_deriv_oauth(self):
        """Test DerivOAuth can be imported."""
        from api.deriv_client import DerivOAuth

        assert DerivOAuth is not None

    def test_import_trading_ai(self):
        """Test TradingAIv3 can be imported."""
        from ml.trading_ai_v3 import TradingAIv3

        assert TradingAIv3 is not None

    def test_import_auto_trader(self):
        """Test AutoTrader can be imported."""
        from ml.auto_trader import AutoTrader

        assert AutoTrader is not None

    def test_import_dart_config(self):
        """Test dart_config can be imported."""
        from config.dart_config import get_config, load_config

        assert get_config is not None
        assert load_config is not None

    def test_config_has_oauth_fields(self):
        """Test config has OAuth 2.0 fields (not legacy api_token)."""
        from config.dart_config import get_config

        config = get_config()
        assert hasattr(config.api, "deriv_access_token")
        assert hasattr(config.api, "deriv_account_id")
        assert hasattr(config.api, "deriv_oauth_client_id")
        assert not hasattr(config.api, "deriv_api_token")


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
