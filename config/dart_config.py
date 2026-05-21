"""
SOTA Configuration Management for DART v3.0
Using Pydantic v2 for type-safe configuration with validation.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RiskManagementSettings(BaseModel):
    """Risk management configuration."""

    max_daily_loss: float = Field(default=100.0, description="Maximum daily loss limit")
    max_portfolio_risk: float = Field(default=0.02, description="Max portfolio risk (2%)")
    max_position_size: float = Field(default=0.1, description="Max position size (10%)")
    max_drawdown: float = Field(default=0.30, description="Max drawdown threshold (30%)")
    max_consecutive_losses: int = Field(default=3, description="Max consecutive losses")
    confidence_threshold: float = Field(default=0.4, description="Min confidence for trade")
    kelly_fraction: float = Field(default=0.25, description="Kelly criterion fraction")

    @field_validator("max_portfolio_risk", "max_position_size", "max_drawdown")
    @classmethod
    def validate_percentages(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Value must be between 0 and 1")
        return v


class AISettings(BaseModel):
    """AI/ML configuration."""

    use_deep_rl: bool = Field(default=True, description="Enable Deep RL (SAC)")
    use_ensemble_ml: bool = Field(default=True, description="Enable ensemble ML")
    use_llm_analysis: bool = Field(default=True, description="Enable LLM market analysis")
    require_real_llm: bool = Field(
        default=True,
        description="Fail startup if real LLM backend is unavailable",
    )
    ensemble_voting_threshold: float = Field(default=0.6, description="Ensemble consensus threshold")
    model_update_frequency_hours: int = Field(default=24, description="Model retraining frequency")
    training_data_days: int = Field(default=7, description="Historical data for training")

    # RL settings
    rl_learning_rate: float = Field(default=3e-4, description="RL actor learning rate")
    rl_gamma: float = Field(default=0.99, description="RL discount factor")
    rl_tau: float = Field(default=0.005, description="RL soft update coefficient")
    rl_buffer_size: int = Field(default=100000, description="RL replay buffer size")
    rl_batch_size: int = Field(default=256, description="RL training batch size")

    # LLM settings
    llm_model: str = Field(default="Qwen/Qwen2.5-1.5B-Instruct", description="LLM model name")
    llm_temperature: float = Field(default=0.3, description="LLM temperature")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model")


class TradingSettings(BaseModel):
    """Trading configuration."""

    symbol: str = Field(default="R_75", description="Default trading symbol")
    granularity: int = Field(default=60, description="Candle granularity in seconds")
    base_trade_amount: float = Field(default=10.0, description="Base trade amount")
    trade_currency: str = Field(default="USD", description="Trading currency")
    min_trade_amount: float = Field(default=1.0, description="Minimum trade amount")
    max_trade_amount: float = Field(default=100.0, description="Maximum trade amount")
    auto_trading_enabled: bool = Field(default=False, description="Enable auto-trading")

    @field_validator("granularity")
    @classmethod
    def validate_granularity(cls, v):
        if v not in [15, 30, 60, 120, 300, 600, 900, 1800, 3600]:
            raise ValueError("Invalid granularity")
        return v


class APISettings(BaseModel):
    """API configuration — OAuth 2.0."""

    deriv_app_id: str = Field(default="72212", description="Deriv app ID")
    deriv_oauth_client_id: str = Field(default="", description="OAuth 2.0 client ID from Deriv dashboard")
    deriv_oauth_redirect_uri: str = Field(
        default="http://localhost:8080/callback",
        description="OAuth 2.0 redirect URI",
    )
    deriv_access_token: str = Field(default="", description="OAuth 2.0 access token")
    deriv_account_id: str = Field(default="", description="Deriv Options account ID")
    llm_base_url: str = Field(default="http://localhost:8000/v1", description="vLLM (OpenAI-compatible) API URL")
    news_api_key: Optional[str] = Field(default=None, description="NewsAPI key")
    alpha_vantage_key: Optional[str] = Field(default=None, description="Alpha Vantage key")


class MonitoringSettings(BaseModel):
    """Monitoring and evaluation configuration."""

    wandb_enabled: bool = Field(default=False, description="Enable Weights & Biases")
    wandb_project: str = Field(default="DART-v3", description="W&B project name")
    mlflow_enabled: bool = Field(default=False, description="Enable MLflow")
    prometheus_enabled: bool = Field(default=False, description="Enable Prometheus metrics")
    log_level: str = Field(default="INFO", description="Logging level")
    evaluation_interval_trades: int = Field(default=50, description="Trades between evaluations")


class DARTSettings(BaseSettings):
    """Main DART v3.0 configuration."""

    # Settings management
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    # Components
    risk: RiskManagementSettings = Field(default_factory=RiskManagementSettings)
    ai: AISettings = Field(default_factory=AISettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    api: APISettings = Field(default_factory=APISettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    # Application settings
    app_name: str = "DART"
    app_version: str = "3.0.0"
    app_description: str = "Deep Adaptive Reinforcement Trader - State-of-the-Art AI Trading"
    debug_mode: bool = False
    model_dir: str = "./models"
    data_dir: str = "./data"
    logs_dir: str = "./logs"
    reports_dir: str = "./reports"

    def to_dict(self) -> dict:
        """Export configuration to dictionary."""
        return self.model_dump(mode="json")

    def validate_configuration(self) -> bool:
        """Validate critical configuration."""
        warnings = []

        if not self.api.deriv_access_token:
            warnings.append("Deriv OAuth access_token not set. Trading will not work.")
            warnings.append("  → Run OAuth login or set DERIV_ACCESS_TOKEN in .env")

        if not self.api.deriv_account_id:
            warnings.append("Deriv account_id not set. Will auto-detect from REST API if token is available.")

        if self.ai.use_llm_analysis:
            print(f"Info: LLM analysis enabled with model: {self.ai.llm_model}")
            print(f"  Requires vLLM running at: {self.api.llm_base_url}")

        for w in warnings:
            print(f"Warning: {w}")

        return len(warnings) == 0

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DART Configuration v{self.app_version}\n"
            f"  Trading: {self.trading.symbol} ({self.trading.granularity}s)\n"
            f"  AI: Deep RL={self.ai.use_deep_rl}, "
            f"Ensemble={self.ai.use_ensemble_ml}, LLM={self.ai.use_llm_analysis}\n"
            f"  Risk: Max Loss=${self.risk.max_daily_loss}, "
            f"Max Drawdown={self.risk.max_drawdown:.1%}\n"
            f"  Monitoring: W&B={self.monitoring.wandb_enabled}, "
            f"MLflow={self.monitoring.mlflow_enabled}"
        )


# Global configuration instance
_config: Optional[DARTSettings] = None


def get_config() -> DARTSettings:
    """Get or create global configuration."""
    global _config
    if _config is None:
        _config = DARTSettings()
    return _config


def load_config(path: str = ".env") -> DARTSettings:
    """Load configuration from file."""
    global _config
    _config = DARTSettings(_env_file=path)
    return _config
