"""DART v2.0 Configuration Settings."""

# Deriv API settings
DERIV_APP_ID = "72212"  # Change it to your app ID
# You need a valid API token from Deriv.com to execute trades
# Get your API token from: https://app.deriv.com/account/api-token
DERIV_API_TOKEN = "RIw8hz5LA6WYdnM"  # Replace with your valid API token

# Auto-trading settings
AUTO_TRADE_ENABLED = False  # Default to disabled
TRADE_DURATION_SECONDS = 60  # Default duration for trades (1 minute)
TRADE_AMOUNT = 10.0  # Default trade amount
TRADE_CURRENCY = "USD"  # Default currency
MAX_DAILY_LOSS = 100.0  # Maximum daily loss limit
MAX_CONSECUTIVE_LOSSES = 3  # Maximum consecutive losses before recalculating strategy

# ML model settings
TRAINING_DAYS = 7  # Number of days of historical data to use for training
MODEL_UPDATE_FREQUENCY = 24  # Hours between model updates
CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence level to execute a trade

# Enhanced AI settings (v2.0 - enabled by default)
USE_DEEP_RL = True  # Enable deep reinforcement learning (SAC v2.0)
USE_ENHANCED_FEATURES = True  # Enable multi-modal feature extraction
USE_ADVANCED_RISK_MANAGEMENT = True  # Enable sophisticated risk management

# Risk management settings
MAX_PORTFOLIO_RISK = 0.02  # Maximum portfolio risk (2%)
MAX_POSITION_SIZE = 0.1  # Maximum position size (10% of portfolio)
MAX_CORRELATION_EXPOSURE = 0.5  # Maximum correlation exposure (50%)
VAR_CONFIDENCE_LEVELS = [0.95, 0.99]  # Value at Risk confidence levels
STRESS_TEST_ENABLED = True  # Enable stress testing

# Feature extraction settings
INCLUDE_SENTIMENT_ANALYSIS = False  # Requires API keys
INCLUDE_FUNDAMENTAL_DATA = False  # Requires API keys
NEWS_API_KEY = None  # Get from newsapi.org
ALPHA_VANTAGE_KEY = None  # Get from alphavantage.co

# Deep learning settings (if enabled)
RL_LEARNING_RATE_ACTOR = 3e-4
RL_LEARNING_RATE_CRITIC = 3e-4
RL_GAMMA = 0.99  # Discount factor
RL_TAU = 0.005  # Soft update parameter
RL_BUFFER_SIZE = 100000  # Experience replay buffer size
RL_BATCH_SIZE = 256  # Training batch size

# Deep RL Configuration (v2.0 enhanced)
DEEP_RL_CONFIG = {
    "learning_rate_actor": RL_LEARNING_RATE_ACTOR,
    "learning_rate_critic": RL_LEARNING_RATE_CRITIC,
    "gamma": RL_GAMMA,
    "tau": RL_TAU,
    "buffer_size": RL_BUFFER_SIZE,
    "batch_size": RL_BATCH_SIZE,
    "enabled": USE_DEEP_RL,
    # v2.0 additions
    "n_steps": 3,  # N-step returns
    "use_curiosity": True,  # Curiosity-driven exploration
    "curiosity_coef": 0.01,  # Intrinsic reward coefficient
    "target_entropy_scale": 1.0,  # Entropy tuning scale
}

# Risk Management Configuration
RISK_MANAGEMENT_CONFIG = {
    "max_portfolio_risk": MAX_PORTFOLIO_RISK,
    "max_position_size": MAX_POSITION_SIZE,
    "max_correlation_exposure": MAX_CORRELATION_EXPOSURE,
    "var_confidence_levels": VAR_CONFIDENCE_LEVELS,
    "stress_test_enabled": STRESS_TEST_ENABLED,
    "advanced_enabled": USE_ADVANCED_RISK_MANAGEMENT,
}

# UI settings
DEFAULT_THEME = "dark"  # change it to 'light' if you prefer light background
DEFAULT_TIMEFRAME = "1 minute"
DEFAULT_CANDLE_COUNT = 50

# Chart settings
CHART_STYLES = {
    "dark": {
        "base_mpl_style": "dark_background",
        "up_color": "green",
        "down_color": "red",
        "facecolor": "#121212",
        "gridcolor": "#2A2A2A",
        "gridstyle": "--",
        "mavcolors": ["#1f77b4", "#ff7f0e", "#2ca02c"],
    },
    "light": {
        "base_mpl_style": "default",
        "up_color": "green",
        "down_color": "red",
        "facecolor": "white",
        "gridcolor": "#E6E6E6",
        "gridstyle": "-",
        "mavcolors": ["#1f77b4", "#ff7f0e", "#2ca02c"],
    },
}

# Performance monitoring
ENABLE_PERFORMANCE_LOGGING = True
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
PERFORMANCE_METRICS_RETENTION = 30  # Days to retain performance data

# Model persistence settings
AUTO_SAVE_MODELS = True
MODEL_SAVE_FREQUENCY = 24  # Hours between automatic model saves
KEEP_MODEL_VERSIONS = 5  # Number of model versions to keep
