"""Configuration settings for the DART application."""

# Deriv API settings
DERIV_APP_ID = '00000' # Change it to your app ID
# You need a valid API token from Deriv.com to execute trades
# Get your API token from: https://app.deriv.com/account/api-token
DERIV_API_TOKEN = '0000000'    # Replace with your valid API token

# Auto-trading settings
AUTO_TRADE_ENABLED = False  # Default to disabled
TRADE_DURATION_SECONDS = 60  # Default duration for trades (1 minute)
TRADE_AMOUNT = 10.0  # Default trade amount
TRADE_CURRENCY = 'USD'  # Default currency
MAX_DAILY_LOSS = 100.0  # Maximum daily loss limit
MAX_CONSECUTIVE_LOSSES = 3  # Maximum consecutive losses before recalculating strategy

# ML model settings
TRAINING_DAYS = 7  # Number of days of historical data to use for training
MODEL_UPDATE_FREQUENCY = 24  # Hours between model updates
CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence level to execute a trade

# UI settings
DEFAULT_THEME = 'dark' #change it to 'light' if you prefer light background
DEFAULT_TIMEFRAME = '1 minute'
DEFAULT_CANDLE_COUNT = 50

# Chart settings
CHART_STYLES = {
    'dark': {
        'base_mpl_style': 'dark_background',
        'up_color': 'green',
        'down_color': 'red',
        'facecolor': '#121212',
        'gridcolor': '#2A2A2A',
        'gridstyle': '--',
        'mavcolors': ['#1f77b4', '#ff7f0e', '#2ca02c']
    },
    'light': {
        'base_mpl_style': 'default',
        'up_color': 'green',
        'down_color': 'red',
        'facecolor': 'white',
        'gridcolor': '#E6E6E6',
        'gridstyle': '-',
        'mavcolors': ['#1f77b4', '#ff7f0e', '#2ca02c']
    }
}
