"""Configuration settings for the DART application."""

# Deriv API settings
DERIV_APP_ID = '00000' #Change it to your app ID

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
