
# D.A.R.T - Deep Adaptive Reinforcement Trader

![Project Status](https://img.shields.io/badge/status-active-green)

## Overview

DART (Deep Adaptive Reinforcement Trader) is a Python application designed to visualize, analyze, and automatically trade financial markets using advanced machine learning techniques. The application connects to the Deriv API to fetch real-time market data, train ML models on historical data, generate optimal trading strategies, and execute trades automatically. DART features an adaptive learning system that continuously improves its strategies based on trade outcomes.

## Features

### Market Analysis and Visualization
- **Real-time Market Data**: Connect to Deriv API to fetch live market data
- **Market Status Indicators**: Clearly shows which markets are closed with visual indicators
- **Interactive Charts**: View candlestick charts for various markets and timeframes
- **Price Projections**: Generate and visualize price projections using linear regression
- **Confidence Intervals**: Display upper and lower confidence bounds for price projections
- **Customizable Timeframes**: Select from multiple timeframe options (1 minute to 1 day)
- **Theme Switching**: Toggle between dark and light themes for both UI and charts
- **Responsive Layout**: Automatically adjusts to different window sizes and screen resolutions
- **Enhanced Chart Styling**: Customizable chart colors and styles for better visualization
- **Modern UI**: Clean, responsive interface with professional styling
- **Account Dashboard**: View account connection status and balance in real-time

### AI-Powered Auto-Trading
- **Multi-Model Training**: Train and compare multiple ML models (Gradient Boosting, Random Forest, Logistic Regression)
- **Advanced Technical Indicators**: Utilize over 20 technical indicators from the TA library
- **Intelligent Strategy Generation**: Create optimal trading strategies based on ML predictions and technical analysis
- **Technical Signal Confirmation**: Validate ML predictions with MACD, RSI, and Bollinger Bands signals
- **Short-Time Trading**: Focus on short-duration trades with auto-selected optimal timeframes
- **Adaptive Learning**: Recalculate and improve strategies based on trade outcomes
- **Performance Analysis**: Track win rates by market, time of day, and trade direction
- **Manual Price Setting**: Specify exact trading prices for precise entry and exit points
- **AI-Managed Price**: Let AI calculate optimal entry and exit prices based on market conditions
- **Full AI Trading Management**: Enable AI to fully manage your trading strategy, position sizing, and risk management
- **Adjustable Confidence Threshold**: Fine-tune trading frequency with a slider to set the minimum confidence level for trade execution

### Advanced Risk Management
- **Dynamic Position Sizing**: Automatically adjust trade size based on performance and drawdown
- **Drawdown Protection**: Pause trading when drawdown exceeds configurable thresholds
- **Performance-Based Trading**: Analyze win rates across different markets and timeframes
- **Daily Loss Limits**: Set maximum daily loss limits to protect your account
- **Session Tracking**: Monitor and log performance metrics for each trading session
- **Performance Reports**: Generate detailed reports on trading performance and profitability
- **Trade Logging**: Maintain detailed logs of all trades for analysis and auditing

## Installation

### Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

### Installing uv

If you don't have uv installed, you can install it using:

```bash
# On Windows (PowerShell)
curl -LsSf https://astral.sh/uv/install.ps1 | powershell

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/ItzSwapnil/DART.git
   cd DART
   ```

2. Create and activate a virtual environment (recommended):
   ```
   # Using uv
   uv venv .venv

   # Activate the virtual environment
   # On Windows
   .venv\Scripts\activate
   # On Unix/MacOS
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   # Using uv
   uv pip install -e .
   ```

## Usage

### Basic Usage

1. Run the application:
   ```
   python main.py
   ```

2. Select a market from the dropdown menu
3. Choose a timeframe (1 minute to 1 day)
4. Toggle the "Show Projection" checkbox to enable/disable price projections
5. The chart will update automatically with the selected options

### Auto-Trading
1. Select a market and timeframe for trading
2. Click the "Train Model" button to train the AI on 7 days of historical data
3. Adjust the confidence threshold slider to set the minimum confidence level for trade execution
4. Check your account connection status and balance in the Account Dashboard
5. Once training is complete, click "Start Auto-Trading" to begin automated trading
6. Monitor trade status, win rate, and profit/loss in the trading controls section
7. View detailed live trade monitoring information in the Live Trade Monitoring panel, including:
   - Real-time price updates
   - Current profit/loss percentage
   - Countdown timer showing remaining trade duration
   - Entry and current prices with visual indicators
8. The system automatically adjusts trade durations to match market-specific limitations
9. Click "Stop Auto-Trading" to halt the trading process at any time

## Configuration

You can customize various settings in the `config/settings.py` file:

### API Settings
- `DERIV_APP_ID`: Your Deriv API application ID (register at [Deriv API](https://api.deriv.com/))
- `DERIV_API_TOKEN`: Your Deriv API token for authentication and trading

### API Token Setup (Important for Trading)

To use the auto-trading features and see live charts, you must replace the placeholder API token in `config/settings.py` with a valid token:

1. Create a Deriv.com account if you don't have one
2. Log in to your Deriv.com account
3. Go to [Deriv API Token Page](https://app.deriv.com/account/api-token)
4. Create a new token with the following permissions:
   - Read
   - Trade
   - Payments
   - Admin
   - Trading information
5. Copy your new token
6. Open `config/settings.py` and replace the value of `DERIV_API_TOKEN` with your token:
   ```python
   DERIV_API_TOKEN = 'your_actual_token_here'
   ```

**Note**: Without a valid API token, you will receive "Please log in" errors and charts may not display. The token provided in the default configuration is a placeholder and will not work for trading operations or data retrieval.

### Auto-Trading Settings
- `AUTO_TRADE_ENABLED`: Enable/disable auto-trading by default
- `TRADE_DURATION_SECONDS`: Default duration for trades
- `TRADE_AMOUNT`: Default amount to stake on each trade
- `TRADE_CURRENCY`: Currency to use for trading
- `MAX_DAILY_LOSS`: Maximum daily loss limit
- `MAX_CONSECUTIVE_LOSSES`: Maximum consecutive losses before recalculating strategy

### ML Model Settings
- `TRAINING_DAYS`: Number of days of historical data to use for training
- `MODEL_UPDATE_FREQUENCY`: Hours between model updates
- `CONFIDENCE_THRESHOLD`: Minimum confidence level to execute a trade

### UI Settings
- `DEFAULT_THEME`: UI theme ('dark' or 'light')
- `DEFAULT_TIMEFRAME`: Default timeframe for charts
- `DEFAULT_CANDLE_COUNT`: Number of candles to fetch and display
- `CHART_STYLES`: Customize chart appearance for both dark and light themes

## Project Structure

```
DART/
├── api/                  # API client modules
│   └── deriv_client.py   # Deriv API integration with trading capabilities
├── config/               # Configuration files
│   └── settings.py       # Application settings
├── ml/                   # Machine learning modules
│   ├── trading_ai.py     # AI system for market analysis and strategy generation
│   └── auto_trader.py    # Automated trading manager
├── models/               # Directory for saved ML models (created at runtime)
├── ui/                   # User interface components
│   ├── app.py            # Main application UI with trading controls
│   └── chart_styles.py   # Chart styling utilities
├── utils/                # Utility functions
│   └── timeframe.py      # Timeframe conversion utilities
├── main.py               # Application entry point
├── pyproject.toml        # Project dependencies and metadata
└── uv.lock               # Lock file for uv dependency management
```

## Dependencies

DART uses [uv](https://github.com/astral-sh/uv) for dependency management. The main dependencies include:

- `asyncio`: Asynchronous I/O, event loop, and coroutines
- `customtkinter`: Modern-looking tkinter widgets
- `matplotlib`: Plotting library for Python
- `mplfinance`: Matplotlib utilities for financial chart visualization
- `pandas`: Data analysis and manipulation library
- `python-deriv-api`: Official Deriv API client for Python
- `scikit-learn`: Machine learning library (used for linear regression)
- `sv-ttk`: Sun Valley theme for tkinter
- `tk`: Tkinter GUI toolkit

## Development

### Adding or Updating Dependencies

To add or update dependencies, modify the `pyproject.toml` file and then run:

```
uv pip install -e .
```

This will update the `uv.lock` file with the exact versions of all dependencies.

## Current Status and Future Development

The current implementation includes advanced market data visualization, multi-model machine learning, comprehensive technical analysis, and sophisticated auto-trading capabilities. The system trains and compares multiple ML models (Gradient Boosting, Random Forest, and Logistic Regression) on historical data, combines ML predictions with technical analysis signals, and adapts trading strategies based on performance analytics. It features advanced risk management with dynamic position sizing, drawdown protection, and detailed performance tracking.

### Implemented Features
- ✅ **Multi-Model Machine Learning**: Gradient boosting, Random Forest, and Logistic Regression models
- ✅ **Advanced Technical Indicators**: Comprehensive set of indicators from the TA library (MACD, RSI, Bollinger Bands, etc.)
- ✅ **Technical Analysis Integration**: Combining ML predictions with technical analysis signals
- ✅ **Auto-Trading**: Automated trade execution with the Deriv API
- ✅ **Adaptive Learning**: Strategy recalculation based on trade outcomes and performance analysis
- ✅ **Advanced Risk Management**: Dynamic position sizing, drawdown protection, and performance tracking
- ✅ **Performance Analytics**: Detailed performance reports and trade logging
- ✅ **Theme Switching**: Toggle between dark and light themes for UI and charts
- ✅ **Responsive UI**: Window sizing and layout optimization for different screen sizes
- ✅ **Market Status Indicators**: Visual indicators for closed markets
- ✅ **Market-Specific Duration Handling**: Automatic adjustment of trade durations to match market-specific limitations
- ✅ **Live Trade Monitoring**: Real-time updates with current prices, profit/loss percentage, and countdown timer
- ✅ **Fixed Price Retrieval**: Improved market price retrieval for symbols like WLDAUD to prevent validation errors

### Future Development Plans
1. **Deep Neural Networks**: Implement more advanced neural network architectures for improved prediction accuracy
2. **Reinforcement Learning**: Develop a true RL-based trading agent that learns optimal actions through market interactions
3. **Pattern Recognition**: Add support for candlestick pattern recognition and chart pattern analysis
4. **Backtesting Framework**: Create a comprehensive system for testing strategies on historical data
5. **Portfolio Management**: Add features for managing and tracking multiple positions across different markets
6. **Performance Optimization**: Improve model training and prediction speed for faster strategy generation
7. **Web Interface**: Develop a web-based interface for remote monitoring and control
8. **Mobile App Integration**: Create a companion mobile app for monitoring trades on the go
9. **Social Trading**: Add features for sharing strategies and following other traders
10. **Market Sentiment Analysis**: Incorporate news and social media sentiment into trading decisions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Deriv API](https://api.deriv.com/) for providing market data
- [mplfinance](https://github.com/matplotlib/mplfinance) for financial visualization tools
- [scikit-learn](https://scikit-learn.org/) for machine learning capabilities
- [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management
