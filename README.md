
# D.A.R.T - Deep Adaptive Reinforcement Trader

![Project Status](https://img.shields.io/badge/status-in%20development-yellow)

## Overview

DART (Deep Adaptive Reinforcement Trader) is a Python application designed to visualize and analyze financial market data with predictive capabilities. The current implementation provides a robust market data visualization platform with linear regression-based price projections. The long-term vision is to evolve DART into a full-fledged deep reinforcement learning-based trading agent that adapts to changing market conditions and makes informed trading decisions.

## Features

- **Real-time Market Data**: Connect to Deriv API to fetch live market data
- **Interactive Charts**: View candlestick charts for various markets and timeframes
- **Price Projections**: Generate and visualize price projections using linear regression
- **Confidence Intervals**: Display upper and lower confidence bounds for price projections
- **Customizable Timeframes**: Select from multiple timeframe options (1 minute to 1 day)
- **Modern UI**: Clean, responsive interface with dark mode support

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

1. Run the application:
   ```
   python main.py
   ```

2. Select a market from the dropdown menu
3. Choose a timeframe (1 minute to 1 day)
4. Toggle the "Show Projection" checkbox to enable/disable price projections
5. The chart will update automatically with the selected options

## Configuration

You can customize various settings in the `config/settings.py` file:

- `DERIV_APP_ID`: Your Deriv API application ID (register at [Deriv API](https://api.deriv.com/))
- `DEFAULT_THEME`: UI theme ('dark' or 'light')
- `DEFAULT_TIMEFRAME`: Default timeframe for charts
- `DEFAULT_CANDLE_COUNT`: Number of candles to fetch and display
- `CHART_STYLES`: Customize chart appearance for both dark and light themes

## Project Structure

```
DART/
├── api/                  # API client modules
│   └── deriv_client.py   # Deriv API integration
├── config/               # Configuration files
│   └── settings.py       # Application settings
├── ui/                   # User interface components
│   ├── app.py            # Main application UI
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

## Future Development

The current implementation focuses on market data visualization and basic price projections. Future development plans include:

1. **Deep Learning Integration**: Implement neural network models for more accurate price predictions
2. **Reinforcement Learning**: Develop a trading agent that learns from market interactions
3. **Technical Indicators**: Add support for various technical indicators (RSI, MACD, etc.)
4. **Backtesting**: Create a framework for testing strategies on historical data
5. **Portfolio Management**: Add features for managing and tracking multiple positions
6. **Risk Management**: Implement risk assessment and position sizing algorithms

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