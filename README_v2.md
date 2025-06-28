# 🎯 DART - Deep Adaptive Reinforcement Trader v2.0

![Project Status](https://img.shields.io/badge/status-active-green)
![Version](https://img.shields.io/badge/version-2.0-blue)
![AI Powered](https://img.shields.io/badge/AI-powered-purple)

**Professional AI Trading Platform with Advanced Machine Learning**

DART is a cutting-edge trading platform that combines traditional algorithmic trading with advanced AI techniques including Deep Reinforcement Learning, multi-modal feature extraction, and sophisticated risk management. Experience professional-grade trading with dual interface options - a stunning modern web dashboard and a comprehensive desktop application.

## ✨ What's New in v2.0

### 🚀 Dual Interface System
- **🌐 Modern Web Dashboard**: Sleek web-based interface with real-time interactive charts, AI insights, and professional styling
- **🖥️ Enhanced Desktop Interface**: Completely redesigned native application with tabbed interface and advanced controls
- **🎮 Smart Launcher**: Beautiful interface selector to choose your preferred trading experience

### 🤖 Advanced AI Features
- **🧠 Deep Reinforcement Learning**: Soft Actor-Critic (SAC) algorithm for adaptive market learning
- **🔬 Multi-Modal Analysis**: Technical, fundamental, and sentiment analysis integration
- **🎯 Ensemble Learning**: Multiple ML models working together for robust predictions
- **📊 Real-time Market Sentiment**: AI-powered sentiment analysis with visual indicators
- **⚡ Adaptive Risk Management**: Dynamic position sizing and portfolio optimization

### 📊 Professional Analytics
- **📈 Interactive Charts**: Plotly-powered charts with advanced technical indicators
- **🎛️ Real-time Dashboards**: Live performance metrics and trading status
- **📋 Comprehensive Reporting**: Detailed analytics and trade history tracking
- **📱 Responsive Design**: Works seamlessly across desktop, tablet, and mobile devices

## 🛠️ Installation & Quick Start

### Prerequisites
- Python 3.8+ (recommended: Python 3.11+)
- Valid Deriv.com account and API token
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer (recommended)

### Installation
```bash
# Clone the repository
git clone https://github.com/ItzSwapnil/DART.git
cd DART

# Install with uv (recommended)
uv venv .venv
uv pip install -e .

# Or install with pip
pip install -r requirements.txt
```

### Configuration
1. **Get your Deriv API credentials**:
   - Visit [Deriv API Token Page](https://app.deriv.com/account/api-token)
   - Create a token with Read, Trade, and Payments permissions

2. **Configure DART**:
   ```python
   # Edit config/settings.py
   DERIV_APP_ID = 'your_app_id'
   DERIV_API_TOKEN = 'your_api_token'
   ```

3. **Launch DART**:
   ```bash
   python main.py
   ```

## 🎮 Interface Options

### 🌐 Modern Web Dashboard
Experience the future of trading interfaces with our professional web dashboard:

**Features:**
- 📊 **Real-time Interactive Charts**: Plotly-powered candlestick charts with technical indicators
- 🎛️ **Live Performance Metrics**: Real-time P&L, win rates, and portfolio tracking
- 🤖 **AI Market Sentiment**: Visual sentiment gauge with market insights
- 📱 **Responsive Design**: Optimized for desktop, tablet, and mobile
- 🎨 **Professional Styling**: Modern dark/light themes with gradient effects
- ⚡ **Real-time Updates**: Live data streaming and instant notifications

**Access**: Automatically opens at `http://localhost:8501` in your default browser

### 🖥️ Enhanced Desktop Interface
Comprehensive native desktop application with professional features:

**Features:**
- 📑 **Tabbed Interface**: Organized workspace with Trading, Analytics, AI Management, and Settings tabs
- 📊 **Advanced Charts**: Real-time candlestick charts with AI projections
- 🤖 **AI Model Management**: Train, evaluate, and monitor AI models
- 📈 **Live Trade Monitoring**: Real-time trade tracking with detailed logs
- ⚙️ **Comprehensive Settings**: Full configuration and customization options
- 🎯 **Professional Controls**: Advanced trading controls with risk management

**Access**: Native desktop application window with enhanced UI

## 🤖 AI-Powered Trading

### Deep Learning Models
- **🧠 Soft Actor-Critic (SAC)**: Advanced reinforcement learning for market adaptation
- **🌲 Random Forest**: Ensemble learning for robust market predictions
- **🚀 Gradient Boosting**: Sequential learning with error correction
- **📊 Logistic Regression**: Linear trend analysis and classification

### Technical Analysis
- **📈 20+ Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, and more
- **🔍 Pattern Recognition**: Advanced market pattern detection
- **📊 Multi-timeframe Analysis**: From 1-minute to daily chart analysis
- **🎯 Signal Confirmation**: Multiple indicator confirmation system

### Risk Management
- **💰 Dynamic Position Sizing**: Automated position size calculation
- **⚠️ Portfolio Risk Limits**: Maximum exposure and correlation controls
- **📉 Drawdown Protection**: Automatic risk reduction during losses
- **🛡️ Stop Loss Integration**: Smart stop-loss and take-profit levels

## 📊 Key Features

### Real-time Market Data
- 🔗 **Deriv API Integration**: Direct connection to live market data
- 🌍 **Multiple Markets**: Forex, Cryptocurrencies, Indices, Commodities
- ⏱️ **Multiple Timeframes**: 1-minute to daily analysis
- 📊 **Live Price Feeds**: Real-time price updates and market status

### Advanced Analytics
- 📈 **Performance Tracking**: Comprehensive P&L and performance metrics
- 📊 **Win Rate Analysis**: Detailed success rate tracking by market and strategy
- 📋 **Trade History**: Complete trade log with detailed analytics
- 📱 **Mobile-Friendly Reports**: Access analytics from any device

### User Experience
- 🎨 **Professional Themes**: Dark and light modes for all interfaces
- 📱 **Responsive Design**: Optimized for all screen sizes
- ⚡ **Real-time Updates**: Live data streaming and notifications
- 🎯 **Intuitive Controls**: User-friendly interface with professional styling

## 🎯 How to Use

### Getting Started
1. **Launch DART** and select your preferred interface
2. **Configure API credentials** in settings
3. **Choose a market** (EUR/USD, Bitcoin, Ethereum, etc.)
4. **Select timeframe** (1 minute, 5 minutes, 1 hour, etc.)
5. **Train AI model** with historical market data
6. **Start trading** manually or enable automation

### AI Model Training
```python
# Quick training process
1. Select target market and timeframe
2. Click "🎯 Train Model"
3. Wait for completion (30-60 seconds)
4. Review accuracy metrics (target: 75%+)
5. Model ready for trading
```

### Automated Trading
```python
# Set up automated trading
1. Ensure model is trained
2. Configure risk settings
3. Set confidence threshold (0.6-0.8 recommended)
4. Click "🚀 Start Trading"
5. Monitor live performance
```

### Manual Analysis
```python
# Use AI for manual trading
1. Select market and timeframe
2. Click "📊 Analyze Market"
3. Review AI insights and sentiment
4. Use analysis for trading decisions
```

## 🔧 Configuration

### AI Settings
```python
# Advanced AI configuration
USE_DEEP_RL = True              # Enable Deep RL
USE_ENHANCED_FEATURES = True    # Multi-modal features
CONFIDENCE_THRESHOLD = 0.6      # Trading confidence
```

### Risk Management
```python
# Risk control settings
MAX_DAILY_LOSS = 100.0         # Daily loss limit
TRADE_AMOUNT = 10.0            # Default trade size
MAX_PORTFOLIO_RISK = 0.02      # Portfolio risk (2%)
```

### Interface Settings
```python
# UI customization
DEFAULT_THEME = 'dark'         # Theme preference
DEFAULT_TIMEFRAME = '5 minutes' # Default timeframe
AUTO_REFRESH = True            # Auto-refresh data
```

## 📈 Performance Metrics

DART provides comprehensive performance tracking:

### Trading Metrics
- 📊 **Win Rate**: Percentage of profitable trades
- 💰 **Profit/Loss**: Total and daily P&L tracking
- 📈 **Sharpe Ratio**: Risk-adjusted returns
- 📉 **Maximum Drawdown**: Largest decline measurement
- ⏱️ **Trade Frequency**: Trades per time period

### AI Metrics
- 🎯 **Model Accuracy**: Prediction accuracy percentage
- 🧠 **Confidence Levels**: AI confidence in predictions
- 📊 **Feature Importance**: Most influential indicators
- 🔄 **Adaptation Rate**: Model learning speed

## 🔒 Security & Safety

### Security Features
- 🔐 **Encrypted API Storage**: Secure credential management
- 🛡️ **Input Validation**: Protection against malicious inputs
- 📝 **Audit Logging**: Complete activity tracking
- 🔒 **Secure Connections**: Encrypted API communications

### Risk Controls
- 💰 **Position Limits**: Maximum position size controls
- 🛑 **Stop Losses**: Automatic loss limitation
- 📊 **Portfolio Monitoring**: Real-time exposure tracking
- ⚠️ **Risk Alerts**: Immediate risk notifications

## 🆘 Support & Documentation

### Getting Help
- 📚 **Full Documentation**: Comprehensive guides and tutorials
- 🐛 **Bug Reports**: GitHub Issues for technical problems
- 💬 **Community**: GitHub Discussions for questions
- 📧 **Direct Support**: Email support for urgent issues

### Resources
- 🎥 **Video Tutorials**: Step-by-step usage guides
- 📖 **API Documentation**: Complete API reference
- 🔧 **Troubleshooting**: Common issues and solutions
- 📊 **Best Practices**: Trading strategy recommendations

## 🚀 Roadmap

### Immediate (Q2 2024)
- [ ] 📱 Mobile companion app
- [ ] 🔄 Additional broker integrations
- [ ] 🤝 Social trading features
- [ ] 📊 Advanced portfolio optimization

### Future (Q3-Q4 2024)
- [ ] 🌐 Multi-asset portfolio management
- [ ] 🎨 Custom strategy builder
- [ ] 📰 News sentiment integration
- [ ] 🔍 Advanced backtesting framework
- [ ] 🤖 ML model marketplace

## ⚠️ Disclaimer

**Important Trading Notice**: Trading financial instruments involves substantial risk and may not be suitable for all investors. Past performance does not guarantee future results. The AI models and strategies provided are for educational and research purposes only.

**No Financial Advice**: DART is a technology platform and should not be considered financial advice. Always conduct your own research and consider consulting with a qualified financial advisor before trading.

**Use at Your Own Risk**: All trading decisions and their consequences are solely your responsibility. Start with small amounts and thoroughly understand the risks involved.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions from the community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to contribute.

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/YourUsername/DART.git
cd DART

# Create development environment
uv venv .venv
uv pip install -e ".[dev]"

# Run tests
python -m pytest

# Submit pull request
```

## 🙏 Acknowledgments

- **Deriv.com**: For providing robust trading API
- **Plotly**: For beautiful interactive charts
- **Streamlit**: For rapid web app development
- **scikit-learn**: For machine learning capabilities
- **Community**: For feedback and contributions

---

**🎯 Built with ❤️ by the DART Team**

*Empowering traders with AI-driven insights and professional trading tools.*

**🌟 Star us on GitHub if you find DART useful!**
