# 🤖 DART AI Features Guide

## **What are Deep RL and Enhanced Features?**

### **🧠 Deep RL (Deep Reinforcement Learning)**
- **What it does**: Advanced AI that learns optimal trading strategies through trial and error
- **How it works**: Uses neural networks (SAC - Soft Actor-Critic) to adapt to market conditions
- **Benefits**: 
  - Learns from successful/failed trades
  - Adapts to changing market conditions
  - Develops complex trading strategies
  - Better long-term performance
- **When to use**: For complex markets and adaptive strategies
- **Resource requirements**: More CPU/GPU intensive

### **🔬 Enhanced Features**
- **What it does**: Multi-modal feature extraction using advanced technical analysis
- **Components**:
  - **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, Volume indicators
  - **Pattern Recognition**: Chart patterns, support/resistance levels
  - **Sentiment Analysis**: Market sentiment from multiple sources
  - **Volume Analysis**: Order flow and volume-based signals
- **Benefits**:
  - More comprehensive market analysis
  - Better signal accuracy
  - Reduced false positives
  - Multi-timeframe analysis
- **When to use**: For detailed market analysis and higher accuracy

## **Recommended Settings**

### **For Beginners:**
- Deep RL: ❌ OFF (use traditional ML first)
- Enhanced Features: ✅ ON (better analysis)
- Confidence Threshold: 70-80%

### **For Advanced Users:**
- Deep RL: ✅ ON (adaptive learning)
- Enhanced Features: ✅ ON (maximum analysis)
- Confidence Threshold: 60-70%

### **For High-Frequency Trading:**
- Deep RL: ✅ ON (quick adaptation)
- Enhanced Features: ❌ OFF (faster execution)
- Confidence Threshold: 80-90%

## **Training Process**

1. **Data Collection**: Fetches 7 days of historical data (4000+ candles)
2. **Feature Engineering**: Creates 20+ technical indicators
3. **Model Training**: Trains 3 ML models (Random Forest, Gradient Boosting, Logistic Regression)
4. **Validation**: Cross-validation and performance testing
5. **Model Selection**: Chooses best performing model
6. **Optimization**: Fine-tunes parameters for selected market

## **Performance Metrics**

- **Accuracy**: Overall prediction correctness (50-70% is good)
- **Precision**: True positive rate (higher = fewer false signals)
- **Win Rate**: Percentage of profitable trades
- **F1 Score**: Balance between precision and recall

## **Troubleshooting**

### **Training Fails:**
- Check internet connection
- Try different market (some have more data)
- Wait and retry (API rate limits)

### **Low Accuracy:**
- Use Enhanced Features
- Try longer timeframes (5min, 15min)
- Train on more volatile markets

### **No Trading Signals:**
- Lower confidence threshold
- Check market hours (some markets closed)
- Retrain model with recent data
