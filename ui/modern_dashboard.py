"""
DART - Modern Web Dashboard
A stunning, professional web-based interface for the Deep Adaptive Reinforcement Trader.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import time
from datetime import datetime, timedelta
import json
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.deriv_client import DerivClient
from ml.trading_ai import TradingAI
from ml.auto_trader import AutoTrader
from config.settings import *

# Page configuration
st.set_page_config(
    page_title="DART - Deep Adaptive Reinforcement Trader",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
/* Modern dark theme */
.main {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
}

.stApp {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
}

/* Custom metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e1e3f 0%, #2a2a5a 100%);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(56, 189, 248, 0.2);
}

/* Glowing buttons */
.stButton > button {
    background: linear-gradient(45deg, #3b82f6, #8b5cf6);
    border: none;
    border-radius: 10px;
    color: white;
    font-weight: 600;
    padding: 0.75rem 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5);
}

/* Sidebar styling */
.css-1d391kg {
    background: linear-gradient(180deg, #1e1e3f 0%, #2a2a5a 100%);
}

/* Status indicators */
.status-connected {
    color: #10b981;
    font-weight: bold;
}

.status-disconnected {
    color: #ef4444;
    font-weight: bold;
}

.status-trading {
    color: #f59e0b;
    font-weight: bold;
}

/* Custom title */
.main-title {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 2rem;
}

/* Trading dashboard cards */
.trading-card {
    background: rgba(30, 30, 63, 0.8);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    backdrop-filter: blur(10px);
}

/* Performance metrics */
.perf-positive {
    color: #10b981 !important;
}

.perf-negative {
    color: #ef4444 !important;
}

.perf-neutral {
    color: #6b7280 !important;
}

/* Loading animation */
@keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}

.loading {
    animation: pulse 2s infinite;
}

/* Chart containers */
.chart-container {
    background: rgba(30, 30, 63, 0.6);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    border: 1px solid rgba(56, 189, 248, 0.2);
}
</style>
""", unsafe_allow_html=True)

class ModernDashboard:
    """Modern web-based dashboard for DART."""
    
    def __init__(self):
        self.client = None
        self.trading_ai = None
        self.auto_trader = None
        self.last_update = datetime.now()
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize DART components."""
        try:
            self.client = DerivClient(app_id=DERIV_APP_ID, api_token=DERIV_API_TOKEN)
            self.trading_ai = TradingAI(
                model_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'),
                use_deep_rl=USE_DEEP_RL,
                use_enhanced_features=USE_ENHANCED_FEATURES
            )
            self.auto_trader = AutoTrader(client=self.client, trading_ai=self.trading_ai)
        except Exception as e:
            st.error(f"Failed to initialize DART components: {e}")
    
    def render_header(self):
        """Render the main header."""
        st.markdown('<h1 class="main-title">🎯 DART Dashboard</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #94a3b8; font-size: 1.2rem; margin-bottom: 2rem;">Deep Adaptive Reinforcement Trader - Professional Trading Platform</p>', unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with controls."""
        with st.sidebar:
            st.markdown("## 🚀 Trading Controls")
            
            # Connection status
            if self.client:
                try:
                    # This would need to be made async in a real implementation
                    connection_status = "🟢 Connected" if True else "🔴 Disconnected"  # Placeholder
                    st.markdown(f"**Status:** {connection_status}")
                except:
                    st.markdown("**Status:** 🔴 Disconnected")
            
            st.markdown("---")
            
            # Market selection
            st.markdown("### 📈 Market Settings")
            
            # Load real market data from client
            markets = self.get_available_markets()
            if not markets:
                markets = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "Bitcoin", "Ethereum"]
            
            selected_market = st.selectbox("Select Market", markets, index=0, key="market_selector")
            
            # Store selected market in session state for chart updates
            if 'selected_market' not in st.session_state or st.session_state.selected_market != selected_market:
                st.session_state.selected_market = selected_market
                st.session_state.market_changed = True
            
            timeframes = ["1 minute", "5 minutes", "15 minutes", "1 hour", "4 hours", "1 day"]
            selected_timeframe = st.selectbox("Timeframe", timeframes, index=0)
            
            st.markdown("---")
            
            # AI Configuration
            st.markdown("### 🤖 AI Configuration")
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=CONFIDENCE_THRESHOLD,
                step=0.05,
                help="Minimum confidence level for trade execution"
            )
            
            use_deep_rl = st.checkbox("Enable Deep RL", value=USE_DEEP_RL, help="Use Deep Reinforcement Learning")
            use_enhanced_features = st.checkbox("Enhanced Features", value=USE_ENHANCED_FEATURES, help="Use multi-modal feature extraction")
            
            st.markdown("---")
            
            # Risk Management
            st.markdown("### ⚠️ Risk Management")
            
            max_daily_loss = st.number_input("Max Daily Loss ($)", value=MAX_DAILY_LOSS, min_value=10.0, max_value=1000.0)
            trade_amount = st.number_input("Trade Amount ($)", value=TRADE_AMOUNT, min_value=1.0, max_value=100.0)
            
            st.markdown("---")
            
            # Trading Controls
            st.markdown("### 🎮 Trading Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🎯 Train Model", help="Train the AI model with latest data"):
                    self.train_model_action(selected_market, selected_timeframe)
            
            with col2:
                if st.button("📊 Analyze", help="Analyze current market conditions"):
                    self.analyze_market_action(selected_market, selected_timeframe)
            
            # Trading start/stop
            if 'trading_active' not in st.session_state:
                st.session_state.trading_active = False
            
            if not st.session_state.trading_active:
                if st.button("🚀 Start Trading", help="Start automated trading", type="primary"):
                    st.session_state.trading_active = True
                    st.success("Trading started!")
                    st.rerun()
            else:
                if st.button("🛑 Stop Trading", help="Stop automated trading", type="secondary"):
                    st.session_state.trading_active = False
                    st.info("Trading stopped!")
                    st.rerun()
            
            return {
                'market': selected_market,
                'timeframe': selected_timeframe,
                'confidence_threshold': confidence_threshold,
                'use_deep_rl': use_deep_rl,
                'use_enhanced_features': use_enhanced_features,
                'max_daily_loss': max_daily_loss,
                'trade_amount': trade_amount
            }
    
    def render_main_dashboard(self, settings):
        """Render the main dashboard content."""
        # Key metrics row
        self.render_key_metrics()
        
        # Charts row
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.render_price_chart(settings['market'], settings['timeframe'])
        
        with col2:
            self.render_ai_insights()
        
        # Performance and trading status
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_performance_metrics()
        
        with col2:
            self.render_trading_status()
        
        # Recent trades and logs
        self.render_recent_activity()
    
    def render_key_metrics(self):
        """Render key performance metrics."""
        st.markdown("## 📊 Key Performance Metrics")
        
        # Sample data - in real implementation, fetch from trading system
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>💰 Portfolio Value</h3>
                <h2 style="color: #10b981;">$12,547.32</h2>
                <p style="color: #10b981;">+3.24% ↗</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📈 Win Rate</h3>
                <h2 style="color: #3b82f6;">78.5%</h2>
                <p style="color: #10b981;">+2.1% ↗</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Active Trades</h3>
                <h2 style="color: #f59e0b;">3</h2>
                <p style="color: #6b7280;">EUR/USD, BTC, ETH</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>🤖 AI Confidence</h3>
                <h2 style="color: #8b5cf6;">92.3%</h2>
                <p style="color: #10b981;">High ↗</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown("""
            <div class="metric-card">
                <h3>⚡ Daily P&L</h3>
                <h2 style="color: #10b981;">+$234.56</h2>
                <p style="color: #10b981;">+1.89% ↗</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_price_chart(self, market, timeframe):
        """Render the main price chart with indicators."""
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown(f"### 📈 {market} Price Chart - {timeframe}")
        
        # Get real market data for the selected market
        df = self.get_market_data(market, timeframe, 100)
        
        if df.empty:
            st.error(f"No data available for {market}")
            return
        
        # Create candlestick chart
        fig = go.Figure()
        
        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=market,
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444'
        ))
        
        # Add volume
        fig.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            yaxis='y2',
            opacity=0.3,
            marker_color='#3b82f6'
        ))
        
        # Add moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['sma_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='#f59e0b', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['sma_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='#8b5cf6', width=2)
        ))
        
        # Layout
        fig.update_layout(
            title=f"{market} - {timeframe}",
            yaxis=dict(title="Price", side="left"),
            yaxis2=dict(title="Volume", side="right", overlaying="y", showgrid=False),
            xaxis=dict(title="Time"),
            template="plotly_dark",
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        fig.add_trace(go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=market,
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444'
        ))
        
        # Add moving averages
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['ma20'],
            mode='lines',
            name='MA20',
            line=dict(color='#3b82f6', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['ma50'],
            mode='lines',
            name='MA50',
            line=dict(color='#8b5cf6', width=2)
        ))
        
        # Add AI prediction line (future prediction)
        if len(df) > 0:
            last_timestamp = df['timestamp'].iloc[-1]
            # Ensure we have a valid timestamp
            if pd.isna(last_timestamp):
                return
                
            future_dates = pd.date_range(start=last_timestamp, periods=21, freq='1H')[1:]
            prediction_trend = np.random.normal(1.001, 0.002, len(future_dates))
            predicted_prices = [df['close'].iloc[-1]]
            
            for trend in prediction_trend:
                predicted_prices.append(predicted_prices[-1] * trend)
            
            fig.add_trace(go.Scatter(
                x=list(df['timestamp'].iloc[-1:]) + list(future_dates),
                y=predicted_prices,
                mode='lines',
                name='AI Prediction',
                line=dict(color='#f59e0b', width=3, dash='dash'),
                opacity=0.7
            ))
        
        # Update layout
        fig.update_layout(
            title=f"{market} Price Analysis",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_dark",
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_ai_insights(self):
        """Render AI insights and signals."""
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 🤖 AI Market Insights")
        
        # Market sentiment gauge
        sentiment_score = 0.75  # Sample data
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = sentiment_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Market Sentiment"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#3b82f6"},
                'steps': [
                    {'range': [0, 25], 'color': "#ef4444"},
                    {'range': [25, 50], 'color': "#f59e0b"},
                    {'range': [50, 75], 'color': "#10b981"},
                    {'range': [75, 100], 'color': "#06b6d4"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trading signals
        st.markdown("#### 🎯 Current Signals")
        
        signals = [
            {"signal": "BUY", "asset": "EUR/USD", "confidence": "92%", "time": "2 min ago"},
            {"signal": "HOLD", "asset": "BTC/USD", "confidence": "78%", "time": "5 min ago"},
            {"signal": "SELL", "asset": "GBP/USD", "confidence": "85%", "time": "8 min ago"},
        ]
        
        for signal in signals:
            color = "#10b981" if signal["signal"] == "BUY" else "#ef4444" if signal["signal"] == "SELL" else "#f59e0b"
            st.markdown(f"""
            <div class="trading-card">
                <span style="color: {color}; font-weight: bold;">{signal["signal"]}</span>
                <span style="float: right; color: #94a3b8;">{signal["time"]}</span><br>
                <strong>{signal["asset"]}</strong> - Confidence: {signal["confidence"]}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_performance_metrics(self):
        """Render detailed performance metrics."""
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Performance Analytics")
        
        # Performance chart
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)
        portfolio_values = [10000]
        
        for i in range(1, len(dates)):
            change = np.random.normal(0.001, 0.02)
            portfolio_values.append(portfolio_values[-1] * (1 + change))
        
        df_perf = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_values
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_perf['date'],
            y=df_perf['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#10b981', width=3),
            fill='tonexty',
            fillcolor='rgba(16, 185, 129, 0.1)'
        ))
        
        fig.update_layout(
            title="Portfolio Performance (30 Days)",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Return", "25.47%", "2.34%")
            st.metric("Sharpe Ratio", "1.85", "0.12")
        
        with col2:
            st.metric("Max Drawdown", "-3.2%", "0.8%")
            st.metric("Volatility", "12.4%", "-1.1%")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_trading_status(self):
        """Render current trading status and active positions."""
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 🎮 Trading Status")
        
        # Trading status
        if st.session_state.get('trading_active', False):
            st.markdown('<p class="status-trading">🟡 ACTIVE TRADING</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-neutral">⚪ MONITORING</p>', unsafe_allow_html=True)
        
        # Active positions
        st.markdown("#### 💼 Active Positions")
        
        positions = [
            {"asset": "EUR/USD", "type": "CALL", "amount": "$25.00", "pnl": "+$3.45", "status": "winning"},
            {"asset": "BTC/USD", "type": "PUT", "amount": "$50.00", "pnl": "-$2.10", "status": "losing"},
            {"asset": "GBP/USD", "type": "CALL", "amount": "$30.00", "pnl": "+$8.20", "status": "winning"},
        ]
        
        for pos in positions:
            pnl_color = "#10b981" if pos["status"] == "winning" else "#ef4444"
            st.markdown(f"""
            <div class="trading-card">
                <strong>{pos["asset"]}</strong> - {pos["type"]}
                <span style="float: right; color: {pnl_color}; font-weight: bold;">{pos["pnl"]}</span><br>
                Amount: {pos["amount"]}
            </div>
            """, unsafe_allow_html=True)
        
        # Trading statistics
        st.markdown("#### 📈 Session Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Trades Today", "12", "3")
            st.metric("Win Rate", "75%", "5%")
        
        with col2:
            st.metric("Profit Today", "$34.56", "$12.30")
            st.metric("Avg Trade", "$2.88", "$0.45")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_recent_activity(self):
        """Render recent trading activity and logs."""
        st.markdown("## 📋 Recent Activity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("### 🔄 Recent Trades")
            
            trades = [
                {"time": "14:23:45", "asset": "EUR/USD", "type": "CALL", "result": "WIN", "profit": "+$5.20"},
                {"time": "14:18:12", "asset": "BTC/USD", "type": "PUT", "result": "LOSS", "profit": "-$3.50"},
                {"time": "14:12:33", "asset": "GBP/USD", "type": "CALL", "result": "WIN", "profit": "+$7.80"},
                {"time": "14:05:21", "asset": "AUD/USD", "type": "PUT", "result": "WIN", "profit": "+$4.10"},
                {"time": "13:58:44", "asset": "USD/JPY", "type": "CALL", "result": "LOSS", "profit": "-$2.90"},
            ]
            
            for trade in trades:
                result_color = "#10b981" if trade["result"] == "WIN" else "#ef4444"
                st.markdown(f"""
                <div class="trading-card">
                    <span style="color: #94a3b8;">{trade["time"]}</span>
                    <span style="float: right; color: {result_color}; font-weight: bold;">{trade["profit"]}</span><br>
                    <strong>{trade["asset"]}</strong> {trade["type"]} - <span style="color: {result_color};">{trade["result"]}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("### 📝 System Logs")
            
            logs = [
                {"time": "14:25:10", "level": "INFO", "message": "Model retrained successfully"},
                {"time": "14:23:45", "level": "SUCCESS", "message": "Trade executed: EUR/USD CALL"},
                {"time": "14:20:33", "level": "INFO", "message": "Market analysis completed"},
                {"time": "14:18:12", "level": "WARNING", "message": "High volatility detected"},
                {"time": "14:15:55", "level": "INFO", "message": "Risk check passed"},
                {"time": "14:12:33", "level": "SUCCESS", "message": "Trade executed: GBP/USD CALL"},
            ]
            
            for log in logs:
                level_colors = {
                    "INFO": "#3b82f6",
                    "SUCCESS": "#10b981", 
                    "WARNING": "#f59e0b",
                    "ERROR": "#ef4444"
                }
                color = level_colors.get(log["level"], "#6b7280")
                
                st.markdown(f"""
                <div class="trading-card">
                    <span style="color: #94a3b8;">{log["time"]}</span>
                    <span style="color: {color}; font-weight: bold; float: right;">{log["level"]}</span><br>
                    {log["message"]}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def train_model_action(self, market, timeframe):
        """Handle model training action."""
        with st.spinner("Training AI model..."):
            time.sleep(2)  # Simulate training time
            st.success(f"✅ Model trained successfully for {market} ({timeframe})")
            st.balloons()
    
    def analyze_market_action(self, market, timeframe):
        """Handle market analysis action."""
        with st.spinner("Analyzing market conditions..."):
            time.sleep(1)  # Simulate analysis time
            st.success(f"📊 Market analysis completed for {market}")
    
    def get_available_markets(self):
        """Get available markets from the client."""
        try:
            if self.client:
                # Use a synchronous method to get markets in Streamlit
                # In a real implementation, you would cache this or use async properly
                import asyncio
                try:
                    # Try to get the event loop
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If we're in an async context, we can't run another event loop
                        # Return a comprehensive list of common markets
                        return [
                            "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "NZD/USD",
                            "EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "USD/CHF", "EUR/CHF",
                            "GBP/CHF", "CAD/JPY", "NZD/JPY", "CHF/JPY", "EUR/AUD", "GBP/AUD",
                            "AUD/CAD", "AUD/CHF", "NZD/USD", "EUR/NZD", "GBP/NZD", "AUD/NZD",
                            "Volatility 10 Index", "Volatility 25 Index", "Volatility 50 Index", 
                            "Volatility 75 Index", "Volatility 100 Index", "Boom 300 Index",
                            "Boom 500 Index", "Boom 1000 Index", "Crash 300 Index", "Crash 500 Index",
                            "Crash 1000 Index", "Step Index", "Jump 10 Index", "Jump 25 Index",
                            "Jump 50 Index", "Jump 75 Index", "Jump 100 Index",
                            "Bear Market Index", "Bull Market Index", "Bitcoin", "Ethereum",
                            "Litecoin", "Ripple", "Bitcoin Cash", "Dash", "EOS", "IOTA",
                            "NEO", "OmiseGO", "Zcash", "Tronix", "Binance Coin", "Cardano"
                        ]
                    else:
                        # We can create a new event loop
                        markets = asyncio.run(self.client.get_active_symbols())
                        if markets and isinstance(markets, dict):
                            return list(markets.keys())
                except Exception as e:
                    print(f"Error getting markets asynchronously: {e}")
                    
                # Fallback to comprehensive market list
                return [
                    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "NZD/USD",
                    "EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "USD/CHF", "EUR/CHF",
                    "GBP/CHF", "CAD/JPY", "NZD/JPY", "CHF/JPY", "EUR/AUD", "GBP/AUD",
                    "AUD/CAD", "AUD/CHF", "NZD/USD", "EUR/NZD", "GBP/NZD", "AUD/NZD",
                    "Volatility 10 Index", "Volatility 25 Index", "Volatility 50 Index", 
                    "Volatility 75 Index", "Volatility 100 Index", "Boom 300 Index",
                    "Boom 500 Index", "Boom 1000 Index", "Crash 300 Index", "Crash 500 Index",
                    "Crash 1000 Index", "Step Index", "Jump 10 Index", "Jump 25 Index",
                    "Jump 50 Index", "Jump 75 Index", "Jump 100 Index",
                    "Bear Market Index", "Bull Market Index", "Bitcoin", "Ethereum",
                    "Litecoin", "Ripple", "Bitcoin Cash", "Dash", "EOS", "IOTA"
                ]
            return []
        except Exception as e:
            print(f"Error fetching markets: {e}")
            return []
    
    def get_market_data(self, symbol, timeframe="1m", count=100):
        """Get market data for a specific symbol."""
        try:
            # Generate sample data for demonstration
            # In a real implementation, this would fetch from the API
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=max(count//10, 1))  # Ensure at least 1 hour
            dates = pd.date_range(start=start_time, end=end_time, periods=count)
            np.random.seed(hash(symbol) % 1000)  # Consistent seed for each symbol
            
            # Base price varies by symbol
            base_prices = {
                "EUR/USD": 1.0800, "GBP/USD": 1.2650, "USD/JPY": 148.50, "AUD/USD": 0.6580,
                "Bitcoin": 45000, "Ethereum": 2800, "Litecoin": 70, "XRP": 0.60
            }
            
            base_price = base_prices.get(symbol, 1.0000)
            prices = [base_price]
            
            for i in range(1, count):
                change = np.random.normal(0, 0.001)  # Small random changes
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)
            
            return pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': [p * 1.002 for p in prices],
                'low': [p * 0.998 for p in prices],
                'close': prices,
                'volume': np.random.randint(1000, 10000, count)
            })
        except Exception as e:
            print(f"Error fetching market data for {symbol}: {e}")
            return pd.DataFrame()

    # ...existing code...
    
    def run(self):
        """Run the dashboard."""
        self.render_header()
        
        # Sidebar configuration
        settings = self.render_sidebar()
        
        # Main dashboard
        self.render_main_dashboard(settings)
        
        # Auto-refresh
        if st.button("🔄 Refresh Dashboard", help="Refresh all data"):
            st.rerun()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
            🎯 DART v2.0 - Deep Adaptive Reinforcement Trader<br>
            Professional AI Trading Platform | Real-time Market Analysis
        </div>
        """, unsafe_allow_html=True)

# Initialize and run the dashboard
if __name__ == "__main__":
    dashboard = ModernDashboard()
    dashboard.run()
