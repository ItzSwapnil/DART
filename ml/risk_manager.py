"""
Enhanced Risk Management Layer for DART
Implements sophisticated risk controls described in the project report.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import datetime
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('risk_management')

class RiskLevel(Enum):
    """Risk levels for position sizing and exposure management."""
    VERY_LOW = 0.5
    LOW = 0.75
    MEDIUM = 1.0
    HIGH = 1.25
    VERY_HIGH = 1.5

class MarketCondition(Enum):
    """Market condition classifications."""
    NORMAL = "normal"
    VOLATILE = "volatile"
    TRENDING = "trending"
    RANGING = "ranging"
    CRISIS = "crisis"

@dataclass
class RiskMetrics:
    """Container for risk calculation results."""
    value_at_risk_95: float
    value_at_risk_99: float
    conditional_var_95: float
    conditional_var_99: float
    maximum_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    volatility: float
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float

@dataclass
class PositionLimits:
    """Position and exposure limits."""
    max_position_size: float
    max_portfolio_exposure: float
    max_sector_exposure: float
    max_single_trade_risk: float
    max_daily_trades: int
    max_correlation_exposure: float

class AdvancedRiskManager:
    """
    Advanced risk management system implementing multiple risk control mechanisms
    as described in the DART project report.
    """
    
    def __init__(self, initial_capital=10000, max_portfolio_risk=0.02, 
                 var_confidence_levels=[0.95, 0.99], lookback_period=252):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_portfolio_risk = max_portfolio_risk
        self.var_confidence_levels = var_confidence_levels
        self.lookback_period = lookback_period
        
        # Risk tracking
        self.trade_history = []
        self.portfolio_returns = []
        self.daily_pnl = []
        self.drawdown_history = []
        self.exposure_history = []
        
        # Position tracking
        self.current_positions = {}
        self.sector_exposures = {}
        self.correlation_matrix = None
        
        # Risk limits
        self.position_limits = PositionLimits(
            max_position_size=0.1,  # 10% of portfolio
            max_portfolio_exposure=0.8,  # 80% of capital
            max_sector_exposure=0.3,  # 30% per sector
            max_single_trade_risk=0.02,  # 2% per trade
            max_daily_trades=20,
            max_correlation_exposure=0.5  # 50% in correlated assets
        )
        
        # Dynamic risk adjustment
        self.volatility_regime = "normal"
        self.market_condition = MarketCondition.NORMAL
        self.stress_test_scenarios = self._initialize_stress_scenarios()
        
        # Performance tracking
        self.performance_metrics = {}
        self.risk_alerts = []
        
        # Kelly criterion parameters
        self.kelly_lookback = 50
        self.kelly_fraction = 0.25  # Conservative fractional Kelly
        
    def _initialize_stress_scenarios(self) -> Dict:
        """Initialize stress test scenarios."""
        return {
            "market_crash": {"market_decline": -0.20, "volatility_spike": 3.0},
            "flash_crash": {"market_decline": -0.10, "volatility_spike": 5.0},
            "interest_rate_shock": {"rate_change": 0.02, "duration_impact": -0.15},
            "liquidity_crisis": {"bid_ask_widening": 2.0, "volume_decline": -0.50},
            "sector_rotation": {"sector_decline": -0.30, "correlation_breakdown": 0.8},
            "currency_crisis": {"fx_volatility": 2.0, "correlation_spike": 0.9}
        }
    
    def calculate_position_size(self, signal_strength: float, confidence: float,
                              expected_return: float, expected_volatility: float,
                              symbol: str, current_price: float) -> Dict:
        """
        Calculate optimal position size using multiple methodologies.
        
        Args:
            signal_strength: Strength of the trading signal (-1 to 1)
            confidence: Model confidence (0 to 1)
            expected_return: Expected return of the trade
            expected_volatility: Expected volatility
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Dictionary with position sizing recommendations
        """
        
        # Base position size calculation
        base_position = self._calculate_base_position_size(
            signal_strength, confidence, expected_return, expected_volatility
        )
        
        # Kelly criterion adjustment
        kelly_position = self._calculate_kelly_position(
            expected_return, expected_volatility, symbol
        )
        
        # Risk parity adjustment
        risk_parity_position = self._calculate_risk_parity_position(
            symbol, expected_volatility
        )
        
        # Volatility targeting
        vol_target_position = self._calculate_volatility_targeted_position(
            expected_volatility, signal_strength
        )
        
        # Combine methodologies with weights
        weights = {
            'base': 0.3,
            'kelly': 0.3,
            'risk_parity': 0.2,
            'vol_target': 0.2
        }
        
        final_position = (
            weights['base'] * base_position +
            weights['kelly'] * kelly_position +
            weights['risk_parity'] * risk_parity_position +
            weights['vol_target'] * vol_target_position
        )
        
        # Apply risk limits and constraints
        final_position = self._apply_risk_constraints(
            final_position, symbol, current_price
        )
        
        return {
            'recommended_size': final_position,
            'base_size': base_position,
            'kelly_size': kelly_position,
            'risk_parity_size': risk_parity_position,
            'vol_target_size': vol_target_position,
            'risk_utilization': abs(final_position) / self.position_limits.max_position_size,
            'capital_allocation': final_position * current_price,
            'max_loss': self._calculate_max_loss(final_position, current_price)
        }
    
    def _calculate_base_position_size(self, signal_strength: float, confidence: float,
                                    expected_return: float, expected_volatility: float) -> float:
        """Calculate base position size using signal strength and confidence."""
        # Normalize signal strength and confidence
        normalized_signal = np.clip(signal_strength, -1, 1)
        normalized_confidence = np.clip(confidence, 0, 1)
        
        # Base size proportional to signal strength and confidence
        base_size = normalized_signal * normalized_confidence
        
        # Adjust for expected risk-return profile
        if expected_volatility > 0:
            risk_adjusted_size = base_size * (expected_return / expected_volatility)
        else:
            risk_adjusted_size = base_size
        
        # Apply maximum position limit
        max_position = self.position_limits.max_position_size
        return np.clip(risk_adjusted_size, -max_position, max_position)
    
    def _calculate_kelly_position(self, expected_return: float, 
                                expected_volatility: float, symbol: str) -> float:
        """Calculate position size using Kelly criterion."""
        
        # Get historical performance for this symbol
        recent_trades = [t for t in self.trade_history[-self.kelly_lookback:] 
                        if t.get('symbol') == symbol]
        
        if len(recent_trades) < 10:
            return 0.0  # Insufficient data
        
        # Calculate win rate and average win/loss
        wins = [t['return'] for t in recent_trades if t['return'] > 0]
        losses = [abs(t['return']) for t in recent_trades if t['return'] < 0]
        
        if not wins or not losses:
            return 0.0
        
        win_rate = len(wins) / len(recent_trades)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        if avg_loss > 0:
            b = avg_win / avg_loss
            kelly_fraction = (b * win_rate - (1 - win_rate)) / b
        else:
            kelly_fraction = 0.0
        
        # Apply fractional Kelly for safety
        kelly_position = kelly_fraction * self.kelly_fraction
        
        # Apply direction based on expected return
        if expected_return < 0:
            kelly_position = -abs(kelly_position)
        
        return np.clip(kelly_position, -self.position_limits.max_position_size,
                      self.position_limits.max_position_size)
    
    def _calculate_risk_parity_position(self, symbol: str, expected_volatility: float) -> float:
        """Calculate position size using risk parity approach."""
        
        # Target risk contribution per position
        target_risk_contribution = self.max_portfolio_risk / max(len(self.current_positions) + 1, 1)
        
        # Position size = target_risk / volatility
        if expected_volatility > 0:
            risk_parity_size = target_risk_contribution / expected_volatility
        else:
            risk_parity_size = 0.0
        
        return np.clip(risk_parity_size, -self.position_limits.max_position_size,
                      self.position_limits.max_position_size)
    
    def _calculate_volatility_targeted_position(self, expected_volatility: float,
                                              signal_strength: float) -> float:
        """Calculate position size targeting specific volatility level."""
        
        target_volatility = 0.15  # 15% annual volatility target
        
        if expected_volatility > 0:
            vol_scalar = target_volatility / expected_volatility
            vol_target_size = signal_strength * vol_scalar * 0.1  # 10% base allocation
        else:
            vol_target_size = 0.0
        
        return np.clip(vol_target_size, -self.position_limits.max_position_size,
                      self.position_limits.max_position_size)
    
    def _apply_risk_constraints(self, position_size: float, symbol: str, 
                              current_price: float) -> float:
        """Apply risk constraints and limits to position size."""
        
        # Check maximum single trade risk
        max_trade_value = self.current_capital * self.position_limits.max_single_trade_risk
        max_shares = max_trade_value / current_price
        position_size = np.clip(position_size, -max_shares, max_shares)
        
        # Check portfolio exposure limits
        total_exposure = sum(abs(pos['size'] * pos['price']) 
                           for pos in self.current_positions.values())
        new_exposure = abs(position_size * current_price)
        max_portfolio_value = self.current_capital * self.position_limits.max_portfolio_exposure
        
        if total_exposure + new_exposure > max_portfolio_value:
            adjustment_factor = (max_portfolio_value - total_exposure) / new_exposure
            position_size *= max(0, adjustment_factor)
        
        # Check correlation limits
        if self.correlation_matrix is not None:
            position_size = self._apply_correlation_limits(position_size, symbol)
        
        # Market condition adjustments
        position_size = self._apply_market_condition_adjustments(position_size)
        
        return position_size
    
    def _apply_correlation_limits(self, position_size: float, symbol: str) -> float:
        """Apply correlation-based position limits."""
        
        # Calculate correlation-weighted exposure
        corr_weighted_exposure = 0
        for existing_symbol, position in self.current_positions.items():
            if existing_symbol in self.correlation_matrix.columns and symbol in self.correlation_matrix.index:
                correlation = abs(self.correlation_matrix.loc[symbol, existing_symbol])
                if correlation > 0.7:  # High correlation threshold
                    corr_weighted_exposure += abs(position['size']) * correlation
        
        # Reduce position if correlation exposure is too high
        max_corr_exposure = self.current_capital * self.position_limits.max_correlation_exposure
        if corr_weighted_exposure > max_corr_exposure:
            reduction_factor = max_corr_exposure / corr_weighted_exposure
            position_size *= reduction_factor
        
        return position_size
    
    def _apply_market_condition_adjustments(self, position_size: float) -> float:
        """Adjust position size based on current market conditions."""
        
        adjustment_factors = {
            MarketCondition.NORMAL: 1.0,
            MarketCondition.VOLATILE: 0.7,
            MarketCondition.TRENDING: 1.2,
            MarketCondition.RANGING: 0.8,
            MarketCondition.CRISIS: 0.3
        }
        
        factor = adjustment_factors.get(self.market_condition, 1.0)
        return position_size * factor
    
    def _calculate_max_loss(self, position_size: float, current_price: float) -> float:
        """Calculate maximum potential loss for the position."""
        position_value = abs(position_size * current_price)
        
        # Assume maximum loss scenarios
        max_loss_scenarios = {
            'normal': 0.05,  # 5% loss
            'volatile': 0.15,  # 15% loss
            'crisis': 0.30   # 30% loss
        }
        
        scenario_prob = {
            'normal': 0.8,
            'volatile': 0.15,
            'crisis': 0.05
        }
        
        expected_max_loss = sum(
            position_value * loss * prob 
            for (scenario, loss), (_, prob) in zip(max_loss_scenarios.items(), scenario_prob.items())
        )
        
        return expected_max_loss
    
    def calculate_portfolio_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        
        if len(self.portfolio_returns) < 30:
            # Insufficient data for meaningful risk calculations
            return RiskMetrics(
                value_at_risk_95=0, value_at_risk_99=0,
                conditional_var_95=0, conditional_var_99=0,
                maximum_drawdown=0, sharpe_ratio=0,
                sortino_ratio=0, calmar_ratio=0,
                volatility=0, beta=0, alpha=0,
                information_ratio=0, tracking_error=0
            )
        
        returns = np.array(self.portfolio_returns[-self.lookback_period:])
        
        # Value at Risk calculations
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Performance ratios
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        # Sharpe ratio (assuming risk-free rate of 0.02/252 daily)
        risk_free_rate = 0.02 / 252
        sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < risk_free_rate]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else volatility
        sortino_ratio = (mean_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio
        annualized_return = mean_return * 252
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Beta and Alpha (against market benchmark - simplified)
        # In practice, you would use actual market data
        beta = 1.0  # Placeholder
        alpha = annualized_return - (risk_free_rate * 252 + beta * 0.08)  # Assuming 8% market return
        
        # Information ratio and tracking error
        tracking_error = volatility  # Simplified
        information_ratio = alpha / tracking_error if tracking_error > 0 else 0
        
        return RiskMetrics(
            value_at_risk_95=var_95,
            value_at_risk_99=var_99,
            conditional_var_95=cvar_95,
            conditional_var_99=cvar_99,
            maximum_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            volatility=volatility,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            tracking_error=tracking_error
        )
    
    def run_stress_tests(self) -> Dict:
        """Run stress tests on current portfolio."""
        
        stress_results = {}
        
        for scenario_name, scenario_params in self.stress_test_scenarios.items():
            portfolio_impact = 0
            
            for symbol, position in self.current_positions.items():
                position_value = position['size'] * position['price']
                
                # Apply scenario-specific shocks
                if 'market_decline' in scenario_params:
                    market_shock = scenario_params['market_decline']
                    position_impact = position_value * market_shock
                    portfolio_impact += position_impact
                
                # Add volatility impact
                if 'volatility_spike' in scenario_params:
                    vol_multiplier = scenario_params['volatility_spike']
                    # Increased volatility typically reduces position values due to higher risk
                    vol_impact = position_value * -0.05 * vol_multiplier
                    portfolio_impact += vol_impact
            
            # Calculate impact as percentage of portfolio
            portfolio_impact_pct = portfolio_impact / self.current_capital
            
            stress_results[scenario_name] = {
                'absolute_impact': portfolio_impact,
                'percentage_impact': portfolio_impact_pct,
                'scenario_params': scenario_params,
                'severity': self._classify_stress_severity(portfolio_impact_pct)
            }
        
        return stress_results
    
    def _classify_stress_severity(self, impact_pct: float) -> str:
        """Classify the severity of stress test impact."""
        if abs(impact_pct) <= 0.05:
            return "Low"
        elif abs(impact_pct) <= 0.15:
            return "Medium"
        elif abs(impact_pct) <= 0.30:
            return "High"
        else:
            return "Severe"
    
    def check_risk_limits(self) -> List[Dict]:
        """Check all risk limits and return violations."""
        violations = []
        
        # Calculate current portfolio metrics
        total_exposure = sum(abs(pos['size'] * pos['price']) 
                           for pos in self.current_positions.values())
        
        # Portfolio exposure check
        exposure_ratio = total_exposure / self.current_capital
        if exposure_ratio > self.position_limits.max_portfolio_exposure:
            violations.append({
                'type': 'portfolio_exposure',
                'current': exposure_ratio,
                'limit': self.position_limits.max_portfolio_exposure,
                'severity': 'high' if exposure_ratio > 1.2 * self.position_limits.max_portfolio_exposure else 'medium'
            })
        
        # Individual position size checks
        for symbol, position in self.current_positions.items():
            position_value = abs(position['size'] * position['price'])
            position_ratio = position_value / self.current_capital
            
            if position_ratio > self.position_limits.max_position_size:
                violations.append({
                    'type': 'position_size',
                    'symbol': symbol,
                    'current': position_ratio,
                    'limit': self.position_limits.max_position_size,
                    'severity': 'high' if position_ratio > 1.5 * self.position_limits.max_position_size else 'medium'
                })
        
        # Drawdown checks
        if len(self.drawdown_history) > 0:
            current_drawdown = self.drawdown_history[-1]
            max_acceptable_drawdown = -0.20  # 20% maximum drawdown
            
            if current_drawdown < max_acceptable_drawdown:
                violations.append({
                    'type': 'drawdown',
                    'current': current_drawdown,
                    'limit': max_acceptable_drawdown,
                    'severity': 'high'
                })
        
        # Daily loss checks
        if len(self.daily_pnl) > 0:
            today_pnl = self.daily_pnl[-1]
            max_daily_loss = -0.05 * self.current_capital  # 5% daily loss limit
            
            if today_pnl < max_daily_loss:
                violations.append({
                    'type': 'daily_loss',
                    'current': today_pnl,
                    'limit': max_daily_loss,
                    'severity': 'high'
                })
        
        return violations
    
    def update_market_condition(self, market_data: Dict):
        """Update market condition assessment based on current market data."""
        
        # Simple market condition detection based on volatility and trends
        volatility = market_data.get('volatility', 0)
        trend_strength = market_data.get('trend_strength', 0)
        volume_ratio = market_data.get('volume_ratio', 1.0)
        
        # Classify market condition
        if volatility > 0.03:  # High volatility threshold
            if volume_ratio < 0.5:  # Low volume
                self.market_condition = MarketCondition.CRISIS
            else:
                self.market_condition = MarketCondition.VOLATILE
        elif abs(trend_strength) > 0.7:  # Strong trend
            self.market_condition = MarketCondition.TRENDING
        elif abs(trend_strength) < 0.3:  # Weak trend
            self.market_condition = MarketCondition.RANGING
        else:
            self.market_condition = MarketCondition.NORMAL
        
        logger.info(f"Market condition updated to: {self.market_condition.value}")
    
    def update_position(self, symbol: str, size: float, price: float, 
                       trade_return: float = None):
        """Update position tracking and risk metrics."""
        
        # Update position
        if size == 0:
            # Close position
            if symbol in self.current_positions:
                del self.current_positions[symbol]
        else:
            self.current_positions[symbol] = {
                'size': size,
                'price': price,
                'timestamp': datetime.datetime.now()
            }
        
        # Update trade history if return provided
        if trade_return is not None:
            self.trade_history.append({
                'symbol': symbol,
                'return': trade_return,
                'timestamp': datetime.datetime.now(),
                'size': size,
                'price': price
            })
            
            # Update portfolio returns
            portfolio_return = trade_return * abs(size) / self.current_capital
            self.portfolio_returns.append(portfolio_return)
            
            # Update capital
            self.current_capital += trade_return * abs(size)
        
        # Calculate current drawdown
        if len(self.portfolio_returns) > 0:
            cumulative_return = np.prod([1 + r for r in self.portfolio_returns])
            peak_value = max([np.prod([1 + r for r in self.portfolio_returns[:i+1]]) 
                            for i in range(len(self.portfolio_returns))])
            current_drawdown = (cumulative_return - peak_value) / peak_value
            self.drawdown_history.append(current_drawdown)
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary."""
        
        risk_metrics = self.calculate_portfolio_risk_metrics()
        stress_results = self.run_stress_tests()
        violations = self.check_risk_limits()
        
        return {
            'risk_metrics': risk_metrics.__dict__,
            'stress_tests': stress_results,
            'violations': violations,
            'market_condition': self.market_condition.value,
            'portfolio_exposure': sum(abs(pos['size'] * pos['price']) 
                                    for pos in self.current_positions.values()) / self.current_capital,
            'current_positions': len(self.current_positions),
            'current_capital': self.current_capital,
            'total_trades': len(self.trade_history),
            'recent_performance': {
                'last_30_days': np.mean(self.portfolio_returns[-30:]) if len(self.portfolio_returns) >= 30 else 0,
                'last_7_days': np.mean(self.portfolio_returns[-7:]) if len(self.portfolio_returns) >= 7 else 0,
                'ytd': np.mean(self.portfolio_returns) if self.portfolio_returns else 0
            }
        }
    
    def save_risk_state(self, filepath: str):
        """Save current risk management state."""
        
        state = {
            'current_capital': self.current_capital,
            'portfolio_returns': self.portfolio_returns,
            'trade_history': self.trade_history,
            'current_positions': self.current_positions,
            'market_condition': self.market_condition.value,
            'position_limits': self.position_limits.__dict__,
            'performance_metrics': self.performance_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, default=str, indent=2)
        
        logger.info(f"Risk state saved to {filepath}")
    
    def load_risk_state(self, filepath: str):
        """Load risk management state from file."""
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.current_capital = state.get('current_capital', self.initial_capital)
            self.portfolio_returns = state.get('portfolio_returns', [])
            self.trade_history = state.get('trade_history', [])
            self.current_positions = state.get('current_positions', {})
            self.market_condition = MarketCondition(state.get('market_condition', 'normal'))
            self.performance_metrics = state.get('performance_metrics', {})
            
            logger.info(f"Risk state loaded from {filepath}")
        else:
            logger.warning(f"No risk state file found at {filepath}")
