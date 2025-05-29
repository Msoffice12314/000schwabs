"""
Schwab AI Trading System - Risk Management System
Comprehensive risk management with dynamic position sizing, correlation limits, and drawdown protection.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import math
from scipy import stats

from models.signal_detector import TradingSignal, SignalType
from utils.cache_manager import CacheManager
from utils.database import DatabaseManager
from config.settings import get_settings

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk levels for different trading modes"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    SCALPING = "scalping"

class RiskEventType(Enum):
    """Types of risk events"""
    POSITION_SIZE_EXCEEDED = "position_size_exceeded"
    PORTFOLIO_RISK_EXCEEDED = "portfolio_risk_exceeded"
    CORRELATION_LIMIT_EXCEEDED = "correlation_limit_exceeded"
    DRAWDOWN_LIMIT_EXCEEDED = "drawdown_limit_exceeded"
    VOLATILITY_SPIKE = "volatility_spike"
    MARGIN_CALL_RISK = "margin_call_risk"

@dataclass
class RiskMetrics:
    """Risk metrics for portfolio and positions"""
    portfolio_value: float
    total_risk: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    var_95: float  # Value at Risk 95%
    expected_shortfall: float
    beta: float
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    sector_exposure: Dict[str, float] = field(default_factory=dict)
    position_risks: Dict[str, float] = field(default_factory=dict)

@dataclass
class PositionSize:
    """Position sizing recommendation"""
    symbol: str
    recommended_shares: int
    recommended_dollars: float
    max_shares: int
    max_dollars: float
    risk_percentage: float
    kelly_fraction: float
    confidence_adjustment: float
    reason: str
    warnings: List[str] = field(default_factory=list)

@dataclass
class RiskEvent:
    """Risk event notification"""
    event_type: RiskEventType
    symbol: Optional[str]
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    message: str
    timestamp: datetime
    metrics: Dict[str, float]
    recommended_actions: List[str]

class RiskManager:
    """
    Advanced risk management system with dynamic position sizing,
    correlation analysis, and portfolio-level risk controls.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache_manager = CacheManager()
        self.db_manager = DatabaseManager()
        
        # Risk parameters by mode
        self.risk_params = {
            RiskLevel.CONSERVATIVE: {
                'max_position_size': 0.05,  # 5% per position
                'max_portfolio_risk': 0.02,  # 2% portfolio risk
                'max_correlation': 0.7,
                'max_drawdown': 0.10,  # 10% max drawdown
                'stop_loss': 0.015,  # 1.5% stop loss
                'take_profit': 0.03,  # 3% take profit
                'max_daily_trades': 3,
                'volatility_threshold': 0.3
            },
            RiskLevel.MODERATE: {
                'max_position_size': 0.10,  # 10% per position
                'max_portfolio_risk': 0.05,  # 5% portfolio risk
                'max_correlation': 0.8,
                'max_drawdown': 0.15,  # 15% max drawdown
                'stop_loss': 0.02,  # 2% stop loss
                'take_profit': 0.04,  # 4% take profit
                'max_daily_trades': 10,
                'volatility_threshold': 0.4
            },
            RiskLevel.AGGRESSIVE: {
                'max_position_size': 0.15,  # 15% per position
                'max_portfolio_risk': 0.08,  # 8% portfolio risk
                'max_correlation': 0.85,
                'max_drawdown': 0.20,  # 20% max drawdown
                'stop_loss': 0.03,  # 3% stop loss
                'take_profit': 0.06,  # 6% take profit
                'max_daily_trades': 20,
                'volatility_threshold': 0.5
            },
            RiskLevel.SCALPING: {
                'max_position_size': 0.08,  # 8% per position
                'max_portfolio_risk': 0.06,  # 6% portfolio risk
                'max_correlation': 0.9,
                'max_drawdown': 0.12,  # 12% max drawdown
                'stop_loss': 0.005,  # 0.5% stop loss
                'take_profit': 0.01,  # 1% take profit
                'max_daily_trades': 50,
                'volatility_threshold': 0.6
            }
        }
        
        # Current risk level
        self.current_risk_level = RiskLevel.MODERATE
        
        # Portfolio state
        self.portfolio_value = 0.0
        self.positions = {}
        self.price_history = {}
        self.returns_history = []
        
        # Risk events
        self.risk_events = []
        
        logger.info("RiskManager initialized")
    
    def set_risk_level(self, risk_level: RiskLevel):
        """Set the current risk level"""
        self.current_risk_level = risk_level
        logger.info(f"Risk level set to: {risk_level.value}")
    
    def get_current_params(self) -> Dict[str, float]:
        """Get current risk parameters"""
        return self.risk_params[self.current_risk_level]
    
    async def calculate_position_size(self, signal: TradingSignal, 
                                    portfolio_value: float,
                                    current_positions: Dict[str, Dict]) -> PositionSize:
        """
        Calculate optimal position size using Kelly Criterion with risk adjustments
        
        Args:
            signal: Trading signal from AI model
            portfolio_value: Current portfolio value
            current_positions: Current position holdings
            
        Returns:
            PositionSize object with recommendations
        """
        try:
            params = self.get_current_params()
            symbol = signal.symbol
            price = signal.price
            confidence = signal.confidence
            
            warnings = []
            
            # Kelly Criterion calculation
            kelly_fraction = await self._calculate_kelly_fraction(signal)
            
            # Confidence adjustment
            confidence_adjustment = confidence * 0.8 + 0.2  # Scale confidence
            
            # Volatility adjustment
            volatility = await self._get_symbol_volatility(symbol)
            volatility_adjustment = max(0.5, 1.0 - volatility)
            
            # Base position size from Kelly
            base_fraction = kelly_fraction * confidence_adjustment * volatility_adjustment
            
            # Apply maximum position size limit
            max_position_fraction = params['max_position_size']
            position_fraction = min(base_fraction, max_position_fraction)
            
            if base_fraction > max_position_fraction:
                warnings.append(f"Kelly fraction {base_fraction:.3f} capped at {max_position_fraction:.3f}")
            
            # Calculate dollar amounts
            recommended_dollars = portfolio_value * position_fraction
            max_dollars = portfolio_value * max_position_fraction
            
            # Calculate shares
            recommended_shares = int(recommended_dollars / price)
            max_shares = int(max_dollars / price)
            
            # Portfolio risk check
            portfolio_risk = await self._calculate_portfolio_risk(
                symbol, recommended_shares * price, current_positions
            )
            
            if portfolio_risk > params['max_portfolio_risk']:
                # Reduce position size to meet portfolio risk limit
                risk_adjustment = params['max_portfolio_risk'] / portfolio_risk
                recommended_shares = int(recommended_shares * risk_adjustment)
                recommended_dollars = recommended_shares * price
                warnings.append(f"Position reduced for portfolio risk limit")
            
            # Correlation check
            correlation_adjustment = await self._check_correlation_limits(
                symbol, recommended_dollars, current_positions
            )
            
            if correlation_adjustment < 1.0:
                recommended_shares = int(recommended_shares * correlation_adjustment)
                recommended_dollars = recommended_shares * price
                warnings.append(f"Position reduced due to correlation limits")
            
            # Final risk percentage
            risk_percentage = (recommended_dollars / portfolio_value) if portfolio_value > 0 else 0
            
            position_size = PositionSize(
                symbol=symbol,
                recommended_shares=recommended_shares,
                recommended_dollars=recommended_dollars,
                max_shares=max_shares,
                max_dollars=max_dollars,
                risk_percentage=risk_percentage,
                kelly_fraction=kelly_fraction,
                confidence_adjustment=confidence_adjustment,
                reason=f"Kelly: {kelly_fraction:.3f}, Confidence: {confidence:.3f}, Volatility: {volatility:.3f}",
                warnings=warnings
            )
            
            logger.info(f"Position size calculated for {symbol}: {recommended_shares} shares (${recommended_dollars:.2f})")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {signal.symbol}: {str(e)}")
            return PositionSize(
                symbol=signal.symbol,
                recommended_shares=0,
                recommended_dollars=0.0,
                max_shares=0,
                max_dollars=0.0,
                risk_percentage=0.0,
                kelly_fraction=0.0,
                confidence_adjustment=0.0,
                reason="Error in calculation",
                warnings=["Calculation error occurred"]
            )
    
    async def _calculate_kelly_fraction(self, signal: TradingSignal) -> float:
        """Calculate Kelly Criterion fraction for position sizing"""
        try:
            # Get historical win rate and average win/loss for this signal type
            win_rate, avg_win, avg_loss = await self._get_signal_statistics(signal)
            
            if avg_loss == 0:
                return 0.0
            
            # Kelly formula: f = (bp - q) / b
            # where b = odds (avg_win/avg_loss), p = win_rate, q = 1-p
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # Cap Kelly fraction for safety
            kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # Max 25%
            
            # Apply signal confidence adjustment
            kelly_fraction *= signal.confidence
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Error calculating Kelly fraction: {str(e)}")
            return 0.02  # Conservative default
    
    async def _get_signal_statistics(self, signal: TradingSignal) -> Tuple[float, float, float]:
        """Get historical statistics for signal type and symbol"""
        try:
            # Query historical signal performance from database
            query = """
                SELECT 
                    AVG(CASE WHEN actual_return > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                    AVG(CASE WHEN actual_return > 0 THEN actual_return ELSE 0 END) as avg_win,
                    AVG(CASE WHEN actual_return <= 0 THEN ABS(actual_return) ELSE 0 END) as avg_loss
                FROM signal_performance 
                WHERE symbol = %s AND signal_type = %s 
                AND created_date >= %s
            """
            
            # Look back 90 days
            lookback_date = datetime.now() - timedelta(days=90)
            
            result = await self.db_manager.execute_query(
                query, (signal.symbol, signal.signal_type.value, lookback_date)
            )
            
            if result and len(result) > 0:
                win_rate = result[0]['win_rate'] or 0.5
                avg_win = result[0]['avg_win'] or 0.02
                avg_loss = result[0]['avg_loss'] or 0.02
            else:
                # Default values for new signals
                win_rate = 0.55  # Slightly optimistic
                avg_win = 0.025  # 2.5% average win
                avg_loss = 0.02   # 2% average loss
            
            return win_rate, avg_win, avg_loss
            
        except Exception as e:
            logger.error(f"Error getting signal statistics: {str(e)}")
            return 0.5, 0.02, 0.02  # Conservative defaults
    
    async def _get_symbol_volatility(self, symbol: str) -> float:
        """Get symbol volatility (annualized)"""
        try:
            # Try cache first
            cache_key = f"volatility_{symbol}"
            cached_vol = await self.cache_manager.get(cache_key)
            if cached_vol:
                return cached_vol
            
            # Calculate volatility from price history
            if symbol in self.price_history:
                prices = self.price_history[symbol]
                if len(prices) > 20:
                    returns = np.diff(np.log(prices))
                    volatility = np.std(returns) * np.sqrt(252)  # Annualized
                    
                    # Cache for 1 hour
                    await self.cache_manager.set(cache_key, volatility, expire=3600)
                    return volatility
            
            # Default volatility based on asset type
            default_volatilities = {
                'SPY': 0.15, 'QQQ': 0.20, 'IWM': 0.25,
                'AAPL': 0.25, 'MSFT': 0.25, 'GOOGL': 0.30,
                'TSLA': 0.50, 'NVDA': 0.45
            }
            
            return default_volatilities.get(symbol, 0.30)  # 30% default
            
        except Exception as e:
            logger.error(f"Error getting volatility for {symbol}: {str(e)}")
            return 0.30
    
    async def _calculate_portfolio_risk(self, symbol: str, position_value: float, 
                                      current_positions: Dict[str, Dict]) -> float:
        """Calculate total portfolio risk including new position"""
        try:
            total_risk = 0.0
            
            # Risk from current positions
            for pos_symbol, position in current_positions.items():
                pos_value = position.get('market_value', 0)
                pos_volatility = await self._get_symbol_volatility(pos_symbol)
                pos_risk = (pos_value / self.portfolio_value) * pos_volatility
                total_risk += pos_risk ** 2  # Variance
            
            # Risk from new position
            if self.portfolio_value > 0:
                new_volatility = await self._get_symbol_volatility(symbol)
                new_risk = (position_value / self.portfolio_value) * new_volatility
                total_risk += new_risk ** 2
                
                # Add correlation effects (simplified)
                for pos_symbol in current_positions:
                    correlation = await self._get_correlation(symbol, pos_symbol)
                    pos_value = current_positions[pos_symbol].get('market_value', 0)
                    pos_volatility = await self._get_symbol_volatility(pos_symbol)
                    
                    cross_term = 2 * correlation * new_risk * (pos_value / self.portfolio_value) * pos_volatility
                    total_risk += cross_term
            
            return math.sqrt(max(0, total_risk))  # Standard deviation
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {str(e)}")
            return 0.0
    
    async def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        try:
            if symbol1 == symbol2:
                return 1.0
            
            # Try cache first
            cache_key = f"corr_{min(symbol1, symbol2)}_{max(symbol1, symbol2)}"
            cached_corr = await self.cache_manager.get(cache_key)
            if cached_corr is not None:
                return cached_corr
            
            # Calculate correlation from price history
            if symbol1 in self.price_history and symbol2 in self.price_history:
                prices1 = self.price_history[symbol1]
                prices2 = self.price_history[symbol2]
                
                if len(prices1) > 30 and len(prices2) > 30:
                    # Use common length
                    min_len = min(len(prices1), len(prices2))
                    returns1 = np.diff(np.log(prices1[-min_len:]))
                    returns2 = np.diff(np.log(prices2[-min_len:]))
                    
                    correlation = np.corrcoef(returns1, returns2)[0, 1]
                    
                    # Handle NaN
                    if np.isnan(correlation):
                        correlation = 0.0
                    
                    # Cache for 1 hour
                    await self.cache_manager.set(cache_key, correlation, expire=3600)
                    return correlation
            
            # Default correlations based on sector/market
            sector_correlations = {
                ('SPY', 'QQQ'): 0.85,
                ('AAPL', 'MSFT'): 0.6,
                ('GOOGL', 'MSFT'): 0.7,
                ('TSLA', 'NVDA'): 0.4,
            }
            
            key = (min(symbol1, symbol2), max(symbol1, symbol2))
            return sector_correlations.get(key, 0.3)  # Default moderate correlation
            
        except Exception as e:
            logger.error(f"Error getting correlation: {str(e)}")
            return 0.3
    
    async def _check_correlation_limits(self, symbol: str, position_value: float, 
                                      current_positions: Dict[str, Dict]) -> float:
        """Check if new position violates correlation limits"""
        try:
            params = self.get_current_params()
            max_correlation = params['max_correlation']
            
            adjustment_factor = 1.0
            
            for pos_symbol, position in current_positions.items():
                correlation = await self._get_correlation(symbol, pos_symbol)
                
                if correlation > max_correlation:
                    # Reduce position size based on correlation excess
                    excess_correlation = correlation - max_correlation
                    reduction = excess_correlation / correlation
                    adjustment_factor = min(adjustment_factor, 1.0 - reduction)
            
            return max(0.1, adjustment_factor)  # Minimum 10% of intended size
            
        except Exception as e:
            logger.error(f"Error checking correlation limits: {str(e)}")
            return 1.0
    
    async def validate_trade(self, signal: TradingSignal, position_size: PositionSize,
                           current_positions: Dict[str, Dict]) -> Tuple[bool, List[str]]:
        """
        Validate if trade can be executed based on risk rules
        
        Returns:
            (is_valid, list_of_violations)
        """
        try:
            violations = []
            params = self.get_current_params()
            
            # Check position size limits
            if position_size.risk_percentage > params['max_position_size']:
                violations.append(f"Position size {position_size.risk_percentage:.3f} exceeds limit {params['max_position_size']:.3f}")
            
            # Check daily trade limit
            daily_trades = await self._get_daily_trade_count()
            if daily_trades >= params['max_daily_trades']:
                violations.append(f"Daily trade limit reached: {daily_trades}/{params['max_daily_trades']}")
            
            # Check portfolio risk
            portfolio_risk = await self._calculate_portfolio_risk(
                signal.symbol, position_size.recommended_dollars, current_positions
            )
            if portfolio_risk > params['max_portfolio_risk']:
                violations.append(f"Portfolio risk {portfolio_risk:.3f} exceeds limit {params['max_portfolio_risk']:.3f}")
            
            # Check current drawdown
            current_drawdown = await self._get_current_drawdown()
            if current_drawdown > params['max_drawdown']:
                violations.append(f"Current drawdown {current_drawdown:.3f} exceeds limit {params['max_drawdown']:.3f}")
            
            # Check volatility conditions
            volatility = await self._get_symbol_volatility(signal.symbol)
            if volatility > params['volatility_threshold']:
                violations.append(f"Symbol volatility {volatility:.3f} exceeds threshold {params['volatility_threshold']:.3f}")
            
            # Check signal confidence
            if signal.confidence < 0.6:
                violations.append(f"Signal confidence {signal.confidence:.3f} too low")
            
            is_valid = len(violations) == 0
            
            if not is_valid:
                logger.warning(f"Trade validation failed for {signal.symbol}: {violations}")
            
            return is_valid, violations
            
        except Exception as e:
            logger.error(f"Error validating trade: {str(e)}")
            return False, ["Validation error occurred"]
    
    async def _get_daily_trade_count(self) -> int:
        """Get number of trades executed today"""
        try:
            today = datetime.now().date()
            query = """
                SELECT COUNT(*) as trade_count 
                FROM trades 
                WHERE DATE(execution_time) = %s
            """
            
            result = await self.db_manager.execute_query(query, (today,))
            return result[0]['trade_count'] if result else 0
            
        except Exception as e:
            logger.error(f"Error getting daily trade count: {str(e)}")
            return 0
    
    async def _get_current_drawdown(self) -> float:
        """Calculate current portfolio drawdown"""
        try:
            # Get portfolio value history
            query = """
                SELECT portfolio_value, date 
                FROM portfolio_history 
                WHERE date >= %s 
                ORDER BY date
            """
            
            lookback_date = datetime.now() - timedelta(days=252)  # 1 year
            result = await self.db_manager.execute_query(query, (lookback_date,))
            
            if not result or len(result) < 2:
                return 0.0
            
            values = [row['portfolio_value'] for row in result]
            current_value = values[-1]
            
            # Calculate maximum drawdown
            peak = values[0]
            max_drawdown = 0.0
            
            for value in values:
                if value > peak:
                    peak = value
                
                drawdown = (peak - value) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            # Current drawdown from peak
            recent_peak = max(values[-30:]) if len(values) >= 30 else peak
            current_drawdown = (recent_peak - current_value) / recent_peak if recent_peak > 0 else 0
            
            return current_drawdown
            
        except Exception as e:
            logger.error(f"Error calculating current drawdown: {str(e)}")
            return 0.0
    
    async def calculate_stop_loss_take_profit(self, signal: TradingSignal, 
                                            entry_price: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        try:
            params = self.get_current_params()
            
            # Base stop loss and take profit from risk parameters
            stop_loss_pct = params['stop_loss']
            take_profit_pct = params['take_profit']
            
            # Adjust based on volatility
            volatility = await self._get_symbol_volatility(signal.symbol)
            
            # Higher volatility = wider stops
            volatility_multiplier = 1.0 + (volatility - 0.2) * 0.5
            
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                stop_loss = entry_price * (1 - stop_loss_pct * volatility_multiplier)
                take_profit = entry_price * (1 + take_profit_pct * volatility_multiplier)
            else:  # SELL signals
                stop_loss = entry_price * (1 + stop_loss_pct * volatility_multiplier)
                take_profit = entry_price * (1 - take_profit_pct * volatility_multiplier)
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating stop loss/take profit: {str(e)}")
            # Return safe defaults
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                return entry_price * 0.98, entry_price * 1.04
            else:
                return entry_price * 1.02, entry_price * 0.96
    
    async def calculate_risk_metrics(self, portfolio_data: Dict[str, Any]) -> RiskMetrics:
        """Calculate comprehensive risk metrics for the portfolio"""
        try:
            portfolio_value = portfolio_data.get('total_value', 0)
            positions = portfolio_data.get('positions', {})
            
            # Calculate portfolio returns
            returns = await self._get_portfolio_returns()
            
            # Basic risk metrics
            total_risk = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
            
            # Maximum drawdown
            max_drawdown = await self._calculate_max_drawdown(returns)
            
            # Sharpe ratio
            risk_free_rate = 0.02  # 2% risk-free rate
            excess_returns = np.array(returns) - risk_free_rate / 252
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = [r for r in excess_returns if r < 0]
            downside_std = np.std(downside_returns) if downside_returns else np.std(excess_returns)
            sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
            
            # Value at Risk (95%)
            var_95 = np.percentile(returns, 5) * portfolio_value if len(returns) > 0 else 0
            
            # Expected Shortfall (Conditional VaR)
            returns_sorted = sorted(returns)
            tail_size = max(1, int(len(returns) * 0.05))
            expected_shortfall = np.mean(returns_sorted[:tail_size]) * portfolio_value if tail_size > 0 else 0
            
            # Portfolio beta (vs SPY)
            beta = await self._calculate_portfolio_beta()
            
            # Correlation matrix
            correlation_matrix = await self._build_correlation_matrix(positions)
            
            # Sector exposure
            sector_exposure = await self._calculate_sector_exposure(positions)
            
            # Individual position risks
            position_risks = {}
            for symbol, position in positions.items():
                vol = await self._get_symbol_volatility(symbol)
                position_value = position.get('market_value', 0)
                position_weight = position_value / portfolio_value if portfolio_value > 0 else 0
                position_risks[symbol] = position_weight * vol
            
            return RiskMetrics(
                portfolio_value=portfolio_value,
                total_risk=total_risk,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                beta=beta,
                correlation_matrix=correlation_matrix,
                sector_exposure=sector_exposure,
                position_risks=position_risks
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return RiskMetrics(
                portfolio_value=0,
                total_risk=0,
                max_drawdown=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                var_95=0,
                expected_shortfall=0,
                beta=1.0
            )
    
    async def _get_portfolio_returns(self) -> List[float]:
        """Get historical portfolio returns"""
        try:
            query = """
                SELECT portfolio_value, date 
                FROM portfolio_history 
                WHERE date >= %s 
                ORDER BY date
            """
            
            lookback_date = datetime.now() - timedelta(days=252)
            result = await self.db_manager.execute_query(query, (lookback_date,))
            
            if len(result) < 2:
                return []
            
            values = [row['portfolio_value'] for row in result]
            returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
            
            return returns
            
        except Exception as e:
            logger.error(f"Error getting portfolio returns: {str(e)}")
            return []
    
    async def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        try:
            if not returns:
                return 0.0
            
            cumulative = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (running_max - cumulative) / running_max
            
            return float(np.max(drawdown))
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0
    
    async def _calculate_portfolio_beta(self) -> float:
        """Calculate portfolio beta vs market (SPY)"""
        try:
            portfolio_returns = await self._get_portfolio_returns()
            
            # Get SPY returns for same period
            # This would need market data integration
            # For now, return default beta
            return 1.0
            
        except Exception as e:
            logger.error(f"Error calculating portfolio beta: {str(e)}")
            return 1.0
    
    async def _build_correlation_matrix(self, positions: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
        """Build correlation matrix for portfolio positions"""
        try:
            symbols = list(positions.keys())
            correlation_matrix = {}
            
            for symbol1 in symbols:
                correlation_matrix[symbol1] = {}
                for symbol2 in symbols:
                    correlation = await self._get_correlation(symbol1, symbol2)
                    correlation_matrix[symbol1][symbol2] = correlation
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error building correlation matrix: {str(e)}")
            return {}
    
    async def _calculate_sector_exposure(self, positions: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate sector exposure percentages"""
        try:
            # This would need sector mapping data
            # For now, return basic tech/finance categorization
            sector_map = {
                'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
                'TSLA': 'Consumer', 'NVDA': 'Technology', 'META': 'Technology',
                'JPM': 'Finance', 'BAC': 'Finance', 'GS': 'Finance',
                'SPY': 'Market', 'QQQ': 'Technology', 'IWM': 'Market'
            }
            
            sector_exposure = {}
            total_value = sum(pos.get('market_value', 0) for pos in positions.values())
            
            for symbol, position in positions.items():
                sector = sector_map.get(symbol, 'Other')
                value = position.get('market_value', 0)
                weight = value / total_value if total_value > 0 else 0
                
                if sector in sector_exposure:
                    sector_exposure[sector] += weight
                else:
                    sector_exposure[sector] = weight
            
            return sector_exposure
            
        except Exception as e:
            logger.error(f"Error calculating sector exposure: {str(e)}")
            return {}
    
    async def generate_risk_alerts(self, portfolio_data: Dict[str, Any]) -> List[RiskEvent]:
        """Generate risk alerts based on current portfolio state"""
        try:
            alerts = []
            params = self.get_current_params()
            
            # Check drawdown alert
            current_drawdown = await self._get_current_drawdown()
            if current_drawdown > params['max_drawdown'] * 0.8:  # 80% of limit
                severity = "HIGH" if current_drawdown > params['max_drawdown'] else "MEDIUM"
                alerts.append(RiskEvent(
                    event_type=RiskEventType.DRAWDOWN_LIMIT_EXCEEDED,
                    symbol=None,
                    severity=severity,
                    message=f"Portfolio drawdown {current_drawdown:.2%} approaching limit {params['max_drawdown']:.2%}",
                    timestamp=datetime.now(),
                    metrics={'current_drawdown': current_drawdown, 'limit': params['max_drawdown']},
                    recommended_actions=["Reduce position sizes", "Consider defensive positions", "Review stop losses"]
                ))
            
            # Check position concentration
            positions = portfolio_data.get('positions', {})
            total_value = portfolio_data.get('total_value', 1)
            
            for symbol, position in positions.items():
                position_weight = position.get('market_value', 0) / total_value
                if position_weight > params['max_position_size']:
                    alerts.append(RiskEvent(
                        event_type=RiskEventType.POSITION_SIZE_EXCEEDED,
                        symbol=symbol,
                        severity="MEDIUM",
                        message=f"Position {symbol} weight {position_weight:.2%} exceeds limit {params['max_position_size']:.2%}",
                        timestamp=datetime.now(),
                        metrics={'position_weight': position_weight, 'limit': params['max_position_size']},
                        recommended_actions=[f"Reduce {symbol} position", "Rebalance portfolio"]
                    ))
            
            # Check volatility spikes
            for symbol in positions.keys():
                volatility = await self._get_symbol_volatility(symbol)
                if volatility > params['volatility_threshold']:
                    alerts.append(RiskEvent(
                        event_type=RiskEventType.VOLATILITY_SPIKE,
                        symbol=symbol,
                        severity="MEDIUM",
                        message=f"High volatility detected for {symbol}: {volatility:.2%}",
                        timestamp=datetime.now(),
                        metrics={'volatility': volatility, 'threshold': params['volatility_threshold']},
                        recommended_actions=[f"Monitor {symbol} closely", "Consider tighter stops"]
                    ))
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating risk alerts: {str(e)}")
            return []
    
    def update_portfolio_state(self, portfolio_value: float, positions: Dict[str, Dict]):
        """Update internal portfolio state"""
        self.portfolio_value = portfolio_value
        self.positions = positions
        
        # Update price history for volatility calculations
        for symbol, position in positions.items():
            current_price = position.get('current_price', 0)
            if current_price > 0:
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                
                self.price_history[symbol].append(current_price)
                
                # Keep last 252 prices (1 year of daily data)
                if len(self.price_history[symbol]) > 252:
                    self.price_history[symbol] = self.price_history[symbol][-252:]
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk management summary"""
        params = self.get_current_params()
        
        return {
            'risk_level': self.current_risk_level.value,
            'parameters': params,
            'portfolio_value': self.portfolio_value,
            'position_count': len(self.positions),
            'recent_events': len(self.risk_events),
            'last_update': datetime.now().isoformat()
        }
