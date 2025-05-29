"""
Schwab AI Trading System - Portfolio Management System
Advanced portfolio optimization with dynamic rebalancing, sector allocation, and performance tracking.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from models.signal_detector import TradingSignal, SignalType
from trading.risk_manager import RiskManager, RiskLevel
from utils.cache_manager import CacheManager
from utils.database import DatabaseManager
from config.settings import get_settings

logger = logging.getLogger(__name__)

class RebalanceReason(Enum):
    """Reasons for portfolio rebalancing"""
    SCHEDULED = "scheduled"
    DRIFT_THRESHOLD = "drift_threshold"
    SIGNAL_BASED = "signal_based"
    RISK_MANAGEMENT = "risk_management"
    VOLATILITY_REGIME = "volatility_regime"

class AllocationStrategy(Enum):
    """Portfolio allocation strategies"""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MEAN_VARIANCE = "mean_variance"
    KELLY_OPTIMAL = "kelly_optimal"
    AI_ADAPTIVE = "ai_adaptive"

@dataclass
class Position:
    """Individual position in portfolio"""
    symbol: str
    shares: int
    avg_cost: float
    current_price: float
    market_value: float
    weight: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    day_change: float
    day_change_pct: float
    sector: str
    last_updated: datetime

@dataclass 
class PortfolioAllocation:
    """Target portfolio allocation"""
    symbol: str
    target_weight: float
    current_weight: float
    target_value: float
    current_value: float
    shares_to_trade: int
    trade_value: float
    priority: int
    reasoning: str

@dataclass
class RebalanceRecommendation:
    """Portfolio rebalancing recommendation"""
    reason: RebalanceReason
    allocations: List[PortfolioAllocation]
    total_trades: int
    total_trade_value: float
    expected_improvement: float
    risk_impact: float
    costs_estimate: float
    urgency: str  # LOW, MEDIUM, HIGH
    reasoning: List[str]

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_value: float
    cash_balance: float
    invested_value: float
    total_return: float
    total_return_pct: float
    day_change: float
    day_change_pct: float
    ytd_return: float
    ytd_return_pct: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    position_count: int
    sector_allocation: Dict[str, float]
    top_holdings: List[Dict[str, Any]]

class PortfolioManager:
    """
    Advanced portfolio management system with dynamic allocation,
    risk-adjusted optimization, and automated rebalancing.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.risk_manager = RiskManager()
        self.cache_manager = CacheManager()
        self.db_manager = DatabaseManager()
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.cash_balance = 0.0
        self.total_value = 0.0
        self.allocation_strategy = AllocationStrategy.AI_ADAPTIVE
        
        # Rebalancing parameters
        self.rebalance_threshold = 0.05  # 5% drift threshold
        self.min_rebalance_value = 1000  # Minimum trade value
        self.rebalance_frequency = 7  # Days between scheduled rebalances
        self.last_rebalance = None
        
        # Sector limits
        self.sector_limits = {
            'Technology': 0.40,  # Max 40% in tech
            'Healthcare': 0.20,
            'Finance': 0.20,
            'Consumer': 0.20,
            'Energy': 0.10,
            'Real Estate': 0.10,
            'Utilities': 0.10,
            'Other': 0.15
        }
        
        # Performance tracking
        self.performance_history = []
        self.benchmark_symbol = 'SPY'
        
        logger.info("PortfolioManager initialized")
    
    async def initialize(self) -> bool:
        """Initialize the portfolio manager"""
        try:
            await self.risk_manager.initialize()
            await self._load_portfolio_state()
            await self._load_performance_history()
            
            logger.info("PortfolioManager initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PortfolioManager: {str(e)}")
            return False
    
    async def _load_portfolio_state(self):
        """Load current portfolio state from database"""
        try:
            # Load positions
            query = """
                SELECT symbol, shares, avg_cost, current_price, sector,
                       market_value, unrealized_pnl, last_updated
                FROM positions 
                WHERE shares > 0
            """
            
            positions_data = await self.db_manager.execute_query(query)
            
            for row in positions_data:
                position = Position(
                    symbol=row['symbol'],
                    shares=row['shares'],
                    avg_cost=row['avg_cost'],
                    current_price=row['current_price'],
                    market_value=row['market_value'],
                    weight=0.0,  # Will be calculated
                    unrealized_pnl=row['unrealized_pnl'],
                    unrealized_pnl_pct=row['unrealized_pnl'] / (row['shares'] * row['avg_cost']) * 100,
                    day_change=0.0,  # Will be updated
                    day_change_pct=0.0,
                    sector=row['sector'],
                    last_updated=row['last_updated']
                )
                self.positions[row['symbol']] = position
            
            # Load cash balance
            cash_query = "SELECT cash_balance FROM account_summary ORDER BY date DESC LIMIT 1"
            cash_result = await self.db_manager.execute_query(cash_query)
            self.cash_balance = cash_result[0]['cash_balance'] if cash_result else 0.0
            
            # Calculate total value and weights
            await self._calculate_portfolio_metrics()
            
            logger.info(f"Loaded portfolio: {len(self.positions)} positions, ${self.total_value:.2f} total value")
            
        except Exception as e:
            logger.error(f"Error loading portfolio state: {str(e)}")
    
    async def _load_performance_history(self):
        """Load historical performance data"""
        try:
            query = """
                SELECT date, portfolio_value, benchmark_value, 
                       daily_return, cumulative_return
                FROM portfolio_performance
                WHERE date >= %s
                ORDER BY date
            """
            
            lookback_date = datetime.now() - timedelta(days=365)
            self.performance_history = await self.db_manager.execute_query(query, (lookback_date,))
            
        except Exception as e:
            logger.error(f"Error loading performance history: {str(e)}")
            self.performance_history = []
    
    async def update_positions(self, market_data: Dict[str, Dict[str, float]]):
        """Update position values with current market data"""
        try:
            total_market_value = 0.0
            
            for symbol, position in self.positions.items():
                if symbol in market_data:
                    current_price = market_data[symbol]['price']
                    prev_close = market_data[symbol].get('prev_close', current_price)
                    
                    # Update position values
                    position.current_price = current_price
                    position.market_value = position.shares * current_price
                    position.unrealized_pnl = position.market_value - (position.shares * position.avg_cost)
                    position.unrealized_pnl_pct = (position.unrealized_pnl / (position.shares * position.avg_cost)) * 100
                    position.day_change = position.shares * (current_price - prev_close)
                    position.day_change_pct = ((current_price - prev_close) / prev_close) * 100
                    position.last_updated = datetime.now()
                    
                    total_market_value += position.market_value
            
            # Update total value and weights
            self.total_value = total_market_value + self.cash_balance
            
            for position in self.positions.values():
                position.weight = position.market_value / self.total_value if self.total_value > 0 else 0
            
            # Update risk manager
            self.risk_manager.update_portfolio_state(self.total_value, 
                {symbol: {'market_value': pos.market_value, 'current_price': pos.current_price} 
                 for symbol, pos in self.positions.items()})
            
            logger.debug(f"Updated portfolio positions, total value: ${self.total_value:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
    
    async def calculate_optimal_allocation(self, signals: Dict[str, TradingSignal],
                                         available_cash: float) -> Dict[str, float]:
        """
        Calculate optimal portfolio allocation based on AI signals and risk constraints
        
        Args:
            signals: Dictionary of trading signals by symbol
            available_cash: Available cash for new positions
            
        Returns:
            Dictionary of target weights by symbol
        """
        try:
            if self.allocation_strategy == AllocationStrategy.AI_ADAPTIVE:
                return await self._ai_adaptive_allocation(signals, available_cash)
            elif self.allocation_strategy == AllocationStrategy.RISK_PARITY:
                return await self._risk_parity_allocation(signals)
            elif self.allocation_strategy == AllocationStrategy.KELLY_OPTIMAL:
                return await self._kelly_optimal_allocation(signals)
            elif self.allocation_strategy == AllocationStrategy.MEAN_VARIANCE:
                return await self._mean_variance_allocation(signals)
            else:  # EQUAL_WEIGHT
                return await self._equal_weight_allocation(signals)
                
        except Exception as e:
            logger.error(f"Error calculating optimal allocation: {str(e)}")
            return {}
    
    async def _ai_adaptive_allocation(self, signals: Dict[str, TradingSignal], 
                                    available_cash: float) -> Dict[str, float]:
        """AI-adaptive allocation based on signal strength and confidence"""
        try:
            allocation = {}
            
            # Filter valid signals
            valid_signals = {symbol: signal for symbol, signal in signals.items() 
                           if signal.confidence > 0.6 and signal.signal_type != SignalType.HOLD}
            
            if not valid_signals:
                return {}
            
            # Calculate signal scores
            signal_scores = {}
            for symbol, signal in valid_signals.items():
                # Base score from confidence and signal strength
                confidence_score = signal.confidence
                strength_score = signal.strength.value / 4.0  # Normalize to 0-1
                
                # Adjust for signal type
                direction_multiplier = 1.0
                if signal.signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
                    direction_multiplier = 1.5
                elif signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                    direction_multiplier = 1.0
                
                # Risk adjustment
                risk_adjustment = 1.0 - signal.risk_score
                
                # Final score
                signal_scores[symbol] = confidence_score * strength_score * direction_multiplier * risk_adjustment
            
            # Normalize scores to weights
            total_score = sum(signal_scores.values())
            if total_score > 0:
                base_weights = {symbol: score / total_score for symbol, score in signal_scores.items()}
            else:
                return {}
            
            # Apply sector constraints
            allocation = await self._apply_sector_constraints(base_weights, valid_signals)
            
            # Apply position size limits
            max_position_weight = self.risk_manager.get_current_params()['max_position_size']
            for symbol in allocation:
                allocation[symbol] = min(allocation[symbol], max_position_weight)
            
            # Renormalize after constraints
            total_weight = sum(allocation.values())
            if total_weight > 0:
                allocation = {symbol: weight / total_weight for symbol, weight in allocation.items()}
            
            return allocation
            
        except Exception as e:
            logger.error(f"Error in AI adaptive allocation: {str(e)}")
            return {}
    
    async def _apply_sector_constraints(self, base_weights: Dict[str, float], 
                                      signals: Dict[str, TradingSignal]) -> Dict[str, float]:
        """Apply sector allocation constraints"""
        try:
            # Get sector mapping for each symbol
            sector_weights = {}
            symbol_sectors = {}
            
            for symbol, weight in base_weights.items():
                sector = await self._get_symbol_sector(symbol)
                symbol_sectors[symbol] = sector
                
                if sector not in sector_weights:
                    sector_weights[sector] = 0
                sector_weights[sector] += weight
            
            # Check sector limits and adjust if needed
            adjusted_weights = base_weights.copy()
            
            for sector, current_weight in sector_weights.items():
                sector_limit = self.sector_limits.get(sector, 0.15)
                
                if current_weight > sector_limit:
                    # Reduce weights proportionally for this sector
                    reduction_factor = sector_limit / current_weight
                    
                    for symbol, weight in base_weights.items():
                        if symbol_sectors[symbol] == sector:
                            adjusted_weights[symbol] = weight * reduction_factor
            
            return adjusted_weights
            
        except Exception as e:
            logger.error(f"Error applying sector constraints: {str(e)}")
            return base_weights
    
    async def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector classification for symbol"""
        try:
            # Try cache first
            cache_key = f"sector_{symbol}"
            cached_sector = await self.cache_manager.get(cache_key)
            if cached_sector:
                return cached_sector
            
            # Check database
            query = "SELECT sector FROM symbol_info WHERE symbol = %s"
            result = await self.db_manager.execute_query(query, (symbol,))
            
            if result:
                sector = result[0]['sector']
                await self.cache_manager.set(cache_key, sector, expire=86400)  # Cache for 24 hours
                return sector
            
            # Default sector mapping
            sector_map = {
                'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
                'AMZN': 'Consumer', 'TSLA': 'Consumer', 'META': 'Technology',
                'NVDA': 'Technology', 'NFLX': 'Technology', 'CRM': 'Technology',
                'JPM': 'Finance', 'BAC': 'Finance', 'GS': 'Finance', 'V': 'Finance',
                'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
                'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
                'SPY': 'Market', 'QQQ': 'Technology', 'IWM': 'Market',
                'VTI': 'Market', 'VOO': 'Market'
            }
            
            sector = sector_map.get(symbol, 'Other')
            await self.cache_manager.set(cache_key, sector, expire=86400)
            
            return sector
            
        except Exception as e:
            logger.error(f"Error getting sector for {symbol}: {str(e)}")
            return 'Other'
    
    async def _risk_parity_allocation(self, signals: Dict[str, TradingSignal]) -> Dict[str, float]:
        """Risk parity allocation - equal risk contribution"""
        try:
            symbols = list(signals.keys())
            if not symbols:
                return {}
            
            # Get volatilities for each symbol
            volatilities = {}
            for symbol in symbols:
                vol = await self.risk_manager._get_symbol_volatility(symbol)
                volatilities[symbol] = vol
            
            # Calculate inverse volatility weights
            inv_vol_weights = {symbol: 1.0 / vol for symbol, vol in volatilities.items()}
            
            # Normalize to sum to 1
            total_inv_vol = sum(inv_vol_weights.values())
            allocation = {symbol: weight / total_inv_vol for symbol, weight in inv_vol_weights.items()}
            
            return allocation
            
        except Exception as e:
            logger.error(f"Error in risk parity allocation: {str(e)}")
            return {}
    
    async def _kelly_optimal_allocation(self, signals: Dict[str, TradingSignal]) -> Dict[str, float]:
        """Kelly optimal allocation based on expected returns and volatilities"""
        try:
            allocation = {}
            
            for symbol, signal in signals.items():
                # Get expected return from signal predictions
                expected_return = signal.model_predictions.get('price_change', 0)
                confidence = signal.confidence
                
                # Get volatility
                volatility = await self.risk_manager._get_symbol_volatility(symbol)
                
                # Kelly fraction: f = (μ - r) / σ²
                # where μ is expected return, r is risk-free rate, σ is volatility
                risk_free_rate = 0.02 / 252  # Daily risk-free rate
                excess_return = expected_return - risk_free_rate
                
                kelly_fraction = excess_return / (volatility ** 2) if volatility > 0 else 0
                
                # Apply confidence adjustment
                kelly_fraction *= confidence
                
                # Cap Kelly fraction for safety
                kelly_fraction = max(0, min(kelly_fraction, 0.25))
                
                allocation[symbol] = kelly_fraction
            
            # Normalize
            total_kelly = sum(allocation.values())
            if total_kelly > 0:
                allocation = {symbol: weight / total_kelly for symbol, weight in allocation.items()}
            
            return allocation
            
        except Exception as e:
            logger.error(f"Error in Kelly optimal allocation: {str(e)}")
            return {}
    
    async def _mean_variance_allocation(self, signals: Dict[str, TradingSignal]) -> Dict[str, float]:
        """Mean-variance optimization (Markowitz)"""
        try:
            symbols = list(signals.keys())
            n_assets = len(symbols)
            
            if n_assets == 0:
                return {}
            
            # Get expected returns
            expected_returns = np.array([
                signals[symbol].model_predictions.get('price_change', 0) for symbol in symbols
            ])
            
            # Build covariance matrix
            cov_matrix = np.zeros((n_assets, n_assets))
            for i, symbol1 in enumerate(symbols):
                vol1 = await self.risk_manager._get_symbol_volatility(symbol1)
                for j, symbol2 in enumerate(symbols):
                    vol2 = await self.risk_manager._get_symbol_volatility(symbol2)
                    correlation = await self.risk_manager._get_correlation(symbol1, symbol2)
                    cov_matrix[i, j] = correlation * vol1 * vol2
            
            # Optimization objective: minimize risk for given return
            def objective(weights):
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                return portfolio_variance
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            ]
            
            # Bounds
            bounds = [(0, 0.25) for _ in range(n_assets)]  # Max 25% per asset
            
            # Initial guess
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                allocation = {symbols[i]: float(result.x[i]) for i in range(n_assets)}
                return allocation
            else:
                # Fall back to equal weight
                return {symbol: 1.0 / n_assets for symbol in symbols}
                
        except Exception as e:
            logger.error(f"Error in mean-variance allocation: {str(e)}")
            return {}
    
    async def _equal_weight_allocation(self, signals: Dict[str, TradingSignal]) -> Dict[str, float]:
        """Simple equal weight allocation"""
        try:
            symbols = list(signals.keys())
            if not symbols:
                return {}
            
            weight = 1.0 / len(symbols)
            return {symbol: weight for symbol in symbols}
            
        except Exception as e:
            logger.error(f"Error in equal weight allocation: {str(e)}")
            return {}
    
    async def generate_rebalance_recommendation(self, signals: Dict[str, TradingSignal],
                                             force_reason: Optional[RebalanceReason] = None) -> Optional[RebalanceRecommendation]:
        """
        Generate portfolio rebalancing recommendation
        
        Args:
            signals: Current trading signals
            force_reason: Force rebalancing for specific reason
            
        Returns:
            RebalanceRecommendation or None if no rebalancing needed
        """
        try:
            # Check if rebalancing is needed
            rebalance_reason = force_reason or await self._should_rebalance()
            
            if not rebalance_reason:
                return None
            
            # Calculate target allocation
            target_allocation = await self.calculate_optimal_allocation(signals, self.cash_balance)
            
            if not target_allocation:
                return None
            
            # Calculate current allocation
            current_allocation = {symbol: pos.weight for symbol, pos in self.positions.items()}
            
            # Generate allocation changes
            allocations = []
            total_trade_value = 0
            
            # Add new positions
            for symbol, target_weight in target_allocation.items():
                current_weight = current_allocation.get(symbol, 0)
                current_value = current_weight * self.total_value
                target_value = target_weight * self.total_value
                
                trade_value = target_value - current_value
                
                if abs(trade_value) > self.min_rebalance_value:
                    # Get current price
                    current_price = self.positions[symbol].current_price if symbol in self.positions else signals[symbol].price
                    shares_to_trade = int(trade_value / current_price)
                    
                    allocation = PortfolioAllocation(
                        symbol=symbol,
                        target_weight=target_weight,
                        current_weight=current_weight,
                        target_value=target_value,
                        current_value=current_value,
                        shares_to_trade=shares_to_trade,
                        trade_value=trade_value,
                        priority=self._calculate_priority(symbol, signals),
                        reasoning=self._generate_allocation_reasoning(symbol, signals, target_weight, current_weight)
                    )
                    
                    allocations.append(allocation)
                    total_trade_value += abs(trade_value)
            
            # Handle positions to close
            for symbol, current_weight in current_allocation.items():
                if symbol not in target_allocation and current_weight > 0:
                    current_value = current_weight * self.total_value
                    shares_to_trade = -self.positions[symbol].shares
                    
                    allocation = PortfolioAllocation(
                        symbol=symbol,
                        target_weight=0.0,
                        current_weight=current_weight,
                        target_value=0.0,
                        current_value=current_value,
                        shares_to_trade=shares_to_trade,
                        trade_value=-current_value,
                        priority=1,  # High priority to close
                        reasoning="Position not in target allocation"
                    )
                    
                    allocations.append(allocation)
                    total_trade_value += current_value
            
            if not allocations:
                return None
            
            # Sort by priority
            allocations.sort(key=lambda x: x.priority, reverse=True)
            
            # Calculate expected improvement and costs
            expected_improvement = await self._calculate_expected_improvement(allocations, signals)
            costs_estimate = await self._estimate_trading_costs(allocations)
            risk_impact = await self._calculate_risk_impact(target_allocation)
            
            # Determine urgency
            urgency = self._determine_urgency(rebalance_reason, expected_improvement, costs_estimate)
            
            # Generate reasoning
            reasoning = await self._generate_rebalance_reasoning(rebalance_reason, allocations, expected_improvement)
            
            recommendation = RebalanceRecommendation(
                reason=rebalance_reason,
                allocations=allocations,
                total_trades=len(allocations),
                total_trade_value=total_trade_value,
                expected_improvement=expected_improvement,
                risk_impact=risk_impact,
                costs_estimate=costs_estimate,
                urgency=urgency,
                reasoning=reasoning
            )
            
            logger.info(f"Generated rebalance recommendation: {len(allocations)} trades, ${total_trade_value:.2f} total value")
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating rebalance recommendation: {str(e)}")
            return None
    
    async def _should_rebalance(self) -> Optional[RebalanceReason]:
        """Check if portfolio should be rebalanced"""
        try:
            # Check scheduled rebalancing
            if self.last_rebalance is None or (datetime.now() - self.last_rebalance).days >= self.rebalance_frequency:
                return RebalanceReason.SCHEDULED
            
            # Check drift threshold
            max_drift = 0
            for symbol, position in self.positions.items():
                # This would need target weights to calculate drift
                # For now, assume some threshold logic
                if position.weight > self.risk_manager.get_current_params()['max_position_size']:
                    max_drift = max(max_drift, position.weight - self.risk_manager.get_current_params()['max_position_size'])
            
            if max_drift > self.rebalance_threshold:
                return RebalanceReason.DRIFT_THRESHOLD
            
            # Check risk management triggers
            risk_metrics = await self.risk_manager.calculate_risk_metrics({
                'total_value': self.total_value,
                'positions': {symbol: {'market_value': pos.market_value} for symbol, pos in self.positions.items()}
            })
            
            if risk_metrics.max_drawdown > self.risk_manager.get_current_params()['max_drawdown'] * 0.8:
                return RebalanceReason.RISK_MANAGEMENT
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking rebalance conditions: {str(e)}")
            return None
    
    def _calculate_priority(self, symbol: str, signals: Dict[str, TradingSignal]) -> int:
        """Calculate priority for allocation (1-10, higher is more important)"""
        try:
            if symbol not in signals:
                return 5  # Medium priority
            
            signal = signals[symbol]
            
            # Base priority from signal strength
            priority = signal.strength.value * 2  # 2-8 range
            
            # Adjust for confidence
            if signal.confidence > 0.8:
                priority += 1
            elif signal.confidence < 0.6:
                priority -= 1
            
            # Adjust for signal type
            if signal.signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
                priority += 1
            
            return max(1, min(10, priority))
            
        except Exception as e:
            logger.error(f"Error calculating priority: {str(e)}")
            return 5
    
    def _generate_allocation_reasoning(self, symbol: str, signals: Dict[str, TradingSignal],
                                     target_weight: float, current_weight: float) -> str:
        """Generate reasoning for allocation change"""
        try:
            if symbol not in signals:
                return "No active signal"
            
            signal = signals[symbol]
            
            reasons = []
            
            # Signal information
            reasons.append(f"{signal.signal_type.value} signal with {signal.confidence:.1%} confidence")
            
            # Weight change
            weight_change = target_weight - current_weight
            if weight_change > 0:
                reasons.append(f"Increasing allocation by {weight_change:.1%}")
            elif weight_change < 0:
                reasons.append(f"Decreasing allocation by {abs(weight_change):.1%}")
            
            # Key factors
            if signal.model_predictions.get('price_change', 0) > 0.03:
                reasons.append("Strong positive price prediction")
            elif signal.model_predictions.get('price_change', 0) < -0.03:
                reasons.append("Strong negative price prediction")
            
            return "; ".join(reasons)
            
        except Exception as e:
            logger.error(f"Error generating allocation reasoning: {str(e)}")
            return "Allocation adjustment needed"
    
    async def _calculate_expected_improvement(self, allocations: List[PortfolioAllocation],
                                           signals: Dict[str, TradingSignal]) -> float:
        """Calculate expected improvement from rebalancing"""
        try:
            total_improvement = 0.0
            
            for allocation in allocations:
                symbol = allocation.symbol
                if symbol in signals:
                    signal = signals[symbol]
                    expected_return = signal.model_predictions.get('price_change', 0)
                    weight_change = allocation.target_weight - allocation.current_weight
                    
                    # Improvement = weight_change * expected_return * confidence
                    improvement = weight_change * expected_return * signal.confidence
                    total_improvement += improvement
            
            return total_improvement
            
        except Exception as e:
            logger.error(f"Error calculating expected improvement: {str(e)}")
            return 0.0
    
    async def _estimate_trading_costs(self, allocations: List[PortfolioAllocation]) -> float:
        """Estimate trading costs for rebalancing"""
        try:
            total_costs = 0.0
            
            # Schwab commission (assume $0 for stocks, small fee for options)
            commission_per_trade = 0.0
            
            # Bid-ask spread cost (estimate)
            spread_cost_rate = 0.001  # 0.1% per trade
            
            for allocation in allocations:
                # Commission
                if allocation.shares_to_trade != 0:
                    total_costs += commission_per_trade
                
                # Spread cost
                total_costs += abs(allocation.trade_value) * spread_cost_rate
            
            return total_costs
            
        except Exception as e:
            logger.error(f"Error estimating trading costs: {str(e)}")
            return 0.0
    
    async def _calculate_risk_impact(self, target_allocation: Dict[str, float]) -> float:
        """Calculate risk impact of new allocation"""
        try:
            # Simple risk calculation based on concentration
            max_weight = max(target_allocation.values()) if target_allocation else 0
            n_positions = len(target_allocation)
            
            # Risk increases with concentration and decreases with diversification
            concentration_risk = max_weight
            diversification_benefit = 1.0 / np.sqrt(n_positions) if n_positions > 0 else 1.0
            
            risk_impact = concentration_risk - diversification_benefit
            
            return risk_impact
            
        except Exception as e:
            logger.error(f"Error calculating risk impact: {str(e)}")
            return 0.0
    
    def _determine_urgency(self, reason: RebalanceReason, expected_improvement: float,
                          costs_estimate: float) -> str:
        """Determine urgency level for rebalancing"""
        try:
            # Cost-benefit ratio
            if costs_estimate > 0:
                benefit_cost_ratio = expected_improvement / costs_estimate
            else:
                benefit_cost_ratio = expected_improvement * 100  # High ratio if no costs
            
            if reason == RebalanceReason.RISK_MANAGEMENT:
                return "HIGH"
            elif reason == RebalanceReason.DRIFT_THRESHOLD and benefit_cost_ratio > 5:
                return "HIGH"
            elif reason == RebalanceReason.SIGNAL_BASED and benefit_cost_ratio > 3:
                return "MEDIUM"
            elif reason == RebalanceReason.SCHEDULED and benefit_cost_ratio > 2:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception as e:
            logger.error(f"Error determining urgency: {str(e)}")
            return "MEDIUM"
    
    async def _generate_rebalance_reasoning(self, reason: RebalanceReason,
                                         allocations: List[PortfolioAllocation],
                                         expected_improvement: float) -> List[str]:
        """Generate reasoning for rebalancing recommendation"""
        try:
            reasoning = []
            
            # Main reason
            reason_text = {
                RebalanceReason.SCHEDULED: "Scheduled rebalancing due",
                RebalanceReason.DRIFT_THRESHOLD: "Portfolio allocation has drifted from targets",
                RebalanceReason.SIGNAL_BASED: "New AI signals indicate better opportunities",
                RebalanceReason.RISK_MANAGEMENT: "Risk management controls triggered",
                RebalanceReason.VOLATILITY_REGIME: "Market volatility regime change detected"
            }
            reasoning.append(reason_text.get(reason, "Rebalancing recommended"))
            
            # Key changes
            buy_count = len([a for a in allocations if a.shares_to_trade > 0])
            sell_count = len([a for a in allocations if a.shares_to_trade < 0])
            
            if buy_count > 0:
                reasoning.append(f"Adding/increasing {buy_count} positions")
            if sell_count > 0:
                reasoning.append(f"Reducing/closing {sell_count} positions")
            
            # Expected benefit
            if expected_improvement > 0.01:
                reasoning.append(f"Expected improvement: {expected_improvement:.1%}")
            elif expected_improvement < -0.01:
                reasoning.append(f"Risk reduction priority (expected cost: {abs(expected_improvement):.1%})")
            
            return reasoning
            
        except Exception as e:
            logger.error(f"Error generating rebalance reasoning: {str(e)}")
            return ["Rebalancing recommended"]
    
    async def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio performance metrics"""
        try:
            # Basic metrics
            total_value = self.total_value
            cash_balance = self.cash_balance
            invested_value = sum(pos.market_value for pos in self.positions.values())
            
            # Daily changes
            day_change = sum(pos.day_change for pos in self.positions.values())
            day_change_pct = (day_change / (total_value - day_change)) * 100 if total_value > day_change else 0
            
            # Performance calculation
            total_return, total_return_pct = await self._calculate_total_return()
            ytd_return, ytd_return_pct = await self._calculate_ytd_return()
            annualized_return = await self._calculate_annualized_return()
            
            # Risk metrics
            volatility = await self._calculate_portfolio_volatility()
            sharpe_ratio = await self._calculate_sharpe_ratio(annualized_return, volatility)
            max_drawdown = await self._calculate_max_drawdown()
            
            # Trading metrics  
            win_rate = await self._calculate_win_rate()
            
            # Allocation metrics
            sector_allocation = await self._calculate_sector_allocation()
            top_holdings = await self._get_top_holdings()
            
            return PortfolioMetrics(
                total_value=total_value,
                cash_balance=cash_balance,
                invested_value=invested_value,
                total_return=total_return,
                total_return_pct=total_return_pct,
                day_change=day_change,
                day_change_pct=day_change_pct,
                ytd_return=ytd_return,
                ytd_return_pct=ytd_return_pct,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                position_count=len(self.positions),
                sector_allocation=sector_allocation,
                top_holdings=top_holdings
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return PortfolioMetrics(
                total_value=0, cash_balance=0, invested_value=0,
                total_return=0, total_return_pct=0, day_change=0, day_change_pct=0,
                ytd_return=0, ytd_return_pct=0, annualized_return=0,
                volatility=0, sharpe_ratio=0, max_drawdown=0, win_rate=0,
                position_count=0, sector_allocation={}, top_holdings=[]
            )
    
    async def _calculate_total_return(self) -> Tuple[float, float]:
        """Calculate total return since inception"""
        try:
            # Get initial portfolio value
            query = "SELECT portfolio_value FROM portfolio_history ORDER BY date LIMIT 1"
            result = await self.db_manager.execute_query(query)
            
            if result:
                initial_value = result[0]['portfolio_value']
                total_return = self.total_value - initial_value
                total_return_pct = (total_return / initial_value) * 100 if initial_value > 0 else 0
                return total_return, total_return_pct
            
            return 0.0, 0.0
            
        except Exception as e:
            logger.error(f"Error calculating total return: {str(e)}")
            return 0.0, 0.0
    
    async def _calculate_ytd_return(self) -> Tuple[float, float]:
        """Calculate year-to-date return"""
        try:
            year_start = datetime(datetime.now().year, 1, 1)
            query = "SELECT portfolio_value FROM portfolio_history WHERE date >= %s ORDER BY date LIMIT 1"
            result = await self.db_manager.execute_query(query, (year_start,))
            
            if result:
                ytd_start_value = result[0]['portfolio_value']
                ytd_return = self.total_value - ytd_start_value
                ytd_return_pct = (ytd_return / ytd_start_value) * 100 if ytd_start_value > 0 else 0
                return ytd_return, ytd_return_pct
            
            return 0.0, 0.0
            
        except Exception as e:
            logger.error(f"Error calculating YTD return: {str(e)}")
            return 0.0, 0.0
    
    async def _calculate_annualized_return(self) -> float:
        """Calculate annualized return"""
        try:
            if not self.performance_history:
                return 0.0
            
            # Get daily returns
            returns = [row['daily_return'] for row in self.performance_history if row['daily_return'] is not None]
            
            if len(returns) < 2:
                return 0.0
            
            # Calculate compound annual growth rate
            total_return = np.prod(1 + np.array(returns))
            days = len(returns)
            annualized_return = (total_return ** (252 / days)) - 1
            
            return annualized_return * 100
            
        except Exception as e:
            logger.error(f"Error calculating annualized return: {str(e)}")
            return 0.0
    
    async def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility (annualized)"""
        try:
            if not self.performance_history:
                return 0.0
            
            returns = [row['daily_return'] for row in self.performance_history if row['daily_return'] is not None]
            
            if len(returns) < 2:
                return 0.0
            
            volatility = np.std(returns) * np.sqrt(252) * 100
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {str(e)}")
            return 0.0
    
    async def _calculate_sharpe_ratio(self, annualized_return: float, volatility: float) -> float:
        """Calculate Sharpe ratio"""
        try:
            risk_free_rate = 2.0  # 2% risk-free rate
            
            if volatility > 0:
                sharpe_ratio = (annualized_return - risk_free_rate) / volatility
                return sharpe_ratio
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
    
    async def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            if not self.performance_history:
                return 0.0
            
            values = [row['portfolio_value'] for row in self.performance_history]
            
            if len(values) < 2:
                return 0.0
            
            cumulative = np.array(values)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (running_max - cumulative) / running_max
            
            return float(np.max(drawdown)) * 100
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0
    
    async def _calculate_win_rate(self) -> float:
        """Calculate win rate from closed positions"""
        try:
            query = """
                SELECT COUNT(*) as total_trades,
                       SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades
                FROM trades 
                WHERE status = 'CLOSED' AND realized_pnl IS NOT NULL
                AND execution_time >= %s
            """
            
            lookback_date = datetime.now() - timedelta(days=90)
            result = await self.db_manager.execute_query(query, (lookback_date,))
            
            if result and result[0]['total_trades'] > 0:
                win_rate = (result[0]['winning_trades'] / result[0]['total_trades']) * 100
                return win_rate
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {str(e)}")
            return 0.0
    
    async def _calculate_sector_allocation(self) -> Dict[str, float]:
        """Calculate current sector allocation"""
        try:
            sector_allocation = {}
            
            for symbol, position in self.positions.items():
                sector = await self._get_symbol_sector(symbol)
                weight = position.weight
                
                if sector in sector_allocation:
                    sector_allocation[sector] += weight
                else:
                    sector_allocation[sector] = weight
            
            # Convert to percentages
            return {sector: weight * 100 for sector, weight in sector_allocation.items()}
            
        except Exception as e:
            logger.error(f"Error calculating sector allocation: {str(e)}")
            return {}
    
    async def _get_top_holdings(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top portfolio holdings"""
        try:
            holdings = []
            
            # Sort positions by market value
            sorted_positions = sorted(self.positions.items(), 
                                    key=lambda x: x[1].market_value, reverse=True)
            
            for symbol, position in sorted_positions[:top_n]:
                holding = {
                    'symbol': symbol,
                    'weight': position.weight * 100,
                    'market_value': position.market_value,
                    'unrealized_pnl_pct': position.unrealized_pnl_pct,
                    'day_change_pct': position.day_change_pct,
                    'sector': position.sector
                }
                holdings.append(holding)
            
            return holdings
            
        except Exception as e:
            logger.error(f"Error getting top holdings: {str(e)}")
            return []
    
    async def execute_rebalance(self, recommendation: RebalanceRecommendation) -> bool:
        """
        Execute portfolio rebalancing (this would integrate with order management)
        
        Returns:
            True if rebalancing was successful
        """
        try:
            logger.info(f"Executing rebalance: {recommendation.total_trades} trades")
            
            # This would integrate with the order manager
            # For now, just log the intended trades
            
            for allocation in recommendation.allocations:
                if allocation.shares_to_trade != 0:
                    action = "BUY" if allocation.shares_to_trade > 0 else "SELL"
                    logger.info(f"  {action} {abs(allocation.shares_to_trade)} shares of {allocation.symbol}")
            
            # Update last rebalance time
            self.last_rebalance = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing rebalance: {str(e)}")
            return False
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary for API/UI"""
        return {
            'total_value': self.total_value,
            'cash_balance': self.cash_balance,
            'position_count': len(self.positions),
            'allocation_strategy': self.allocation_strategy.value,
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
            'positions': {
                symbol: {
                    'shares': pos.shares,
                    'market_value': pos.market_value,
                    'weight': pos.weight,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                    'day_change_pct': pos.day_change_pct
                }
                for symbol, pos in self.positions.items()
            }
        }
