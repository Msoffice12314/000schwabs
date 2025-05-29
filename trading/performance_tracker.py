import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import asyncio
from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import threading
from collections import defaultdict, deque
import time
import psutil

Base = declarative_base()

@dataclass
class PerformanceSnapshot:
    """Snapshot of portfolio performance at a point in time"""
    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    daily_return: float
    cumulative_return: float
    drawdown: float
    sharpe_ratio: float
    volatility: float
    num_positions: int
    sector_allocation: Dict[str, float] = field(default_factory=dict)
    position_details: Dict[str, Dict] = field(default_factory=dict)

@dataclass
class TradeRecord:
    """Individual trade record for performance tracking"""
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    commission: float
    total_cost: float
    strategy: str
    confidence: float
    market_conditions: Dict[str, Any] = field(default_factory=dict)

class PortfolioMetricsDB(Base):
    """SQLAlchemy model for portfolio metrics storage"""
    __tablename__ = 'portfolio_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    total_value = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    positions_value = Column(Float, nullable=False)
    daily_return = Column(Float, nullable=True)
    cumulative_return = Column(Float, nullable=True)
    drawdown = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    volatility = Column(Float, nullable=True)
    num_positions = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class TradesDB(Base):
    """SQLAlchemy model for trades storage"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, nullable=False)
    total_cost = Column(Float, nullable=False)
    strategy = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    market_data = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

class PerformanceTracker:
    """Advanced performance tracking system for live trading"""
    
    def __init__(self, initial_capital: float = 100000.0, 
                 db_path: str = "performance.db"):
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)
        
        # Database setup
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Performance data storage
        self.snapshots: List[PerformanceSnapshot] = []
        self.trades: List[TradeRecord] = []
        self.returns: deque = deque(maxlen=252)  # Keep last 252 days
        
        # Real-time metrics
        self.current_value = initial_capital
        self.peak_value = initial_capital
        self.current_drawdown = 0.0
        self.daily_returns = deque(maxlen=252)
        
        # Performance caching
        self.last_calculation_time = time.time()
        self.cached_metrics = {}
        self.cache_duration = 300  # 5 minutes
        
        # Threading for background calculations
        self.calculation_thread = None
        self.stop_calculation = threading.Event()
        
        # Benchmark tracking
        self.benchmark_returns = deque(maxlen=252)
        
        # Risk management alerts
        self.risk_thresholds = {
            'max_drawdown': 0.15,
            'daily_var_95': 0.05,
            'portfolio_heat': 0.1,
            'concentration_limit': 0.25
        }
        
        self.start_background_calculations()
    
    def start_background_calculations(self):
        """Start background thread for performance calculations"""
        if self.calculation_thread is None or not self.calculation_thread.is_alive():
            self.stop_calculation.clear()
            self.calculation_thread = threading.Thread(
                target=self._background_calculation_loop,
                daemon=True
            )
            self.calculation_thread.start()
    
    def stop_background_calculations(self):
        """Stop background calculations"""
        self.stop_calculation.set()
        if self.calculation_thread and self.calculation_thread.is_alive():
            self.calculation_thread.join(timeout=5.0)
    
    def _background_calculation_loop(self):
        """Background loop for continuous performance calculations"""
        while not self.stop_calculation.is_set():
            try:
                self._update_cached_metrics()
                time.sleep(60)  # Update every minute
            except Exception as e:
                self.logger.error(f"Error in background calculation: {e}")
                time.sleep(60)
    
    def record_trade(self, symbol: str, side: str, quantity: int, price: float,
                    commission: float = 0.0, strategy: str = "unknown",
                    confidence: float = 0.0, market_conditions: Dict = None):
        """Record a new trade and update performance metrics"""
        try:
            timestamp = datetime.now()
            total_cost = quantity * price + commission
            
            trade = TradeRecord(
                timestamp=timestamp,
                symbol=symbol,
                side=side.upper(),
                quantity=quantity,
                price=price,
                commission=commission,
                total_cost=total_cost,
                strategy=strategy,
                confidence=confidence,
                market_conditions=market_conditions or {}
            )
            
            self.trades.append(trade)
            
            # Store in database
            self._store_trade_in_db(trade)
            
            # Update portfolio value based on trade
            if side.upper() == 'BUY':
                self.current_value -= total_cost
            else:
                self.current_value += (quantity * price - commission)
            
            self.logger.info(f"Recorded trade: {side} {quantity} {symbol} @ ${price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
    
    def update_portfolio_value(self, positions: Dict[str, Dict], cash: float):
        """Update current portfolio value with latest position data"""
        try:
            positions_value = sum(
                pos['quantity'] * pos['current_price'] 
                for pos in positions.values()
            )
            
            new_total_value = cash + positions_value
            
            # Calculate daily return
            if self.current_value > 0:
                daily_return = (new_total_value - self.current_value) / self.current_value
                self.daily_returns.append(daily_return)
            
            self.current_value = new_total_value
            
            # Update peak and drawdown
            if new_total_value > self.peak_value:
                self.peak_value = new_total_value
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (new_total_value - self.peak_value) / self.peak_value
            
            # Create performance snapshot
            snapshot = self._create_performance_snapshot(
                cash, positions_value, positions
            )
            self.snapshots.append(snapshot)
            
            # Store in database
            self._store_snapshot_in_db(snapshot)
            
            # Check risk thresholds
            self._check_risk_alerts(snapshot)
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio value: {e}")
    
    def _create_performance_snapshot(self, cash: float, positions_value: float,
                                   positions: Dict[str, Dict]) -> PerformanceSnapshot:
        """Create a comprehensive performance snapshot"""
        timestamp = datetime.now()
        total_value = cash + positions_value
        
        # Calculate metrics
        daily_return = self.daily_returns[-1] if self.daily_returns else 0.0
        cumulative_return = (total_value - self.initial_capital) / self.initial_capital
        
        # Calculate rolling metrics
        sharpe_ratio = self._calculate_rolling_sharpe()
        volatility = self._calculate_rolling_volatility()
        
        # Sector allocation
        sector_allocation = self._calculate_sector_allocation(positions)
        
        # Position details
        position_details = {
            symbol: {
                'quantity': pos['quantity'],
                'price': pos['current_price'],
                'value': pos['quantity'] * pos['current_price'],
                'weight': (pos['quantity'] * pos['current_price']) / total_value,
                'unrealized_pnl': pos.get('unrealized_pnl', 0),
                'days_held': pos.get('days_held', 0)
            }
            for symbol, pos in positions.items()
        }
        
        return PerformanceSnapshot(
            timestamp=timestamp,
            total_value=total_value,
            cash=cash,
            positions_value=positions_value,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            drawdown=self.current_drawdown,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            num_positions=len(positions),
            sector_allocation=sector_allocation,
            position_details=position_details
        )
    
    def get_performance_metrics(self, period_days: int = 30) -> Dict[str, Any]:
        """Get comprehensive performance metrics for specified period"""
        try:
            # Check cache first
            cache_key = f"metrics_{period_days}"
            if (cache_key in self.cached_metrics and 
                time.time() - self.last_calculation_time < self.cache_duration):
                return self.cached_metrics[cache_key]
            
            # Calculate fresh metrics
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            # Filter snapshots for period
            period_snapshots = [
                s for s in self.snapshots 
                if start_date <= s.timestamp <= end_date
            ]
            
            if not period_snapshots:
                return {}
            
            # Calculate metrics
            metrics = self._calculate_comprehensive_metrics(period_snapshots)
            
            # Cache results
            self.cached_metrics[cache_key] = metrics
            self.last_calculation_time = time.time()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _calculate_comprehensive_metrics(self, snapshots: List[PerformanceSnapshot]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics from snapshots"""
        if not snapshots:
            return {}
        
        # Basic metrics
        start_value = snapshots[0].total_value
        end_value = snapshots[-1].total_value
        total_return = (end_value - start_value) / start_value
        
        # Time-based metrics
        days = (snapshots[-1].timestamp - snapshots[0].timestamp).days
        annualized_return = (1 + total_return) ** (365.25 / max(days, 1)) - 1 if days > 0 else 0
        
        # Return series
        returns = []
        for i in range(1, len(snapshots)):
            prev_val = snapshots[i-1].total_value
            curr_val = snapshots[i].total_value
            ret = (curr_val - prev_val) / prev_val if prev_val > 0 else 0
            returns.append(ret)
        
        returns_array = np.array(returns)
        
        # Risk metrics
        volatility = np.std(returns_array) * np.sqrt(252) if len(returns_array) > 1 else 0
        downside_returns = returns_array[returns_array < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 else 0
        
        # Drawdown metrics
        drawdowns = [s.drawdown for s in snapshots]
        max_drawdown = min(drawdowns) if drawdowns else 0
        
        # Risk-adjusted metrics
        risk_free_rate = 0.02  # 2% annual
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0
        
        # Value at Risk
        var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0
        var_99 = np.percentile(returns_array, 1) if len(returns_array) > 0 else 0
        
        # Trading metrics
        trade_metrics = self._calculate_trade_metrics()
        
        # Position metrics
        position_metrics = self._calculate_position_metrics(snapshots)
        
        return {
            'period': {
                'start_date': snapshots[0].timestamp.strftime('%Y-%m-%d'),
                'end_date': snapshots[-1].timestamp.strftime('%Y-%m-%d'),
                'days': days
            },
            'returns': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'best_day': max(returns_array) if len(returns_array) > 0 else 0,
                'worst_day': min(returns_array) if len(returns_array) > 0 else 0,
                'positive_days': sum(1 for r in returns_array if r > 0),
                'negative_days': sum(1 for r in returns_array if r < 0),
                'win_rate': sum(1 for r in returns_array if r > 0) / len(returns_array) if returns_array.size > 0 else 0
            },
            'risk': {
                'volatility': volatility,
                'downside_volatility': downside_volatility,
                'max_drawdown': max_drawdown,
                'current_drawdown': snapshots[-1].drawdown,
                'var_95': var_95,
                'var_99': var_99,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio
            },
            'portfolio': {
                'current_value': snapshots[-1].total_value,
                'cash': snapshots[-1].cash,
                'positions_value': snapshots[-1].positions_value,
                'num_positions': snapshots[-1].num_positions,
                'sector_allocation': snapshots[-1].sector_allocation
            },
            'trading': trade_metrics,
            'positions': position_metrics
        }
    
    def _calculate_trade_metrics(self) -> Dict[str, Any]:
        """Calculate trading-specific performance metrics"""
        if not self.trades:
            return {}
        
        # Group trades by symbol for P&L calculation
        symbol_trades = defaultdict(list)
        for trade in self.trades:
            symbol_trades[trade.symbol].append(trade)
        
        # Calculate realized P&L for closed positions
        realized_pnl = []
        total_commission = sum(trade.commission for trade in self.trades)
        
        for symbol, trades in symbol_trades.items():
            position = 0
            avg_price = 0
            
            for trade in sorted(trades, key=lambda x: x.timestamp):
                if trade.side == 'BUY':
                    if position == 0:
                        position = trade.quantity
                        avg_price = trade.price
                    else:
                        # Average up
                        total_cost = (position * avg_price) + (trade.quantity * trade.price)
                        position += trade.quantity
                        avg_price = total_cost / position if position > 0 else 0
                
                elif trade.side == 'SELL':
                    if position > 0:
                        sell_quantity = min(trade.quantity, position)
                        pnl = (trade.price - avg_price) * sell_quantity - trade.commission
                        realized_pnl.append(pnl)
                        position -= sell_quantity
        
        if not realized_pnl:
            return {'total_trades': len(self.trades), 'total_commission': total_commission}
        
        # Calculate trading statistics
        winning_trades = [pnl for pnl in realized_pnl if pnl > 0]
        losing_trades = [pnl for pnl in realized_pnl if pnl < 0]
        
        return {
            'total_trades': len(realized_pnl),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(realized_pnl) if realized_pnl else 0,
            'total_pnl': sum(realized_pnl),
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'largest_win': max(winning_trades) if winning_trades else 0,
            'largest_loss': min(losing_trades) if losing_trades else 0,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else float('inf'),
            'total_commission': total_commission
        }
    
    def _calculate_position_metrics(self, snapshots: List[PerformanceSnapshot]) -> Dict[str, Any]:
        """Calculate position-specific metrics"""
        if not snapshots:
            return {}
        
        latest_snapshot = snapshots[-1]
        
        # Portfolio concentration
        position_weights = [
            pos['weight'] for pos in latest_snapshot.position_details.values()
        ]
        
        max_position_weight = max(position_weights) if position_weights else 0
        top_5_concentration = sum(sorted(position_weights, reverse=True)[:5])
        
        # Sector concentration
        sector_weights = list(latest_snapshot.sector_allocation.values())
        max_sector_weight = max(sector_weights) if sector_weights else 0
        
        return {
            'portfolio_concentration': {
                'max_position_weight': max_position_weight,
                'top_5_concentration': top_5_concentration,
                'num_positions': len(position_weights)
            },
            'sector_concentration': {
                'max_sector_weight': max_sector_weight,
                'sector_allocation': latest_snapshot.sector_allocation
            },
            'unrealized_pnl': {
                'total': sum(pos['unrealized_pnl'] for pos in latest_snapshot.position_details.values()),
                'by_position': {
                    symbol: pos['unrealized_pnl'] 
                    for symbol, pos in latest_snapshot.position_details.items()
                }
            }
        }
    
    def _calculate_rolling_sharpe(self, window: int = 30) -> float:
        """Calculate rolling Sharpe ratio"""
        if len(self.daily_returns) < window:
            return 0.0
        
        recent_returns = list(self.daily_returns)[-window:]
        if not recent_returns:
            return 0.0
        
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize
        annual_return = mean_return * 252
        annual_volatility = std_return * np.sqrt(252)
        
        risk_free_rate = 0.02  # 2% annual
        return (annual_return - risk_free_rate) / annual_volatility
    
    def _calculate_rolling_volatility(self, window: int = 30) -> float:
        """Calculate rolling volatility"""
        if len(self.daily_returns) < window:
            return 0.0
        
        recent_returns = list(self.daily_returns)[-window:]
        if not recent_returns:
            return 0.0
        
        return np.std(recent_returns) * np.sqrt(252)
    
    def _calculate_sector_allocation(self, positions: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate sector allocation (placeholder - would need sector mapping)"""
        # This is a simplified implementation
        # In practice, you would map symbols to sectors
        total_value = sum(pos['quantity'] * pos['current_price'] for pos in positions.values())
        
        if total_value == 0:
            return {}
        
        # Placeholder sector mapping
        sector_map = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'TSLA': 'Automotive',
            'JPM': 'Financial',
            'BAC': 'Financial',
            'JNJ': 'Healthcare',
            'PFE': 'Healthcare'
        }
        
        sector_values = defaultdict(float)
        for symbol, pos in positions.items():
            value = pos['quantity'] * pos['current_price']
            sector = sector_map.get(symbol, 'Other')
            sector_values[sector] += value
        
        return {
            sector: value / total_value 
            for sector, value in sector_values.items()
        }
    
    def _check_risk_alerts(self, snapshot: PerformanceSnapshot):
        """Check for risk threshold breaches and generate alerts"""
        alerts = []
        
        # Drawdown alert
        if abs(snapshot.drawdown) > self.risk_thresholds['max_drawdown']:
            alerts.append({
                'type': 'MAX_DRAWDOWN_BREACH',
                'message': f"Maximum drawdown threshold breached: {abs(snapshot.drawdown)*100:.2f}%",
                'severity': 'HIGH',
                'timestamp': snapshot.timestamp
            })
        
        # Concentration alert
        max_position_weight = max(
            pos['weight'] for pos in snapshot.position_details.values()
        ) if snapshot.position_details else 0
        
        if max_position_weight > self.risk_thresholds['concentration_limit']:
            alerts.append({
                'type': 'CONCENTRATION_ALERT',
                'message': f"Position concentration too high: {max_position_weight*100:.2f}%",
                'severity': 'MEDIUM',
                'timestamp': snapshot.timestamp
            })
        
        # Daily VaR alert
        if len(self.daily_returns) > 0:
            recent_returns = list(self.daily_returns)[-30:]  # Last 30 days
            if len(recent_returns) >= 10:
                var_95 = np.percentile(recent_returns, 5)
                if abs(var_95) > self.risk_thresholds['daily_var_95']:
                    alerts.append({
                        'type': 'VAR_BREACH',
                        'message': f"Daily VaR 95% exceeded: {abs(var_95)*100:.2f}%",
                        'severity': 'MEDIUM',
                        'timestamp': snapshot.timestamp
                    })
        
        # Log alerts
        for alert in alerts:
            self.logger.warning(f"Risk Alert: {alert['message']}")
    
    def _store_snapshot_in_db(self, snapshot: PerformanceSnapshot):
        """Store performance snapshot in database"""
        try:
            with self.SessionLocal() as session:
                db_snapshot = PortfolioMetricsDB(
                    timestamp=snapshot.timestamp,
                    total_value=snapshot.total_value,
                    cash=snapshot.cash,
                    positions_value=snapshot.positions_value,
                    daily_return=snapshot.daily_return,
                    cumulative_return=snapshot.cumulative_return,
                    drawdown=snapshot.drawdown,
                    sharpe_ratio=snapshot.sharpe_ratio,
                    volatility=snapshot.volatility,
                    num_positions=snapshot.num_positions
                )
                session.add(db_snapshot)
                session.commit()
        except Exception as e:
            self.logger.error(f"Error storing snapshot in database: {e}")
    
    def _store_trade_in_db(self, trade: TradeRecord):
        """Store trade record in database"""
        try:
            with self.SessionLocal() as session:
                db_trade = TradesDB(
                    timestamp=trade.timestamp,
                    symbol=trade.symbol,
                    side=trade.side,
                    quantity=trade.quantity,
                    price=trade.price,
                    commission=trade.commission,
                    total_cost=trade.total_cost,
                    strategy=trade.strategy,
                    confidence=trade.confidence,
                    market_data=json.dumps(trade.market_conditions)
                )
                session.add(db_trade)
                session.commit()
        except Exception as e:
            self.logger.error(f"Error storing trade in database: {e}")
    
    def _update_cached_metrics(self):
        """Update cached metrics in background"""
        try:
            # Update various period metrics
            for period in [1, 7, 30, 90, 365]:
                self.get_performance_metrics(period)
        except Exception as e:
            self.logger.error(f"Error updating cached metrics: {e}")
    
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time data for dashboard display"""
        try:
            if not self.snapshots:
                return {}
            
            latest = self.snapshots[-1]
            
            # System performance
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            return {
                'portfolio': {
                    'total_value': latest.total_value,
                    'cash': latest.cash,
                    'positions_value': latest.positions_value,
                    'daily_return': latest.daily_return,
                    'cumulative_return': latest.cumulative_return,
                    'current_drawdown': latest.drawdown,
                    'num_positions': latest.num_positions
                },
                'performance': {
                    'sharpe_ratio': latest.sharpe_ratio,
                    'volatility': latest.volatility,
                    'peak_value': self.peak_value,
                    'days_trading': len(self.snapshots)
                },
                'recent_trades': [
                    {
                        'timestamp': trade.timestamp.strftime('%H:%M:%S'),
                        'symbol': trade.symbol,
                        'side': trade.side,
                        'quantity': trade.quantity,
                        'price': trade.price
                    }
                    for trade in self.trades[-5:]  # Last 5 trades
                ],
                'top_positions': sorted([
                    {
                        'symbol': symbol,
                        'value': pos['value'],
                        'weight': pos['weight'],
                        'unrealized_pnl': pos['unrealized_pnl']
                    }
                    for symbol, pos in latest.position_details.items()
                ], key=lambda x: x['value'], reverse=True)[:10],
                'system': {
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory_percent,
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {e}")
            return {}
    
    def export_performance_report(self, format: str = 'json', 
                                 period_days: int = 30) -> str:
        """Export comprehensive performance report"""
        try:
            metrics = self.get_performance_metrics(period_days)
            
            if format.lower() == 'json':
                return json.dumps(metrics, indent=2, default=str)
            
            elif format.lower() == 'csv':
                # Export snapshots as CSV
                if not self.snapshots:
                    return ""
                
                df_data = []
                for snapshot in self.snapshots:
                    df_data.append({
                        'timestamp': snapshot.timestamp,
                        'total_value': snapshot.total_value,
                        'cash': snapshot.cash,
                        'positions_value': snapshot.positions_value,
                        'daily_return': snapshot.daily_return,
                        'cumulative_return': snapshot.cumulative_return,
                        'drawdown': snapshot.drawdown,
                        'sharpe_ratio': snapshot.sharpe_ratio,
                        'volatility': snapshot.volatility,
                        'num_positions': snapshot.num_positions
                    })
                
                df = pd.DataFrame(df_data)
                return df.to_csv(index=False)
            
            else:
                return json.dumps(metrics, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error exporting performance report: {e}")
            return ""
    
    def get_attribution_analysis(self) -> Dict[str, Any]:
        """Perform performance attribution analysis"""
        try:
            if not self.snapshots or len(self.snapshots) < 2:
                return {}
            
            # Get recent snapshots for analysis
            recent_snapshots = self.snapshots[-30:]  # Last 30 snapshots
            
            # Calculate position contributions
            position_contributions = defaultdict(list)
            
            for i in range(1, len(recent_snapshots)):
                prev_snapshot = recent_snapshots[i-1]
                curr_snapshot = recent_snapshots[i]
                
                # Calculate contribution by position
                for symbol in curr_snapshot.position_details:
                    if symbol in prev_snapshot.position_details:
                        prev_value = prev_snapshot.position_details[symbol]['value']
                        curr_value = curr_snapshot.position_details[symbol]['value']
                        
                        if prev_snapshot.total_value > 0:
                            contribution = (curr_value - prev_value) / prev_snapshot.total_value
                            position_contributions[symbol].append(contribution)
            
            # Aggregate contributions
            attribution = {}
            for symbol, contributions in position_contributions.items():
                attribution[symbol] = {
                    'total_contribution': sum(contributions),
                    'average_contribution': np.mean(contributions),
                    'volatility': np.std(contributions) if len(contributions) > 1 else 0,
                    'periods': len(contributions)
                }
            
            return {
                'position_attribution': attribution,
                'period_analyzed': len(recent_snapshots),
                'start_date': recent_snapshots[0].timestamp.strftime('%Y-%m-%d'),
                'end_date': recent_snapshots[-1].timestamp.strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            self.logger.error(f"Error in attribution analysis: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old performance data to manage storage"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean snapshots
            self.snapshots = [
                s for s in self.snapshots 
                if s.timestamp > cutoff_date
            ]
            
            # Clean trades
            self.trades = [
                t for t in self.trades 
                if t.timestamp > cutoff_date
            ]
            
            # Clean database
            with self.SessionLocal() as session:
                session.query(PortfolioMetricsDB).filter(
                    PortfolioMetricsDB.timestamp < cutoff_date
                ).delete()
                
                session.query(TradesDB).filter(
                    TradesDB.timestamp < cutoff_date
                ).delete()
                
                session.commit()
            
            self.logger.info(f"Cleaned up data older than {days_to_keep} days")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_background_calculations()
