import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

from .metrics_calculator import MetricsCalculator
from .report_generator import ReportGenerator

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_positions: int = 10
    position_sizing: str = "equal_weight"  # equal_weight, kelly, risk_parity
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    benchmark: str = "SPY"

@dataclass
class Trade:
    """Individual trade record"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    price: float
    timestamp: datetime
    commission: float
    strategy_signal: str
    confidence: float

@dataclass
class Position:
    """Current position state"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    entry_date: datetime
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0

class BacktestEngine:
    """Advanced backtesting engine for trading strategies"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Portfolio state
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_history: List[Dict] = []
        
        # Performance tracking
        self.metrics_calculator = MetricsCalculator()
        self.report_generator = ReportGenerator()
        
        # Market data
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None
        
    async def load_market_data(self, symbols: List[str]) -> None:
        """Load historical market data for backtesting"""
        try:
            # This would integrate with your data provider
            self.logger.info(f"Loading market data for {len(symbols)} symbols")
            
            for symbol in symbols:
                # Placeholder for actual data loading
                # You would replace this with actual Schwab API calls
                data = await self._fetch_historical_data(symbol)
                self.market_data[symbol] = data
                
            # Load benchmark data
            if self.config.benchmark:
                self.benchmark_data = await self._fetch_historical_data(self.config.benchmark)
                
        except Exception as e:
            self.logger.error(f"Error loading market data: {e}")
            raise
    
    async def _fetch_historical_data(self, symbol: str) -> pd.DataFrame:
        """Fetch historical data for a symbol"""
        # This is a placeholder - replace with actual Schwab API integration
        date_range = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq='D'
        )
        
        # Generate mock data for demonstration
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(len(date_range)) * 0.01))
        
        return pd.DataFrame({
            'timestamp': date_range,
            'open': prices * (1 + np.random.randn(len(date_range)) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(len(date_range))) * 0.005),
            'low': prices * (1 - np.abs(np.random.randn(len(date_range))) * 0.005),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(date_range))
        }).set_index('timestamp')
    
    def run_backtest(self, strategy_signals: pd.DataFrame) -> Dict[str, Any]:
        """Run the main backtesting simulation"""
        try:
            self.logger.info("Starting backtesting simulation")
            
            # Initialize portfolio tracking
            self._initialize_portfolio()
            
            # Process each trading day
            trading_dates = pd.date_range(
                start=self.config.start_date,
                end=self.config.end_date,
                freq='B'  # Business days only
            )
            
            for date in trading_dates:
                self._process_trading_day(date, strategy_signals)
                self._update_portfolio_history(date)
            
            # Calculate final performance metrics
            results = self._calculate_backtest_results()
            
            self.logger.info("Backtesting simulation completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in backtesting simulation: {e}")
            raise
    
    def _initialize_portfolio(self) -> None:
        """Initialize portfolio state"""
        self.cash = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
    
    def _process_trading_day(self, date: datetime, strategy_signals: pd.DataFrame) -> None:
        """Process trading signals for a single day"""
        try:
            # Get signals for this date
            if date not in strategy_signals.index:
                return
            
            day_signals = strategy_signals.loc[date]
            
            # Update current positions with market prices
            self._update_position_prices(date)
            
            # Process sell signals first
            self._process_sell_signals(date, day_signals)
            
            # Process buy signals
            self._process_buy_signals(date, day_signals)
            
        except Exception as e:
            self.logger.error(f"Error processing trading day {date}: {e}")
    
    def _update_position_prices(self, date: datetime) -> None:
        """Update current positions with latest market prices"""
        for symbol, position in self.positions.items():
            if symbol in self.market_data and date in self.market_data[symbol].index:
                current_price = self.market_data[symbol].loc[date, 'close']
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
    
    def _process_sell_signals(self, date: datetime, signals: pd.Series) -> None:
        """Process sell signals for the day"""
        for symbol in self.positions.keys():
            if symbol in signals and signals[symbol] < 0:  # Sell signal
                self._execute_sell_order(date, symbol, abs(signals[symbol]))
    
    def _process_buy_signals(self, date: datetime, signals: pd.Series) -> None:
        """Process buy signals for the day"""
        buy_signals = signals[signals > 0]
        
        if len(buy_signals) == 0:
            return
        
        # Calculate position sizes
        available_cash = self.cash * 0.95  # Keep 5% cash buffer
        position_sizes = self._calculate_position_sizes(buy_signals, available_cash)
        
        for symbol, signal_strength in buy_signals.items():
            if symbol in position_sizes and position_sizes[symbol] > 0:
                self._execute_buy_order(date, symbol, position_sizes[symbol], signal_strength)
    
    def _calculate_position_sizes(self, signals: pd.Series, available_cash: float) -> Dict[str, float]:
        """Calculate position sizes based on strategy"""
        position_sizes = {}
        
        if self.config.position_sizing == "equal_weight":
            # Equal weight allocation
            position_value = available_cash / len(signals)
            for symbol in signals.index:
                position_sizes[symbol] = position_value
                
        elif self.config.position_sizing == "signal_weighted":
            # Weight by signal strength
            total_signal = signals.sum()
            for symbol, signal in signals.items():
                weight = signal / total_signal
                position_sizes[symbol] = available_cash * weight
        
        return position_sizes
    
    def _execute_buy_order(self, date: datetime, symbol: str, position_value: float, confidence: float) -> None:
        """Execute a buy order"""
        try:
            if symbol not in self.market_data or date not in self.market_data[symbol].index:
                return
            
            market_data = self.market_data[symbol].loc[date]
            price = market_data['close']
            
            # Apply slippage
            slippage = price * self.config.slippage_rate
            execution_price = price + slippage
            
            # Calculate quantity
            quantity = int(position_value / execution_price)
            if quantity <= 0:
                return
            
            # Calculate commission
            commission = position_value * self.config.commission_rate
            total_cost = quantity * execution_price + commission
            
            if total_cost > self.cash:
                return  # Insufficient funds
            
            # Update cash
            self.cash -= total_cost
            
            # Create or update position
            if symbol in self.positions:
                # Average down existing position
                existing_pos = self.positions[symbol]
                total_quantity = existing_pos.quantity + quantity
                avg_price = ((existing_pos.quantity * existing_pos.entry_price) + 
                           (quantity * execution_price)) / total_quantity
                
                existing_pos.quantity = total_quantity
                existing_pos.entry_price = avg_price
                existing_pos.current_price = execution_price
                existing_pos.market_value = total_quantity * execution_price
            else:
                # New position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=execution_price,
                    current_price=execution_price,
                    entry_date=date,
                    market_value=quantity * execution_price
                )
            
            # Record trade
            trade = Trade(
                symbol=symbol,
                side='buy',
                quantity=quantity,
                price=execution_price,
                timestamp=date,
                commission=commission,
                strategy_signal='buy',
                confidence=confidence
            )
            self.trades.append(trade)
            
        except Exception as e:
            self.logger.error(f"Error executing buy order for {symbol}: {e}")
    
    def _execute_sell_order(self, date: datetime, symbol: str, signal_strength: float) -> None:
        """Execute a sell order"""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            if symbol not in self.market_data or date not in self.market_data[symbol].index:
                return
            
            market_data = self.market_data[symbol].loc[date]
            price = market_data['close']
            
            # Apply slippage
            slippage = price * self.config.slippage_rate
            execution_price = price - slippage
            
            # Determine quantity to sell (partial or full)
            if signal_strength >= 1.0:
                quantity_to_sell = position.quantity  # Full exit
            else:
                quantity_to_sell = int(position.quantity * signal_strength)
            
            if quantity_to_sell <= 0:
                return
            
            # Calculate proceeds
            gross_proceeds = quantity_to_sell * execution_price
            commission = gross_proceeds * self.config.commission_rate
            net_proceeds = gross_proceeds - commission
            
            # Update cash
            self.cash += net_proceeds
            
            # Calculate realized P&L
            realized_pnl = (execution_price - position.entry_price) * quantity_to_sell
            
            # Update position
            if quantity_to_sell >= position.quantity:
                # Full exit
                del self.positions[symbol]
            else:
                # Partial exit
                position.quantity -= quantity_to_sell
                position.market_value = position.quantity * execution_price
                position.realized_pnl += realized_pnl
            
            # Record trade
            trade = Trade(
                symbol=symbol,
                side='sell',
                quantity=quantity_to_sell,
                price=execution_price,
                timestamp=date,
                commission=commission,
                strategy_signal='sell',
                confidence=signal_strength
            )
            self.trades.append(trade)
            
        except Exception as e:
            self.logger.error(f"Error executing sell order for {symbol}: {e}")
    
    def _update_portfolio_history(self, date: datetime) -> None:
        """Update portfolio history for the day"""
        total_value = self.cash
        positions_value = 0
        
        for position in self.positions.values():
            positions_value += position.market_value
            total_value += position.market_value
        
        portfolio_record = {
            'date': date,
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'num_positions': len(self.positions),
            'positions': {symbol: {
                'quantity': pos.quantity,
                'price': pos.current_price,
                'value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl
            } for symbol, pos in self.positions.items()}
        }
        
        self.portfolio_history.append(portfolio_record)
    
    def _calculate_backtest_results(self) -> Dict[str, Any]:
        """Calculate comprehensive backtesting results"""
        try:
            # Convert portfolio history to DataFrame
            portfolio_df = pd.DataFrame(self.portfolio_history)
            portfolio_df.set_index('date', inplace=True)
            
            # Calculate returns
            portfolio_df['returns'] = portfolio_df['total_value'].pct_change()
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(
                portfolio_df, self.benchmark_data, self.config.initial_capital
            )
            
            # Generate trade analysis
            trade_analysis = self._analyze_trades()
            
            # Generate report
            report = self.report_generator.generate_report(
                portfolio_df, metrics, trade_analysis, self.config
            )
            
            return {
                'portfolio_history': portfolio_df,
                'trades': self.trades,
                'metrics': metrics,
                'trade_analysis': trade_analysis,
                'report': report,
                'final_value': portfolio_df['total_value'].iloc[-1] if len(portfolio_df) > 0 else self.config.initial_capital,
                'total_return': (portfolio_df['total_value'].iloc[-1] / self.config.initial_capital - 1) if len(portfolio_df) > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating backtest results: {e}")
            raise
    
    def _analyze_trades(self) -> Dict[str, Any]:
        """Analyze trading performance"""
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame([{
            'symbol': trade.symbol,
            'side': trade.side,
            'quantity': trade.quantity,
            'price': trade.price,
            'timestamp': trade.timestamp,
            'commission': trade.commission,
            'confidence': trade.confidence
        } for trade in self.trades])
        
        # Group trades by symbol to calculate P&L
        trade_pairs = []
        position_tracker = {}
        
        for _, trade in trades_df.iterrows():
            symbol = trade['symbol']
            
            if symbol not in position_tracker:
                position_tracker[symbol] = {'quantity': 0, 'avg_price': 0}
            
            if trade['side'] == 'buy':
                old_qty = position_tracker[symbol]['quantity']
                old_price = position_tracker[symbol]['avg_price']
                new_qty = trade['quantity']
                new_price = trade['price']
                
                total_qty = old_qty + new_qty
                if total_qty > 0:
                    avg_price = ((old_qty * old_price) + (new_qty * new_price)) / total_qty
                    position_tracker[symbol] = {'quantity': total_qty, 'avg_price': avg_price}
            
            elif trade['side'] == 'sell' and position_tracker[symbol]['quantity'] > 0:
                entry_price = position_tracker[symbol]['avg_price']
                exit_price = trade['price']
                quantity = min(trade['quantity'], position_tracker[symbol]['quantity'])
                
                pnl = (exit_price - entry_price) * quantity - trade['commission']
                
                trade_pairs.append({
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': quantity,
                    'pnl': pnl,
                    'return_pct': (exit_price - entry_price) / entry_price,
                    'timestamp': trade['timestamp']
                })
                
                position_tracker[symbol]['quantity'] -= quantity
        
        if not trade_pairs:
            return {}
        
        trade_pairs_df = pd.DataFrame(trade_pairs)
        
        return {
            'total_trades': len(trade_pairs),
            'winning_trades': len(trade_pairs_df[trade_pairs_df['pnl'] > 0]),
            'losing_trades': len(trade_pairs_df[trade_pairs_df['pnl'] < 0]),
            'win_rate': len(trade_pairs_df[trade_pairs_df['pnl'] > 0]) / len(trade_pairs_df),
            'avg_win': trade_pairs_df[trade_pairs_df['pnl'] > 0]['pnl'].mean() if len(trade_pairs_df[trade_pairs_df['pnl'] > 0]) > 0 else 0,
            'avg_loss': trade_pairs_df[trade_pairs_df['pnl'] < 0]['pnl'].mean() if len(trade_pairs_df[trade_pairs_df['pnl'] < 0]) > 0 else 0,
            'largest_win': trade_pairs_df['pnl'].max(),
            'largest_loss': trade_pairs_df['pnl'].min(),
            'profit_factor': abs(trade_pairs_df[trade_pairs_df['pnl'] > 0]['pnl'].sum() / trade_pairs_df[trade_pairs_df['pnl'] < 0]['pnl'].sum()) if trade_pairs_df[trade_pairs_df['pnl'] < 0]['pnl'].sum() != 0 else float('inf'),
            'total_pnl': trade_pairs_df['pnl'].sum()
        }

class PortfolioOptimizer:
    """Portfolio optimization for backtesting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def optimize_weights(self, returns: pd.DataFrame, method: str = "mean_variance") -> Dict[str, float]:
        """Optimize portfolio weights"""
        try:
            if method == "mean_variance":
                return self._mean_variance_optimization(returns)
            elif method == "risk_parity":
                return self._risk_parity_optimization(returns)
            elif method == "equal_weight":
                return self._equal_weight_optimization(returns)
            else:
                self.logger.warning(f"Unknown optimization method: {method}")
                return self._equal_weight_optimization(returns)
                
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            return self._equal_weight_optimization(returns)
    
    def _mean_variance_optimization(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Mean-variance optimization (Markowitz)"""
        import scipy.optimize as sco
        
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        num_assets = len(mean_returns)
        
        # Objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.sum(weights * mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            return -portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess
        x0 = np.array([1/num_assets] * num_assets)
        
        # Optimization
        result = sco.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return dict(zip(returns.columns, result.x))
        else:
            return self._equal_weight_optimization(returns)
    
    def _risk_parity_optimization(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Risk parity optimization"""
        cov_matrix = returns.cov()
        inv_vol = 1 / np.sqrt(np.diag(cov_matrix))
        weights = inv_vol / np.sum(inv_vol)
        
        return dict(zip(returns.columns, weights))
    
    def _equal_weight_optimization(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Equal weight optimization"""
        num_assets = len(returns.columns)
        weight = 1.0 / num_assets
        
        return {col: weight for col in returns.columns}
