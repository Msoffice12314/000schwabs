"""
Trading Strategy Engine - AI-Driven Trading Strategy Implementation
Combines BiConNet predictions with risk management for automated trading
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from enum import Enum
from dataclasses import dataclass
import json

from config.settings import settings
from models.biconnet_core import BiConNet, create_biconnet_model
from models.signal_detector import SignalDetector
from models.market_predictor import MarketPredictor
from schwab_api.market_data import market_data_client, get_quotes, is_market_open
from schwab_api.trading_client import trading_client
from trading.risk_manager import risk_manager
from trading.portfolio_manager import portfolio_manager
from data.market_regime import MarketRegimeDetector
from utils.logger import get_logger

logger = get_logger(__name__)

class SignalType(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

class StrategyMode(Enum):
    """Strategy execution modes"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    SCALPING = "scalping"

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    signal_type: SignalType
    confidence: float
    price_target: float
    stop_loss: float
    take_profit: float
    position_size: float
    timestamp: datetime
    reasoning: str
    technical_indicators: Dict[str, float]
    ai_prediction: Dict[str, Any]
    risk_score: float

@dataclass
class Position:
    """Trading position data structure"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    entry_time: datetime
    position_type: str  # 'long' or 'short'
    stop_loss: float
    take_profit: float
    unrealized_pnl: float
    realized_pnl: float = 0.0

class StrategyEngine:
    """Main trading strategy engine"""
    
    def __init__(self, mode: StrategyMode = StrategyMode.MODERATE):
        self.mode = mode
        self.running = False
        self.positions: Dict[str, Position] = {}
        self.active_orders: Dict[str, Dict] = {}
        
        # AI Components
        self.signal_detector = SignalDetector()
        self.market_predictor = MarketPredictor()
        self.market_regime_detector = MarketRegimeDetector()
        
        # Trading parameters based on mode
        self.strategy_params = self._get_strategy_parameters()
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0
        }
        
        # Watchlist of symbols to trade
        self.watchlist = self._get_default_watchlist()
        
        logger.info(f"Strategy Engine initialized in {mode.value} mode")
    
    def _get_strategy_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters based on mode"""
        base_params = {
            'min_confidence_threshold': settings.trading.min_confidence_score,
            'max_position_size': settings.trading.max_position_size,
            'stop_loss_pct': settings.trading.stop_loss_pct,
            'take_profit_pct': settings.trading.take_profit_pct,
            'max_daily_trades': 10,
            'max_positions': 5,
            'rebalance_frequency': 'daily'
        }
        
        if self.mode == StrategyMode.CONSERVATIVE:
            base_params.update({
                'min_confidence_threshold': 0.8,
                'max_position_size': 0.05,  # 5% max position
                'stop_loss_pct': 0.015,     # 1.5% stop loss
                'take_profit_pct': 0.03,    # 3% take profit
                'max_daily_trades': 3,
                'max_positions': 3
            })
        elif self.mode == StrategyMode.AGGRESSIVE:
            base_params.update({
                'min_confidence_threshold': 0.6,
                'max_position_size': 0.15,  # 15% max position
                'stop_loss_pct': 0.03,      # 3% stop loss
                'take_profit_pct': 0.06,    # 6% take profit
                'max_daily_trades': 20,
                'max_positions': 8
            })
        elif self.mode == StrategyMode.SCALPING:
            base_params.update({
                'min_confidence_threshold': 0.65,
                'max_position_size': 0.08,  # 8% max position
                'stop_loss_pct': 0.005,     # 0.5% stop loss
                'take_profit_pct': 0.01,    # 1% take profit
                'max_daily_trades': 50,
                'max_positions': 10,
                'rebalance_frequency': 'hourly'
            })
        
        return base_params
    
    def _get_default_watchlist(self) -> List[str]:
        """Get default watchlist based on strategy mode"""
        if self.mode == StrategyMode.CONSERVATIVE:
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        elif self.mode == StrategyMode.AGGRESSIVE:
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
                   'AMD', 'CRM', 'ADBE', 'PYPL', 'SHOP', 'SQ', 'ROKU', 'ZOOM']
        elif self.mode == StrategyMode.SCALPING:
            return ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'SQQQ', 'TQQQ']
        else:  # MODERATE
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
                   'JPM', 'V', 'JNJ', 'PG']
    
    async def start_trading(self):
        """Start the trading strategy engine"""
        logger.info("Starting trading strategy engine")
        self.running = True
        
        try:
            # Main trading loop
            while self.running:
                if is_market_open():
                    await self._trading_loop()
                    await asyncio.sleep(60)  # Check every minute during market hours
                else:
                    await self._after_hours_tasks()
                    await asyncio.sleep(300)  # Check every 5 minutes after hours
                    
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            await self._cleanup()
    
    async def _trading_loop(self):
        """Main trading logic loop"""
        try:
            # Update market regime
            current_regime = await self.market_regime_detector.detect_regime()
            logger.debug(f"Current market regime: {current_regime}")
            
            # Get current positions and portfolio state
            await self._update_positions()
            
            # Risk management check
            risk_check = await risk_manager.pre_trade_risk_check()
            if not risk_check['approved']:
                logger.warning(f"Trading halted due to risk: {risk_check['reason']}")
                return
            
            # Generate signals for watchlist
            signals = await self._generate_signals()
            
            # Execute trades based on signals
            await self._execute_signals(signals)
            
            # Update stop losses and take profits
            await self._update_stop_losses()
            
            # Portfolio rebalancing if needed
            if self._should_rebalance():
                await self._rebalance_portfolio()
            
            # Update performance metrics
            await self._update_performance_metrics()
            
        except Exception as e:
            logger.error(f"Error in trading loop iteration: {e}")
    
    async def _generate_signals(self) -> List[TradingSignal]:
        """Generate trading signals for watchlist symbols"""
        signals = []
        
        try:
            # Get current quotes for all watchlist symbols
            quotes = get_quotes(self.watchlist)
            
            for symbol in self.watchlist:
                if symbol not in quotes:
                    continue
                    
                try:
                    # Get AI prediction
                    prediction = await self.market_predictor.predict_async(symbol)
                    
                    # Get technical signals
                    technical_signal = await self.signal_detector.get_signals_async(symbol)
                    
                    # Combine predictions to generate trading signal
                    signal = await self._create_trading_signal(
                        symbol, quotes[symbol], prediction, technical_signal
                    )
                    
                    if signal and signal.confidence >= self.strategy_params['min_confidence_threshold']:
                        signals.append(signal)
                        
                except Exception as e:
                    logger.error(f"Error generating signal for {symbol}: {e}")
            
            # Sort signals by confidence
            signals.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.info(f"Generated {len(signals)} trading signals")
            return signals[:self.strategy_params['max_daily_trades']]
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    async def _create_trading_signal(self, symbol: str, quote: Any, 
                                   prediction: Dict, technical_signal: Dict) -> Optional[TradingSignal]:
        """Create a trading signal from AI predictions and technical analysis"""
        try:
            current_price = quote.last_price
            
            # Extract prediction data
            predicted_price = prediction.get('predicted_price', current_price)
            confidence = prediction.get('confidence', 0.0)
            direction = prediction.get('direction', 'hold')
            
            # Extract technical indicators
            tech_indicators = technical_signal.get('indicators', {})
            tech_signal = technical_signal.get('signal', 'hold')
            tech_confidence = technical_signal.get('confidence', 0.0)
            
            # Combine AI and technical analysis
            combined_confidence = (confidence * 0.7) + (tech_confidence * 0.3)
            
            # Determine signal type
            price_change_pct = (predicted_price - current_price) / current_price
            
            if direction == 'buy' and tech_signal in ['buy', 'strong_buy'] and price_change_pct > 0.02:
                signal_type = SignalType.STRONG_BUY if combined_confidence > 0.8 else SignalType.BUY
            elif direction == 'sell' and tech_signal in ['sell', 'strong_sell'] and price_change_pct < -0.02:
                signal_type = SignalType.STRONG_SELL if combined_confidence > 0.8 else SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            # Skip if signal is HOLD or confidence too low
            if signal_type == SignalType.HOLD or combined_confidence < 0.5:
                return None
            
            # Calculate position size based on confidence and volatility
            volatility = tech_indicators.get('volatility', 0.02)
            base_position_size = self.strategy_params['max_position_size']
            confidence_multiplier = min(combined_confidence * 1.2, 1.0)
            volatility_adjustment = max(0.5, 1 - (volatility * 10))  # Reduce size for high volatility
            
            position_size = base_position_size * confidence_multiplier * volatility_adjustment
            
            # Calculate stop loss and take profit
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                stop_loss = current_price * (1 - self.strategy_params['stop_loss_pct'])
                take_profit = current_price * (1 + self.strategy_params['take_profit_pct'])
                price_target = predicted_price
            else:  # SELL signals
                stop_loss = current_price * (1 + self.strategy_params['stop_loss_pct'])
                take_profit = current_price * (1 - self.strategy_params['take_profit_pct'])
                price_target = predicted_price
            
            # Calculate risk score
            risk_score = await risk_manager.calculate_position_risk(
                symbol, position_size, current_price, stop_loss
            )
            
            # Create reasoning string
            reasoning = f"AI Confidence: {confidence:.2f}, Tech Signal: {tech_signal}, "
            reasoning += f"Price Target: ${price_target:.2f}, Current: ${current_price:.2f}"
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=combined_confidence,
                price_target=price_target,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                timestamp=datetime.now(),
                reasoning=reasoning,
                technical_indicators=tech_indicators,
                ai_prediction=prediction,
                risk_score=risk_score
            )
            
        except Exception as e:
            logger.error(f"Error creating trading signal for {symbol}: {e}")
            return None
    
    async def _execute_signals(self, signals: List[TradingSignal]):
        """Execute trading signals"""
        executed_trades = 0
        
        for signal in signals:
            if executed_trades >= self.strategy_params['max_daily_trades']:
                break
            
            # Skip if we already have a position in this symbol
            if signal.symbol in self.positions:
                continue
            
            # Skip if too many open positions
            if len(self.positions) >= self.strategy_params['max_positions']:
                break
            
            # Risk check for individual position
            if signal.risk_score > 0.7:  # High risk threshold
                logger.warning(f"Skipping {signal.symbol} due to high risk score: {signal.risk_score}")
                continue
            
            try:
                # Execute the trade
                success = await self._execute_trade(signal)
                if success:
                    executed_trades += 1
                    logger.info(f"Executed {signal.signal_type.value} signal for {signal.symbol}")
                
            except Exception as e:
                logger.error(f"Failed to execute signal for {signal.symbol}: {e}")
        
        logger.info(f"Executed {executed_trades} trades out of {len(signals)} signals")
    
    async def _execute_trade(self, signal: TradingSignal) -> bool:
        """Execute individual trade based on signal"""
        try:
            # Calculate shares to buy/sell
            portfolio_value = await portfolio_manager.get_total_value()
            position_value = portfolio_value * signal.position_size
            current_price = (await get_quotes([signal.symbol]))[signal.symbol].last_price
            shares = int(position_value / current_price)
            
            if shares < 1:
                logger.warning(f"Position size too small for {signal.symbol}: {shares} shares")
                return False
            
            # Determine order side
            side = 'buy' if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else 'sell'
            
            # Place market order
            order = await trading_client.place_order(
                symbol=signal.symbol,
                quantity=shares,
                side=side,
                order_type='market'
            )
            
            if order and order.get('status') == 'filled':
                # Create position record
                position = Position(
                    symbol=signal.symbol,
                    quantity=shares if side == 'buy' else -shares,
                    entry_price=current_price,
                    current_price=current_price,
                    entry_time=datetime.now(),
                    position_type='long' if side == 'buy' else 'short',
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    unrealized_pnl=0.0
                )
                
                self.positions[signal.symbol] = position
                
                # Place stop loss and take profit orders
                await self._place_exit_orders(signal.symbol, position)
                
                # Update performance tracking
                self.performance_metrics['total_trades'] += 1
                
                return True
            else:
                logger.error(f"Order failed for {signal.symbol}: {order}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade for {signal.symbol}: {e}")
            return False
    
    async def _place_exit_orders(self, symbol: str, position: Position):
        """Place stop loss and take profit orders"""
        try:
            # Stop loss order
            stop_loss_side = 'sell' if position.quantity > 0 else 'buy'
            stop_order = await trading_client.place_order(
                symbol=symbol,
                quantity=abs(position.quantity),
                side=stop_loss_side,
                order_type='stop_loss',
                stop_price=position.stop_loss
            )
            
            # Take profit order
            take_profit_side = 'sell' if position.quantity > 0 else 'buy'
            take_profit_order = await trading_client.place_order(
                symbol=symbol,
                quantity=abs(position.quantity),
                side=take_profit_side,
                order_type='limit',
                limit_price=position.take_profit
            )
            
            # Store order IDs for tracking
            self.active_orders[symbol] = {
                'stop_loss': stop_order.get('order_id') if stop_order else None,
                'take_profit': take_profit_order.get('order_id') if take_profit_order else None
            }
            
        except Exception as e:
            logger.error(f"Error placing exit orders for {symbol}: {e}")
    
    async def _update_positions(self):
        """Update current positions with latest prices"""
        if not self.positions:
            return
        
        try:
            symbols = list(self.positions.keys())
            quotes = get_quotes(symbols)
            
            for symbol, position in self.positions.items():
                if symbol in quotes:
                    current_price = quotes[symbol].last_price
                    position.current_price = current_price
                    
                    # Calculate unrealized P&L
                    if position.quantity > 0:  # Long position
                        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    else:  # Short position
                        position.unrealized_pnl = (position.entry_price - current_price) * abs(position.quantity)
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    async def _update_stop_losses(self):
        """Update stop losses for open positions (trailing stops)"""
        for symbol, position in self.positions.items():
            try:
                current_price = position.current_price
                
                # Trailing stop logic for profitable positions
                if position.quantity > 0 and current_price > position.entry_price:  # Long position in profit
                    new_stop = current_price * (1 - self.strategy_params['stop_loss_pct'])
                    if new_stop > position.stop_loss:
                        position.stop_loss = new_stop
                        # Update stop loss order
                        await self._update_stop_loss_order(symbol, new_stop)
                        
                elif position.quantity < 0 and current_price < position.entry_price:  # Short position in profit
                    new_stop = current_price * (1 + self.strategy_params['stop_loss_pct'])
                    if new_stop < position.stop_loss:
                        position.stop_loss = new_stop
                        # Update stop loss order
                        await self._update_stop_loss_order(symbol, new_stop)
                
            except Exception as e:
                logger.error(f"Error updating stop loss for {symbol}: {e}")
    
    async def _update_stop_loss_order(self, symbol: str, new_stop_price: float):
        """Update stop loss order with new price"""
        try:
            if symbol in self.active_orders and self.active_orders[symbol].get('stop_loss'):
                # Cancel existing stop loss order
                await trading_client.cancel_order(self.active_orders[symbol]['stop_loss'])
                
                # Place new stop loss order
                position = self.positions[symbol]
                stop_loss_side = 'sell' if position.quantity > 0 else 'buy'
                
                new_order = await trading_client.place_order(
                    symbol=symbol,
                    quantity=abs(position.quantity),
                    side=stop_loss_side,
                    order_type='stop_loss',
                    stop_price=new_stop_price
                )
                
                if new_order:
                    self.active_orders[symbol]['stop_loss'] = new_order.get('order_id')
                
        except Exception as e:
            logger.error(f"Error updating stop loss order for {symbol}: {e}")
    
    def _should_rebalance(self) -> bool:
        """Check if portfolio rebalancing is needed"""
        frequency = self.strategy_params['rebalance_frequency']
        
        if frequency == 'hourly':
            return datetime.now().minute == 0
        elif frequency == 'daily':
            return datetime.now().hour == 9 and datetime.now().minute == 30  # Market open
        
        return False
    
    async def _rebalance_portfolio(self):
        """Rebalance portfolio based on current signals and risk"""
        logger.info("Starting portfolio rebalancing")
        
        try:
            # Get current portfolio composition
            portfolio_summary = await portfolio_manager.get_portfolio_summary()
            
            # Check for positions that should be closed
            for symbol in list(self.positions.keys()):
                position = self.positions[symbol]
                
                # Close position if holding too long (based on strategy mode)
                holding_time = datetime.now() - position.entry_time
                max_holding_time = timedelta(days=7) if self.mode != StrategyMode.SCALPING else timedelta(hours=4)
                
                if holding_time > max_holding_time:
                    await self._close_position(symbol, "Max holding time reached")
                
                # Close position if unrealized loss is too high
                if position.unrealized_pnl < -1000:  # $1000 loss threshold
                    await self._close_position(symbol, "Stop loss triggered")
            
        except Exception as e:
            logger.error(f"Error during portfolio rebalancing: {e}")
    
    async def _close_position(self, symbol: str, reason: str):
        """Close a position"""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            side = 'sell' if position.quantity > 0 else 'buy'
            
            # Place market order to close position
            order = await trading_client.place_order(
                symbol=symbol,
                quantity=abs(position.quantity),
                side=side,
                order_type='market'
            )
            
            if order and order.get('status') == 'filled':
                # Update performance metrics
                if position.unrealized_pnl > 0:
                    self.performance_metrics['winning_trades'] += 1
                else:
                    self.performance_metrics['losing_trades'] += 1
                
                self.performance_metrics['total_pnl'] += position.unrealized_pnl
                
                # Remove position
                del self.positions[symbol]
                
                # Cancel any active orders
                if symbol in self.active_orders:
                    for order_id in self.active_orders[symbol].values():
                        if order_id:
                            await trading_client.cancel_order(order_id)
                    del self.active_orders[symbol]
                
                logger.info(f"Closed position for {symbol}: {reason}, P&L: ${position.unrealized_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
    
    async def _update_performance_metrics(self):
        """Update strategy performance metrics"""
        try:
            total_trades = self.performance_metrics['total_trades']
            if total_trades > 0:
                win_rate = self.performance_metrics['winning_trades'] / total_trades
                self.performance_metrics['win_rate'] = win_rate
            
            # Calculate Sharpe ratio (simplified)
            if len(self.positions) > 0:
                returns = [pos.unrealized_pnl for pos in self.positions.values()]
                if len(returns) > 1:
                    avg_return = np.mean(returns)
                    return_std = np.std(returns)
                    if return_std > 0:
                        self.performance_metrics['sharpe_ratio'] = avg_return / return_std
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _after_hours_tasks(self):
        """Tasks to perform when market is closed"""
        try:
            # Update any pending orders
            await self._check_pending_orders()
            
            # Prepare for next trading day
            await self._prepare_next_session()
            
            # Log daily performance
            if datetime.now().hour == 17:  # 5 PM
                await self._log_daily_performance()
            
        except Exception as e:
            logger.error(f"Error in after hours tasks: {e}")
    
    async def _check_pending_orders(self):
        """Check status of pending orders"""
        try:
            for symbol, orders in self.active_orders.items():
                for order_type, order_id in orders.items():
                    if order_id:
                        order_status = await trading_client.get_order_status(order_id)
                        if order_status.get('status') == 'filled':
                            logger.info(f"{order_type} order filled for {symbol}")
                            
                            # Handle position closure if exit order filled
                            if order_type in ['stop_loss', 'take_profit'] and symbol in self.positions:
                                position = self.positions[symbol]
                                self.performance_metrics['total_pnl'] += position.unrealized_pnl
                                
                                if position.unrealized_pnl > 0:
                                    self.performance_metrics['winning_trades'] += 1
                                else:
                                    self.performance_metrics['losing_trades'] += 1
                                
                                del self.positions[symbol]
                                del self.active_orders[symbol]
        
        except Exception as e:
            logger.error(f"Error checking pending orders: {e}")
    
    async def _prepare_next_session(self):
        """Prepare for next trading session"""
        # Update watchlist based on performance
        # Retrain models if needed
        # Adjust strategy parameters
        pass
    
    async def _log_daily_performance(self):
        """Log daily performance summary"""
        try:
            total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_pnl += self.performance_metrics['total_pnl']
            
            logger.info("=== Daily Performance Summary ===")
            logger.info(f"Total P&L: ${total_pnl:.2f}")
            logger.info(f"Total Trades: {self.performance_metrics['total_trades']}")
            logger.info(f"Win Rate: {self.performance_metrics['win_rate']:.2%}")
            logger.info(f"Active Positions: {len(self.positions)}")
            logger.info(f"Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.2f}")
            
        except Exception as e:
            logger.error(f"Error logging daily performance: {e}")
    
    async def _cleanup(self):
        """Cleanup resources when stopping"""
        logger.info("Cleaning up strategy engine")
        self.running = False
        
        # Close all positions if configured to do so
        # Cancel all pending orders
        # Save state for restart
    
    def stop(self):
        """Stop the strategy engine"""
        logger.info("Stopping strategy engine")
        self.running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current strategy status"""
        return {
            'running': self.running,
            'mode': self.mode.value,
            'active_positions': len(self.positions),
            'active_orders': len(self.active_orders),
            'performance': self.performance_metrics.copy(),
            'watchlist': self.watchlist.copy(),
            'strategy_params': self.strategy_params.copy()
        }

# Global strategy engine instance
strategy_engine = StrategyEngine()

# Export key classes and functions
__all__ = [
    'StrategyEngine',
    'TradingSignal',
    'Position',
    'SignalType',
    'StrategyMode',
    'strategy_engine'
]