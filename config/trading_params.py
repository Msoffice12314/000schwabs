"""
Trading strategy parameters and configuration
"""

from config.settings import settings
from trading.strategy_engine import StrategyMode

# Strategy Mode Configurations
STRATEGY_CONFIGS = {
    StrategyMode.CONSERVATIVE: {
        'min_confidence_threshold': 0.8,
        'max_position_size': 0.05,
        'stop_loss_pct': 0.015,
        'take_profit_pct': 0.03,
        'max_daily_trades': 3,
        'max_positions': 3,
        'rebalance_frequency': 'daily'
    },
    StrategyMode.MODERATE: {
        'min_confidence_threshold': 0.7,
        'max_position_size': 0.1,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04,
        'max_daily_trades': 10,
        'max_positions': 5,
        'rebalance_frequency': 'daily'
    },
    StrategyMode.AGGRESSIVE: {
        'min_confidence_threshold': 0.6,
        'max_position_size': 0.15,
        'stop_loss_pct': 0.03,
        'take_profit_pct': 0.06,
        'max_daily_trades': 20,
        'max_positions': 8,
        'rebalance_frequency': 'daily'
    },
    StrategyMode.SCALPING: {
        'min_confidence_threshold': 0.65,
        'max_position_size': 0.08,
        'stop_loss_pct': 0.005,
        'take_profit_pct': 0.01,
        'max_daily_trades': 50,
        'max_positions': 10,
        'rebalance_frequency': 'hourly'
    }
}

# Risk Management Parameters
RISK_PARAMS = {
    'max_portfolio_risk': settings.trading.max_portfolio_risk,
    'max_sector_allocation': 0.3,
    'max_correlation_threshold': 0.7,
    'volatility_lookback': 20,
    'var_confidence_level': 0.05,
    'max_drawdown_limit': 0.15
}

# Default Watchlists by Strategy Mode
DEFAULT_WATCHLISTS = {
    StrategyMode.CONSERVATIVE: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
    StrategyMode.MODERATE: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'JPM', 'V', 'JNJ', 'PG'],
    StrategyMode.AGGRESSIVE: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'CRM', 'ADBE', 'PYPL', 'SHOP', 'SQ', 'ROKU', 'ZOOM'],
    StrategyMode.SCALPING: ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'SQQQ', 'TQQQ']
}
