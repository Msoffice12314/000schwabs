from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, Index, UniqueConstraint, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

Base = declarative_base()

class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

class User(Base, TimestampMixin):
    """User accounts and authentication"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(50))
    last_name = Column(String(50))
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    last_login = Column(DateTime)
    login_count = Column(Integer, default=0)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    password_changed_at = Column(DateTime, default=datetime.utcnow)
    settings = Column(JSON, default=dict)
    
    # Relationships
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="user")
    alerts = relationship("Alert", back_populates="user")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'is_active': self.is_active,
            'is_admin': self.is_admin,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat(),
            'settings': self.settings
        }

class APIKey(Base, TimestampMixin):
    """API keys for external service integration"""
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    service_name = Column(String(50), nullable=False)  # 'schwab', 'alpaca', etc.
    key_name = Column(String(100), nullable=False)
    encrypted_key = Column(Text, nullable=False)
    encrypted_secret = Column(Text)
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime)
    last_used = Column(DateTime)
    usage_count = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    __table_args__ = (
        UniqueConstraint('user_id', 'service_name', 'key_name'),
        Index('idx_api_keys_user_service', 'user_id', 'service_name')
    )

class Portfolio(Base, TimestampMixin):
    """User portfolios"""
    __tablename__ = 'portfolios'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    initial_cash = Column(Float, nullable=False, default=100000.0)
    current_cash = Column(Float, nullable=False, default=100000.0)
    total_value = Column(Float, nullable=False, default=100000.0)
    is_active = Column(Boolean, default=True)
    is_paper_trading = Column(Boolean, default=True)
    risk_tolerance = Column(String(20), default='moderate')  # conservative, moderate, aggressive
    max_position_size = Column(Float, default=0.1)  # 10% max per position
    stop_loss_percent = Column(Float, default=0.05)  # 5% stop loss
    settings = Column(JSON, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="portfolio")
    performance_snapshots = relationship("PerformanceSnapshot", back_populates="portfolio", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_portfolios_user_active', 'user_id', 'is_active'),
        CheckConstraint('initial_cash > 0'),
        CheckConstraint('current_cash >= 0'),
        CheckConstraint('max_position_size > 0 AND max_position_size <= 1'),
        CheckConstraint('stop_loss_percent > 0 AND stop_loss_percent <= 1')
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'initial_cash': self.initial_cash,
            'current_cash': self.current_cash,
            'total_value': self.total_value,
            'is_active': self.is_active,
            'is_paper_trading': self.is_paper_trading,
            'risk_tolerance': self.risk_tolerance,
            'created_at': self.created_at.isoformat(),
            'settings': self.settings
        }

class Symbol(Base, TimestampMixin):
    """Stock symbols and metadata"""
    __tablename__ = 'symbols'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    company_name = Column(String(200))
    sector = Column(String(50))
    industry = Column(String(100))
    market_cap = Column(Float)
    exchange = Column(String(20))
    currency = Column(String(3), default='USD')
    is_active = Column(Boolean, default=True)
    is_tradeable = Column(Boolean, default=True)
    last_price = Column(Float)
    price_updated_at = Column(DateTime)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    positions = relationship("Position", back_populates="symbol")
    trades = relationship("Trade", back_populates="symbol")
    market_data = relationship("MarketData", back_populates="symbol", cascade="all, delete-orphan")
    ai_predictions = relationship("AIPrediction", back_populates="symbol", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_symbols_sector', 'sector'),
        Index('idx_symbols_active_tradeable', 'is_active', 'is_tradeable')
    )

class Position(Base, TimestampMixin):
    """Current portfolio positions"""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'), nullable=False)
    symbol_id = Column(Integer, ForeignKey('symbols.id'), nullable=False)
    quantity = Column(Integer, nullable=False)
    average_cost = Column(Float, nullable=False)
    current_price = Column(Float)
    market_value = Column(Float)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    position_type = Column(String(10), default='LONG')  # LONG, SHORT
    entry_date = Column(DateTime, default=datetime.utcnow)
    stop_loss_price = Column(Float)
    take_profit_price = Column(Float)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")
    symbol = relationship("Symbol", back_populates="positions")
    
    __table_args__ = (
        UniqueConstraint('portfolio_id', 'symbol_id'),
        Index('idx_positions_portfolio_active', 'portfolio_id', 'is_active'),
        CheckConstraint('quantity != 0')
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol.symbol if self.symbol else None,
            'quantity': self.quantity,
            'average_cost': self.average_cost,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'position_type': self.position_type,
            'entry_date': self.entry_date.isoformat(),
            'is_active': self.is_active
        }

class Trade(Base, TimestampMixin):
    """Trade execution records"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'), nullable=False)
    symbol_id = Column(Integer, ForeignKey('symbols.id'), nullable=False)
    order_id = Column(String(50), unique=True, index=True)
    trade_type = Column(String(10), nullable=False)  # BUY, SELL
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    fees = Column(Float, default=0.0)
    total_amount = Column(Float, nullable=False)
    strategy_name = Column(String(50))
    signal_confidence = Column(Float)
    execution_time = Column(DateTime, default=datetime.utcnow)
    order_status = Column(String(20), default='FILLED')  # PENDING, FILLED, CANCELLED, REJECTED
    market_conditions = Column(JSON, default=dict)
    notes = Column(Text)
    
    # Relationships
    user = relationship("User", back_populates="trades")
    portfolio = relationship("Portfolio", back_populates="trades")
    symbol = relationship("Symbol", back_populates="trades")
    
    __table_args__ = (
        Index('idx_trades_portfolio_symbol', 'portfolio_id', 'symbol_id'),
        Index('idx_trades_execution_time', 'execution_time'),
        Index('idx_trades_strategy', 'strategy_name'),
        CheckConstraint('quantity > 0'),
        CheckConstraint('price > 0'),
        CheckConstraint('total_amount > 0')
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'order_id': self.order_id,
            'symbol': self.symbol.symbol if self.symbol else None,
            'trade_type': self.trade_type,
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'fees': self.fees,
            'total_amount': self.total_amount,
            'strategy_name': self.strategy_name,
            'signal_confidence': self.signal_confidence,
            'execution_time': self.execution_time.isoformat(),
            'order_status': self.order_status,
            'notes': self.notes
        }

class MarketData(Base, TimestampMixin):
    """Historical and real-time market data"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)  # 1m, 5m, 1h, 1d, etc.
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, default=0)
    vwap = Column(Float)  # Volume Weighted Average Price
    trade_count = Column(Integer)
    
    # Relationships
    symbol = relationship("Symbol", back_populates="market_data")
    
    __table_args__ = (
        UniqueConstraint('symbol_id', 'timestamp', 'timeframe'),
        Index('idx_market_data_symbol_timeframe', 'symbol_id', 'timeframe', 'timestamp'),
        CheckConstraint('open_price > 0'),
        CheckConstraint('high_price > 0'),
        CheckConstraint('low_price > 0'), 
        CheckConstraint('close_price > 0'),
        CheckConstraint('high_price >= low_price')
    )

class AIPrediction(Base, TimestampMixin):
    """AI model predictions"""
    __tablename__ = 'ai_predictions'
    
    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'), nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20))
    prediction_type = Column(String(50), nullable=False)  # price, direction, volatility, etc.
    prediction_horizon = Column(String(20))  # 1h, 1d, 1w, 1m
    predicted_value = Column(Float)
    confidence_score = Column(Float)
    probability_up = Column(Float)
    probability_down = Column(Float) 
    target_price = Column(Float)
    support_level = Column(Float)
    resistance_level = Column(Float)
    prediction_features = Column(JSON, default=dict)
    actual_value = Column(Float)  # For backtesting
    accuracy_score = Column(Float)
    prediction_date = Column(DateTime, default=datetime.utcnow)
    expiry_date = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    symbol = relationship("Symbol", back_populates="ai_predictions")
    
    __table_args__ = (
        Index('idx_ai_predictions_symbol_model', 'symbol_id', 'model_name'),
        Index('idx_ai_predictions_date_active', 'prediction_date', 'is_active'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1'),
        CheckConstraint('probability_up >= 0 AND probability_up <= 1'),
        CheckConstraint('probability_down >= 0 AND probability_down <= 1')
    )

class PerformanceSnapshot(Base, TimestampMixin):
    """Portfolio performance snapshots"""
    __tablename__ = 'performance_snapshots'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'), nullable=False)
    snapshot_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    total_value = Column(Float, nullable=False)
    cash_value = Column(Float, nullable=False)
    positions_value = Column(Float, nullable=False)
    daily_return = Column(Float)
    cumulative_return = Column(Float)
    drawdown = Column(Float)
    volatility = Column(Float)
    sharpe_ratio = Column(Float)
    beta = Column(Float)
    alpha = Column(Float)
    max_drawdown = Column(Float)
    num_positions = Column(Integer, default=0)
    sector_allocation = Column(JSON, default=dict)
    top_holdings = Column(JSON, default=dict)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="performance_snapshots")
    
    __table_args__ = (
        Index('idx_performance_portfolio_date', 'portfolio_id', 'snapshot_date'),
        CheckConstraint('total_value >= 0'),
        CheckConstraint('cash_value >= 0'),
        CheckConstraint('positions_value >= 0')
    )

class Alert(Base, TimestampMixin):
    """User alerts and notifications"""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    alert_type = Column(String(50), nullable=False)  # price, portfolio, system, etc.
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    severity = Column(String(20), default='INFO')  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    symbol_id = Column(Integer, ForeignKey('symbols.id'))
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'))
    trigger_condition = Column(JSON)
    is_read = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    triggered_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="alerts")
    symbol = relationship("Symbol")
    portfolio = relationship("Portfolio")
    
    __table_args__ = (
        Index('idx_alerts_user_read', 'user_id', 'is_read'),
        Index('idx_alerts_severity_active', 'severity', 'is_active'),
        Index('idx_alerts_triggered_at', 'triggered_at')
    )

class Strategy(Base, TimestampMixin):
    """Trading strategies"""
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    strategy_type = Column(String(50))  # momentum, mean_reversion, ml_based, etc.
    version = Column(String(20), default='1.0')
    parameters = Column(JSON, default=dict)
    risk_parameters = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    is_paper_only = Column(Boolean, default=True)
    performance_metrics = Column(JSON, default=dict)
    backtest_results = Column(JSON, default=dict)
    last_used = Column(DateTime)
    usage_count = Column(Integer, default=0)
    
    # Relationships
    strategy_runs = relationship("StrategyRun", back_populates="strategy", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_strategies_type_active', 'strategy_type', 'is_active'),
    )

class StrategyRun(Base, TimestampMixin):
    """Strategy execution runs"""
    __tablename__ = 'strategy_runs'
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'), nullable=False)
    run_name = Column(String(100))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    status = Column(String(20), default='RUNNING')  # RUNNING, COMPLETED, FAILED, STOPPED
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    win_rate = Column(Float)
    error_message = Column(Text)
    execution_log = Column(Text)
    
    # Relationships
    strategy = relationship("Strategy", back_populates="strategy_runs")
    portfolio = relationship("Portfolio")
    
    __table_args__ = (
        Index('idx_strategy_runs_strategy_portfolio', 'strategy_id', 'portfolio_id'),
        Index('idx_strategy_runs_start_time', 'start_time'),
        Index('idx_strategy_runs_status', 'status')
    )

class SystemEvent(Base, TimestampMixin):
    """System events and audit log"""
    __tablename__ = 'system_events'
    
    id = Column(Integer, primary_key=True)
    event_type = Column(String(50), nullable=False)  # login, trade, error, etc.
    event_category = Column(String(50))  # security, trading, system, etc.
    user_id = Column(Integer, ForeignKey('users.id'))
    description = Column(Text, nullable=False)
    severity = Column(String(20), default='INFO')
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(String(500))
    session_id = Column(String(100))
    request_id = Column(String(100))
    additional_data = Column(JSON, default=dict)
    event_time = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User")
    
    __table_args__ = (
        Index('idx_system_events_type_time', 'event_type', 'event_time'),
        Index('idx_system_events_user_time', 'user_id', 'event_time'),
        Index('idx_system_events_severity', 'severity')
    )

class DataFeed(Base, TimestampMixin):
    """Data feed configuration and status"""
    __tablename__ = 'data_feeds'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    provider = Column(String(50), nullable=False)  # schwab, alpaca, yahoo, etc.
    feed_type = Column(String(50))  # real_time, historical, news, etc.
    endpoint_url = Column(String(500))
    is_active = Column(Boolean, default=True)
    is_connected = Column(Boolean, default=False)
    last_heartbeat = Column(DateTime)
    connection_errors = Column(Integer, default=0)
    total_messages = Column(Integer, default=0)
    messages_per_second = Column(Float, default=0.0)
    latency_ms = Column(Float)
    configuration = Column(JSON, default=dict)
    error_log = Column(Text)
    
    __table_args__ = (
        Index('idx_data_feeds_provider_active', 'provider', 'is_active'),
    )

class ModelTraining(Base, TimestampMixin):
    """ML model training records"""
    __tablename__ = 'model_training'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50))  # lstm, transformer, random_forest, etc.
    version = Column(String(20))
    training_start = Column(DateTime, default=datetime.utcnow)
    training_end = Column(DateTime)
    training_duration = Column(Float)  # seconds
    dataset_size = Column(Integer)
    num_features = Column(Integer)
    training_parameters = Column(JSON, default=dict)
    validation_score = Column(Float)
    test_score = Column(Float)
    cross_validation_scores = Column(JSON)
    feature_importance = Column(JSON)
    model_path = Column(String(500))
    training_log = Column(Text)
    status = Column(String(20), default='TRAINING')  # TRAINING, COMPLETED, FAILED
    error_message = Column(Text)
    
    __table_args__ = (
        Index('idx_model_training_name_version', 'model_name', 'version'),
        Index('idx_model_training_status', 'status'),
        Index('idx_model_training_start', 'training_start')
    )

class Configuration(Base, TimestampMixin):
    """System configuration settings"""
    __tablename__ = 'configurations'
    
    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(Text)
    value_type = Column(String(20), default='string')  # string, int, float, bool, json
    category = Column(String(50))
    description = Column(Text)
    is_sensitive = Column(Boolean, default=False)  # For passwords, API keys, etc.
    is_system = Column(Boolean, default=False)  # System vs user configurable
    validation_regex = Column(String(500))
    default_value = Column(Text)
    
    __table_args__ = (
        Index('idx_configurations_category', 'category'),
    )
    
    def get_typed_value(self):
        """Return value converted to appropriate type"""
        if self.value is None:
            return None
            
        if self.value_type == 'int':
            return int(self.value)
        elif self.value_type == 'float':
            return float(self.value)
        elif self.value_type == 'bool':
            return self.value.lower() in ('true', '1', 'yes')
        elif self.value_type == 'json':
            return json.loads(self.value)
        else:
            return self.value

# Create indexes for better query performance
def create_additional_indexes():
    """Create additional database indexes for performance"""
    pass

# Utility functions for model operations
def create_user(session, username: str, email: str, password_hash: str, **kwargs) -> User:
    """Create a new user"""
    user = User(
        username=username,
        email=email,
        password_hash=password_hash,
        **kwargs
    )
    session.add(user)
    session.commit()
    return user

def get_user_by_username(session, username: str) -> Optional[User]:
    """Get user by username"""
    return session.query(User).filter(User.username == username).first()

def get_user_portfolios(session, user_id: int, active_only: bool = True) -> List[Portfolio]:
    """Get user's portfolios"""
    query = session.query(Portfolio).filter(Portfolio.user_id == user_id)
    if active_only:
        query = query.filter(Portfolio.is_active == True)
    return query.all()

def get_portfolio_positions(session, portfolio_id: int, active_only: bool = True) -> List[Position]:
    """Get portfolio positions"""
    query = session.query(Position).filter(Portfolio.id == portfolio_id)
    if active_only:
        query = query.filter(Position.is_active == True)
    return query.all()

def create_trade(session, user_id: int, portfolio_id: int, symbol_id: int, 
                trade_type: str, quantity: int, price: float, **kwargs) -> Trade:
    """Create a new trade record"""
    trade = Trade(
        user_id=user_id,
        portfolio_id=portfolio_id,
        symbol_id=symbol_id,
        trade_type=trade_type,
        quantity=quantity,
        price=price,
        total_amount=quantity * price,
        **kwargs
    )
    session.add(trade)
    session.commit()
    return trade

def get_symbol_by_ticker(session, symbol: str) -> Optional[Symbol]:
    """Get symbol by ticker"""
    return session.query(Symbol).filter(Symbol.symbol == symbol.upper()).first()

def get_or_create_symbol(session, symbol: str, **kwargs) -> Symbol:
    """Get existing symbol or create new one"""
    existing = get_symbol_by_ticker(session, symbol)
    if existing:
        return existing
    
    new_symbol = Symbol(symbol=symbol.upper(), **kwargs)
    session.add(new_symbol)
    session.commit()
    return new_symbol

def log_system_event(session, event_type: str, description: str, 
                    user_id: Optional[int] = None, **kwargs):
    """Log system event"""
    event = SystemEvent(
        event_type=event_type,
        description=description,
        user_id=user_id,
        **kwargs
    )
    session.add(event)
    session.commit()

def get_configuration(session, key: str, default=None):
    """Get configuration value"""
    config = session.query(Configuration).filter(Configuration.key == key).first()
    if config:
        return config.get_typed_value()
    return default

def set_configuration(session, key: str, value: Any, value_type: str = 'string', 
                     category: str = None, description: str = None):
    """Set configuration value"""
    config = session.query(Configuration).filter(Configuration.key == key).first()
    
    if config:
        config.value = str(value)
        config.value_type = value_type
        config.updated_at = datetime.utcnow()
    else:
        config = Configuration(
            key=key,
            value=str(value),
            value_type=value_type,
            category=category,
            description=description
        )
        session.add(config)
    
    session.commit()

# Database migration utilities
def upgrade_database_schema(session):
    """Apply database schema upgrades"""
    # This would contain migration logic
    pass

def get_database_version(session) -> str:
    """Get current database schema version"""
    return get_configuration(session, 'database_version', '1.0')

def set_database_version(session, version: str):
    """Set database schema version"""
    set_configuration(session, 'database_version', version, 'string', 'system', 'Database schema version')
