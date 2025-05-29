"""
Core configuration management for Schwab AI Trading System
Handles all application settings and environment configuration
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', '5432'))
    username: str = os.getenv('DB_USERNAME', 'postgres')
    password: str = os.getenv('DB_PASSWORD', '')
    database: str = os.getenv('DB_NAME', 'schwab_ai')
    pool_size: int = int(os.getenv('DB_POOL_SIZE', '10'))
    pool_timeout: int = int(os.getenv('DB_POOL_TIMEOUT', '30'))

@dataclass
class RedisConfig:
    """Redis configuration for caching and sessions"""
    host: str = os.getenv('REDIS_HOST', 'localhost')
    port: int = int(os.getenv('REDIS_PORT', '6379'))
    password: str = os.getenv('REDIS_PASSWORD', '')
    db: int = int(os.getenv('REDIS_DB', '0'))
    max_connections: int = int(os.getenv('REDIS_MAX_CONNECTIONS', '50'))

@dataclass
class SchwabAPIConfig:
    """Schwab API configuration settings"""
    base_url: str = "https://api.schwabapi.com"
    auth_url: str = "https://api.schwabapi.com/v1/oauth"
    client_id: str = os.getenv('SCHWAB_CLIENT_ID', '')
    client_secret: str = os.getenv('SCHWAB_CLIENT_SECRET', '')
    redirect_uri: str = os.getenv('SCHWAB_REDIRECT_URI', 'https://127.0.0.1:8182')
    token_path: str = os.getenv('SCHWAB_TOKEN_PATH', './data/schwab_token.json')
    rate_limit_per_second: int = int(os.getenv('SCHWAB_RATE_LIMIT', '120'))
    timeout: int = int(os.getenv('SCHWAB_TIMEOUT', '30'))

@dataclass
class TradingConfig:
    """Trading strategy configuration"""
    max_portfolio_risk: float = float(os.getenv('MAX_PORTFOLIO_RISK', '0.02'))
    max_position_size: float = float(os.getenv('MAX_POSITION_SIZE', '0.1'))
    stop_loss_pct: float = float(os.getenv('STOP_LOSS_PCT', '0.02'))
    take_profit_pct: float = float(os.getenv('TAKE_PROFIT_PCT', '0.04'))
    min_confidence_score: float = float(os.getenv('MIN_CONFIDENCE_SCORE', '0.7'))
    rebalance_frequency: str = os.getenv('REBALANCE_FREQUENCY', 'daily')
    trading_hours_start: str = os.getenv('TRADING_HOURS_START', '09:30')
    trading_hours_end: str = os.getenv('TRADING_HOURS_END', '16:00')
    timezone: str = os.getenv('TRADING_TIMEZONE', 'America/New_York')

@dataclass
class ModelConfig:
    """AI Model configuration settings"""
    sequence_length: int = int(os.getenv('MODEL_SEQUENCE_LENGTH', '60'))
    cnn_filters: int = int(os.getenv('MODEL_CNN_FILTERS', '64'))
    lstm_units: int = int(os.getenv('MODEL_LSTM_UNITS', '50'))
    dropout_rate: float = float(os.getenv('MODEL_DROPOUT_RATE', '0.2'))
    learning_rate: float = float(os.getenv('MODEL_LEARNING_RATE', '0.001'))
    batch_size: int = int(os.getenv('MODEL_BATCH_SIZE', '32'))
    epochs: int = int(os.getenv('MODEL_EPOCHS', '100'))
    validation_split: float = float(os.getenv('MODEL_VALIDATION_SPLIT', '0.2'))
    early_stopping_patience: int = int(os.getenv('MODEL_EARLY_STOPPING_PATIENCE', '10'))
    model_save_path: str = os.getenv('MODEL_SAVE_PATH', './data_storage/models/')

@dataclass
class WebAppConfig:
    """Web application configuration"""
    host: str = os.getenv('WEB_HOST', '0.0.0.0')
    port: int = int(os.getenv('WEB_PORT', '8000'))
    debug: bool = os.getenv('WEB_DEBUG', 'False').lower() == 'true'
    secret_key: str = os.getenv('WEB_SECRET_KEY', 'your-secret-key-here')
    session_timeout: int = int(os.getenv('WEB_SESSION_TIMEOUT', '3600'))
    cors_origins: list = os.getenv('WEB_CORS_ORIGINS', 'http://localhost:3000').split(',')

@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    level: str = os.getenv('LOG_LEVEL', 'INFO')
    format: str = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_path: str = os.getenv('LOG_FILE_PATH', './logs/schwab_ai.log')
    max_file_size: int = int(os.getenv('LOG_MAX_FILE_SIZE', '10485760'))  # 10MB
    backup_count: int = int(os.getenv('LOG_BACKUP_COUNT', '5'))
    enable_console: bool = os.getenv('LOG_ENABLE_CONSOLE', 'True').lower() == 'true'

@dataclass
class DataConfig:
    """Data storage and processing configuration"""
    raw_data_path: str = os.getenv('RAW_DATA_PATH', './data_storage/raw_data/')
    processed_data_path: str = os.getenv('PROCESSED_DATA_PATH', './data_storage/processed_data/')
    backup_path: str = os.getenv('BACKUP_PATH', './data_storage/backups/')
    data_retention_days: int = int(os.getenv('DATA_RETENTION_DAYS', '365'))
    batch_size: int = int(os.getenv('DATA_BATCH_SIZE', '1000'))
    compression: str = os.getenv('DATA_COMPRESSION', 'gzip')

@dataclass
class MonitoringConfig:
    """System monitoring configuration"""
    enable_metrics: bool = os.getenv('MONITORING_ENABLE_METRICS', 'True').lower() == 'true'
    metrics_port: int = int(os.getenv('MONITORING_METRICS_PORT', '9090'))
    health_check_interval: int = int(os.getenv('MONITORING_HEALTH_CHECK_INTERVAL', '60'))
    alert_email: str = os.getenv('MONITORING_ALERT_EMAIL', '')
    slack_webhook: str = os.getenv('MONITORING_SLACK_WEBHOOK', '')

class Settings:
    """Main settings class that aggregates all configuration"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.schwab_api = SchwabAPIConfig()
        self.trading = TradingConfig()
        self.model = ModelConfig()
        self.web_app = WebAppConfig()
        self.logging = LoggingConfig()
        self.data = DataConfig()
        self.monitoring = MonitoringConfig()
        
        # Ensure required directories exist
        self._create_directories()
        
        # Validate critical settings
        self._validate_settings()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            Path(self.data.raw_data_path),
            Path(self.data.processed_data_path),
            Path(self.data.backup_path),
            Path(self.model.model_save_path),
            Path(self.logging.file_path).parent,
            Path(self.schwab_api.token_path).parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _validate_settings(self):
        """Validate critical configuration settings"""
        if not self.schwab_api.client_id:
            raise ValueError("SCHWAB_CLIENT_ID environment variable is required")
        
        if not self.schwab_api.client_secret:
            raise ValueError("SCHWAB_CLIENT_SECRET environment variable is required")
        
        if not self.web_app.secret_key or self.web_app.secret_key == 'your-secret-key-here':
            raise ValueError("WEB_SECRET_KEY must be set to a secure random value")
        
        if self.trading.max_portfolio_risk > 0.1:
            raise ValueError("MAX_PORTFOLIO_RISK should not exceed 10% (0.1)")
        
        if self.trading.max_position_size > 0.5:
            raise ValueError("MAX_POSITION_SIZE should not exceed 50% (0.5)")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            'database': self.database.__dict__,
            'redis': self.redis.__dict__,
            'schwab_api': {k: v for k, v in self.schwab_api.__dict__.items() 
                          if k not in ['client_secret']},  # Exclude sensitive data
            'trading': self.trading.__dict__,
            'model': self.model.__dict__,
            'web_app': {k: v for k, v in self.web_app.__dict__.items() 
                       if k not in ['secret_key']},  # Exclude sensitive data
            'logging': self.logging.__dict__,
            'data': self.data.__dict__,
            'monitoring': self.monitoring.__dict__
        }
    
    def save_config(self, file_path: str):
        """Save configuration to file"""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load_config(cls, file_path: str) -> 'Settings':
        """Load configuration from file"""
        if not os.path.exists(file_path):
            return cls()
        
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        
        settings = cls()
        
        # Update settings with loaded data
        for section, values in config_data.items():
            if hasattr(settings, section):
                section_obj = getattr(settings, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
        
        return settings
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        return (f"postgresql://{self.database.username}:{self.database.password}"
                f"@{self.database.host}:{self.database.port}/{self.database.database}")
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.redis.password:
            return f"redis://:{self.redis.password}@{self.redis.host}:{self.redis.port}/{self.redis.db}"
        return f"redis://{self.redis.host}:{self.redis.port}/{self.redis.db}"
    
    def is_trading_hours(self, current_time=None) -> bool:
        """Check if current time is within trading hours"""
        from datetime import datetime, time
        import pytz
        
        if current_time is None:
            tz = pytz.timezone(self.trading.timezone)
            current_time = datetime.now(tz).time()
        
        start_time = time.fromisoformat(self.trading.trading_hours_start)
        end_time = time.fromisoformat(self.trading.trading_hours_end)
        
        return start_time <= current_time <= end_time
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters for training"""
        return {
            'sequence_length': self.model.sequence_length,
            'cnn_filters': self.model.cnn_filters,
            'lstm_units': self.model.lstm_units,
            'dropout_rate': self.model.dropout_rate,
            'learning_rate': self.model.learning_rate,
            'batch_size': self.model.batch_size,
            'epochs': self.model.epochs,
            'validation_split': self.model.validation_split,
            'early_stopping_patience': self.model.early_stopping_patience
        }

# Global settings instance
settings = Settings()

# Export commonly used configurations
DATABASE_URL = settings.get_database_url()
REDIS_URL = settings.get_redis_url()
MODEL_PARAMS = settings.get_model_params()

# Environment detection
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
IS_PRODUCTION = ENVIRONMENT.lower() == 'production'
IS_DEVELOPMENT = ENVIRONMENT.lower() == 'development'
IS_TESTING = ENVIRONMENT.lower() == 'testing'