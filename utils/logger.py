"""
Advanced Logging System for Schwab AI Trading System
Provides structured logging with multiple handlers and formatters
"""

import logging
import logging.handlers
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import structlog
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

from config.settings import settings

# Install rich traceback handler
install(show_locals=True)

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Create base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)

class TradingContextFormatter(logging.Formatter):
    """Formatter that includes trading context information"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Add trading context if available
        trading_context = getattr(record, 'trading_context', {})
        
        # Base format
        base_format = super().format(record)
        
        if trading_context:
            context_str = " | ".join([f"{k}={v}" for k, v in trading_context.items()])
            return f"{base_format} | {context_str}"
        
        return base_format

class SchwabLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds trading context to log records"""
    
    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        super().__init__(logger, extra or {})
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        # Add extra fields to the record
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        # Add adapter's extra fields
        kwargs['extra'].update(self.extra)
        
        # Add trading context if available
        if hasattr(self, 'trading_context'):
            kwargs['extra']['trading_context'] = self.trading_context
        
        return msg, kwargs
    
    def set_trading_context(self, **context):
        """Set trading context for this logger"""
        self.trading_context = context
    
    def clear_trading_context(self):
        """Clear trading context"""
        if hasattr(self, 'trading_context'):
            delattr(self, 'trading_context')

def setup_logging():
    """Setup comprehensive logging system"""
    
    # Create logs directory
    log_dir = Path(settings.logging.file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.logging.level))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with Rich formatting
    if settings.logging.enable_console:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_level=True,
            show_path=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True
        )
        console_handler.setLevel(getattr(logging, settings.logging.level))
        
        # Use structured formatter for production, rich for development
        if settings.IS_PRODUCTION:
            console_handler.setFormatter(StructuredFormatter())
        
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=settings.logging.file_path,
        maxBytes=settings.logging.max_file_size,
        backupCount=settings.logging.backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, settings.logging.level))
    
    # Use structured format for file logging
    file_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(file_handler)
    
    # Separate handler for trading activities
    trading_log_path = log_dir / 'trading.log'
    trading_handler = logging.handlers.RotatingFileHandler(
        filename=trading_log_path,
        maxBytes=settings.logging.max_file_size,
        backupCount=settings.logging.backup_count,
        encoding='utf-8'
    )
    trading_handler.setLevel(logging.INFO)
    trading_handler.setFormatter(TradingContextFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Add trading handler to specific loggers
    trading_loggers = [
        'trading.strategy_engine',
        'trading.portfolio_manager',
        'trading.risk_manager',
        'schwab_api.trading_client'
    ]
    
    for logger_name in trading_loggers:
        logger = logging.getLogger(logger_name)
        logger.addHandler(trading_handler)
    
    # Error handler for critical errors
    error_log_path = log_dir / 'errors.log'
    error_handler = logging.handlers.RotatingFileHandler(
        filename=error_log_path,
        maxBytes=settings.logging.max_file_size,
        backupCount=settings.logging.backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(error_handler)
    
    # Performance handler for performance metrics
    perf_log_path = log_dir / 'performance.log'
    perf_handler = logging.handlers.RotatingFileHandler(
        filename=perf_log_path,
        maxBytes=settings.logging.max_file_size,
        backupCount=settings.logging.backup_count,
        encoding='utf-8'
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(StructuredFormatter())
    
    # Add performance handler to performance logger
    perf_logger = logging.getLogger('performance')
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)
    perf_logger.propagate = False  # Don't propagate to root logger
    
    # Configure third-party loggers
    configure_third_party_loggers()
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized", extra={
        'log_level': settings.logging.level,
        'log_file': str(settings.logging.file_path),
        'console_enabled': settings.logging.enable_console
    })

def configure_third_party_loggers():
    """Configure logging levels for third-party libraries"""
    
    # Reduce noise from third-party libraries
    third_party_configs = {
        'urllib3.connectionpool': logging.WARNING,
        'requests.packages.urllib3': logging.WARNING,
        'websockets': logging.WARNING,
        'asyncio': logging.WARNING,
        'matplotlib': logging.WARNING,
        'PIL': logging.WARNING,
        'tornado.access': logging.WARNING,
        'uvicorn.access': logging.WARNING if settings.IS_PRODUCTION else logging.INFO,
        'uvicorn.error': logging.INFO,
        'fastapi': logging.INFO,
        'sqlalchemy.engine': logging.WARNING if settings.IS_PRODUCTION else logging.INFO,
        'alembic': logging.INFO,
    }
    
    for logger_name, level in third_party_configs.items():
        logging.getLogger(logger_name).setLevel(level)

def get_logger(name: str, **context) -> SchwabLoggerAdapter:
    """
    Get a logger with optional trading context
    
    Args:
        name: Logger name (usually __name__)
        **context: Additional context to include in all log messages
        
    Returns:
        Configured logger adapter
    """
    base_logger = logging.getLogger(name)
    adapter = SchwabLoggerAdapter(base_logger, context)
    
    return adapter

def get_trading_logger(symbol: str = None, strategy: str = None, **context) -> SchwabLoggerAdapter:
    """
    Get a logger specifically for trading activities
    
    Args:
        symbol: Trading symbol
        strategy: Strategy name
        **context: Additional trading context
        
    Returns:
        Trading logger with context
    """
    logger = get_logger('trading')
    
    trading_context = {}
    if symbol:
        trading_context['symbol'] = symbol
    if strategy:
        trading_context['strategy'] = strategy
    
    trading_context.update(context)
    logger.set_trading_context(**trading_context)
    
    return logger

def get_performance_logger() -> SchwabLoggerAdapter:
    """Get logger for performance metrics"""
    return get_logger('performance')

class LoggingContext:
    """Context manager for adding temporary logging context"""
    
    def __init__(self, logger: SchwabLoggerAdapter, **context):
        self.logger = logger
        self.context = context
        self.original_context = getattr(logger, 'trading_context', {})
    
    def __enter__(self):
        # Merge contexts
        new_context = self.original_context.copy()
        new_context.update(self.context)
        self.logger.set_trading_context(**new_context)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original context
        if self.original_context:
            self.logger.set_trading_context(**self.original_context)
        else:
            self.logger.clear_trading_context()

def log_function_call(func):
    """Decorator to log function calls and execution time"""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        # Log function entry
        logger.debug(f"Entering {func.__name__}", extra={
            'function': func.__name__,
            'args_count': len(args),
            'kwargs_keys': list(kwargs.keys())
        })
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log successful completion
            logger.debug(f"Completed {func.__name__}", extra={
                'function': func.__name__,
                'execution_time': execution_time,
                'success': True
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log exception
            logger.error(f"Exception in {func.__name__}: {e}", extra={
                'function': func.__name__,
                'execution_time': execution_time,
                'success': False,
                'exception_type': type(e).__name__
            }, exc_info=True)
            
            raise
    
    return wrapper

def log_trade_execution(symbol: str, action: str, quantity: int, price: float, **context):
    """Log trade execution with structured data"""
    logger = get_trading_logger(symbol=symbol)
    
    trade_data = {
        'action': action,
        'quantity': quantity,
        'price': price,
        'timestamp': datetime.now().isoformat(),
        **context
    }
    
    logger.info(f"Trade executed: {action} {quantity} {symbol} @ ${price:.2f}", extra={
        'trade_data': trade_data,
        'event_type': 'trade_execution'
    })

def log_performance_metric(metric_name: str, value: float, **context):
    """Log performance metrics"""
    perf_logger = get_performance_logger()
    
    perf_logger.info(f"Performance metric: {metric_name}", extra={
        'metric_name': metric_name,
        'metric_value': value,
        'timestamp': datetime.now().isoformat(),
        'event_type': 'performance_metric',
        **context
    })

def log_error_with_context(logger: logging.Logger, message: str, **context):
    """Log error with additional context"""
    logger.error(message, extra={
        'error_context': context,
        'timestamp': datetime.now().isoformat(),
        'event_type': 'error'
    }, exc_info=True)

# Convenience function for quick setup
def quick_setup():
    """Quick logging setup for simple use cases"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('schwab_ai.log')
        ]
    )

# Export commonly used items
__all__ = [
    'setup_logging',
    'get_logger',
    'get_trading_logger', 
    'get_performance_logger',
    'LoggingContext',
    'log_function_call',
    'log_trade_execution',
    'log_performance_metric',
    'log_error_with_context',
    'quick_setup'
]