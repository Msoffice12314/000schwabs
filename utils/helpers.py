import os
import json
import yaml
import pickle
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import pandas as pd
import numpy as np
from pathlib import Path
import time
import threading
from functools import wraps, lru_cache
from dataclasses import dataclass, asdict
import re
import uuid
import zipfile
import tarfile
import requests
from urllib.parse import urlparse
import sqlite3
import csv
import tempfile
import shutil
import math
from decimal import Decimal, ROUND_HALF_UP
import pytz
from dateutil import parser as date_parser
import schedule
import psutil
import platform
import socket
import subprocess

# Logging configuration
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None,
                 log_format: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Configuration management
def load_config(config_path: str, config_type: str = "auto") -> Dict[str, Any]:
    """Load configuration from file (JSON, YAML, or Python dict)"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Auto-detect config type from extension
    if config_type == "auto":
        ext = Path(config_path).suffix.lower()
        if ext in ['.json']:
            config_type = "json"
        elif ext in ['.yaml', '.yml']:
            config_type = "yaml"
        elif ext in ['.py']:
            config_type = "python"
        else:
            config_type = "json"  # default
    
    try:
        with open(config_path, 'r') as f:
            if config_type == "json":
                return json.load(f)
            elif config_type == "yaml":
                return yaml.safe_load(f)
            elif config_type == "python":
                # Execute Python file and extract variables
                config_globals = {}
                exec(f.read(), config_globals)
                return {k: v for k, v in config_globals.items() if not k.startswith('__')}
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")

def save_config(config: Dict[str, Any], config_path: str, config_type: str = "auto"):
    """Save configuration to file"""
    # Auto-detect config type from extension
    if config_type == "auto":
        ext = Path(config_path).suffix.lower()
        if ext in ['.json']:
            config_type = "json"
        elif ext in ['.yaml', '.yml']:
            config_type = "yaml"
        else:
            config_type = "json"  # default
    
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_type == "json":
                json.dump(config, f, indent=2, default=str)
            elif config_type == "yaml":
                yaml.dump(config, f, default_flow_style=False)
    except Exception as e:
        raise ValueError(f"Error saving configuration: {e}")

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries"""
    result = {}
    for config in configs:
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    return result

# Data validation and cleaning
def validate_data_types(data: Dict[str, Any], schema: Dict[str, type]) -> bool:
    """Validate data types against schema"""
    for key, expected_type in schema.items():
        if key not in data:
            return False
        if not isinstance(data[key], expected_type):
            return False
    return True

def clean_data(data: Union[Dict, List, str, float, int]) -> Union[Dict, List, str, float, int, None]:
    """Clean data by removing None values and empty structures"""
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            cleaned_value = clean_data(value)
            if cleaned_value is not None:
                cleaned[key] = cleaned_value
        return cleaned if cleaned else None
    elif isinstance(data, list):
        cleaned = [clean_data(item) for item in data]
        cleaned = [item for item in cleaned if item is not None]
        return cleaned if cleaned else None
    elif isinstance(data, str):
        return data.strip() if data.strip() else None
    else:
        return data

def sanitize_string(text: str, max_length: int = None, allowed_chars: str = None) -> str:
    """Sanitize string by removing invalid characters"""
    if not isinstance(text, str):
        return str(text)
    
    # Remove control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Filter allowed characters if specified
    if allowed_chars:
        sanitized = ''.join(c for c in sanitized if c in allowed_chars)
    
    # Truncate if max_length specified
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized.strip()

# Financial calculations
def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100

def calculate_compound_return(returns: List[float]) -> float:
    """Calculate compound return from list of returns"""
    if not returns:
        return 0.0
    
    compound = 1.0
    for r in returns:
        compound *= (1 + r)
    return compound - 1.0

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02,
                         periods_per_year: int = 252) -> float:
    """Calculate Sharpe ratio"""
    if not returns or len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - (risk_free_rate / periods_per_year)
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(periods_per_year)

def calculate_max_drawdown(values: List[float]) -> Tuple[float, int, int]:
    """Calculate maximum drawdown and its duration"""
    if not values:
        return 0.0, 0, 0
    
    peak = values[0]
    max_dd = 0.0
    max_dd_start = 0
    max_dd_end = 0
    current_dd_start = 0
    
    for i, value in enumerate(values):
        if value > peak:
            peak = value
            current_dd_start = i
        else:
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_start = current_dd_start
                max_dd_end = i
    
    return max_dd, max_dd_start, max_dd_end

def normalize_price(price: float, decimals: int = 2) -> float:
    """Normalize price to specified decimal places"""
    return float(Decimal(str(price)).quantize(
        Decimal('0.' + '0' * decimals), 
        rounding=ROUND_HALF_UP
    ))

def calculate_position_size(account_value: float, risk_per_trade: float,
                          entry_price: float, stop_loss_price: float) -> int:
    """Calculate position size based on risk management"""
    if entry_price <= 0 or stop_loss_price <= 0 or entry_price == stop_loss_price:
        return 0
    
    risk_amount = account_value * risk_per_trade
    price_risk = abs(entry_price - stop_loss_price)
    
    if price_risk == 0:
        return 0
    
    position_size = risk_amount / price_risk
    return int(position_size / entry_price)

# Date and time utilities
def get_market_timezone() -> pytz.timezone:
    """Get US Eastern timezone for market hours"""
    return pytz.timezone('US/Eastern')

def is_market_hours(dt: datetime = None, timezone_str: str = 'US/Eastern') -> bool:
    """Check if given datetime is during market hours"""
    if dt is None:
        dt = datetime.now()
    
    if dt.tzinfo is None:
        dt = pytz.timezone(timezone_str).localize(dt)
    else:
        dt = dt.astimezone(pytz.timezone(timezone_str))
    
    # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
    if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= dt <= market_close

def get_next_market_open(dt: datetime = None, timezone_str: str = 'US/Eastern') -> datetime:
    """Get next market open datetime"""
    if dt is None:
        dt = datetime.now()
    
    if dt.tzinfo is None:
        dt = pytz.timezone(timezone_str).localize(dt)
    else:
        dt = dt.astimezone(pytz.timezone(timezone_str))
    
    # If it's weekend, move to Monday
    while dt.weekday() >= 5:
        dt += timedelta(days=1)
    
    # Set to market open time
    market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
    
    # If we're past market open today, move to next day
    if dt > market_open:
        dt += timedelta(days=1)
        while dt.weekday() >= 5:
            dt += timedelta(days=1)
        market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
    
    return market_open

def parse_date_string(date_str: str, default_timezone: str = 'UTC') -> datetime:
    """Parse date string with various formats"""
    try:
        # Try parsing with dateutil
        dt = date_parser.parse(date_str)
        
        # Add timezone if naive
        if dt.tzinfo is None:
            dt = pytz.timezone(default_timezone).localize(dt)
        
        return dt
    except Exception as e:
        raise ValueError(f"Unable to parse date string '{date_str}': {e}")

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"

# File and I/O utilities
def ensure_directory(path: str) -> str:
    """Ensure directory exists, create if not"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def get_file_hash(file_path: str, hash_algorithm: str = 'md5') -> str:
    """Get file hash"""
    hash_func = getattr(hashlib, hash_algorithm)()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

def compress_file(source_path: str, target_path: str, compression: str = 'gzip') -> bool:
    """Compress file"""
    try:
        if compression == 'gzip':
            import gzip
            with open(source_path, 'rb') as f_in:
                with gzip.open(target_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif compression == 'zip':
            with zipfile.ZipFile(target_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(source_path, os.path.basename(source_path))
        elif compression == 'tar':
            with tarfile.open(target_path, 'w:gz') as tar:
                tar.add(source_path, arcname=os.path.basename(source_path))
        else:
            raise ValueError(f"Unsupported compression: {compression}")
        
        return True
    except Exception as e:
        logging.error(f"Compression failed: {e}")
        return False

def read_csv_chunks(file_path: str, chunk_size: int = 10000) -> pd.DataFrame:
    """Read large CSV file in chunks"""
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

def backup_file(file_path: str, backup_dir: str = None, max_backups: int = 5) -> str:
    """Create backup of file with rotation"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if backup_dir is None:
        backup_dir = os.path.dirname(file_path)
    
    ensure_directory(backup_dir)
    
    filename = os.path.basename(file_path)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_filename = f"{filename}.backup_{timestamp}"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    # Copy file
    shutil.copy2(file_path, backup_path)
    
    # Rotate old backups
    backup_pattern = f"{filename}.backup_*"
    existing_backups = sorted([
        f for f in os.listdir(backup_dir) 
        if f.startswith(f"{filename}.backup_")
    ])
    
    # Remove excess backups
    while len(existing_backups) > max_backups:
        old_backup = existing_backups.pop(0)
        os.remove(os.path.join(backup_dir, old_backup))
    
    return backup_path

# Performance and monitoring
def measure_execution_time(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger = logging.getLogger(func.__module__)
        logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    return wrapper

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, 
                    exponential_backoff: bool = True) -> Callable:
    """Decorator to retry function on failure"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        wait_time = delay * (2 ** attempt if exponential_backoff else 1)
                        logging.warning(f"{func.__name__} failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"{func.__name__} failed after {max_retries + 1} attempts: {e}")
            
            raise last_exception
        return wrapper
    return decorator

def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'cpu_percent': psutil.cpu_percent(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': {
            path: {
                'total': psutil.disk_usage(path).total,
                'used': psutil.disk_usage(path).used,
                'free': psutil.disk_usage(path).free,
                'percent': (psutil.disk_usage(path).used / psutil.disk_usage(path).total) * 100
            } for path in ['/'] if os.path.exists(path)
        },
        'hostname': socket.gethostname(),
        'ip_address': socket.gethostbyname(socket.gethostname())
    }

def monitor_resource_usage(threshold_cpu: float = 80.0, threshold_memory: float = 80.0) -> Dict[str, Any]:
    """Monitor system resource usage and return alerts"""
    alerts = []
    
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    
    if cpu_percent > threshold_cpu:
        alerts.append(f"High CPU usage: {cpu_percent:.1f}%")
    
    if memory_percent > threshold_memory:
        alerts.append(f"High memory usage: {memory_percent:.1f}%")
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'alerts': alerts,
        'timestamp': datetime.now().isoformat()
    }

# Network utilities
def check_internet_connection(host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
    """Check internet connectivity"""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

def download_file(url: str, local_path: str, chunk_size: int = 8192) -> bool:
    """Download file from URL"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        ensure_directory(os.path.dirname(local_path))
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        
        return True
    except Exception as e:
        logging.error(f"Download failed: {e}")
        return False

def validate_url(url: str) -> bool:
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

# Data processing utilities
def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def remove_duplicates(lst: List[Any], key: Callable = None) -> List[Any]:
    """Remove duplicates from list while preserving order"""
    if key is None:
        seen = set()
        return [x for x in lst if not (x in seen or seen.add(x))]
    else:
        seen = set()
        return [x for x in lst if not (key(x) in seen or seen.add(key(x)))]

def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

# Scheduling utilities
class TaskScheduler:
    """Simple task scheduler"""
    
    def __init__(self):
        self.tasks = []
        self.running = False
        self.thread = None
    
    def add_task(self, func: Callable, interval: int, *args, **kwargs):
        """Add recurring task"""
        self.tasks.append({
            'func': func,
            'interval': interval,
            'args': args,
            'kwargs': kwargs,
            'last_run': 0
        })
    
    def start(self):
        """Start scheduler"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop scheduler"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _run_scheduler(self):
        """Scheduler main loop"""
        while self.running:
            current_time = time.time()
            
            for task in self.tasks:
                if current_time - task['last_run'] >= task['interval']:
                    try:
                        task['func'](*task['args'], **task['kwargs'])
                        task['last_run'] = current_time
                    except Exception as e:
                        logging.error(f"Scheduled task failed: {e}")
            
            time.sleep(1)

# Utility functions for common operations
def generate_unique_id(prefix: str = "") -> str:
    """Generate unique identifier"""
    return f"{prefix}{uuid.uuid4().hex[:8]}"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that handles division by zero"""
    return numerator / denominator if denominator != 0 else default

def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp value between min and max"""
    return max(min_value, min(max_value, value))

def interpolate(x: float, x1: float, x2: float, y1: float, y2: float) -> float:
    """Linear interpolation"""
    if x2 == x1:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

def moving_average(values: List[float], window: int) -> List[float]:
    """Calculate moving average"""
    if len(values) < window:
        return values
    
    result = []
    for i in range(len(values)):
        if i < window - 1:
            result.append(values[i])
        else:
            avg = sum(values[i - window + 1:i + 1]) / window
            result.append(avg)
    
    return result

@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    """Calculate Fibonacci number with caching"""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def format_number(number: float, decimals: int = 2, thousands_sep: str = ",") -> str:
    """Format number with thousands separator"""
    format_str = f"{{:,.{decimals}f}}"
    formatted = format_str.format(number)
    
    if thousands_sep != ",":
        formatted = formatted.replace(",", thousands_sep)
    
    return formatted

# Environment and configuration helpers
def get_env_var(var_name: str, default: Any = None, var_type: type = str) -> Any:
    """Get environment variable with type conversion"""
    value = os.getenv(var_name, default)
    
    if value is None:
        return default
    
    try:
        if var_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif var_type == list:
            return [item.strip() for item in value.split(',')]
        else:
            return var_type(value)
    except (ValueError, TypeError):
        return default

def load_env_file(env_path: str = '.env'):
    """Load environment variables from .env file"""
    if not os.path.exists(env_path):
        return
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Context managers
class temporary_directory:
    """Context manager for temporary directory"""
    
    def __init__(self, prefix: str = "temp_"):
        self.prefix = prefix
        self.path = None
    
    def __enter__(self) -> str:
        self.path = tempfile.mkdtemp(prefix=self.prefix)
        return self.path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path and os.path.exists(self.path):
            shutil.rmtree(self.path)

class suppress_stdout:
    """Context manager to suppress stdout"""
    
    def __enter__(self):
        self._original_stdout = os.dup(1)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 1)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self._original_stdout, 1)
        os.close(self._devnull)
        os.close(self._original_stdout)

# Testing utilities
def create_mock_data(n_samples: int = 100, n_features: int = 10, 
                    random_seed: int = 42) -> pd.DataFrame:
    """Create mock data for testing"""
    np.random.seed(random_seed)
    
    data = {}
    for i in range(n_features):
        if i % 3 == 0:
            # Categorical feature
            data[f'cat_feature_{i}'] = np.random.choice(['A', 'B', 'C'], n_samples)
        elif i % 3 == 1:
            # Numeric feature (normal distribution)
            data[f'num_feature_{i}'] = np.random.normal(0, 1, n_samples)
        else:
            # Numeric feature (uniform distribution)
            data[f'uniform_feature_{i}'] = np.random.uniform(0, 100, n_samples)
    
    # Add timestamp column
    start_date = datetime.now() - timedelta(days=n_samples)
    data['timestamp'] = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    return pd.DataFrame(data)

def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, 
                          check_dtype: bool = True, rtol: float = 1e-5):
    """Assert two DataFrames are equal with better error messages"""
    try:
        pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype, rtol=rtol)
    except AssertionError as e:
        # Provide more detailed error information
        if df1.shape != df2.shape:
            raise AssertionError(f"DataFrame shapes differ: {df1.shape} vs {df2.shape}")
        
        if not df1.columns.equals(df2.columns):
            raise AssertionError(f"Column names differ: {list(df1.columns)} vs {list(df2.columns)}")
        
        raise e

# Global utilities
class Singleton:
    """Singleton metaclass"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

def deprecated(reason: str = ""):
    """Decorator to mark functions as deprecated"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import warnings
            warnings.warn(
                f"{func.__name__} is deprecated. {reason}",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator
