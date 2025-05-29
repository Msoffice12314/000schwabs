"""
Schwab AI Trading System - Data Collection Pipeline
Comprehensive data collection system for real-time and historical market data with multiple sources.
"""

import logging
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from concurrent.futures import ThreadPoolExecutor
import requests
from pathlib import Path

from schwab_api.market_data import MarketDataClient
from schwab_api.rate_limiter import RateLimiter
from utils.cache_manager import CacheManager
from utils.database import DatabaseManager
from config.settings import get_settings

logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Data source types"""
    SCHWAB_API = "schwab_api"
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    FINNHUB = "finnhub"
    IEX_CLOUD = "iex_cloud"

class DataType(Enum):
    """Types of market data"""
    BARS = "bars"  # OHLCV bars
    QUOTES = "quotes"  # Real-time quotes
    TRADES = "trades"  # Individual trades
    OPTIONS = "options"  # Options data
    FUNDAMENTALS = "fundamentals"  # Company fundamentals
    NEWS = "news"  # Market news
    ECONOMIC = "economic"  # Economic indicators

class TimeFrame(Enum):
    """Data time frames"""
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1m"

@dataclass
class DataRequest:
    """Data collection request"""
    symbols: List[str]
    data_type: DataType
    timeframe: TimeFrame
    start_date: datetime
    end_date: datetime
    source: DataSource = DataSource.SCHWAB_API
    extended_hours: bool = False
    include_dividends: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataPoint:
    """Individual data point"""
    symbol: str
    timestamp: datetime
    data_type: DataType
    timeframe: TimeFrame
    data: Dict[str, Any]
    source: DataSource
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CollectionResult:
    """Data collection result"""
    request: DataRequest
    success: bool
    data_points: List[DataPoint]
    error_message: Optional[str] = None
    collection_time: float = 0.0
    records_collected: int = 0
    data_quality: float = 1.0

class DataCollector:
    """
    Comprehensive market data collection system supporting multiple data sources
    with real-time streaming, historical data, and data quality management.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache_manager = CacheManager()
        self.db_manager = DatabaseManager()
        self.rate_limiter = RateLimiter()
        
        # Data sources
        self.market_data_client = MarketDataClient()
        self.data_sources = {}
        
        # Collection configuration
        self.collection_config = {
            'batch_size': 100,
            'max_concurrent_requests': 10,
            'retry_attempts': 3,
            'retry_delay': 1.0,
            'cache_duration': 300,  # 5 minutes
            'quality_threshold': 0.8
        }
        
        # Symbol universe
        self.symbol_universe = []
        self.watchlist_symbols = []
        
        # Collection queues
        self.real_time_queue = asyncio.Queue(maxsize=1000)
        self.historical_queue = asyncio.Queue(maxsize=500)
        
        # Collection tasks
        self.collection_tasks = []
        self.is_collecting = False
        
        # Data validation
        self.validation_rules = {}
        
        logger.info("DataCollector initialized")
    
    async def initialize(self) -> bool:
        """Initialize the data collector"""
        try:
            # Initialize market data client
            await self.market_data_client.initialize()
            
            # Load symbol universe
            await self._load_symbol_universe()
            
            # Initialize data sources
            await self._initialize_data_sources()
            
            # Setup validation rules
            self._setup_validation_rules()
            
            # Load collection configuration
            await self._load_collection_config()
            
            logger.info("DataCollector initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DataCollector: {str(e)}")
            return False
    
    async def _load_symbol_universe(self):
        """Load symbol universe from database/configuration"""
        try:
            # Load from database
            query = "SELECT symbol, sector, market_cap, active FROM symbols WHERE active = TRUE"
            results = await self.db_manager.execute_query(query)
            
            if results:
                self.symbol_universe = [row['symbol'] for row in results]
            else:
                # Default symbol list
                self.symbol_universe = [
                    'SPY', 'QQQ', 'IWM', 'VTI', 'VOO',
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                    'META', 'NVDA', 'NFLX', 'CRM', 'ADBE',
                    'JPM', 'BAC', 'GS', 'V', 'MA',
                    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'
                ]
            
            # Load watchlist symbols
            watchlist_query = "SELECT symbol FROM watchlist WHERE active = TRUE"
            watchlist_results = await self.db_manager.execute_query(watchlist_query)
            
            if watchlist_results:
                self.watchlist_symbols = [row['symbol'] for row in watchlist_results]
            else:
                self.watchlist_symbols = self.symbol_universe[:10]  # Top 10 as default
            
            logger.info(f"Loaded {len(self.symbol_universe)} symbols, {len(self.watchlist_symbols)} in watchlist")
            
        except Exception as e:
            logger.error(f"Error loading symbol universe: {str(e)}")
            # Use default symbols
            self.symbol_universe = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'TSLA']
            self.watchlist_symbols = self.symbol_universe
    
    async def _initialize_data_sources(self):
        """Initialize different data source adapters"""
        try:
            # Schwab API (primary)
            self.data_sources[DataSource.SCHWAB_API] = self.market_data_client
            
            # Yahoo Finance (backup)
            self.data_sources[DataSource.YAHOO_FINANCE] = YahooFinanceAdapter()
            
            # Add other sources as needed
            # self.data_sources[DataSource.ALPHA_VANTAGE] = AlphaVantageAdapter()
            # self.data_sources[DataSource.POLYGON] = PolygonAdapter()
            
            logger.info(f"Initialized {len(self.data_sources)} data sources")
            
        except Exception as e:
            logger.error(f"Error initializing data sources: {str(e)}")
    
    def _setup_validation_rules(self):
        """Setup data validation rules"""
        self.validation_rules = {
            DataType.BARS: {
                'required_fields': ['open', 'high', 'low', 'close', 'volume'],
                'logical_checks': ['high >= low', 'high >= open', 'high >= close', 'low <= open', 'low <= close'],
                'range_checks': {'volume': (0, float('inf'))}
            },
            DataType.QUOTES: {
                'required_fields': ['bid', 'ask', 'last'],
                'logical_checks': ['ask >= bid', 'last > 0'],
                'range_checks': {'bid': (0, float('inf')), 'ask': (0, float('inf'))}
            }
        }
    
    async def _load_collection_config(self):
        """Load collection configuration from database"""
        try:
            query = "SELECT config_key, config_value FROM data_collection_config"
            results = await self.db_manager.execute_query(query)
            
            for row in results:
                key = row['config_key']
                value = row['config_value']
                
                if key in self.collection_config:
                    # Parse value based on type
                    if isinstance(self.collection_config[key], int):
                        self.collection_config[key] = int(value)
                    elif isinstance(self.collection_config[key], float):
                        self.collection_config[key] = float(value)
                    else:
                        self.collection_config[key] = value
            
            logger.info("Loaded collection configuration")
            
        except Exception as e:
            logger.error(f"Error loading collection config: {str(e)}")
    
    async def collect_historical_data(self, request: DataRequest) -> CollectionResult:
        """
        Collect historical market data
        
        Args:
            request: Data collection request
            
        Returns:
            CollectionResult with collected data
        """
        try:
            start_time = time.time()
            
            # Validate request
            if not self._validate_request(request):
                return CollectionResult(
                    request=request,
                    success=False,
                    data_points=[],
                    error_message="Invalid request parameters"
                )
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_data = await self.cache_manager.get(cache_key)
            
            if cached_data and not self._is_cache_stale(cached_data, request):
                logger.info(f"Using cached data for {len(request.symbols)} symbols")
                return cached_data
            
            # Collect data from primary source
            data_points = []
            
            try:
                primary_source = request.source
                if primary_source in self.data_sources:
                    data_points = await self._collect_from_source(request, primary_source)
                else:
                    logger.warning(f"Primary source {primary_source} not available")
            
            except Exception as e:
                logger.error(f"Error collecting from primary source: {str(e)}")
            
            # Try backup sources if primary failed
            if not data_points and request.source != DataSource.YAHOO_FINANCE:
                try:
                    logger.info("Trying backup data source")
                    backup_request = request
                    backup_request.source = DataSource.YAHOO_FINANCE
                    data_points = await self._collect_from_source(backup_request, DataSource.YAHOO_FINANCE)
                except Exception as e:
                    logger.error(f"Error collecting from backup source: {str(e)}")
            
            # Validate and clean data
            validated_data = await self._validate_and_clean_data(data_points, request)
            
            # Calculate collection metrics
            collection_time = time.time() - start_time
            records_collected = len(validated_data)
            data_quality = self._calculate_data_quality(validated_data)
            
            # Store data in database
            if validated_data:
                await self._store_data_points(validated_data)
            
            # Create result
            result = CollectionResult(
                request=request,
                success=len(validated_data) > 0,
                data_points=validated_data,
                error_message=None if validated_data else "No data collected",
                collection_time=collection_time,
                records_collected=records_collected,
                data_quality=data_quality
            )
            
            # Cache result
            if result.success:
                await self.cache_manager.set(cache_key, result, expire=self.collection_config['cache_duration'])
            
            logger.info(f"Collected {records_collected} data points in {collection_time:.2f}s "
                       f"with {data_quality:.2%} quality")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in historical data collection: {str(e)}")
            return CollectionResult(
                request=request,
                success=False,
                data_points=[],
                error_message=str(e),
                collection_time=time.time() - start_time if 'start_time' in locals() else 0.0
            )
    
    async def _collect_from_source(self, request: DataRequest, 
                                 source: DataSource) -> List[DataPoint]:
        """Collect data from specific source"""
        try:
            data_adapter = self.data_sources[source]
            
            if request.data_type == DataType.BARS:
                return await self._collect_bars(request, data_adapter)
            elif request.data_type == DataType.QUOTES:
                return await self._collect_quotes(request, data_adapter)
            elif request.data_type == DataType.FUNDAMENTALS:
                return await self._collect_fundamentals(request, data_adapter)
            else:
                logger.warning(f"Data type {request.data_type} not supported")
                return []
                
        except Exception as e:
            logger.error(f"Error collecting from {source}: {str(e)}")
            return []
    
    async def _collect_bars(self, request: DataRequest, data_adapter) -> List[DataPoint]:
        """Collect OHLCV bar data"""
        try:
            data_points = []
            
            # Process symbols in batches
            for i in range(0, len(request.symbols), self.collection_config['batch_size']):
                batch_symbols = request.symbols[i:i + self.collection_config['batch_size']]
                
                # Apply rate limiting
                await self.rate_limiter.acquire()
                
                # Collect data for batch
                batch_tasks = []
                for symbol in batch_symbols:
                    task = self._collect_symbol_bars(symbol, request, data_adapter)
                    batch_tasks.append(task)
                
                # Execute batch
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for symbol, result in zip(batch_symbols, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error collecting {symbol}: {str(result)}")
                    elif result:
                        data_points.extend(result)
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error collecting bars: {str(e)}")
            return []
    
    async def _collect_symbol_bars(self, symbol: str, request: DataRequest, 
                                 data_adapter) -> List[DataPoint]:
        """Collect bars for a single symbol"""
        try:
            # Get bars from adapter
            if hasattr(data_adapter, 'get_bars'):
                bars_data = await data_adapter.get_bars(
                    symbol=symbol,
                    timeframe=request.timeframe.value,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    extended_hours=request.extended_hours
                )
            else:
                # Use alternative method for different adapters
                bars_data = await self._get_bars_generic(
                    data_adapter, symbol, request
                )
            
            if not bars_data:
                return []
            
            # Convert to DataPoint objects
            data_points = []
            for bar in bars_data:
                data_point = DataPoint(
                    symbol=symbol,
                    timestamp=bar['timestamp'],
                    data_type=DataType.BARS,
                    timeframe=request.timeframe,
                    data={
                        'open': bar['open'],
                        'high': bar['high'],
                        'low': bar['low'],
                        'close': bar['close'],
                        'volume': bar['volume']
                    },
                    source=request.source,
                    quality_score=self._calculate_bar_quality(bar),
                    metadata={'extended_hours': request.extended_hours}
                )
                data_points.append(data_point)
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error collecting bars for {symbol}: {str(e)}")
            return []
    
    async def _get_bars_generic(self, data_adapter, symbol: str, 
                              request: DataRequest) -> List[Dict]:
        """Generic method to get bars from different adapters"""
        try:
            # This would be implemented differently for each data source
            # For now, return empty list
            logger.warning(f"Generic bar collection not implemented for {type(data_adapter)}")
            return []
            
        except Exception as e:
            logger.error(f"Error in generic bar collection: {str(e)}")
            return []
    
    def _calculate_bar_quality(self, bar: Dict) -> float:
        """Calculate quality score for a bar"""
        try:
            quality_score = 1.0
            
            # Check for missing data
            required_fields = ['open', 'high', 'low', 'close', 'volume']
            for field in required_fields:
                if field not in bar or bar[field] is None:
                    quality_score -= 0.2
            
            # Check for logical consistency
            if bar.get('high', 0) < bar.get('low', 0):
                quality_score -= 0.3
            
            if bar.get('high', 0) < max(bar.get('open', 0), bar.get('close', 0)):
                quality_score -= 0.2
            
            if bar.get('low', float('inf')) > min(bar.get('open', 0), bar.get('close', 0)):
                quality_score -= 0.2
            
            # Check for unusual values
            if bar.get('volume', 0) == 0:
                quality_score -= 0.1
            
            return max(0.0, quality_score)
            
        except Exception as e:
            logger.error(f"Error calculating bar quality: {str(e)}")
            return 0.5
    
    async def _collect_quotes(self, request: DataRequest, data_adapter) -> List[DataPoint]:
        """Collect real-time quote data"""
        try:
            data_points = []
            
            # Get quotes for all symbols
            if hasattr(data_adapter, 'get_quotes'):
                quotes_data = await data_adapter.get_quotes(request.symbols)
                
                for symbol, quote in quotes_data.items():
                    data_point = DataPoint(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        data_type=DataType.QUOTES,
                        timeframe=TimeFrame.MINUTE_1,  # Real-time
                        data={
                            'bid': quote.get('bid', 0),
                            'ask': quote.get('ask', 0),
                            'last': quote.get('last', 0),
                            'bid_size': quote.get('bid_size', 0),
                            'ask_size': quote.get('ask_size', 0),
                            'volume': quote.get('volume', 0)
                        },
                        source=request.source,
                        quality_score=self._calculate_quote_quality(quote)
                    )
                    data_points.append(data_point)
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error collecting quotes: {str(e)}")
            return []
    
    def _calculate_quote_quality(self, quote: Dict) -> float:
        """Calculate quality score for a quote"""
        try:
            quality_score = 1.0
            
            # Check for missing data
            required_fields = ['bid', 'ask', 'last']
            for field in required_fields:
                if field not in quote or quote[field] is None or quote[field] <= 0:
                    quality_score -= 0.3
            
            # Check bid/ask spread reasonableness
            bid = quote.get('bid', 0)
            ask = quote.get('ask', 0)
            
            if bid > 0 and ask > 0:
                spread_pct = (ask - bid) / ((ask + bid) / 2)
                if spread_pct > 0.05:  # > 5% spread
                    quality_score -= 0.2
                elif spread_pct > 0.02:  # > 2% spread
                    quality_score -= 0.1
            
            return max(0.0, quality_score)
            
        except Exception as e:
            logger.error(f"Error calculating quote quality: {str(e)}")
            return 0.5
    
    async def _collect_fundamentals(self, request: DataRequest, data_adapter) -> List[DataPoint]:
        """Collect fundamental data"""
        try:
            # This would implement fundamental data collection
            # For now, return empty list
            logger.info("Fundamental data collection not yet implemented")
            return []
            
        except Exception as e:
            logger.error(f"Error collecting fundamentals: {str(e)}")
            return []
    
    async def _validate_and_clean_data(self, data_points: List[DataPoint], 
                                     request: DataRequest) -> List[DataPoint]:
        """Validate and clean collected data"""
        try:
            validated_data = []
            
            for data_point in data_points:
                # Basic validation
                if not self._validate_data_point(data_point):
                    continue
                
                # Data type specific validation
                if data_point.data_type in self.validation_rules:
                    if not self._validate_with_rules(data_point, self.validation_rules[data_point.data_type]):
                        continue
                
                # Clean data
                cleaned_data_point = self._clean_data_point(data_point)
                validated_data.append(cleaned_data_point)
            
            logger.info(f"Validated {len(validated_data)} out of {len(data_points)} data points")
            return validated_data
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return data_points  # Return original data if validation fails
    
    def _validate_data_point(self, data_point: DataPoint) -> bool:
        """Basic data point validation"""
        try:
            # Check required fields
            if not data_point.symbol or not data_point.timestamp or not data_point.data:
                return False
            
            # Check timestamp reasonableness
            if data_point.timestamp > datetime.now() + timedelta(hours=1):
                return False
            
            if data_point.timestamp < datetime.now() - timedelta(days=365 * 10):
                return False
            
            # Check quality score
            if data_point.quality_score < self.collection_config['quality_threshold']:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data point: {str(e)}")
            return False
    
    def _validate_with_rules(self, data_point: DataPoint, rules: Dict) -> bool:
        """Validate data point against specific rules"""
        try:
            data = data_point.data
            
            # Check required fields
            for field in rules.get('required_fields', []):
                if field not in data or data[field] is None:
                    return False
            
            # Check logical constraints
            for check in rules.get('logical_checks', []):
                try:
                    # Simple evaluation (in production, use safer evaluation)
                    check_result = eval(check, {"__builtins__": {}}, data)
                    if not check_result:
                        return False
                except:
                    return False
            
            # Check range constraints
            for field, (min_val, max_val) in rules.get('range_checks', {}).items():
                if field in data:
                    value = data[field]
                    if value < min_val or value > max_val:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating with rules: {str(e)}")
            return False
    
    def _clean_data_point(self, data_point: DataPoint) -> DataPoint:
        """Clean and normalize data point"""
        try:
            # Make a copy
            cleaned_data = data_point.data.copy()
            
            # Round numerical values
            for key, value in cleaned_data.items():
                if isinstance(value, float):
                    if key in ['open', 'high', 'low', 'close', 'bid', 'ask', 'last']:
                        cleaned_data[key] = round(value, 4)  # Price precision
                    elif key == 'volume':
                        cleaned_data[key] = int(value)  # Volume as integer
                    else:
                        cleaned_data[key] = round(value, 6)  # General precision
            
            # Create cleaned data point
            cleaned_data_point = DataPoint(
                symbol=data_point.symbol,
                timestamp=data_point.timestamp,
                data_type=data_point.data_type,
                timeframe=data_point.timeframe,
                data=cleaned_data,
                source=data_point.source,
                quality_score=data_point.quality_score,
                metadata=data_point.metadata
            )
            
            return cleaned_data_point
            
        except Exception as e:
            logger.error(f"Error cleaning data point: {str(e)}")
            return data_point
    
    def _calculate_data_quality(self, data_points: List[DataPoint]) -> float:
        """Calculate overall data quality score"""
        try:
            if not data_points:
                return 0.0
            
            quality_scores = [dp.quality_score for dp in data_points]
            return np.mean(quality_scores)
            
        except Exception as e:
            logger.error(f"Error calculating data quality: {str(e)}")
            return 0.5
    
    async def _store_data_points(self, data_points: List[DataPoint]):
        """Store data points in database"""
        try:
            # Group by data type for efficient storage
            bars_data = [dp for dp in data_points if dp.data_type == DataType.BARS]
            quotes_data = [dp for dp in data_points if dp.data_type == DataType.QUOTES]
            
            # Store bars
            if bars_data:
                await self._store_bars_data(bars_data)
            
            # Store quotes
            if quotes_data:
                await self._store_quotes_data(quotes_data)
            
            logger.info(f"Stored {len(data_points)} data points to database")
            
        except Exception as e:
            logger.error(f"Error storing data points: {str(e)}")
    
    async def _store_bars_data(self, bars_data: List[DataPoint]):
        """Store bar data in database"""
        try:
            insert_query = """
                INSERT INTO market_data (symbol, timestamp, timeframe, open, high, low, close, 
                                       volume, source, quality_score, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    open = VALUES(open), high = VALUES(high), low = VALUES(low),
                    close = VALUES(close), volume = VALUES(volume),
                    quality_score = GREATEST(quality_score, VALUES(quality_score))
            """
            
            values = []
            for dp in bars_data:
                data = dp.data
                values.append((
                    dp.symbol,
                    dp.timestamp,
                    dp.timeframe.value,
                    data['open'],
                    data['high'],
                    data['low'],
                    data['close'],
                    data['volume'],
                    dp.source.value,
                    dp.quality_score,
                    datetime.now()
                ))
            
            await self.db_manager.execute_many(insert_query, values)
            
        except Exception as e:
            logger.error(f"Error storing bars data: {str(e)}")
    
    async def _store_quotes_data(self, quotes_data: List[DataPoint]):
        """Store quote data in database"""
        try:
            insert_query = """
                INSERT INTO quotes (symbol, timestamp, bid, ask, last, bid_size, ask_size,
                                  volume, source, quality_score, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = []
            for dp in quotes_data:
                data = dp.data
                values.append((
                    dp.symbol,
                    dp.timestamp,
                    data.get('bid', 0),
                    data.get('ask', 0),
                    data.get('last', 0),
                    data.get('bid_size', 0),
                    data.get('ask_size', 0),
                    data.get('volume', 0),
                    dp.source.value,
                    dp.quality_score,
                    datetime.now()
                ))
            
            await self.db_manager.execute_many(insert_query, values)
            
        except Exception as e:
            logger.error(f"Error storing quotes data: {str(e)}")
    
    def _validate_request(self, request: DataRequest) -> bool:
        """Validate data collection request"""
        try:
            # Check symbols
            if not request.symbols:
                return False
            
            # Check date range
            if request.start_date >= request.end_date:
                return False
            
            # Check if date range is reasonable
            if request.end_date - request.start_date > timedelta(days=365 * 5):
                logger.warning("Very large date range requested")
            
            # Check data source availability
            if request.source not in self.data_sources:
                logger.warning(f"Data source {request.source} not available")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating request: {str(e)}")
            return False
    
    def _generate_cache_key(self, request: DataRequest) -> str:
        """Generate cache key for request"""
        try:
            key_components = [
                "_".join(sorted(request.symbols)),
                request.data_type.value,
                request.timeframe.value,
                request.start_date.strftime("%Y%m%d"),
                request.end_date.strftime("%Y%m%d"),
                request.source.value
            ]
            
            return "_".join(key_components)
            
        except Exception as e:
            logger.error(f"Error generating cache key: {str(e)}")
            return f"cache_key_{hash(str(request))}"
    
    def _is_cache_stale(self, cached_data: CollectionResult, request: DataRequest) -> bool:
        """Check if cached data is stale"""
        try:
            # If request is for real-time data, cache is always stale
            if request.data_type == DataType.QUOTES:
                return True
            
            # Check if end date is recent (today)
            if request.end_date.date() >= datetime.now().date():
                return True
            
            # Check cache age
            cache_age = datetime.now() - cached_data.data_points[0].timestamp
            if cache_age > timedelta(hours=1):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking cache staleness: {str(e)}")
            return True
    
    async def start_real_time_collection(self, symbols: Optional[List[str]] = None):
        """Start real-time data collection"""
        try:
            if self.is_collecting:
                logger.warning("Real-time collection already running")
                return
            
            symbols_to_collect = symbols or self.watchlist_symbols
            
            logger.info(f"Starting real-time collection for {len(symbols_to_collect)} symbols")
            
            # Create collection task
            collection_task = asyncio.create_task(
                self._real_time_collection_loop(symbols_to_collect)
            )
            self.collection_tasks.append(collection_task)
            
            self.is_collecting = True
            
        except Exception as e:
            logger.error(f"Error starting real-time collection: {str(e)}")
    
    async def _real_time_collection_loop(self, symbols: List[str]):
        """Main real-time collection loop"""
        try:
            while self.is_collecting:
                # Create quote request
                quote_request = DataRequest(
                    symbols=symbols,
                    data_type=DataType.QUOTES,
                    timeframe=TimeFrame.MINUTE_1,
                    start_date=datetime.now(),
                    end_date=datetime.now(),
                    source=DataSource.SCHWAB_API
                )
                
                # Collect quotes
                result = await self.collect_historical_data(quote_request)
                
                if result.success:
                    # Add to real-time queue
                    for data_point in result.data_points:
                        try:
                            await self.real_time_queue.put(data_point)
                        except asyncio.QueueFull:
                            logger.warning("Real-time queue full, dropping data point")
                
                # Wait before next collection
                await asyncio.sleep(5)  # 5 second interval
                
        except Exception as e:
            logger.error(f"Error in real-time collection loop: {str(e)}")
        finally:
            self.is_collecting = False
    
    async def stop_real_time_collection(self):
        """Stop real-time data collection"""
        try:
            self.is_collecting = False
            
            # Cancel collection tasks
            for task in self.collection_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.collection_tasks, return_exceptions=True)
            
            self.collection_tasks.clear()
            
            logger.info("Real-time collection stopped")
            
        except Exception as e:
            logger.error(f"Error stopping real-time collection: {str(e)}")
    
    async def get_latest_data(self, symbols: List[str], 
                            data_type: DataType = DataType.BARS,
                            timeframe: TimeFrame = TimeFrame.DAILY,
                            limit: int = 100) -> Dict[str, pd.DataFrame]:
        """Get latest data for symbols"""
        try:
            data_dict = {}
            
            for symbol in symbols:
                query = """
                    SELECT timestamp, open, high, low, close, volume
                    FROM market_data 
                    WHERE symbol = %s AND timeframe = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """
                
                results = await self.db_manager.execute_query(
                    query, (symbol, timeframe.value, limit)
                )
                
                if results:
                    df = pd.DataFrame(results)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    df.sort_index(inplace=True)
                    data_dict[symbol] = df
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error getting latest data: {str(e)}")
            return {}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get data collection statistics"""
        try:
            stats = {
                'symbol_universe_size': len(self.symbol_universe),
                'watchlist_size': len(self.watchlist_symbols),
                'data_sources_available': len(self.data_sources),
                'is_collecting_realtime': self.is_collecting,
                'active_collection_tasks': len(self.collection_tasks),
                'real_time_queue_size': self.real_time_queue.qsize(),
                'historical_queue_size': self.historical_queue.qsize(),
                'collection_config': self.collection_config
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {'error': str(e)}
    
    async def cleanup(self):
        """Cleanup data collector resources"""
        try:
            # Stop real-time collection
            await self.stop_real_time_collection()
            
            # Clear queues
            while not self.real_time_queue.empty():
                self.real_time_queue.get_nowait()
            
            while not self.historical_queue.empty():
                self.historical_queue.get_nowait()
            
            logger.info("DataCollector cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


class YahooFinanceAdapter:
    """Yahoo Finance data adapter (backup source)"""
    
    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
        self.session = None
    
    async def initialize(self):
        """Initialize the adapter"""
        self.session = aiohttp.ClientSession()
    
    async def get_bars(self, symbol: str, timeframe: str, 
                      start_date: datetime, end_date: datetime,
                      extended_hours: bool = False) -> List[Dict]:
        """Get bar data from Yahoo Finance"""
        try:
            # Convert timeframe
            interval_map = {
                '1min': '1m',
                '5min': '5m',
                '15min': '15m',
                '30min': '30m',
                '1h': '1h',
                '1d': '1d',
                '1w': '1wk',
                '1m': '1mo'
            }
            
            interval = interval_map.get(timeframe, '1d')
            
            # Build URL
            params = {
                'symbol': symbol,
                'interval': interval,
                'period1': int(start_date.timestamp()),
                'period2': int(end_date.timestamp()),
                'includePrePost': extended_hours
            }
            
            url = f"{self.base_url}/{symbol}"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_yahoo_response(data)
                else:
                    logger.error(f"Yahoo Finance API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting Yahoo Finance data: {str(e)}")
            return []
    
    def _parse_yahoo_response(self, data: Dict) -> List[Dict]:
        """Parse Yahoo Finance API response"""
        try:
            chart_data = data.get('chart', {}).get('result', [])
            
            if not chart_data:
                return []
            
            result = chart_data[0]
            timestamps = result.get('timestamp', [])
            indicators = result.get('indicators', {})
            quote = indicators.get('quote', [{}])[0]
            
            bars = []
            
            for i, timestamp in enumerate(timestamps):
                try:
                    bar = {
                        'timestamp': datetime.fromtimestamp(timestamp),
                        'open': quote.get('open', [])[i],
                        'high': quote.get('high', [])[i],
                        'low': quote.get('low', [])[i],
                        'close': quote.get('close', [])[i],
                        'volume': quote.get('volume', [])[i] or 0
                    }
                    
                    # Skip bars with missing data
                    if None not in [bar['open'], bar['high'], bar['low'], bar['close']]:
                        bars.append(bar)
                        
                except (IndexError, TypeError):
                    continue
            
            return bars
            
        except Exception as e:
            logger.error(f"Error parsing Yahoo response: {str(e)}")
            return []
    
    async def get_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get quotes from Yahoo Finance"""
        try:
            # Yahoo Finance doesn't provide real-time quotes in the same way
            # This would need to be implemented differently
            return {}
            
        except Exception as e:
            logger.error(f"Error getting Yahoo quotes: {str(e)}")
            return {}
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
