#!/usr/bin/env python3
"""
Data Collector Daemon for Schwab AI Trading System
Continuously collects market data, news, and other relevant information for trading decisions
"""

import os
import sys
import time
import signal
import threading
import asyncio
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import requests
from dataclasses import dataclass, asdict

import schedule
import redis
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import websockets

from config import Config
from database import Database
from api.schwab_api import SchwabAPI
from utils.logger import setup_logger
from utils.market_calendar import MarketCalendar
from utils.rate_limiter import RateLimiter
from utils.data_validator import DataValidator
from utils.websocket_manager import WebSocketManager

@dataclass
class MarketDataPoint:
    """Data structure for market data points"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None

@dataclass
class NewsItem:
    """Data structure for news items"""
    title: str
    content: str
    source: str
    published_at: datetime
    symbols: List[str]
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    url: Optional[str] = None

class DataCollectorDaemon:
    """
    Background daemon for collecting and processing financial data
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the data collector daemon"""
        self.config = Config(config_path)
        self.logger = setup_logger('data_collector', self.config.LOG_LEVEL)
        
        # Initialize components
        self.db = Database(self.config.DATABASE_URL)
        self.schwab_api = SchwabAPI(self.config)
        self.redis_client = redis.Redis.from_url(self.config.REDIS_URL)
        self.market_calendar = MarketCalendar()
        self.rate_limiter = RateLimiter(self.config.SCHWAB_API_RATE_LIMIT, 60)
        self.data_validator = DataValidator()
        self.websocket_manager = WebSocketManager()
        
        # Runtime state
        self.is_running = False
        self.is_market_hours = False
        self.subscribed_symbols = set()
        self.data_threads = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Data collection settings
        self.symbols_to_track = self.config.DEFAULT_SYMBOLS or ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        self.market_data_interval = self.config.MARKET_DATA_UPDATE_INTERVAL or 1  # seconds
        self.news_update_interval = self.config.NEWS_UPDATE_INTERVAL or 300  # seconds
        self.economic_data_interval = 3600  # 1 hour
        
        # Performance tracking
        self.stats = {
            'market_data_points': 0,
            'news_items': 0,
            'errors': 0,
            'last_update': None,
            'uptime_start': datetime.now()
        }
        
        self.logger.info("DataCollectorDaemon initialized successfully")

    def start(self):
        """Start the data collection daemon"""
        self.logger.info("Starting Data Collector Daemon...")
        
        try:
            self.is_running = True
            
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Initialize data sources
            self._initialize_data_sources()
            
            # Start scheduled tasks
            self._setup_schedules()
            
            # Start main collection loop
            self._run_collection_loop()
            
        except Exception as e:
            self.logger.error(f"Failed to start daemon: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def stop(self):
        """Stop the data collection daemon"""
        self.logger.info("Stopping Data Collector Daemon...")
        
        self.is_running = False
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Close connections
        self._cleanup_connections()
        
        self.logger.info("Data Collector Daemon stopped")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    def _initialize_data_sources(self):
        """Initialize all data sources"""
        self.logger.info("Initializing data sources...")
        
        try:
            # Test Schwab API connection
            if self.schwab_api.test_connection():
                self.logger.info("Schwab API connection verified")
            else:
                self.logger.warning("Schwab API connection failed")
            
            # Initialize Redis connection
            self.redis_client.ping()
            self.logger.info("Redis connection verified")
            
            # Initialize database connection
            self.db.test_connection()
            self.logger.info("Database connection verified")
            
            # Load symbols from database
            self._load_tracked_symbols()
            
            # Initialize WebSocket connections
            self._initialize_websockets()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data sources: {str(e)}")
            raise

    def _load_tracked_symbols(self):
        """Load symbols to track from database"""
        try:
            # Get symbols from portfolio positions
            portfolio_symbols = self.db.get_portfolio_symbols()
            
            # Get symbols from watchlist
            watchlist_symbols = self.db.get_watchlist_symbols()
            
            # Combine with default symbols
            all_symbols = set(self.symbols_to_track)
            all_symbols.update(portfolio_symbols)
            all_symbols.update(watchlist_symbols)
            
            self.subscribed_symbols = all_symbols
            self.logger.info(f"Tracking {len(self.subscribed_symbols)} symbols: {sorted(list(self.subscribed_symbols))}")
            
        except Exception as e:
            self.logger.error(f"Failed to load tracked symbols: {str(e)}")
            # Fall back to default symbols
            self.subscribed_symbols = set(self.symbols_to_track)

    def _initialize_websockets(self):
        """Initialize WebSocket connections for real-time data"""
        try:
            if self.config.ENABLE_WEBSOCKET:
                # Start WebSocket manager
                self.websocket_manager.start()
                
                # Subscribe to symbols
                for symbol in self.subscribed_symbols:
                    self.websocket_manager.subscribe_symbol(symbol)
                
                self.logger.info("WebSocket connections initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize WebSockets: {str(e)}")

    def _setup_schedules(self):
        """Setup scheduled tasks"""
        # Market data collection (every second during market hours)
        schedule.every(self.market_data_interval).seconds.do(self._collect_market_data_batch)
        
        # News collection (every 5 minutes)
        schedule.every(self.news_update_interval).seconds.do(self._collect_news_data)
        
        # Economic data collection (hourly)
        schedule.every(self.economic_data_interval).seconds.do(self._collect_economic_data)
        
        # Portfolio updates (every minute during market hours)
        schedule.every(60).seconds.do(self._update_portfolio_data)
        
        # System health check (every 5 minutes)
        schedule.every(300).seconds.do(self._health_check)
        
        # Data cleanup (daily at 2 AM)
        schedule.every().day.at("02:00").do(self._cleanup_old_data)
        
        # Symbol list refresh (every hour)
        schedule.every().hour.do(self._refresh_symbol_list)
        
        self.logger.info("Scheduled tasks configured")

    def _run_collection_loop(self):
        """Main collection loop"""
        self.logger.info("Starting main collection loop...")
        
        while self.is_running:
            try:
                # Check market hours
                self.is_market_hours = self.market_calendar.is_market_open()
                
                # Run scheduled tasks
                schedule.run_pending()
                
                # Process WebSocket data if available
                self._process_websocket_data()
                
                # Update statistics
                self.stats['last_update'] = datetime.now()
                
                # Sleep briefly to prevent excessive CPU usage
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt")
                break
            except Exception as e:
                self.logger.error(f"Error in collection loop: {str(e)}")
                self.stats['errors'] += 1
                time.sleep(1)  # Brief pause before continuing

    def _collect_market_data_batch(self):
        """Collect market data for all symbols in batch"""
        if not self.is_market_hours and not self.config.COLLECT_AFTER_HOURS:
            return
        
        try:
            self.logger.debug("Collecting market data batch...")
            
            # Split symbols into chunks to respect rate limits
            symbol_chunks = list(self._chunk_symbols(self.subscribed_symbols, 50))
            
            # Collect data for each chunk
            futures = []
            for chunk in symbol_chunks:
                future = self.executor.submit(self._collect_market_data_chunk, chunk)
                futures.append(future)
            
            # Process results
            total_points = 0
            for future in as_completed(futures):
                try:
                    points = future.result()
                    total_points += points
                except Exception as e:
                    self.logger.error(f"Market data collection failed: {str(e)}")
                    self.stats['errors'] += 1
            
            self.stats['market_data_points'] += total_points
            
            if total_points > 0:
                self.logger.debug(f"Collected {total_points} market data points")
            
        except Exception as e:
            self.logger.error(f"Error in batch market data collection: {str(e)}")
            self.stats['errors'] += 1

    def _collect_market_data_chunk(self, symbols: List[str]) -> int:
        """Collect market data for a chunk of symbols"""
        try:
            # Apply rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Get quotes from Schwab API
            quotes = self.schwab_api.get_quotes(symbols)
            
            if not quotes:
                return 0
            
            data_points = []
            timestamp = datetime.now()
            
            for symbol, quote in quotes.items():
                try:
                    # Validate data
                    if not self.data_validator.validate_market_data(quote):
                        self.logger.warning(f"Invalid market data for {symbol}")
                        continue
                    
                    # Create data point
                    data_point = MarketDataPoint(
                        symbol=symbol,
                        timestamp=timestamp,
                        open_price=float(quote.get('openPrice', 0)),
                        high_price=float(quote.get('highPrice', 0)),
                        low_price=float(quote.get('lowPrice', 0)),
                        close_price=float(quote.get('lastPrice', 0)),
                        volume=int(quote.get('totalVolume', 0)),
                        bid=float(quote.get('bidPrice', 0)) or None,
                        ask=float(quote.get('askPrice', 0)) or None,
                        bid_size=int(quote.get('bidSize', 0)) or None,
                        ask_size=int(quote.get('askSize', 0)) or None
                    )
                    
                    data_points.append(data_point)
                    
                except Exception as e:
                    self.logger.error(f"Error processing quote for {symbol}: {str(e)}")
            
            # Store data points in database
            if data_points:
                self._store_market_data(data_points)
                
                # Cache latest data in Redis
                self._cache_market_data(data_points)
                
                # Broadcast to WebSocket clients
                self._broadcast_market_data(data_points)
            
            return len(data_points)
            
        except Exception as e:
            self.logger.error(f"Error collecting market data chunk: {str(e)}")
            return 0

    def _collect_news_data(self):
        """Collect news data"""
        try:
            self.logger.debug("Collecting news data...")
            
            # Get news from various sources
            news_items = []
            
            # Collect from different news sources
            sources = [
                self._collect_schwab_news,
                self._collect_alpha_vantage_news,
                self._collect_polygon_news
            ]
            
            for source_func in sources:
                try:
                    items = source_func()
                    news_items.extend(items)
                except Exception as e:
                    self.logger.error(f"Error collecting from news source: {str(e)}")
            
            # Process and store news items
            if news_items:
                processed_items = self._process_news_items(news_items)
                self._store_news_data(processed_items)
                self.stats['news_items'] += len(processed_items)
                
                self.logger.debug(f"Collected {len(processed_items)} news items")
            
        except Exception as e:
            self.logger.error(f"Error collecting news data: {str(e)}")
            self.stats['errors'] += 1

    def _collect_schwab_news(self) -> List[NewsItem]:
        """Collect news from Schwab API"""
        try:
            news_items = []
            
            for symbol in self.subscribed_symbols:
                news_data = self.schwab_api.get_news(symbol)
                
                for item in news_data:
                    news_item = NewsItem(
                        title=item.get('headline', ''),
                        content=item.get('story', ''),
                        source='Schwab',
                        published_at=datetime.fromisoformat(item.get('datetime', '')),
                        symbols=[symbol],
                        url=item.get('storyUrl')
                    )
                    news_items.append(news_item)
            
            return news_items
            
        except Exception as e:
            self.logger.error(f"Error collecting Schwab news: {str(e)}")
            return []

    def _collect_alpha_vantage_news(self) -> List[NewsItem]:
        """Collect news from Alpha Vantage"""
        if not self.config.ALPHA_VANTAGE_API_KEY:
            return []
        
        try:
            news_items = []
            
            # Alpha Vantage news API call
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ','.join(list(self.subscribed_symbols)[:10]),  # Limit to 10 symbols
                'apikey': self.config.ALPHA_VANTAGE_API_KEY,
                'limit': 50
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'feed' in data:
                for item in data['feed']:
                    news_item = NewsItem(
                        title=item.get('title', ''),
                        content=item.get('summary', ''),
                        source='Alpha Vantage',
                        published_at=datetime.strptime(item.get('time_published', ''), '%Y%m%dT%H%M%S'),
                        symbols=[ticker['ticker'] for ticker in item.get('ticker_sentiment', [])],
                        sentiment_score=float(item.get('overall_sentiment_score', 0)),
                        url=item.get('url')
                    )
                    news_items.append(news_item)
            
            return news_items
            
        except Exception as e:
            self.logger.error(f"Error collecting Alpha Vantage news: {str(e)}")
            return []

    def _collect_polygon_news(self) -> List[NewsItem]:
        """Collect news from Polygon.io"""
        if not self.config.POLYGON_API_KEY:
            return []
        
        try:
            news_items = []
            
            # Polygon news API call
            url = "https://api.polygon.io/v2/reference/news"
            params = {
                'ticker.gte': ','.join(list(self.subscribed_symbols)[:10]),
                'published_utc.gte': (datetime.now() - timedelta(hours=24)).isoformat(),
                'order': 'desc',
                'limit': 100,
                'apikey': self.config.POLYGON_API_KEY
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'results' in data:
                for item in data['results']:
                    news_item = NewsItem(
                        title=item.get('title', ''),
                        content=item.get('description', ''),
                        source='Polygon',
                        published_at=datetime.fromisoformat(item.get('published_utc', '')),
                        symbols=item.get('tickers', []),
                        url=item.get('article_url')
                    )
                    news_items.append(news_item)
            
            return news_items
            
        except Exception as e:
            self.logger.error(f"Error collecting Polygon news: {str(e)}")
            return []

    def _collect_economic_data(self):
        """Collect economic indicators and market data"""
        try:
            self.logger.debug("Collecting economic data...")
            
            # Collect various economic indicators
            economic_data = {}
            
            # VIX (Volatility Index)
            vix_data = self._get_market_indicator('VIX')
            if vix_data:
                economic_data['VIX'] = vix_data
            
            # Treasury rates
            treasury_data = self._get_treasury_rates()
            if treasury_data:
                economic_data.update(treasury_data)
            
            # Market indices
            indices = ['SPY', 'QQQ', 'IWM', 'DIA']
            for index in indices:
                index_data = self._get_market_indicator(index)
                if index_data:
                    economic_data[index] = index_data
            
            # Store economic data
            if economic_data:
                self._store_economic_data(economic_data)
                self.logger.debug(f"Collected {len(economic_data)} economic indicators")
            
        except Exception as e:
            self.logger.error(f"Error collecting economic data: {str(e)}")
            self.stats['errors'] += 1

    def _get_market_indicator(self, symbol: str) -> Optional[Dict]:
        """Get market indicator data"""
        try:
            quote = self.schwab_api.get_quote(symbol)
            if quote:
                return {
                    'symbol': symbol,
                    'price': float(quote.get('lastPrice', 0)),
                    'change': float(quote.get('netChange', 0)),
                    'changePercent': float(quote.get('netPercentChangeInDouble', 0)),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            self.logger.error(f"Error getting indicator for {symbol}: {str(e)}")
        return None

    def _get_treasury_rates(self) -> Dict[str, Any]:
        """Get treasury rates from various sources"""
        try:
            # This would integrate with FRED API or similar
            # Placeholder implementation
            return {
                'TNX': {'rate': 4.5, 'timestamp': datetime.now()},  # 10-year treasury
                'FVX': {'rate': 4.2, 'timestamp': datetime.now()},  # 5-year treasury
                'IRX': {'rate': 5.1, 'timestamp': datetime.now()}   # 3-month treasury
            }
        except Exception as e:
            self.logger.error(f"Error getting treasury rates: {str(e)}")
            return {}

    def _update_portfolio_data(self):
        """Update portfolio positions and account data"""
        if not self.is_market_hours:
            return
        
        try:
            self.logger.debug("Updating portfolio data...")
            
            # Get account information
            account_info = self.schwab_api.get_account_info()
            if account_info:
                self._store_account_data(account_info)
            
            # Get positions
            positions = self.schwab_api.get_positions()
            if positions:
                self._store_position_data(positions)
            
            # Get orders
            orders = self.schwab_api.get_orders()
            if orders:
                self._store_order_data(orders)
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio data: {str(e)}")
            self.stats['errors'] += 1

    def _process_news_items(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """Process and enrich news items"""
        processed_items = []
        
        for item in news_items:
            try:
                # Skip duplicates
                if self._is_duplicate_news(item):
                    continue
                
                # Enhance with sentiment analysis
                if not item.sentiment_score and self.config.ENABLE_NEWS_SENTIMENT:
                    item.sentiment_score = self._analyze_sentiment(item.content)
                
                # Calculate relevance score
                item.relevance_score = self._calculate_relevance(item)
                
                processed_items.append(item)
                
            except Exception as e:
                self.logger.error(f"Error processing news item: {str(e)}")
        
        return processed_items

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text (placeholder implementation)"""
        # This would integrate with a sentiment analysis service
        # For now, return a random sentiment score
        import random
        return random.uniform(-1.0, 1.0)

    def _calculate_relevance(self, news_item: NewsItem) -> float:
        """Calculate relevance score for news item"""
        relevance = 0.5  # Base relevance
        
        # Boost relevance if symbols are in our tracked list
        matching_symbols = set(news_item.symbols) & self.subscribed_symbols
        if matching_symbols:
            relevance += 0.3 * len(matching_symbols) / len(news_item.symbols)
        
        # Boost relevance for recent news
        hours_old = (datetime.now() - news_item.published_at).total_seconds() / 3600
        if hours_old < 1:
            relevance += 0.2
        elif hours_old < 6:
            relevance += 0.1
        
        return min(relevance, 1.0)

    def _is_duplicate_news(self, news_item: NewsItem) -> bool:
        """Check if news item is a duplicate"""
        try:
            # Check in Redis cache
            cache_key = f"news:{hash(news_item.title)}"
            return self.redis_client.exists(cache_key)
        except Exception:
            return False

    def _store_market_data(self, data_points: List[MarketDataPoint]):
        """Store market data in database"""
        try:
            records = [asdict(point) for point in data_points]
            self.db.insert_market_data_batch(records)
        except Exception as e:
            self.logger.error(f"Error storing market data: {str(e)}")

    def _store_news_data(self, news_items: List[NewsItem]):
        """Store news data in database"""
        try:
            records = [asdict(item) for item in news_items]
            self.db.insert_news_data_batch(records)
            
            # Cache in Redis to prevent duplicates
            for item in news_items:
                cache_key = f"news:{hash(item.title)}"
                self.redis_client.setex(cache_key, 86400, "1")  # 24 hour expiry
                
        except Exception as e:
            self.logger.error(f"Error storing news data: {str(e)}")

    def _store_economic_data(self, economic_data: Dict[str, Any]):
        """Store economic data in database"""
        try:
            self.db.insert_economic_data(economic_data)
        except Exception as e:
            self.logger.error(f"Error storing economic data: {str(e)}")

    def _store_account_data(self, account_info: Dict[str, Any]):
        """Store account information"""
        try:
            self.db.update_account_info(account_info)
        except Exception as e:
            self.logger.error(f"Error storing account data: {str(e)}")

    def _store_position_data(self, positions: List[Dict[str, Any]]):
        """Store position data"""
        try:
            self.db.update_positions(positions)
        except Exception as e:
            self.logger.error(f"Error storing position data: {str(e)}")

    def _store_order_data(self, orders: List[Dict[str, Any]]):
        """Store order data"""
        try:
            self.db.update_orders(orders)
        except Exception as e:
            self.logger.error(f"Error storing order data: {str(e)}")

    def _cache_market_data(self, data_points: List[MarketDataPoint]):
        """Cache latest market data in Redis"""
        try:
            pipe = self.redis_client.pipeline()
            
            for point in data_points:
                cache_key = f"quote:{point.symbol}"
                data = asdict(point)
                data['timestamp'] = data['timestamp'].isoformat()
                pipe.setex(cache_key, 60, json.dumps(data))  # 1 minute expiry
            
            pipe.execute()
            
        except Exception as e:
            self.logger.error(f"Error caching market data: {str(e)}")

    def _broadcast_market_data(self, data_points: List[MarketDataPoint]):
        """Broadcast market data to WebSocket clients"""
        try:
            if self.websocket_manager:
                for point in data_points:
                    message = {
                        'type': 'market_data',
                        'symbol': point.symbol,
                        'price': point.close_price,
                        'change': point.close_price - point.open_price,
                        'volume': point.volume,
                        'timestamp': point.timestamp.isoformat()
                    }
                    self.websocket_manager.broadcast(message)
                    
        except Exception as e:
            self.logger.error(f"Error broadcasting market data: {str(e)}")

    def _process_websocket_data(self):
        """Process incoming WebSocket data"""
        try:
            if self.websocket_manager:
                messages = self.websocket_manager.get_pending_messages()
                for message in messages:
                    self._handle_websocket_message(message)
        except Exception as e:
            self.logger.error(f"Error processing WebSocket data: {str(e)}")

    def _handle_websocket_message(self, message: Dict[str, Any]):
        """Handle individual WebSocket message"""
        try:
            msg_type = message.get('type')
            
            if msg_type == 'quote':
                # Handle real-time quote
                self._process_realtime_quote(message)
            elif msg_type == 'trade':
                # Handle trade execution
                self._process_trade_message(message)
            elif msg_type == 'level2':
                # Handle level 2 market data
                self._process_level2_data(message)
            
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {str(e)}")

    def _chunk_symbols(self, symbols: Set[str], chunk_size: int):
        """Split symbols into chunks"""
        symbols_list = list(symbols)
        for i in range(0, len(symbols_list), chunk_size):
            yield symbols_list[i:i + chunk_size]

    def _health_check(self):
        """Perform system health check"""
        try:
            health_status = {
                'timestamp': datetime.now(),
                'uptime': (datetime.now() - self.stats['uptime_start']).total_seconds(),
                'market_data_points': self.stats['market_data_points'],
                'news_items': self.stats['news_items'],
                'errors': self.stats['errors'],
                'memory_usage': self._get_memory_usage(),
                'database_status': self._check_database_health(),
                'redis_status': self._check_redis_health(),
                'api_status': self._check_api_health()
            }
            
            # Store health status
            self.redis_client.setex('health_status', 300, json.dumps(health_status, default=str))
            
            # Log warnings if needed
            if health_status['errors'] > 100:
                self.logger.warning(f"High error count: {health_status['errors']}")
            
            if health_status['memory_usage'] > 1000:  # MB
                self.logger.warning(f"High memory usage: {health_status['memory_usage']} MB")
            
        except Exception as e:
            self.logger.error(f"Error in health check: {str(e)}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _check_database_health(self) -> bool:
        """Check database connectivity"""
        try:
            return self.db.test_connection()
        except Exception:
            return False

    def _check_redis_health(self) -> bool:
        """Check Redis connectivity"""
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False

    def _check_api_health(self) -> bool:
        """Check API connectivity"""
        try:
            return self.schwab_api.test_connection()
        except Exception:
            return False

    def _cleanup_old_data(self):
        """Clean up old data from database"""
        try:
            self.logger.info("Cleaning up old data...")
            
            # Remove old market data (keep last 30 days)
            cutoff_date = datetime.now() - timedelta(days=30)
            self.db.cleanup_old_market_data(cutoff_date)
            
            # Remove old news data (keep last 7 days)
            news_cutoff = datetime.now() - timedelta(days=7)
            self.db.cleanup_old_news_data(news_cutoff)
            
            # Clean up Redis cache
            self._cleanup_redis_cache()
            
            self.logger.info("Data cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error in data cleanup: {str(e)}")

    def _cleanup_redis_cache(self):
        """Clean up expired Redis cache entries"""
        try:
            # Redis handles expiration automatically, but we can force cleanup of specific patterns
            pattern_keys = [
                'quote:*',
                'news:*',
                'temp:*'
            ]
            
            for pattern in pattern_keys:
                keys = self.redis_client.keys(pattern)
                if keys:
                    # Check TTL and remove expired keys
                    pipe = self.redis_client.pipeline()
                    for key in keys:
                        ttl = self.redis_client.ttl(key)
                        if ttl == -1:  # No expiration set
                            pipe.expire(key, 3600)  # Set 1 hour expiration
                    pipe.execute()
                    
        except Exception as e:
            self.logger.error(f"Error cleaning Redis cache: {str(e)}")

    def _refresh_symbol_list(self):
        """Refresh the list of symbols to track"""
        try:
            old_count = len(self.subscribed_symbols)
            self._load_tracked_symbols()
            new_count = len(self.subscribed_symbols)
            
            if new_count != old_count:
                self.logger.info(f"Symbol list updated: {old_count} -> {new_count} symbols")
                
                # Update WebSocket subscriptions
                if self.websocket_manager:
                    for symbol in self.subscribed_symbols:
                        self.websocket_manager.subscribe_symbol(symbol)
            
        except Exception as e:
            self.logger.error(f"Error refreshing symbol list: {str(e)}")

    def _cleanup_connections(self):
        """Clean up all connections"""
        try:
            if self.websocket_manager:
                self.websocket_manager.stop()
            
            if self.redis_client:
                self.redis_client.connection_pool.disconnect()
            
            if self.db:
                self.db.close_connections()
                
        except Exception as e:
            self.logger.error(f"Error cleaning up connections: {str(e)}")

    def get_status(self) -> Dict[str, Any]:
        """Get current daemon status"""
        return {
            'is_running': self.is_running,
            'is_market_hours': self.is_market_hours,
            'tracked_symbols': len(self.subscribed_symbols),
            'stats': self.stats.copy(),
            'uptime': (datetime.now() - self.stats['uptime_start']).total_seconds()
        }

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Collector Daemon for Schwab AI Trading')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--symbols', nargs='+', help='Symbols to track')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    
    args = parser.parse_args()
    
    try:
        # Create daemon instance
        collector = DataCollectorDaemon(args.config)
        
        # Set log level
        collector.logger.setLevel(args.log_level)
        
        # Override symbols if provided
        if args.symbols:
            collector.subscribed_symbols = set(args.symbols)
        
        # Run daemon
        if args.daemon:
            # Run as daemon process
            collector.logger.info("Starting as daemon process...")
        
        collector.start()
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        return 0
    except Exception as e:
        print(f"Failed to start data collector: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())