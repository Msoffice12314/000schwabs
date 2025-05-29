"""
Schwab API Market Data Client
Handles real-time quotes, historical data, options chains, and market fundamentals
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import json
from dataclasses import dataclass
from enum import Enum

from schwab_api.auth_manager import auth_manager, get_auth_headers
from schwab_api.rate_limiter import RateLimiter
from config.settings import settings
from utils.cache_manager import cache_manager

logger = logging.getLogger(__name__)

class PeriodType(Enum):
    """Period types for historical data"""
    DAY = "day"
    MONTH = "month"
    YEAR = "year"
    YTD = "ytd"

class FrequencyType(Enum):
    """Frequency types for historical data"""
    MINUTE = "minute"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class MarketType(Enum):
    """Market types"""
    EQUITY = "equity"
    OPTION = "option"
    MUTUAL_FUND = "mutual_fund"
    BOND = "bond"
    ETF = "etf"

@dataclass
class Quote:
    """Stock quote data structure"""
    symbol: str
    description: str
    bid_price: float
    ask_price: float
    last_price: float
    high_price: float
    low_price: float
    open_price: float
    close_price: float
    volume: int
    quote_time: datetime
    trade_time: datetime
    net_change: float
    net_percent_change: float
    mark: float
    mark_change: float
    mark_percent_change: float
    volatility: Optional[float] = None
    implied_volatility: Optional[float] = None
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'Quote':
        """Create Quote from Schwab API response"""
        return cls(
            symbol=data.get('symbol', ''),
            description=data.get('description', ''),
            bid_price=data.get('bidPrice', 0.0),
            ask_price=data.get('askPrice', 0.0),
            last_price=data.get('lastPrice', 0.0),
            high_price=data.get('highPrice', 0.0),
            low_price=data.get('lowPrice', 0.0),
            open_price=data.get('openPrice', 0.0),
            close_price=data.get('closePrice', 0.0),
            volume=data.get('totalVolume', 0),
            quote_time=datetime.fromtimestamp(data.get('quoteTimeInLong', 0) / 1000, tz=timezone.utc),
            trade_time=datetime.fromtimestamp(data.get('tradeTimeInLong', 0) / 1000, tz=timezone.utc),
            net_change=data.get('netChange', 0.0),
            net_percent_change=data.get('netPercentChangeInDouble', 0.0),
            mark=data.get('mark', 0.0),
            mark_change=data.get('markChange', 0.0),
            mark_percent_change=data.get('markPercentChange', 0.0),
            volatility=data.get('volatility'),
            implied_volatility=data.get('impliedVolatility')
        )

@dataclass
class HistoricalCandle:
    """Historical price candle data"""
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'HistoricalCandle':
        """Create HistoricalCandle from Schwab API response"""
        return cls(
            datetime=datetime.fromtimestamp(data.get('datetime', 0) / 1000, tz=timezone.utc),
            open=data.get('open', 0.0),
            high=data.get('high', 0.0),
            low=data.get('low', 0.0),
            close=data.get('close', 0.0),
            volume=data.get('volume', 0)
        )

class SchwabMarketDataClient:
    """Schwab Market Data API Client"""
    
    def __init__(self):
        self.base_url = f"{settings.schwab_api.base_url}/marketdata/v1"
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(
            calls_per_second=settings.schwab_api.rate_limit_per_second,
            calls_per_minute=settings.schwab_api.rate_limit_per_second * 60
        )
        
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'SchwabAI/1.0'
        })
        
        logger.info("Schwab Market Data Client initialized")
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make authenticated API request with rate limiting"""
        self.rate_limiter.wait_if_needed()
        
        headers = get_auth_headers()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.get(url, headers=headers, params=params, 
                                      timeout=settings.schwab_api.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {endpoint}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")
            raise
    
    def get_quote(self, symbol: str, use_cache: bool = True) -> Quote:
        """
        Get real-time quote for a symbol
        
        Args:
            symbol: Stock symbol
            use_cache: Whether to use cached data if available
            
        Returns:
            Quote object with current market data
        """
        cache_key = f"quote_{symbol.upper()}" if use_cache else None
        
        if use_cache:
            cached_quote = cache_manager.get(cache_key)
            if cached_quote:
                return cached_quote
        
        try:
            data = self._make_request(f"quotes/{symbol.upper()}")
            
            if symbol.upper() in data:
                quote = Quote.from_api_response(data[symbol.upper()])
                
                if use_cache:
                    cache_manager.set(cache_key, quote, ttl=30)  # Cache for 30 seconds
                
                return quote
            else:
                raise ValueError(f"No quote data found for symbol {symbol}")
                
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            raise
    
    def get_quotes(self, symbols: List[str], use_cache: bool = True) -> Dict[str, Quote]:
        """
        Get real-time quotes for multiple symbols
        
        Args:
            symbols: List of stock symbols
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary mapping symbols to Quote objects
        """
        symbols_upper = [s.upper() for s in symbols]
        symbols_param = ",".join(symbols_upper)
        
        try:
            data = self._make_request("quotes", params={"symbols": symbols_param})
            
            quotes = {}
            for symbol in symbols_upper:
                if symbol in data:
                    quote = Quote.from_api_response(data[symbol])
                    quotes[symbol] = quote
                    
                    if use_cache:
                        cache_key = f"quote_{symbol}"
                        cache_manager.set(cache_key, quote, ttl=30)
            
            return quotes
            
        except Exception as e:
            logger.error(f"Failed to get quotes for {symbols}: {e}")
            raise
    
    def get_price_history(self, symbol: str, period_type: PeriodType = PeriodType.YEAR,
                         period: int = 1, frequency_type: FrequencyType = FrequencyType.DAILY,
                         frequency: int = 1, start_date: datetime = None,
                         end_date: datetime = None, use_cache: bool = True) -> pd.DataFrame:
        """
        Get historical price data for a symbol
        
        Args:
            symbol: Stock symbol
            period_type: Type of period (day, month, year, ytd)
            period: Number of periods
            frequency_type: Frequency type (minute, daily, weekly, monthly)
            frequency: Frequency value
            start_date: Start date for historical data
            end_date: End date for historical data
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with historical price data
        """
        cache_key = None
        if use_cache and not start_date and not end_date:
            cache_key = f"history_{symbol}_{period_type.value}_{period}_{frequency_type.value}_{frequency}"
            cached_data = cache_manager.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        params = {
            'periodType': period_type.value,
            'period': period,
            'frequencyType': frequency_type.value,
            'frequency': frequency
        }
        
        if start_date:
            params['startDate'] = int(start_date.timestamp() * 1000)
        if end_date:
            params['endDate'] = int(end_date.timestamp() * 1000)
        
        try:
            data = self._make_request(f"pricehistory/{symbol.upper()}", params=params)
            
            if 'candles' not in data or not data['candles']:
                return pd.DataFrame()
            
            candles = [HistoricalCandle.from_api_response(candle_data) 
                      for candle_data in data['candles']]
            
            df = pd.DataFrame([
                {
                    'datetime': candle.datetime,
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close,
                    'volume': candle.volume
                }
                for candle in candles
            ])
            
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            
            if use_cache and cache_key:
                cache_manager.set(cache_key, df, ttl=3600)  # Cache for 1 hour
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get price history for {symbol}: {e}")
            raise
    
    def get_option_chain(self, symbol: str, strike_count: int = 10,
                        include_quotes: bool = True, strategy: str = "SINGLE",
                        from_date: datetime = None, to_date: datetime = None) -> Dict[str, Any]:
        """
        Get option chain data for a symbol
        
        Args:
            symbol: Underlying stock symbol
            strike_count: Number of strikes to include
            include_quotes: Whether to include underlying quotes
            strategy: Option strategy (SINGLE, ANALYTICAL, COVERED, VERTICAL, etc.)
            from_date: Start date for option expiration
            to_date: End date for option expiration
            
        Returns:
            Option chain data
        """
        params = {
            'symbol': symbol.upper(),
            'strikeCount': strike_count,
            'includeQuotes': str(include_quotes).lower(),
            'strategy': strategy
        }
        
        if from_date:
            params['fromDate'] = from_date.strftime('%Y-%m-%d')
        if to_date:
            params['toDate'] = to_date.strftime('%Y-%m-%d')
        
        try:
            data = self._make_request("chains", params=params)
            return data
            
        except Exception as e:
            logger.error(f"Failed to get option chain for {symbol}: {e}")
            raise
    
    def get_market_hours(self, markets: List[str] = None, date: datetime = None) -> Dict[str, Any]:
        """
        Get market hours for specified markets
        
        Args:
            markets: List of markets (equity, option, bond, etc.)
            date: Date to get market hours for
            
        Returns:
            Market hours data
        """
        if markets is None:
            markets = ['equity', 'option']
        
        params = {
            'markets': ','.join(markets)
        }
        
        if date:
            params['date'] = date.strftime('%Y-%m-%d')
        
        try:
            data = self._make_request("markets", params=params)
            return data
            
        except Exception as e:
            logger.error(f"Failed to get market hours: {e}")
            raise
    
    def search_instruments(self, symbol: str, projection: str = "symbol-search") -> List[Dict[str, Any]]:
        """
        Search for instruments by symbol or description
        
        Args:
            symbol: Symbol or description to search for
            projection: Type of search (symbol-search, symbol-regex, desc-search, desc-regex, fundamental)
            
        Returns:
            List of matching instruments
        """
        params = {
            'symbol': symbol,
            'projection': projection
        }
        
        try:
            data = self._make_request("instruments", params=params)
            
            instruments = []
            for key, value in data.items():
                if isinstance(value, dict):
                    instruments.append(value)
                elif isinstance(value, list):
                    instruments.extend(value)
            
            return instruments
            
        except Exception as e:
            logger.error(f"Failed to search instruments for {symbol}: {e}")
            raise
    
    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Get fundamental data for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Fundamental data
        """
        try:
            instruments = self.search_instruments(symbol, projection="fundamental")
            
            if instruments:
                return instruments[0].get('fundamental', {})
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get fundamentals for {symbol}: {e}")
            raise
    
    async def get_quotes_async(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Asynchronously get quotes for multiple symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to Quote objects
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for symbol in symbols:
                task = self._get_quote_async(session, symbol)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            quotes = {}
            for i, result in enumerate(results):
                if isinstance(result, Quote):
                    quotes[symbols[i]] = result
                else:
                    logger.error(f"Failed to get quote for {symbols[i]}: {result}")
            
            return quotes
    
    async def _get_quote_async(self, session: aiohttp.ClientSession, symbol: str) -> Quote:
        """Async helper to get single quote"""
        await self.rate_limiter.wait_if_needed_async()
        
        headers = get_auth_headers()
        url = f"{self.base_url}/quotes/{symbol.upper()}"
        
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            
            if symbol.upper() in data:
                return Quote.from_api_response(data[symbol.upper()])
            else:
                raise ValueError(f"No quote data found for symbol {symbol}")
    
    def get_bulk_historical_data(self, symbols: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols in parallel
        
        Args:
            symbols: List of stock symbols
            **kwargs: Arguments to pass to get_price_history
            
        Returns:
            Dictionary mapping symbols to historical DataFrames
        """
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self.get_price_history, symbol, **kwargs): symbol
                for symbol in symbols
            }
            
            results = {}
            for future in futures:
                symbol = futures[future]
                try:
                    df = future.result()
                    results[symbol] = df
                except Exception as e:
                    logger.error(f"Failed to get historical data for {symbol}: {e}")
                    results[symbol] = pd.DataFrame()
            
            return results
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for price data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Simple moving averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price change indicators
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        df['volatility'] = df['price_change'].rolling(window=20).std()
        
        return df
    
    def is_market_open(self) -> bool:
        """Check if the market is currently open"""
        try:
            market_hours = self.get_market_hours(['equity'])
            
            if 'equity' in market_hours and 'EQ' in market_hours['equity']:
                eq_hours = market_hours['equity']['EQ']
                
                if 'sessionHours' in eq_hours:
                    session_hours = eq_hours['sessionHours']
                    if 'regularMarket' in session_hours:
                        regular_market = session_hours['regularMarket'][0]
                        
                        start_time = datetime.fromisoformat(regular_market['start'].replace('Z', '+00:00'))
                        end_time = datetime.fromisoformat(regular_market['end'].replace('Z', '+00:00'))
                        
                        now = datetime.now(timezone.utc)
                        return start_time <= now <= end_time
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check market hours: {e}")
            # Fallback to trading hours from settings
            return settings.is_trading_hours()

# Global market data client instance
market_data_client = SchwabMarketDataClient()

# Convenience functions
def get_quote(symbol: str, use_cache: bool = True) -> Quote:
    """Get real-time quote for a symbol"""
    return market_data_client.get_quote(symbol, use_cache)

def get_quotes(symbols: List[str], use_cache: bool = True) -> Dict[str, Quote]:
    """Get real-time quotes for multiple symbols"""
    return market_data_client.get_quotes(symbols, use_cache)

def get_price_history(symbol: str, days: int = 365, frequency: str = 'daily') -> pd.DataFrame:
    """Get historical price data (convenience function)"""
    freq_type = FrequencyType.DAILY if frequency.lower() == 'daily' else FrequencyType.MINUTE
    return market_data_client.get_price_history(
        symbol=symbol,
        period_type=PeriodType.YEAR,
        period=1 if days <= 365 else days // 365,
        frequency_type=freq_type,
        frequency=1
    )

def get_historical_data_with_indicators(symbol: str, days: int = 365) -> pd.DataFrame:
    """Get historical data with technical indicators"""
    df = get_price_history(symbol, days)
    return market_data_client.calculate_technical_indicators(df)

async def get_quotes_async(symbols: List[str]) -> Dict[str, Quote]:
    """Asynchronously get quotes for multiple symbols"""
    return await market_data_client.get_quotes_async(symbols)

def is_market_open() -> bool:
    """Check if the market is currently open"""
    return market_data_client.is_market_open()

# Export key classes and functions
__all__ = [
    'SchwabMarketDataClient',
    'Quote',
    'HistoricalCandle',
    'PeriodType',
    'FrequencyType',
    'MarketType',
    'market_data_client',
    'get_quote',
    'get_quotes',
    'get_price_history',
    'get_historical_data_with_indicators',
    'get_quotes_async',
    'is_market_open'
]