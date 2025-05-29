"""
Schwab AI Trading System - Feature Engineering System
Advanced technical analysis and feature engineering with 150+ indicators for AI model input.
"""

import logging
import numpy as np
import pandas as pd
import talib as ta
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
from scipy import stats
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

from utils.cache_manager import CacheManager
from config.settings import get_settings

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Advanced feature engineering system that creates comprehensive
    technical indicators and market microstructure features for AI models.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache_manager = CacheManager()
        
        # Feature categories
        self.feature_categories = {
            'price_action': True,
            'volume': True,
            'momentum': True,
            'trend': True,
            'volatility': True,
            'support_resistance': True,
            'market_structure': True,
            'statistical': True,
            'pattern': True,
            'sentiment': True
        }
        
        # Indicator parameters
        self.params = {
            'ma_periods': [5, 10, 20, 50, 100, 200],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'stoch_k': 14,
            'stoch_d': 3,
            'adx_period': 14,
            'cci_period': 20,
            'atr_period': 14,
            'ema_periods': [12, 26, 50],
            'williams_r_period': 14,
            'momentum_periods': [10, 20],
            'roc_periods': [10, 20],
            'volume_periods': [10, 20, 50]
        }
        
        # Feature list for indexing
        self.feature_names = []
        self.feature_count = 0
        
        logger.info("FeatureEngineer initialized")
    
    async def initialize(self) -> bool:
        """Initialize the feature engineer"""
        try:
            # Build feature names list
            self._build_feature_names()
            
            logger.info(f"FeatureEngineer initialized with {self.feature_count} features")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize FeatureEngineer: {str(e)}")
            return False
    
    def _build_feature_names(self):
        """Build comprehensive list of feature names"""
        self.feature_names = []
        
        if self.feature_categories['price_action']:
            # Basic OHLCV
            self.feature_names.extend(['open', 'high', 'low', 'close', 'volume'])
            
            # Price ratios and relationships
            self.feature_names.extend([
                'hl_ratio', 'oc_ratio', 'body_to_range', 'upper_shadow', 'lower_shadow',
                'typical_price', 'weighted_close'
            ])
            
            # Log returns and changes
            self.feature_names.extend([
                'log_return', 'price_change', 'price_change_pct',
                'high_low_pct', 'open_close_pct'
            ])
        
        if self.feature_categories['volume']:
            # Volume indicators
            self.feature_names.extend([
                'volume_sma_10', 'volume_sma_20', 'volume_ratio_10', 'volume_ratio_20',
                'volume_ema_10', 'volume_ema_20', 'volume_std_10', 'volume_std_20',
                'ad_line', 'obv', 'cmf', 'vwap', 'volume_profile'
            ])
        
        if self.feature_categories['momentum']:
            # Momentum oscillators
            self.feature_names.extend([
                'rsi', 'rsi_sma_5', 'rsi_ema_5',
                'stoch_k', 'stoch_d', 'stoch_rsi',
                'williams_r', 'cci', 'momentum_10', 'momentum_20',
                'roc_10', 'roc_20', 'trix', 'ultimate_oscillator'
            ])
        
        if self.feature_categories['trend']:
            # Trend indicators
            for period in self.params['ma_periods']:
                self.feature_names.extend([f'sma_{period}', f'sma_{period}_ratio'])
            
            for period in self.params['ema_periods']:
                self.feature_names.extend([f'ema_{period}', f'ema_{period}_ratio'])
            
            self.feature_names.extend([
                'macd', 'macd_signal', 'macd_histogram',
                'adx', 'plus_di', 'minus_di',
                'aroon_up', 'aroon_down', 'aroon_osc',
                'ppo', 'ppo_signal', 'ppo_histogram'
            ])
        
        if self.feature_categories['volatility']:
            # Volatility indicators
            self.feature_names.extend([
                'atr', 'atr_pct', 'natr',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
                'keltner_upper', 'keltner_middle', 'keltner_lower',
                'donchian_upper', 'donchian_lower', 'donchian_middle',
                'historical_volatility_10', 'historical_volatility_20',
                'volatility_ratio', 'true_range'
            ])
        
        if self.feature_categories['support_resistance']:
            # Support/Resistance levels
            self.feature_names.extend([
                'pivot_point', 'resistance_1', 'resistance_2', 'support_1', 'support_2',
                'fib_23_6', 'fib_38_2', 'fib_50', 'fib_61_8',
                'weekly_high', 'weekly_low', 'monthly_high', 'monthly_low'
            ])
        
        if self.feature_categories['market_structure']:
            # Market structure features
            self.feature_names.extend([
                'higher_highs', 'lower_lows', 'inside_bar', 'outside_bar',
                'doji', 'hammer', 'shooting_star', 'engulfing_bull', 'engulfing_bear',
                'gap_up', 'gap_down', 'breakout_up', 'breakout_down'
            ])
        
        if self.feature_categories['statistical']:
            # Statistical features
            self.feature_names.extend([
                'z_score_10', 'z_score_20', 'skewness_10', 'kurtosis_10',
                'correlation_spy_10', 'correlation_vix_10',
                'beta_10', 'alpha_10', 'information_ratio'
            ])
        
        if self.feature_categories['pattern']:
            # Pattern recognition
            self.feature_names.extend([
                'double_top', 'double_bottom', 'head_shoulders', 'inv_head_shoulders',
                'triangle_ascending', 'triangle_descending', 'triangle_symmetrical',
                'channel_up', 'channel_down', 'wedge_rising', 'wedge_falling'
            ])
        
        if self.feature_categories['sentiment']:
            # Sentiment proxies
            self.feature_names.extend([
                'put_call_ratio', 'vix_level', 'fear_greed_index',
                'insider_buying', 'institutional_flow'
            ])
        
        self.feature_count = len(self.feature_names)
    
    def get_feature_count(self) -> int:
        """Get total number of features"""
        return self.feature_count
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return self.feature_names.copy()
    
    async def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features from OHLCV data
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with engineered features
        """
        try:
            if data.empty or len(data) < 50:
                logger.warning("Insufficient data for feature engineering")
                return pd.DataFrame()
            
            # Make a copy to avoid modifying original data
            df = data.copy()
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing required column: {col}")
                    return pd.DataFrame()
            
            # Create feature DataFrame
            features = pd.DataFrame(index=df.index)
            
            # Add basic OHLCV if enabled
            if self.feature_categories['price_action']:
                features = await self._add_price_action_features(features, df)
            
            if self.feature_categories['volume']:
                features = await self._add_volume_features(features, df)
            
            if self.feature_categories['momentum']:
                features = await self._add_momentum_features(features, df)
            
            if self.feature_categories['trend']:
                features = await self._add_trend_features(features, df)
            
            if self.feature_categories['volatility']:
                features = await self._add_volatility_features(features, df)
            
            if self.feature_categories['support_resistance']:
                features = await self._add_support_resistance_features(features, df)
            
            if self.feature_categories['market_structure']:
                features = await self._add_market_structure_features(features, df)
            
            if self.feature_categories['statistical']:
                features = await self._add_statistical_features(features, df)
            
            if self.feature_categories['pattern']:
                features = await self._add_pattern_features(features, df)
            
            if self.feature_categories['sentiment']:
                features = await self._add_sentiment_features(features, df)
            
            # Clean features
            features = await self._clean_features(features)
            
            logger.debug(f"Created {len(features.columns)} features from {len(df)} data points")
            return features
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            return pd.DataFrame()
    
    async def _add_price_action_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add price action features"""
        try:
            # Basic OHLCV
            features['open'] = df['open']
            features['high'] = df['high']
            features['low'] = df['low']
            features['close'] = df['close']
            features['volume'] = df['volume']
            
            # Price relationships
            features['hl_ratio'] = df['high'] / df['low']
            features['oc_ratio'] = df['open'] / df['close']
            
            # Candlestick metrics
            body = abs(df['close'] - df['open'])
            range_val = df['high'] - df['low']
            features['body_to_range'] = body / range_val.replace(0, np.nan)
            features['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            features['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            
            # Price calculations
            features['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            features['weighted_close'] = (df['high'] + df['low'] + 2 * df['close']) / 4
            
            # Returns and changes
            features['log_return'] = np.log(df['close'] / df['close'].shift(1))
            features['price_change'] = df['close'] - df['close'].shift(1)
            features['price_change_pct'] = df['close'].pct_change()
            features['high_low_pct'] = (df['high'] - df['low']) / df['close']
            features['open_close_pct'] = (df['close'] - df['open']) / df['open']
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding price action features: {str(e)}")
            return features
    
    async def _add_volume_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        try:
            volume = df['volume']
            close = df['close']
            high = df['high']
            low = df['low']
            
            # Volume moving averages
            features['volume_sma_10'] = volume.rolling(10).mean()
            features['volume_sma_20'] = volume.rolling(20).mean()
            features['volume_ema_10'] = volume.ewm(span=10).mean()
            features['volume_ema_20'] = volume.ewm(span=20).mean()
            
            # Volume ratios
            features['volume_ratio_10'] = volume / features['volume_sma_10']
            features['volume_ratio_20'] = volume / features['volume_sma_20']
            
            # Volume volatility
            features['volume_std_10'] = volume.rolling(10).std()
            features['volume_std_20'] = volume.rolling(20).std()
            
            # Volume indicators using TA-Lib
            features['ad_line'] = ta.AD(high, low, close, volume)
            features['obv'] = ta.OBV(close, volume)
            
            # Chaikin Money Flow
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = mfm.fillna(0)
            mfv = mfm * volume
            features['cmf'] = mfv.rolling(20).sum() / volume.rolling(20).sum()
            
            # VWAP
            typical_price = (high + low + close) / 3
            features['vwap'] = (typical_price * volume).cumsum() / volume.cumsum()
            
            # Volume profile (simplified)
            features['volume_profile'] = volume.rolling(20).sum() / volume.rolling(50).sum()
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding volume features: {str(e)}")
            return features
    
    async def _add_momentum_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum oscillator features"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            
            # RSI
            features['rsi'] = ta.RSI(close, timeperiod=self.params['rsi_period'])
            features['rsi_sma_5'] = features['rsi'].rolling(5).mean()
            features['rsi_ema_5'] = features['rsi'].ewm(span=5).mean()
            
            # Stochastic
            features['stoch_k'], features['stoch_d'] = ta.STOCH(
                high, low, close, 
                fastk_period=self.params['stoch_k'],
                slowk_period=self.params['stoch_d'],
                slowd_period=self.params['stoch_d']
            )
            
            # Stochastic RSI
            features['stoch_rsi'] = ta.STOCHRSI(close, timeperiod=14)
            
            # Williams %R
            features['williams_r'] = ta.WILLR(high, low, close, timeperiod=self.params['williams_r_period'])
            
            # Commodity Channel Index
            features['cci'] = ta.CCI(high, low, close, timeperiod=self.params['cci_period'])
            
            # Momentum
            features['momentum_10'] = ta.MOM(close, timeperiod=10)
            features['momentum_20'] = ta.MOM(close, timeperiod=20)
            
            # Rate of Change
            features['roc_10'] = ta.ROC(close, timeperiod=10)
            features['roc_20'] = ta.ROC(close, timeperiod=20)
            
            # TRIX
            features['trix'] = ta.TRIX(close, timeperiod=14)
            
            # Ultimate Oscillator
            features['ultimate_oscillator'] = ta.ULTOSC(high, low, close)
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding momentum features: {str(e)}")
            return features
    
    async def _add_trend_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following features"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            
            # Simple Moving Averages
            for period in self.params['ma_periods']:
                sma = ta.SMA(close, timeperiod=period)
                features[f'sma_{period}'] = sma
                features[f'sma_{period}_ratio'] = close / sma
            
            # Exponential Moving Averages
            for period in self.params['ema_periods']:
                ema = ta.EMA(close, timeperiod=period)
                features[f'ema_{period}'] = ema
                features[f'ema_{period}_ratio'] = close / ema
            
            # MACD
            macd, macd_signal, macd_hist = ta.MACD(
                close, 
                fastperiod=self.params['macd_fast'],
                slowperiod=self.params['macd_slow'],
                signalperiod=self.params['macd_signal']
            )
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_hist
            
            # ADX (Average Directional Index)
            features['adx'] = ta.ADX(high, low, close, timeperiod=self.params['adx_period'])
            features['plus_di'] = ta.PLUS_DI(high, low, close, timeperiod=self.params['adx_period'])
            features['minus_di'] = ta.MINUS_DI(high, low, close, timeperiod=self.params['adx_period'])
            
            # Aroon
            features['aroon_up'], features['aroon_down'] = ta.AROON(high, low, timeperiod=14)
            features['aroon_osc'] = ta.AROONOSC(high, low, timeperiod=14)
            
            # Percentage Price Oscillator
            features['ppo'], features['ppo_signal'], features['ppo_histogram'] = ta.PPO(close)
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding trend features: {str(e)}")
            return features
    
    async def _add_volatility_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            
            # Average True Range
            features['atr'] = ta.ATR(high, low, close, timeperiod=self.params['atr_period'])
            features['atr_pct'] = features['atr'] / close
            features['natr'] = ta.NATR(high, low, close, timeperiod=self.params['atr_period'])
            
            # True Range
            features['true_range'] = ta.TRANGE(high, low, close)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = ta.BBANDS(
                close, 
                timeperiod=self.params['bb_period'],
                nbdevup=self.params['bb_std'],
                nbdevdn=self.params['bb_std']
            )
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
            
            # Keltner Channels
            keltner_middle = ta.EMA(close, timeperiod=20)
            atr_20 = ta.ATR(high, low, close, timeperiod=20)
            features['keltner_upper'] = keltner_middle + (2 * atr_20)
            features['keltner_middle'] = keltner_middle
            features['keltner_lower'] = keltner_middle - (2 * atr_20)
            
            # Donchian Channels
            features['donchian_upper'] = high.rolling(20).max()
            features['donchian_lower'] = low.rolling(20).min()
            features['donchian_middle'] = (features['donchian_upper'] + features['donchian_lower']) / 2
            
            # Historical Volatility
            log_returns = np.log(close / close.shift(1))
            features['historical_volatility_10'] = log_returns.rolling(10).std() * np.sqrt(252)
            features['historical_volatility_20'] = log_returns.rolling(20).std() * np.sqrt(252)
            
            # Volatility ratio
            features['volatility_ratio'] = features['historical_volatility_10'] / features['historical_volatility_20']
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding volatility features: {str(e)}")
            return features
    
    async def _add_support_resistance_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add support and resistance level features"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Pivot Points
            prev_high = high.shift(1)
            prev_low = low.shift(1)
            prev_close = close.shift(1)
            
            pivot = (prev_high + prev_low + prev_close) / 3
            features['pivot_point'] = pivot
            features['resistance_1'] = 2 * pivot - prev_low
            features['support_1'] = 2 * pivot - prev_high
            features['resistance_2'] = pivot + (prev_high - prev_low)
            features['support_2'] = pivot - (prev_high - prev_low)
            
            # Fibonacci levels (simplified)
            high_20 = high.rolling(20).max()
            low_20 = low.rolling(20).min()
            range_20 = high_20 - low_20
            
            features['fib_23_6'] = high_20 - 0.236 * range_20
            features['fib_38_2'] = high_20 - 0.382 * range_20
            features['fib_50'] = high_20 - 0.5 * range_20
            features['fib_61_8'] = high_20 - 0.618 * range_20
            
            # Time-based support/resistance
            features['weekly_high'] = high.rolling(7).max()
            features['weekly_low'] = low.rolling(7).min()
            features['monthly_high'] = high.rolling(30).max()
            features['monthly_low'] = low.rolling(30).min()
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding support/resistance features: {str(e)}")
            return features
    
    async def _add_market_structure_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure and pattern features"""
        try:
            open_price = df['open']
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Trend structure
            features['higher_highs'] = (high > high.shift(1)).astype(int)
            features['lower_lows'] = (low < low.shift(1)).astype(int)
            
            # Bar patterns
            prev_high = high.shift(1)
            prev_low = low.shift(1)
            
            features['inside_bar'] = ((high <= prev_high) & (low >= prev_low)).astype(int)
            features['outside_bar'] = ((high >= prev_high) & (low <= prev_low)).astype(int)
            
            # Candlestick patterns (simplified)
            body = abs(close - open_price)
            total_range = high - low
            body_ratio = body / total_range.replace(0, np.nan)
            
            features['doji'] = (body_ratio < 0.1).astype(int)
            
            # Hammer pattern
            lower_shadow = np.minimum(open_price, close) - low
            upper_shadow = high - np.maximum(open_price, close)
            features['hammer'] = ((lower_shadow > 2 * body) & (upper_shadow < 0.1 * total_range)).astype(int)
            
            # Shooting star
            features['shooting_star'] = ((upper_shadow > 2 * body) & (lower_shadow < 0.1 * total_range)).astype(int)
            
            # Engulfing patterns
            prev_body = abs(close.shift(1) - open_price.shift(1))
            bullish_engulf = (close > open_price) & (close.shift(1) < open_price.shift(1)) & (body > prev_body)
            bearish_engulf = (close < open_price) & (close.shift(1) > open_price.shift(1)) & (body > prev_body)
            
            features['engulfing_bull'] = bullish_engulf.astype(int)
            features['engulfing_bear'] = bearish_engulf.astype(int)
            
            # Gaps
            features['gap_up'] = (low > high.shift(1)).astype(int)
            features['gap_down'] = (high < low.shift(1)).astype(int)
            
            # Breakouts (simplified)
            resistance = high.rolling(20).max().shift(1)
            support = low.rolling(20).min().shift(1)
            
            features['breakout_up'] = (close > resistance).astype(int)
            features['breakout_down'] = (close < support).astype(int)
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding market structure features: {str(e)}")
            return features
    
    async def _add_statistical_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical analysis features"""
        try:
            close = df['close']
            returns = close.pct_change()
            
            # Z-scores
            sma_10 = close.rolling(10).mean()
            std_10 = close.rolling(10).std()
            features['z_score_10'] = (close - sma_10) / std_10
            
            sma_20 = close.rolling(20).mean()
            std_20 = close.rolling(20).std()
            features['z_score_20'] = (close - sma_20) / std_20
            
            # Rolling statistics
            features['skewness_10'] = returns.rolling(10).skew()
            features['kurtosis_10'] = returns.rolling(10).kurt()
            
            # Correlation placeholders (would need market data)
            features['correlation_spy_10'] = 0.5  # Default correlation
            features['correlation_vix_10'] = -0.3  # Default VIX correlation
            
            # Beta and alpha placeholders
            features['beta_10'] = 1.0  # Default beta
            features['alpha_10'] = 0.0  # Default alpha
            features['information_ratio'] = 0.0  # Default IR
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding statistical features: {str(e)}")
            return features
    
    async def _add_pattern_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Pattern detection (simplified versions)
            # These would typically require more sophisticated algorithms
            
            # Double top/bottom detection
            features['double_top'] = 0  # Placeholder
            features['double_bottom'] = 0  # Placeholder
            
            # Head and shoulders
            features['head_shoulders'] = 0  # Placeholder
            features['inv_head_shoulders'] = 0  # Placeholder
            
            # Triangle patterns
            features['triangle_ascending'] = 0  # Placeholder
            features['triangle_descending'] = 0  # Placeholder
            features['triangle_symmetrical'] = 0  # Placeholder
            
            # Channel patterns
            features['channel_up'] = 0  # Placeholder
            features['channel_down'] = 0  # Placeholder
            
            # Wedge patterns
            features['wedge_rising'] = 0  # Placeholder
            features['wedge_falling'] = 0  # Placeholder
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding pattern features: {str(e)}")
            return features
    
    async def _add_sentiment_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment indicator features"""
        try:
            # Sentiment indicators (would need external data sources)
            # For now, using placeholders
            
            features['put_call_ratio'] = 1.0  # Neutral
            features['vix_level'] = 20.0  # Average VIX level
            features['fear_greed_index'] = 50.0  # Neutral
            features['insider_buying'] = 0.5  # Neutral
            features['institutional_flow'] = 0.0  # Neutral
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding sentiment features: {str(e)}")
            return features
    
    async def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize features"""
        try:
            # Handle infinite values
            features = features.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill NaN values for up to 5 periods
            features = features.fillna(method='ffill', limit=5)
            
            # Fill remaining NaN with median values
            for col in features.columns:
                if features[col].isna().any():
                    median_val = features[col].median()
                    features[col] = features[col].fillna(median_val)
            
            # Remove any remaining NaN rows
            features = features.dropna()
            
            # Ensure feature count matches expected
            expected_features = set(self.feature_names)
            actual_features = set(features.columns)
            
            missing_features = expected_features - actual_features
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                
                # Add missing features with default values
                for feature in missing_features:
                    features[feature] = 0.0
            
            # Remove extra features
            extra_features = actual_features - expected_features
            if extra_features:
                features = features.drop(columns=list(extra_features))
            
            # Reorder columns to match expected order
            features = features.reindex(columns=self.feature_names)
            
            return features
            
        except Exception as e:
            logger.error(f"Error cleaning features: {str(e)}")
            return features
    
    async def calculate_feature_importance(self, features: pd.DataFrame, 
                                         target: pd.Series) -> Dict[str, float]:
        """Calculate feature importance scores"""
        try:
            from sklearn.feature_selection import mutual_info_regression
            from sklearn.preprocessing import StandardScaler
            
            if features.empty or target.empty:
                return {}
            
            # Align features and target
            common_index = features.index.intersection(target.index)
            if len(common_index) < 10:
                logger.warning("Insufficient aligned data for feature importance")
                return {}
            
            X = features.loc[common_index]
            y = target.loc[common_index]
            
            # Remove any remaining NaN
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 10:
                return {}
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Calculate mutual information
            importance_scores = mutual_info_regression(X_scaled, y, random_state=42)
            
            # Create importance dictionary
            importance_dict = {
                feature: float(score) 
                for feature, score in zip(X.columns, importance_scores)
            }
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), 
                                        key=lambda x: x[1], reverse=True))
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {}
    
    def get_feature_categories(self) -> Dict[str, bool]:
        """Get current feature category settings"""
        return self.feature_categories.copy()
    
    def set_feature_categories(self, categories: Dict[str, bool]):
        """Set which feature categories to include"""
        self.feature_categories.update(categories)
        self._build_feature_names()
        logger.info(f"Updated feature categories, now {self.feature_count} features")
    
    def get_feature_stats(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the features"""
        try:
            if features.empty:
                return {}
            
            stats = {
                'total_features': len(features.columns),
                'total_samples': len(features),
                'missing_values': features.isna().sum().sum(),
                'infinite_values': np.isinf(features.select_dtypes(include=[np.number])).sum().sum(),
                'feature_ranges': {},
                'correlation_matrix': {}
            }
            
            # Feature ranges
            numeric_features = features.select_dtypes(include=[np.number])
            for col in numeric_features.columns[:10]:  # Top 10 for brevity
                stats['feature_ranges'][col] = {
                    'min': float(numeric_features[col].min()),
                    'max': float(numeric_features[col].max()),
                    'mean': float(numeric_features[col].mean()),
                    'std': float(numeric_features[col].std())
                }
            
            # Sample correlation matrix (top 10 features)
            if len(numeric_features.columns) > 1:
                sample_features = numeric_features.iloc[:, :10]
                corr_matrix = sample_features.corr()
                stats['correlation_matrix'] = corr_matrix.to_dict()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting feature stats: {str(e)}")
            return {}
    
    async def batch_create_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Create features for multiple symbols in batch"""
        try:
            results = {}
            
            tasks = []
            for symbol, data in data_dict.items():
                task = self.create_features(data)
                tasks.append((symbol, task))
            
            # Execute all tasks concurrently
            completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (symbol, _), result in zip(tasks, completed_tasks):
                if isinstance(result, Exception):
                    logger.error(f"Error creating features for {symbol}: {str(result)}")
                elif not result.empty:
                    results[symbol] = result
            
            logger.info(f"Created features for {len(results)} symbols")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch feature creation: {str(e)}")
            return {}
