"""
Schwab AI Trading System - Market Regime Detection
Advanced market regime classification using statistical analysis, volatility clustering, and machine learning.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

from utils.cache_manager import CacheManager
from utils.database import DatabaseManager
from config.settings import get_settings

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime types"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"

class TrendDirection(Enum):
    """Trend direction types"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"

class VolatilityLevel(Enum):
    """Volatility level classifications"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class RegimeSignals:
    """Market regime technical signals"""
    adx_value: float
    volatility_percentile: float
    trend_strength: float
    momentum_divergence: float
    volume_trend: float
    price_structure: str
    support_resistance_strength: float
    mean_reversion_signal: float

@dataclass
class RegimeDetectionResult:
    """Market regime detection result"""
    regime: MarketRegime
    confidence: float
    trend_direction: TrendDirection
    volatility_level: VolatilityLevel
    regime_strength: float
    duration_estimate: int  # Expected duration in periods
    transition_probability: float
    signals: RegimeSignals
    characteristics: Dict[str, float]
    reasoning: List[str]
    timestamp: datetime

class MarketRegimeDetector:
    """
    Advanced market regime detection system that identifies and classifies
    market conditions using multiple technical and statistical approaches.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache_manager = CacheManager()
        self.db_manager = DatabaseManager()
        
        # Detection parameters
        self.lookback_periods = {
            'short': 20,
            'medium': 50,
            'long': 200
        }
        
        # Regime thresholds
        self.thresholds = {
            'adx_trending': 25.0,
            'adx_strong_trend': 40.0,
            'volatility_low': 0.15,
            'volatility_high': 0.35,
            'volume_spike': 1.5,
            'momentum_threshold': 0.02,
            'trend_strength_min': 0.6
        }
        
        # Historical regime data for ML model
        self.regime_history = []
        self.ml_model = None
        self.scaler = StandardScaler()
        
        # Regime transition matrix
        self.transition_matrix = {}
        
        self.is_initialized = False
        
        logger.info("MarketRegimeDetector initialized")
    
    async def initialize(self) -> bool:
        """Initialize the regime detector"""
        try:
            # Load historical regime data
            await self._load_regime_history()
            
            # Train ML model if sufficient data
            await self._train_ml_model()
            
            # Build transition matrix
            await self._build_transition_matrix()
            
            self.is_initialized = True
            logger.info("MarketRegimeDetector initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MarketRegimeDetector: {str(e)}")
            return False
    
    async def _load_regime_history(self):
        """Load historical regime classifications"""
        try:
            query = """
                SELECT symbol, regime, confidence, trend_direction, volatility_level,
                       regime_strength, timestamp, characteristics
                FROM market_regimes 
                WHERE timestamp >= %s
                ORDER BY timestamp DESC
                LIMIT 1000
            """
            
            lookback_date = datetime.now() - timedelta(days=365)
            results = await self.db_manager.execute_query(query, (lookback_date,))
            
            self.regime_history = results if results else []
            logger.info(f"Loaded {len(self.regime_history)} historical regime records")
            
        except Exception as e:
            logger.error(f"Error loading regime history: {str(e)}")
            self.regime_history = []
    
    async def _train_ml_model(self):
        """Train machine learning model for regime prediction"""
        try:
            if len(self.regime_history) < 100:
                logger.warning("Insufficient historical data for ML model training")
                return
            
            # Prepare training data
            features = []
            labels = []
            
            for record in self.regime_history:
                if record.get('characteristics'):
                    # Parse characteristics JSON if it's a string
                    chars = record['characteristics']
                    if isinstance(chars, str):
                        import json
                        chars = json.loads(chars)
                    
                    feature_vector = [
                        chars.get('adx_value', 0),
                        chars.get('volatility_percentile', 0),
                        chars.get('trend_strength', 0),
                        chars.get('momentum_divergence', 0),
                        chars.get('volume_trend', 0),
                        chars.get('support_resistance_strength', 0),
                        chars.get('mean_reversion_signal', 0)
                    ]
                    
                    features.append(feature_vector)
                    labels.append(record['regime'])
            
            if len(features) >= 50:
                X = np.array(features)
                y = np.array(labels)
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Train Random Forest model
                self.ml_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                self.ml_model.fit(X_scaled, y)
                
                logger.info(f"Trained ML model on {len(features)} samples")
            
        except Exception as e:
            logger.error(f"Error training ML model: {str(e)}")
            self.ml_model = None
    
    async def _build_transition_matrix(self):
        """Build regime transition probability matrix"""
        try:
            if not self.regime_history:
                return
            
            # Count transitions
            transitions = {}
            
            sorted_history = sorted(self.regime_history, key=lambda x: x['timestamp'])
            
            for i in range(1, len(sorted_history)):
                prev_regime = sorted_history[i-1]['regime']
                curr_regime = sorted_history[i]['regime']
                
                if prev_regime not in transitions:
                    transitions[prev_regime] = {}
                
                if curr_regime not in transitions[prev_regime]:
                    transitions[prev_regime][curr_regime] = 0
                
                transitions[prev_regime][curr_regime] += 1
            
            # Convert counts to probabilities
            for prev_regime in transitions:
                total_transitions = sum(transitions[prev_regime].values())
                for curr_regime in transitions[prev_regime]:
                    transitions[prev_regime][curr_regime] /= total_transitions
            
            self.transition_matrix = transitions
            logger.info(f"Built transition matrix with {len(transitions)} regime states")
            
        except Exception as e:
            logger.error(f"Error building transition matrix: {str(e)}")
    
    async def detect_regime(self, data: pd.DataFrame, 
                          symbol: Optional[str] = None) -> RegimeDetectionResult:
        """
        Detect market regime from price data
        
        Args:
            data: OHLCV DataFrame
            symbol: Symbol identifier (optional)
            
        Returns:
            RegimeDetectionResult with detected regime and metadata
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if data.empty or len(data) < self.lookback_periods['short']:
                logger.warning("Insufficient data for regime detection")
                return self._create_default_result()
            
            # Check cache first
            if symbol:
                cache_key = f"regime_{symbol}_{int(datetime.now().timestamp() / 600)}"  # 10 min cache
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    return cached_result
            
            # Calculate regime signals
            signals = await self._calculate_regime_signals(data)
            
            # Determine primary regime
            regime, confidence = await self._classify_regime(signals, data)
            
            # Determine trend direction
            trend_direction = await self._determine_trend_direction(data, signals)
            
            # Classify volatility level
            volatility_level = await self._classify_volatility(data, signals)
            
            # Calculate regime strength
            regime_strength = await self._calculate_regime_strength(signals, regime)
            
            # Estimate duration
            duration_estimate = await self._estimate_regime_duration(regime, signals)
            
            # Calculate transition probability
            transition_prob = await self._calculate_transition_probability(regime)
            
            # Generate characteristics dictionary
            characteristics = {
                'adx_value': signals.adx_value,
                'volatility_percentile': signals.volatility_percentile,
                'trend_strength': signals.trend_strength,
                'momentum_divergence': signals.momentum_divergence,
                'volume_trend': signals.volume_trend,
                'support_resistance_strength': signals.support_resistance_strength,
                'mean_reversion_signal': signals.mean_reversion_signal
            }
            
            # Generate reasoning
            reasoning = await self._generate_reasoning(regime, signals, trend_direction)
            
            # Create result
            result = RegimeDetectionResult(
                regime=regime,
                confidence=confidence,
                trend_direction=trend_direction,
                volatility_level=volatility_level,
                regime_strength=regime_strength,
                duration_estimate=duration_estimate,
                transition_probability=transition_prob,
                signals=signals,
                characteristics=characteristics,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
            # Cache result
            if symbol:
                await self.cache_manager.set(cache_key, result, expire=600)
            
            # Store result for future analysis
            await self._store_regime_result(result, symbol)
            
            logger.debug(f"Detected regime: {regime.value} (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error detecting regime: {str(e)}")
            return self._create_default_result()
    
    def _create_default_result(self) -> RegimeDetectionResult:
        """Create default regime result for error cases"""
        return RegimeDetectionResult(
            regime=MarketRegime.UNKNOWN,
            confidence=0.5,
            trend_direction=TrendDirection.SIDEWAYS,
            volatility_level=VolatilityLevel.MODERATE,
            regime_strength=0.5,
            duration_estimate=20,
            transition_probability=0.5,
            signals=RegimeSignals(
                adx_value=20.0,
                volatility_percentile=0.5,
                trend_strength=0.5,
                momentum_divergence=0.0,
                volume_trend=0.0,
                price_structure="unknown",
                support_resistance_strength=0.5,
                mean_reversion_signal=0.0
            ),
            characteristics={},
            reasoning=["Insufficient data for regime detection"],
            timestamp=datetime.now()
        )
    
    async def _calculate_regime_signals(self, data: pd.DataFrame) -> RegimeSignals:
        """Calculate technical indicators for regime detection"""
        try:
            close = data['close']
            high = data['high']
            low = data['low']
            volume = data['volume']
            
            # ADX for trend strength
            adx_value = await self._calculate_adx(high, low, close)
            
            # Volatility percentile
            returns = close.pct_change().dropna()
            volatility = returns.rolling(20).std()
            current_vol = volatility.iloc[-1] if not volatility.empty else 0.02
            vol_percentile = (volatility <= current_vol).mean()
            
            # Trend strength using linear regression
            trend_strength = await self._calculate_trend_strength(close)
            
            # Momentum divergence
            momentum_divergence = await self._calculate_momentum_divergence(close, high, low)
            
            # Volume trend
            volume_trend = await self._calculate_volume_trend(volume)
            
            # Price structure analysis
            price_structure = await self._analyze_price_structure(high, low, close)
            
            # Support/Resistance strength
            sr_strength = await self._calculate_support_resistance_strength(high, low, close)
            
            # Mean reversion signal
            mean_reversion = await self._calculate_mean_reversion_signal(close)
            
            return RegimeSignals(
                adx_value=adx_value,
                volatility_percentile=vol_percentile,
                trend_strength=trend_strength,
                momentum_divergence=momentum_divergence,
                volume_trend=volume_trend,
                price_structure=price_structure,
                support_resistance_strength=sr_strength,
                mean_reversion_signal=mean_reversion
            )
            
        except Exception as e:
            logger.error(f"Error calculating regime signals: {str(e)}")
            return RegimeSignals(
                adx_value=20.0, volatility_percentile=0.5, trend_strength=0.5,
                momentum_divergence=0.0, volume_trend=0.0, price_structure="unknown",
                support_resistance_strength=0.5, mean_reversion_signal=0.0
            )
    
    async def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Calculate Average Directional Index"""
        try:
            # Simplified ADX calculation
            period = 14
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            plus_dm = high.diff()
            minus_dm = -low.diff()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            # When both are positive, the larger one is kept, the other is set to 0
            plus_dm[(plus_dm < minus_dm)] = 0
            minus_dm[(minus_dm < plus_dm)] = 0
            
            # Smoothed values
            tr_smooth = tr.rolling(period).mean()
            plus_dm_smooth = plus_dm.rolling(period).mean()
            minus_dm_smooth = minus_dm.rolling(period).mean()
            
            # DI values
            plus_di = 100 * plus_dm_smooth / tr_smooth
            minus_di = 100 * minus_dm_smooth / tr_smooth
            
            # DX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            dx = dx.fillna(0)
            
            # ADX
            adx = dx.rolling(period).mean()
            
            return float(adx.iloc[-1]) if not adx.empty else 20.0
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            return 20.0
    
    async def _calculate_trend_strength(self, close: pd.Series) -> float:
        """Calculate trend strength using linear regression"""
        try:
            # Use last 50 periods for trend analysis
            period = min(50, len(close))
            recent_prices = close.tail(period)
            
            if len(recent_prices) < 10:
                return 0.5
            
            # Linear regression
            x = np.arange(len(recent_prices))
            y = recent_prices.values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Normalize R-squared to 0-1 range
            trend_strength = abs(r_value)
            
            return float(trend_strength)
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            return 0.5
    
    async def _calculate_momentum_divergence(self, close: pd.Series, 
                                          high: pd.Series, low: pd.Series) -> float:
        """Calculate momentum divergence between price and oscillator"""
        try:
            # Simple RSI calculation
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Compare price trend vs RSI trend over last 20 periods
            if len(rsi) < 20:
                return 0.0
            
            price_trend = close.iloc[-1] / close.iloc[-20] - 1
            rsi_trend = rsi.iloc[-1] / rsi.iloc[-20] - 1
            
            # Divergence is when trends move in opposite directions
            divergence = -(price_trend * rsi_trend)  # Negative correlation indicates divergence
            
            return float(np.clip(divergence, -1, 1))
            
        except Exception as e:
            logger.error(f"Error calculating momentum divergence: {str(e)}")
            return 0.0
    
    async def _calculate_volume_trend(self, volume: pd.Series) -> float:
        """Calculate volume trend"""
        try:
            if len(volume) < 20:
                return 0.0
            
            # Compare recent volume to historical average
            recent_avg = volume.tail(10).mean()
            historical_avg = volume.tail(50).mean()
            
            volume_trend = (recent_avg / historical_avg - 1) if historical_avg > 0 else 0
            
            return float(np.clip(volume_trend, -2, 2))
            
        except Exception as e:
            logger.error(f"Error calculating volume trend: {str(e)}")
            return 0.0
    
    async def _analyze_price_structure(self, high: pd.Series, 
                                     low: pd.Series, close: pd.Series) -> str:
        """Analyze price structure patterns"""
        try:
            if len(close) < 20:
                return "insufficient_data"
            
            # Higher highs and higher lows
            recent_highs = high.tail(10)
            recent_lows = low.tail(10)
            
            hh_count = sum(1 for i in range(1, len(recent_highs)) 
                          if recent_highs.iloc[i] > recent_highs.iloc[i-1])
            hl_count = sum(1 for i in range(1, len(recent_lows)) 
                          if recent_lows.iloc[i] > recent_lows.iloc[i-1])
            
            lh_count = sum(1 for i in range(1, len(recent_highs)) 
                          if recent_highs.iloc[i] < recent_highs.iloc[i-1])
            ll_count = sum(1 for i in range(1, len(recent_lows)) 
                          if recent_lows.iloc[i] < recent_lows.iloc[i-1])
            
            # Classify structure
            if hh_count > lh_count and hl_count > ll_count:
                return "higher_highs_lows"
            elif lh_count > hh_count and ll_count > hl_count:
                return "lower_highs_lows"
            elif abs(hh_count - lh_count) <= 1 and abs(hl_count - ll_count) <= 1:
                return "sideways"
            else:
                return "mixed"
                
        except Exception as e:
            logger.error(f"Error analyzing price structure: {str(e)}")
            return "unknown"
    
    async def _calculate_support_resistance_strength(self, high: pd.Series, 
                                                   low: pd.Series, close: pd.Series) -> float:
        """Calculate strength of support/resistance levels"""
        try:
            if len(close) < 50:
                return 0.5
            
            # Find peaks and troughs
            highs = high.values
            lows = low.values
            
            peaks, _ = find_peaks(highs, distance=5)
            troughs, _ = find_peaks(-lows, distance=5)
            
            # Count how many times price has tested these levels
            current_price = close.iloc[-1]
            
            # Check proximity to recent peaks/troughs
            if len(peaks) > 0:
                recent_peaks = highs[peaks[-3:]] if len(peaks) >= 3 else highs[peaks]
                resistance_strength = sum(1 for peak in recent_peaks 
                                        if abs(current_price - peak) / peak < 0.02)
            else:
                resistance_strength = 0
            
            if len(troughs) > 0:
                recent_troughs = lows[troughs[-3:]] if len(troughs) >= 3 else lows[troughs]
                support_strength = sum(1 for trough in recent_troughs 
                                     if abs(current_price - trough) / trough < 0.02)
            else:
                support_strength = 0
            
            # Normalize to 0-1 range
            total_strength = (resistance_strength + support_strength) / 6.0  # Max 6 tests
            
            return float(np.clip(total_strength, 0, 1))
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance strength: {str(e)}")
            return 0.5
    
    async def _calculate_mean_reversion_signal(self, close: pd.Series) -> float:
        """Calculate mean reversion signal"""
        try:
            if len(close) < 20:
                return 0.0
            
            # Bollinger Bands mean reversion
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            
            current_price = close.iloc[-1]
            current_sma = sma.iloc[-1]
            current_std = std.iloc[-1]
            
            if current_std == 0:
                return 0.0
            
            # Z-score
            z_score = (current_price - current_sma) / current_std
            
            # Mean reversion signal is stronger when price is far from mean
            mean_reversion = -z_score / 2.0  # Negative because we expect reversion
            
            return float(np.clip(mean_reversion, -1, 1))
            
        except Exception as e:
            logger.error(f"Error calculating mean reversion signal: {str(e)}")
            return 0.0
    
    async def _classify_regime(self, signals: RegimeSignals, 
                             data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """Classify market regime based on signals"""
        try:
            regime_scores = {
                MarketRegime.TRENDING: 0.0,
                MarketRegime.RANGING: 0.0,
                MarketRegime.VOLATILE: 0.0,
                MarketRegime.BREAKOUT: 0.0,
                MarketRegime.REVERSAL: 0.0
            }
            
            # Trending regime signals
            if signals.adx_value > self.thresholds['adx_trending']:
                regime_scores[MarketRegime.TRENDING] += 0.3
            if signals.trend_strength > self.thresholds['trend_strength_min']:
                regime_scores[MarketRegime.TRENDING] += 0.2
            if signals.price_structure in ['higher_highs_lows', 'lower_highs_lows']:
                regime_scores[MarketRegime.TRENDING] += 0.2
            
            # Ranging regime signals
            if signals.adx_value < self.thresholds['adx_trending']:
                regime_scores[MarketRegime.RANGING] += 0.2
            if signals.support_resistance_strength > 0.6:
                regime_scores[MarketRegime.RANGING] += 0.3
            if signals.price_structure == 'sideways':
                regime_scores[MarketRegime.RANGING] += 0.2
            if abs(signals.mean_reversion_signal) > 0.3:
                regime_scores[MarketRegime.RANGING] += 0.1
            
            # Volatile regime signals
            if signals.volatility_percentile > 0.8:
                regime_scores[MarketRegime.VOLATILE] += 0.3
            if abs(signals.momentum_divergence) > 0.3:
                regime_scores[MarketRegime.VOLATILE] += 0.2
            if signals.volume_trend > 0.5:
                regime_scores[MarketRegime.VOLATILE] += 0.2
            
            # Breakout regime signals
            if signals.volume_trend > 1.0:
                regime_scores[MarketRegime.BREAKOUT] += 0.3
            if signals.adx_value > self.thresholds['adx_strong_trend']:
                regime_scores[MarketRegime.BREAKOUT] += 0.2
            if signals.volatility_percentile > 0.7:
                regime_scores[MarketRegime.BREAKOUT] += 0.1
            
            # Reversal regime signals
            if abs(signals.momentum_divergence) > 0.5:
                regime_scores[MarketRegime.REVERSAL] += 0.3
            if signals.support_resistance_strength > 0.7:
                regime_scores[MarketRegime.REVERSAL] += 0.2
            if abs(signals.mean_reversion_signal) > 0.6:
                regime_scores[MarketRegime.REVERSAL] += 0.2
            
            # Use ML model if available
            if self.ml_model:
                try:
                    feature_vector = np.array([[
                        signals.adx_value,
                        signals.volatility_percentile,
                        signals.trend_strength,
                        signals.momentum_divergence,
                        signals.volume_trend,
                        signals.support_resistance_strength,
                        signals.mean_reversion_signal
                    ]])
                    
                    feature_vector_scaled = self.scaler.transform(feature_vector)
                    ml_prediction = self.ml_model.predict(feature_vector_scaled)[0]
                    ml_probabilities = self.ml_model.predict_proba(feature_vector_scaled)[0]
                    
                    # Boost ML prediction score
                    if ml_prediction in regime_scores:
                        ml_regime = MarketRegime(ml_prediction)
                        max_prob = max(ml_probabilities)
                        regime_scores[ml_regime] += 0.3 * max_prob
                        
                except Exception as e:
                    logger.warning(f"Error using ML model: {str(e)}")
            
            # Find best regime
            best_regime = max(regime_scores, key=regime_scores.get)
            confidence = regime_scores[best_regime]
            
            # Normalize confidence to 0-1 range
            confidence = min(max(confidence, 0.0), 1.0)
            
            # Minimum confidence threshold
            if confidence < 0.3:
                best_regime = MarketRegime.UNKNOWN
                confidence = 0.5
            
            return best_regime, confidence
            
        except Exception as e:
            logger.error(f"Error classifying regime: {str(e)}")
            return MarketRegime.UNKNOWN, 0.5
    
    async def _determine_trend_direction(self, data: pd.DataFrame, 
                                       signals: RegimeSignals) -> TrendDirection:
        """Determine trend direction"""
        try:
            close = data['close']
            
            if len(close) < 20:
                return TrendDirection.SIDEWAYS
            
            # Multiple timeframe analysis
            short_sma = close.rolling(10).mean()
            medium_sma = close.rolling(20).mean()
            long_sma = close.rolling(50).mean() if len(close) >= 50 else medium_sma
            
            current_price = close.iloc[-1]
            
            # Price vs moving averages
            vs_short = current_price > short_sma.iloc[-1]
            vs_medium = current_price > medium_sma.iloc[-1] 
            vs_long = current_price > long_sma.iloc[-1]
            
            # Moving average alignment
            sma_alignment = short_sma.iloc[-1] > medium_sma.iloc[-1] > long_sma.iloc[-1]
            sma_alignment_bear = short_sma.iloc[-1] < medium_sma.iloc[-1] < long_sma.iloc[-1]
            
            # Price structure
            bullish_structure = signals.price_structure == "higher_highs_lows"
            bearish_structure = signals.price_structure == "lower_highs_lows"
            
            # Combine signals
            bullish_signals = sum([vs_short, vs_medium, vs_long, sma_alignment, bullish_structure])
            bearish_signals = sum([not vs_short, not vs_medium, not vs_long, 
                                 sma_alignment_bear, bearish_structure])
            
            if bullish_signals >= 3:
                return TrendDirection.BULLISH
            elif bearish_signals >= 3:
                return TrendDirection.BEARISH
            else:
                return TrendDirection.SIDEWAYS
                
        except Exception as e:
            logger.error(f"Error determining trend direction: {str(e)}")
            return TrendDirection.SIDEWAYS
    
    async def _classify_volatility(self, data: pd.DataFrame, 
                                 signals: RegimeSignals) -> VolatilityLevel:
        """Classify volatility level"""
        try:
            close = data['close']
            
            if len(close) < 20:
                return VolatilityLevel.MODERATE
            
            # Calculate realized volatility
            returns = close.pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)  # Annualized
            
            # Classify based on thresholds
            if volatility < self.thresholds['volatility_low']:
                return VolatilityLevel.LOW
            elif volatility > self.thresholds['volatility_high']:
                if volatility > 0.5:
                    return VolatilityLevel.EXTREME
                else:
                    return VolatilityLevel.HIGH
            else:
                return VolatilityLevel.MODERATE
                
        except Exception as e:
            logger.error(f"Error classifying volatility: {str(e)}")
            return VolatilityLevel.MODERATE
    
    async def _calculate_regime_strength(self, signals: RegimeSignals, 
                                       regime: MarketRegime) -> float:
        """Calculate strength of detected regime"""
        try:
            strength_factors = []
            
            if regime == MarketRegime.TRENDING:
                strength_factors.extend([
                    signals.adx_value / 50.0,  # Normalize ADX
                    signals.trend_strength,
                    1.0 if signals.price_structure in ['higher_highs_lows', 'lower_highs_lows'] else 0.5
                ])
            
            elif regime == MarketRegime.RANGING:
                strength_factors.extend([
                    1.0 - signals.adx_value / 50.0,  # Lower ADX = stronger range
                    signals.support_resistance_strength,
                    1.0 if signals.price_structure == 'sideways' else 0.5,
                    min(abs(signals.mean_reversion_signal), 1.0)
                ])
            
            elif regime == MarketRegime.VOLATILE:
                strength_factors.extend([
                    signals.volatility_percentile,
                    min(abs(signals.momentum_divergence), 1.0),
                    min(signals.volume_trend / 2.0, 1.0)
                ])
            
            elif regime == MarketRegime.BREAKOUT:
                strength_factors.extend([
                    min(signals.volume_trend / 2.0, 1.0),
                    signals.adx_value / 50.0,
                    signals.volatility_percentile
                ])
            
            elif regime == MarketRegime.REVERSAL:
                strength_factors.extend([
                    min(abs(signals.momentum_divergence), 1.0),
                    signals.support_resistance_strength,
                    min(abs(signals.mean_reversion_signal), 1.0)
                ])
            
            else:  # UNKNOWN
                return 0.5
            
            regime_strength = np.mean(strength_factors) if strength_factors else 0.5
            
            return float(np.clip(regime_strength, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating regime strength: {str(e)}")
            return 0.5
    
    async def _estimate_regime_duration(self, regime: MarketRegime, 
                                      signals: RegimeSignals) -> int:
        """Estimate expected duration of regime in periods"""
        try:
            # Base durations by regime type (in periods)
            base_durations = {
                MarketRegime.TRENDING: 50,
                MarketRegime.RANGING: 30,
                MarketRegime.VOLATILE: 15,
                MarketRegime.BREAKOUT: 10,
                MarketRegime.REVERSAL: 5,
                MarketRegime.UNKNOWN: 20
            }
            
            base_duration = base_durations[regime]
            
            # Adjust based on regime strength
            strength_multiplier = 0.5 + signals.trend_strength  # 0.5 to 1.5 range
            
            # Adjust based on volatility (high volatility = shorter duration)
            volatility_adjustment = 1.0 - (signals.volatility_percentile - 0.5) * 0.5
            
            estimated_duration = int(base_duration * strength_multiplier * volatility_adjustment)
            
            return max(5, min(estimated_duration, 100))  # Clamp between 5 and 100
            
        except Exception as e:
            logger.error(f"Error estimating regime duration: {str(e)}")
            return 20
    
    async def _calculate_transition_probability(self, regime: MarketRegime) -> float:
        """Calculate probability of regime transition"""
        try:
            if not self.transition_matrix or regime.value not in self.transition_matrix:
                return 0.2  # Default transition probability
            
            regime_transitions = self.transition_matrix[regime.value]
            
            # Probability of staying in same regime
            stay_probability = regime_transitions.get(regime.value, 0.5)
            
            # Transition probability is complement of staying
            transition_probability = 1.0 - stay_probability
            
            return float(np.clip(transition_probability, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating transition probability: {str(e)}")
            return 0.2
    
    async def _generate_reasoning(self, regime: MarketRegime, signals: RegimeSignals,
                                trend_direction: TrendDirection) -> List[str]:
        """Generate human-readable reasoning for regime detection"""
        try:
            reasoning = []
            
            # Main regime reasoning
            if regime == MarketRegime.TRENDING:
                reasoning.append(f"Strong trending regime detected (ADX: {signals.adx_value:.1f})")
                if signals.trend_strength > 0.7:
                    reasoning.append("High trend strength confirmed by linear regression")
                if signals.price_structure in ['higher_highs_lows', 'lower_highs_lows']:
                    reasoning.append(f"Price structure shows {signals.price_structure}")
            
            elif regime == MarketRegime.RANGING:
                reasoning.append(f"Range-bound market detected (ADX: {signals.adx_value:.1f})")
                if signals.support_resistance_strength > 0.6:
                    reasoning.append("Strong support/resistance levels identified")
                if abs(signals.mean_reversion_signal) > 0.3:
                    reasoning.append("Mean reversion signals present")
            
            elif regime == MarketRegime.VOLATILE:
                reasoning.append(f"High volatility regime (percentile: {signals.volatility_percentile:.2f})")
                if signals.volume_trend > 0.5:
                    reasoning.append("Elevated volume supporting volatility")
                if abs(signals.momentum_divergence) > 0.3:
                    reasoning.append("Momentum divergence detected")
            
            elif regime == MarketRegime.BREAKOUT:
                reasoning.append("Breakout regime identified")
                if signals.volume_trend > 1.0:
                    reasoning.append("Volume surge confirming breakout")
                if signals.adx_value > 40:
                    reasoning.append("Strong directional movement (ADX > 40)")
            
            elif regime == MarketRegime.REVERSAL:
                reasoning.append("Potential reversal regime")
                if abs(signals.momentum_divergence) > 0.5:
                    reasoning.append("Strong momentum divergence suggesting reversal")
                if signals.support_resistance_strength > 0.7:
                    reasoning.append("Testing key support/resistance levels")
            
            # Trend direction reasoning
            reasoning.append(f"Trend direction: {trend_direction.value}")
            
            # Volume context
            if signals.volume_trend > 0.5:
                reasoning.append("Above-average volume supporting the move")
            elif signals.volume_trend < -0.3:
                reasoning.append("Below-average volume - weak participation")
            
            return reasoning
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {str(e)}")
            return [f"Regime detected: {regime.value}"]
    
    async def _store_regime_result(self, result: RegimeDetectionResult, 
                                 symbol: Optional[str] = None):
        """Store regime detection result in database"""
        try:
            insert_query = """
                INSERT INTO market_regimes (symbol, regime, confidence, trend_direction,
                                          volatility_level, regime_strength, duration_estimate,
                                          transition_probability, characteristics, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            import json
            characteristics_json = json.dumps(result.characteristics)
            
            values = (
                symbol or 'MARKET',
                result.regime.value,
                result.confidence,
                result.trend_direction.value,
                result.volatility_level.value,
                result.regime_strength,
                result.duration_estimate,
                result.transition_probability,
                characteristics_json,
                result.timestamp
            )
            
            await self.db_manager.execute_query(insert_query, values)
            
        except Exception as e:
            logger.error(f"Error storing regime result: {str(e)}")
    
    async def get_regime_history(self, symbol: str, 
                               days: int = 30) -> List[RegimeDetectionResult]:
        """Get historical regime classifications for a symbol"""
        try:
            query = """
                SELECT * FROM market_regimes 
                WHERE symbol = %s AND timestamp >= %s 
                ORDER BY timestamp DESC
            """
            
            cutoff_date = datetime.now() - timedelta(days=days)
            results = await self.db_manager.execute_query(query, (symbol, cutoff_date))
            
            regime_history = []
            for row in results:
                # Reconstruct RegimeDetectionResult (simplified)
                regime_result = RegimeDetectionResult(
                    regime=MarketRegime(row['regime']),
                    confidence=row['confidence'],
                    trend_direction=TrendDirection(row['trend_direction']),
                    volatility_level=VolatilityLevel(row['volatility_level']),
                    regime_strength=row['regime_strength'],
                    duration_estimate=row['duration_estimate'],
                    transition_probability=row['transition_probability'],
                    signals=RegimeSignals(0, 0, 0, 0, 0, "", 0, 0),  # Placeholder
                    characteristics=json.loads(row['characteristics']) if row['characteristics'] else {},
                    reasoning=[],
                    timestamp=row['timestamp']
                )
                regime_history.append(regime_result)
            
            return regime_history
            
        except Exception as e:
            logger.error(f"Error getting regime history: {str(e)}")
            return []
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get regime detection statistics"""
        try:
            stats = {
                'total_detections': len(self.regime_history),
                'regime_distribution': {},
                'average_confidence': 0.0,
                'ml_model_available': self.ml_model is not None,
                'transition_matrix_size': len(self.transition_matrix),
                'initialization_status': self.is_initialized
            }
            
            if self.regime_history:
                # Regime distribution
                regimes = [r['regime'] for r in self.regime_history]
                unique_regimes, counts = np.unique(regimes, return_counts=True)
                stats['regime_distribution'] = dict(zip(unique_regimes, counts.tolist()))
                
                # Average confidence
                confidences = [r['confidence'] for r in self.regime_history]
                stats['average_confidence'] = float(np.mean(confidences))
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting regime statistics: {str(e)}")
            return {'error': str(e)}
