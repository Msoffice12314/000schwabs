"""
Schwab AI Trading System - Trading Signal Detector
Advanced AI-powered signal generation using BiConNet architecture with confidence scoring.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum
import asyncio
import warnings
warnings.filterwarnings('ignore')

from models.biconnet import BiConNet
from data.feature_engineer import FeatureEngineer
from data.market_regime import MarketRegimeDetector
from utils.cache_manager import CacheManager
from config.settings import get_settings

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL" 
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"

class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    signal_type: SignalType
    confidence: float
    strength: SignalStrength
    price: float
    timestamp: datetime
    features: Dict[str, float]
    model_predictions: Dict[str, float]
    risk_score: float
    market_regime: str
    reasoning: List[str]
    metadata: Dict[str, Any]

class SignalDetector:
    """
    Advanced trading signal detector using BiConNet neural networks
    with market regime awareness and confidence scoring.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.feature_engineer = FeatureEngineer()
        self.regime_detector = MarketRegimeDetector()
        self.cache_manager = CacheManager()
        
        # Model parameters
        self.model_params = self.settings.model_params
        self.sequence_length = self.model_params['sequence_length']
        self.confidence_threshold = 0.65
        self.signal_threshold = 0.02
        
        # Models dictionary for different timeframes and regimes
        self.models = {}
        self.is_initialized = False
        
        # Signal history for pattern analysis
        self.signal_history = {}
        
        logger.info("SignalDetector initialized")
    
    async def initialize(self) -> bool:
        """Initialize the signal detector with trained models"""
        try:
            # Load pre-trained BiConNet models
            await self._load_models()
            
            # Initialize feature engineer
            await self.feature_engineer.initialize()
            
            # Initialize regime detector
            await self.regime_detector.initialize()
            
            self.is_initialized = True
            logger.info("SignalDetector initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SignalDetector: {str(e)}")
            return False
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame, 
                            current_price: float) -> Optional[TradingSignal]:
        """
        Generate trading signal for a given symbol
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            data: Historical price data
            current_price: Current market price
            
        Returns:
            TradingSignal object or None if no signal
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Check cache first
            cache_key = f"signal_{symbol}_{int(datetime.now().timestamp() / 60)}"
            cached_signal = await self.cache_manager.get(cache_key)
            if cached_signal:
                return cached_signal
            
            # Detect current market regime
            market_regime = await self.regime_detector.detect_regime(data)
            
            # Engineer features
            features = await self.feature_engineer.create_features(data)
            
            # Get model predictions
            predictions = await self._get_model_predictions(symbol, features, market_regime)
            
            # Generate signal based on predictions
            signal = await self._analyze_predictions(
                symbol, predictions, features, current_price, market_regime
            )
            
            # Cache the signal
            if signal:
                await self.cache_manager.set(cache_key, signal, expire=60)
                await self._update_signal_history(symbol, signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {str(e)}")
            return None
    
    async def _load_models(self):
        """Load pre-trained BiConNet models for different regimes"""
        try:
            model_configs = [
                {'regime': 'trending', 'timeframe': '1h'},
                {'regime': 'ranging', 'timeframe': '1h'},
                {'regime': 'volatile', 'timeframe': '1h'},
                {'regime': 'trending', 'timeframe': '4h'},
                {'regime': 'ranging', 'timeframe': '4h'},
            ]
            
            for config in model_configs:
                model_key = f"{config['regime']}_{config['timeframe']}"
                
                try:
                    # Initialize BiConNet model
                    model = BiConNet(
                        input_size=self.feature_engineer.get_feature_count(),
                        sequence_length=self.sequence_length,
                        cnn_filters=self.model_params['cnn_filters'],
                        lstm_units=self.model_params['lstm_units'],
                        dropout_rate=self.model_params['dropout_rate']
                    )
                    
                    # Load model weights if available
                    model_path = f"models/weights/biconnet_{model_key}.pth"
                    try:
                        model.load_state_dict(torch.load(model_path, map_location='cpu'))
                        model.eval()
                        logger.info(f"Loaded model weights for {model_key}")
                    except FileNotFoundError:
                        logger.warning(f"No pre-trained weights found for {model_key}, using random initialization")
                    
                    self.models[model_key] = model
                    
                except Exception as e:
                    logger.error(f"Failed to load model {model_key}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    async def _get_model_predictions(self, symbol: str, features: pd.DataFrame, 
                                   market_regime: str) -> Dict[str, float]:
        """Get predictions from BiConNet models"""
        try:
            predictions = {}
            
            # Select appropriate models based on market regime
            model_keys = [f"{market_regime}_1h", f"{market_regime}_4h"]
            
            for model_key in model_keys:
                if model_key in self.models:
                    model = self.models[model_key]
                    
                    # Prepare input data
                    input_data = await self._prepare_model_input(features)
                    
                    # Get prediction
                    with torch.no_grad():
                        prediction = model(input_data)
                        predictions[model_key] = {
                            'price_change': float(prediction[0][0]),
                            'confidence': float(prediction[0][1]) if prediction.shape[1] > 1 else 0.5,
                            'volatility': float(prediction[0][2]) if prediction.shape[1] > 2 else 0.1
                        }
            
            # Ensemble predictions if multiple models
            if len(predictions) > 1:
                ensemble_pred = await self._ensemble_predictions(predictions)
                predictions['ensemble'] = ensemble_pred
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting model predictions: {str(e)}")
            return {}
    
    async def _prepare_model_input(self, features: pd.DataFrame) -> torch.Tensor:
        """Prepare features for model input"""
        try:
            # Get last sequence_length rows
            if len(features) < self.sequence_length:
                # Pad with zeros if insufficient data
                padding_length = self.sequence_length - len(features)
                padding = pd.DataFrame(
                    np.zeros((padding_length, features.shape[1])),
                    columns=features.columns
                )
                features = pd.concat([padding, features], ignore_index=True)
            
            # Take last sequence_length rows
            sequence_data = features.tail(self.sequence_length).values
            
            # Normalize features
            sequence_data = (sequence_data - np.mean(sequence_data, axis=0)) / (np.std(sequence_data, axis=0) + 1e-8)
            
            # Convert to tensor
            tensor_data = torch.FloatTensor(sequence_data).unsqueeze(0)  # Add batch dimension
            
            return tensor_data
            
        except Exception as e:
            logger.error(f"Error preparing model input: {str(e)}")
            return torch.zeros(1, self.sequence_length, self.feature_engineer.get_feature_count())
    
    async def _ensemble_predictions(self, predictions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Ensemble multiple model predictions"""
        try:
            weights = {'1h': 0.6, '4h': 0.4}  # Short-term bias
            
            ensemble_price_change = 0.0
            ensemble_confidence = 0.0
            ensemble_volatility = 0.0
            total_weight = 0.0
            
            for model_key, pred in predictions.items():
                if model_key != 'ensemble':
                    timeframe = model_key.split('_')[1]
                    weight = weights.get(timeframe, 0.5)
                    
                    ensemble_price_change += pred['price_change'] * weight
                    ensemble_confidence += pred['confidence'] * weight
                    ensemble_volatility += pred['volatility'] * weight
                    total_weight += weight
            
            if total_weight > 0:
                return {
                    'price_change': ensemble_price_change / total_weight,
                    'confidence': ensemble_confidence / total_weight,
                    'volatility': ensemble_volatility / total_weight
                }
            else:
                return {'price_change': 0.0, 'confidence': 0.5, 'volatility': 0.1}
                
        except Exception as e:
            logger.error(f"Error ensembling predictions: {str(e)}")
            return {'price_change': 0.0, 'confidence': 0.5, 'volatility': 0.1}
    
    async def _analyze_predictions(self, symbol: str, predictions: Dict[str, Dict[str, float]], 
                                 features: pd.DataFrame, current_price: float, 
                                 market_regime: str) -> Optional[TradingSignal]:
        """Analyze predictions and generate trading signal"""
        try:
            # Use ensemble prediction if available, otherwise use best single prediction
            main_pred = predictions.get('ensemble', list(predictions.values())[0] if predictions else {})
            
            if not main_pred:
                return None
            
            price_change = main_pred['price_change']
            confidence = main_pred['confidence']
            volatility = main_pred['volatility']
            
            # Determine signal type and strength
            signal_type = SignalType.HOLD
            signal_strength = SignalStrength.WEAK
            reasoning = []
            
            # Apply confidence and magnitude thresholds
            if confidence < self.confidence_threshold:
                reasoning.append(f"Low confidence: {confidence:.3f}")
                return None
            
            # Determine signal direction
            if abs(price_change) > self.signal_threshold:
                if price_change > 0:
                    if price_change > 0.05:  # 5% expected gain
                        signal_type = SignalType.STRONG_BUY
                        signal_strength = SignalStrength.VERY_STRONG
                    elif price_change > 0.03:  # 3% expected gain
                        signal_type = SignalType.BUY
                        signal_strength = SignalStrength.STRONG
                    else:
                        signal_type = SignalType.BUY
                        signal_strength = SignalStrength.MODERATE
                    
                    reasoning.append(f"Bullish prediction: {price_change:.3f}")
                else:
                    if price_change < -0.05:  # 5% expected loss
                        signal_type = SignalType.STRONG_SELL
                        signal_strength = SignalStrength.VERY_STRONG
                    elif price_change < -0.03:  # 3% expected loss
                        signal_type = SignalType.SELL
                        signal_strength = SignalStrength.STRONG
                    else:
                        signal_type = SignalType.SELL
                        signal_strength = SignalStrength.MODERATE
                    
                    reasoning.append(f"Bearish prediction: {price_change:.3f}")
            
            # Additional technical analysis confirmation
            technical_signals = await self._get_technical_confirmations(features)
            reasoning.extend(technical_signals['reasoning'])
            
            # Adjust confidence based on technical confirmation
            technical_confidence = technical_signals['confidence']
            final_confidence = (confidence + technical_confidence) / 2
            
            # Calculate risk score
            risk_score = await self._calculate_risk_score(volatility, market_regime, features)
            
            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=final_confidence,
                strength=signal_strength,
                price=current_price,
                timestamp=datetime.now(),
                features=self._extract_key_features(features),
                model_predictions=main_pred,
                risk_score=risk_score,
                market_regime=market_regime,
                reasoning=reasoning,
                metadata={
                    'model_count': len(predictions),
                    'volatility': volatility,
                    'technical_confirmation': technical_confidence
                }
            )
            
            logger.info(f"Generated signal for {symbol}: {signal_type.value} (confidence: {final_confidence:.3f})")
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing predictions: {str(e)}")
            return None
    
    async def _get_technical_confirmations(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Get technical analysis confirmations"""
        try:
            reasoning = []
            confirmations = 0
            total_indicators = 0
            
            # Check key technical indicators
            if not features.empty:
                latest = features.iloc[-1]
                
                # RSI confirmation
                if 'rsi' in features.columns:
                    rsi = latest['rsi']
                    total_indicators += 1
                    if rsi < 30:
                        confirmations += 1
                        reasoning.append("RSI oversold")
                    elif rsi > 70:
                        confirmations += 1
                        reasoning.append("RSI overbought")
                
                # MACD confirmation
                if 'macd' in features.columns and 'macd_signal' in features.columns:
                    macd = latest['macd']
                    macd_signal = latest['macd_signal']
                    total_indicators += 1
                    if macd > macd_signal:
                        confirmations += 1
                        reasoning.append("MACD bullish crossover")
                    elif macd < macd_signal:
                        confirmations += 1
                        reasoning.append("MACD bearish crossover")
                
                # Bollinger Bands confirmation
                if all(col in features.columns for col in ['bb_upper', 'bb_lower', 'close']):
                    price = latest['close']
                    bb_upper = latest['bb_upper']
                    bb_lower = latest['bb_lower']
                    total_indicators += 1
                    if price < bb_lower:
                        confirmations += 1
                        reasoning.append("Price below lower Bollinger Band")
                    elif price > bb_upper:
                        confirmations += 1
                        reasoning.append("Price above upper Bollinger Band")
            
            # Calculate confirmation confidence
            confirmation_confidence = confirmations / max(total_indicators, 1) if total_indicators > 0 else 0.5
            
            return {
                'confidence': confirmation_confidence,
                'reasoning': reasoning,
                'confirmations': confirmations,
                'total_indicators': total_indicators
            }
            
        except Exception as e:
            logger.error(f"Error getting technical confirmations: {str(e)}")
            return {'confidence': 0.5, 'reasoning': [], 'confirmations': 0, 'total_indicators': 0}
    
    async def _calculate_risk_score(self, volatility: float, market_regime: str, 
                                  features: pd.DataFrame) -> float:
        """Calculate risk score for the signal"""
        try:
            risk_factors = []
            
            # Volatility risk
            vol_risk = min(volatility * 10, 1.0)  # Cap at 1.0
            risk_factors.append(vol_risk)
            
            # Market regime risk
            regime_risk = {
                'trending': 0.3,
                'ranging': 0.5,
                'volatile': 0.8,
                'unknown': 0.6
            }.get(market_regime, 0.6)
            risk_factors.append(regime_risk)
            
            # Technical risk (based on indicator divergence)
            if not features.empty:
                latest = features.iloc[-1]
                tech_risk = 0.4  # Base technical risk
                
                # Check for extreme values
                if 'rsi' in features.columns:
                    rsi = latest['rsi']
                    if rsi > 80 or rsi < 20:
                        tech_risk += 0.2
                
                risk_factors.append(tech_risk)
            
            # Calculate weighted average risk score
            final_risk = np.mean(risk_factors)
            return min(max(final_risk, 0.0), 1.0)  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {str(e)}")
            return 0.5
    
    def _extract_key_features(self, features: pd.DataFrame) -> Dict[str, float]:
        """Extract key features for signal metadata"""
        try:
            if features.empty:
                return {}
            
            latest = features.iloc[-1]
            key_features = {}
            
            # Extract important indicators
            important_cols = ['rsi', 'macd', 'bb_upper', 'bb_lower', 'sma_20', 'ema_12', 'volume_ratio']
            
            for col in important_cols:
                if col in features.columns:
                    key_features[col] = float(latest[col])
            
            return key_features
            
        except Exception as e:
            logger.error(f"Error extracting key features: {str(e)}")
            return {}
    
    async def _update_signal_history(self, symbol: str, signal: TradingSignal):
        """Update signal history for pattern analysis"""
        try:
            if symbol not in self.signal_history:
                self.signal_history[symbol] = []
            
            # Keep last 100 signals
            self.signal_history[symbol].append({
                'timestamp': signal.timestamp,
                'signal_type': signal.signal_type.value,
                'confidence': signal.confidence,
                'price': signal.price
            })
            
            # Trim history
            if len(self.signal_history[symbol]) > 100:
                self.signal_history[symbol] = self.signal_history[symbol][-100:]
                
        except Exception as e:
            logger.error(f"Error updating signal history: {str(e)}")
    
    async def get_signal_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get signal statistics for a symbol"""
        try:
            if symbol not in self.signal_history:
                return {}
            
            history = self.signal_history[symbol]
            
            # Calculate statistics
            total_signals = len(history)
            buy_signals = sum(1 for s in history if 'BUY' in s['signal_type'])
            sell_signals = sum(1 for s in history if 'SELL' in s['signal_type'])
            avg_confidence = np.mean([s['confidence'] for s in history])
            
            return {
                'total_signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'buy_ratio': buy_signals / max(total_signals, 1),
                'sell_ratio': sell_signals / max(total_signals, 1),
                'average_confidence': avg_confidence,
                'last_signal_time': history[-1]['timestamp'] if history else None
            }
            
        except Exception as e:
            logger.error(f"Error getting signal statistics: {str(e)}")
            return {}
    
    async def batch_generate_signals(self, symbols: List[str], 
                                   market_data: Dict[str, pd.DataFrame]) -> Dict[str, TradingSignal]:
        """Generate signals for multiple symbols in batch"""
        try:
            signals = {}
            tasks = []
            
            for symbol in symbols:
                if symbol in market_data:
                    data = market_data[symbol]
                    current_price = data['close'].iloc[-1] if not data.empty else 0.0
                    
                    task = self.generate_signal(symbol, data, current_price)
                    tasks.append((symbol, task))
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (symbol, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    logger.error(f"Error generating signal for {symbol}: {str(result)}")
                elif result is not None:
                    signals[symbol] = result
            
            logger.info(f"Generated {len(signals)} signals from {len(symbols)} symbols")
            return signals
            
        except Exception as e:
            logger.error(f"Error in batch signal generation: {str(e)}")
            return {}
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Clear models from memory
            for model in self.models.values():
                del model
            self.models.clear()
            
            # Clear signal history
            self.signal_history.clear()
            
            logger.info("SignalDetector cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
