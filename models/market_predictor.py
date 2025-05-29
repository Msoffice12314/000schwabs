"""
Schwab AI Trading System - Market Prediction Engine
Advanced AI price prediction using BiConNet neural networks with ensemble methods and uncertainty quantification.
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from models.biconnet import BiConNet
from data.feature_engineer import FeatureEngineer
from data.market_regime import MarketRegimeDetector
from utils.cache_manager import CacheManager
from utils.database import DatabaseManager
from config.settings import get_settings

logger = logging.getLogger(__name__)

class PredictionHorizon(Enum):
    """Prediction time horizons"""
    INTRADAY_1H = "1h"
    INTRADAY_4H = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1m"

class PredictionType(Enum):
    """Types of predictions"""
    PRICE = "price"
    RETURN = "return"
    DIRECTION = "direction"
    VOLATILITY = "volatility"

@dataclass
class PredictionTarget:
    """Target variable configuration"""
    name: str
    horizon: PredictionHorizon
    type: PredictionType
    lookahead_periods: int
    transformation: str = "none"  # none, log, diff, pct_change

@dataclass
class ModelPrediction:
    """Single model prediction result"""
    model_name: str
    prediction: float
    confidence: float
    lower_bound: float
    upper_bound: float
    volatility_forecast: float
    feature_importance: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class EnsemblePrediction:
    """Ensemble prediction result"""
    symbol: str
    target: PredictionTarget
    timestamp: datetime
    prediction: float
    confidence: float
    lower_bound: float
    upper_bound: float
    volatility_forecast: float
    individual_predictions: List[ModelPrediction]
    model_weights: Dict[str, float]
    market_regime: str
    prediction_quality: float
    reasoning: List[str]

@dataclass
class PredictionPerformance:
    """Model performance metrics"""
    model_name: str
    mse: float
    mae: float
    rmse: float
    r2_score: float
    directional_accuracy: float
    sharpe_ratio: float
    max_error: float
    mean_confidence: float
    calibration_error: float

class MarketPredictor:
    """
    Advanced market prediction engine using ensemble of BiConNet models
    with uncertainty quantification and regime-adaptive predictions.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.feature_engineer = FeatureEngineer()
        self.regime_detector = MarketRegimeDetector()
        self.cache_manager = CacheManager()
        self.db_manager = DatabaseManager()
        
        # Model configuration
        self.model_params = self.settings.model_params
        self.sequence_length = self.model_params['sequence_length']
        
        # Prediction targets
        self.prediction_targets = [
            PredictionTarget(
                name="price_1h",
                horizon=PredictionHorizon.INTRADAY_1H,
                type=PredictionType.RETURN,
                lookahead_periods=1,
                transformation="pct_change"
            ),
            PredictionTarget(
                name="price_4h", 
                horizon=PredictionHorizon.INTRADAY_4H,
                type=PredictionType.RETURN,
                lookahead_periods=4,
                transformation="pct_change"
            ),
            PredictionTarget(
                name="price_1d",
                horizon=PredictionHorizon.DAILY,
                type=PredictionType.RETURN,
                lookahead_periods=1,
                transformation="pct_change"
            ),
            PredictionTarget(
                name="volatility_1d",
                horizon=PredictionHorizon.DAILY,
                type=PredictionType.VOLATILITY,
                lookahead_periods=1,
                transformation="log"
            )
        ]
        
        # Model ensemble
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
        
        # Prediction cache
        self.prediction_cache = {}
        
        self.is_initialized = False
        
        logger.info("MarketPredictor initialized")
    
    async def initialize(self) -> bool:
        """Initialize the market predictor"""
        try:
            # Initialize dependencies
            await self.feature_engineer.initialize()
            await self.regime_detector.initialize()
            
            # Load pre-trained models
            await self._load_models()
            
            # Load model performance metrics
            await self._load_performance_metrics()
            
            self.is_initialized = True
            logger.info("MarketPredictor initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MarketPredictor: {str(e)}")
            return False
    
    async def _load_models(self):
        """Load pre-trained models for different regimes and targets"""
        try:
            regimes = ['trending', 'ranging', 'volatile']
            
            for target in self.prediction_targets:
                for regime in regimes:
                    model_key = f"{target.name}_{regime}"
                    
                    # Initialize BiConNet model
                    model = BiConNet(
                        input_size=self.feature_engineer.get_feature_count(),
                        sequence_length=self.sequence_length,
                        cnn_filters=self.model_params['cnn_filters'],
                        lstm_units=self.model_params['lstm_units'],
                        dropout_rate=self.model_params['dropout_rate'],
                        output_size=3  # prediction, confidence, volatility
                    )
                    
                    # Load model weights if available
                    model_path = f"models/weights/predictor_{model_key}.pth"
                    try:
                        state_dict = torch.load(model_path, map_location='cpu')
                        model.load_state_dict(state_dict)
                        model.eval()
                        logger.info(f"Loaded model weights for {model_key}")
                    except FileNotFoundError:
                        logger.warning(f"No pre-trained weights found for {model_key}")
                        # Initialize with random weights
                        pass
                    
                    self.models[model_key] = model
                    
                    # Load corresponding scaler
                    scaler_path = f"models/scalers/scaler_{model_key}.pkl"
                    try:
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)
                        self.scalers[model_key] = scaler
                        logger.info(f"Loaded scaler for {model_key}")
                    except FileNotFoundError:
                        logger.warning(f"No scaler found for {model_key}, using default")
                        self.scalers[model_key] = StandardScaler()
            
            logger.info(f"Loaded {len(self.models)} prediction models")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    async def _load_performance_metrics(self):
        """Load historical model performance metrics"""
        try:
            query = """
                SELECT model_name, mse, mae, rmse, r2_score, directional_accuracy,
                       sharpe_ratio, max_error, mean_confidence, calibration_error,
                       last_updated
                FROM model_performance 
                WHERE last_updated >= %s
            """
            
            # Load metrics from last 30 days
            cutoff_date = datetime.now() - timedelta(days=30)
            results = await self.db_manager.execute_query(query, (cutoff_date,))
            
            for row in results:
                perf = PredictionPerformance(
                    model_name=row['model_name'],
                    mse=row['mse'],
                    mae=row['mae'], 
                    rmse=row['rmse'],
                    r2_score=row['r2_score'],
                    directional_accuracy=row['directional_accuracy'],
                    sharpe_ratio=row['sharpe_ratio'],
                    max_error=row['max_error'],
                    mean_confidence=row['mean_confidence'],
                    calibration_error=row['calibration_error']
                )
                self.model_performance[row['model_name']] = perf
            
            logger.info(f"Loaded performance metrics for {len(self.model_performance)} models")
            
        except Exception as e:
            logger.error(f"Error loading performance metrics: {str(e)}")
    
    async def predict(self, symbol: str, data: pd.DataFrame, 
                     target: Optional[PredictionTarget] = None) -> Optional[EnsemblePrediction]:
        """
        Generate ensemble prediction for a symbol
        
        Args:
            symbol: Trading symbol
            data: Historical OHLCV data
            target: Specific prediction target (if None, uses default)
            
        Returns:
            EnsemblePrediction with ensemble results
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if data.empty or len(data) < self.sequence_length:
                logger.warning(f"Insufficient data for prediction: {len(data)} rows")
                return None
            
            # Use default target if not specified
            if target is None:
                target = self.prediction_targets[0]  # 1h return prediction
            
            # Check cache first
            cache_key = f"pred_{symbol}_{target.name}_{int(datetime.now().timestamp() / 300)}"  # 5 min cache
            cached_prediction = await self.cache_manager.get(cache_key)
            if cached_prediction:
                return cached_prediction
            
            # Detect market regime
            market_regime = await self.regime_detector.detect_regime(data)
            
            # Engineer features
            features = await self.feature_engineer.create_features(data)
            
            if features.empty:
                logger.error("Failed to engineer features")
                return None
            
            # Get individual model predictions
            individual_predictions = await self._get_individual_predictions(
                symbol, features, target, market_regime
            )
            
            if not individual_predictions:
                logger.error("No individual predictions generated")
                return None
            
            # Calculate ensemble prediction
            ensemble_prediction = await self._calculate_ensemble_prediction(
                symbol, target, individual_predictions, market_regime
            )
            
            # Cache the prediction
            await self.cache_manager.set(cache_key, ensemble_prediction, expire=300)
            
            # Store prediction for performance tracking
            await self._store_prediction(ensemble_prediction)
            
            logger.info(f"Generated prediction for {symbol}: {ensemble_prediction.prediction:.4f} "
                       f"(confidence: {ensemble_prediction.confidence:.3f})")
            
            return ensemble_prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction for {symbol}: {str(e)}")
            return None
    
    async def _get_individual_predictions(self, symbol: str, features: pd.DataFrame,
                                        target: PredictionTarget, 
                                        market_regime: str) -> List[ModelPrediction]:
        """Get predictions from individual models"""
        try:
            individual_predictions = []
            
            # Regime-specific models
            regime_models = [
                f"{target.name}_{market_regime}",
                f"{target.name}_trending",
                f"{target.name}_ranging",
                f"{target.name}_volatile"
            ]
            
            for model_key in regime_models:
                if model_key in self.models:
                    prediction = await self._get_model_prediction(
                        model_key, features, target
                    )
                    
                    if prediction:
                        individual_predictions.append(prediction)
            
            return individual_predictions
            
        except Exception as e:
            logger.error(f"Error getting individual predictions: {str(e)}")
            return []
    
    async def _get_model_prediction(self, model_key: str, features: pd.DataFrame,
                                  target: PredictionTarget) -> Optional[ModelPrediction]:
        """Get prediction from a specific model"""
        try:
            model = self.models[model_key]
            scaler = self.scalers[model_key]
            
            # Prepare input data
            input_data = await self._prepare_model_input(features, scaler)
            
            if input_data is None:
                return None
            
            # Get prediction
            with torch.no_grad():
                model_output = model(input_data)
                
                # Extract outputs
                if model_output.shape[1] >= 3:
                    prediction = float(model_output[0][0])
                    confidence = float(torch.sigmoid(model_output[0][1]))  # Convert to probability
                    volatility = float(torch.exp(model_output[0][2]))  # Exp to ensure positive
                else:
                    prediction = float(model_output[0][0])
                    confidence = 0.5  # Default confidence
                    volatility = 0.02  # Default volatility
            
            # Calculate prediction bounds
            std_dev = volatility
            lower_bound = prediction - 1.96 * std_dev  # 95% confidence interval
            upper_bound = prediction + 1.96 * std_dev
            
            # Get feature importance (simplified)
            feature_importance = await self._calculate_feature_importance(
                model, input_data, features.columns[-10:]  # Top 10 features
            )
            
            model_prediction = ModelPrediction(
                model_name=model_key,
                prediction=prediction,
                confidence=confidence,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                volatility_forecast=volatility,
                feature_importance=feature_importance,
                metadata={
                    'model_type': 'BiConNet',
                    'sequence_length': self.sequence_length,
                    'feature_count': len(features.columns)
                }
            )
            
            return model_prediction
            
        except Exception as e:
            logger.error(f"Error getting prediction from {model_key}: {str(e)}")
            return None
    
    async def _prepare_model_input(self, features: pd.DataFrame, 
                                 scaler: StandardScaler) -> Optional[torch.Tensor]:
        """Prepare features for model input"""
        try:
            if len(features) < self.sequence_length:
                logger.warning("Insufficient features for sequence")
                return None
            
            # Get last sequence_length rows
            sequence_data = features.tail(self.sequence_length).values
            
            # Handle NaN values
            if np.isnan(sequence_data).any():
                sequence_data = np.nan_to_num(sequence_data, nan=0.0)
            
            # Scale features
            try:
                sequence_data_scaled = scaler.transform(sequence_data)
            except:
                # If scaler not fitted, fit on current data
                sequence_data_scaled = scaler.fit_transform(sequence_data)
            
            # Convert to tensor
            tensor_data = torch.FloatTensor(sequence_data_scaled).unsqueeze(0)  # Add batch dimension
            
            return tensor_data
            
        except Exception as e:
            logger.error(f"Error preparing model input: {str(e)}")
            return None
    
    async def _calculate_feature_importance(self, model: nn.Module, 
                                          input_data: torch.Tensor,
                                          feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance using gradient-based attribution"""
        try:
            # Simplified feature importance (would use proper attribution methods in production)
            importance = {}
            
            # Use last layer weights as proxy for importance
            if hasattr(model, 'output_layer'):
                weights = model.output_layer.weight.data.abs().mean(dim=0)
                
                # Map to feature names (simplified)
                for i, name in enumerate(feature_names):
                    if i < len(weights):
                        importance[name] = float(weights[i])
                    else:
                        importance[name] = 0.0
            else:
                # Default uniform importance
                for name in feature_names:
                    importance[name] = 1.0 / len(feature_names)
            
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {name: 0.0 for name in feature_names}
    
    async def _calculate_ensemble_prediction(self, symbol: str, target: PredictionTarget,
                                           individual_predictions: List[ModelPrediction],
                                           market_regime: str) -> EnsemblePrediction:
        """Calculate ensemble prediction from individual models"""
        try:
            # Calculate model weights based on performance
            model_weights = await self._calculate_model_weights(individual_predictions, market_regime)
            
            # Weighted ensemble prediction
            weighted_prediction = 0.0
            weighted_confidence = 0.0
            weighted_volatility = 0.0
            total_weight = 0.0
            
            lower_bounds = []
            upper_bounds = []
            
            for pred in individual_predictions:
                weight = model_weights.get(pred.model_name, 1.0)
                
                weighted_prediction += pred.prediction * weight
                weighted_confidence += pred.confidence * weight
                weighted_volatility += pred.volatility_forecast * weight
                total_weight += weight
                
                lower_bounds.append(pred.lower_bound)
                upper_bounds.append(pred.upper_bound)
            
            # Normalize by total weight
            if total_weight > 0:
                ensemble_prediction = weighted_prediction / total_weight
                ensemble_confidence = weighted_confidence / total_weight
                ensemble_volatility = weighted_volatility / total_weight
            else:
                ensemble_prediction = np.mean([p.prediction for p in individual_predictions])
                ensemble_confidence = np.mean([p.confidence for p in individual_predictions])
                ensemble_volatility = np.mean([p.volatility_forecast for p in individual_predictions])
            
            # Ensemble bounds (conservative approach)
            ensemble_lower = np.min(lower_bounds)
            ensemble_upper = np.max(upper_bounds)
            
            # Calculate prediction quality
            prediction_quality = await self._calculate_prediction_quality(
                individual_predictions, ensemble_confidence, market_regime
            )
            
            # Generate reasoning
            reasoning = await self._generate_prediction_reasoning(
                individual_predictions, model_weights, market_regime
            )
            
            ensemble_pred = EnsemblePrediction(
                symbol=symbol,
                target=target,
                timestamp=datetime.now(),
                prediction=ensemble_prediction,
                confidence=ensemble_confidence,
                lower_bound=ensemble_lower,
                upper_bound=ensemble_upper,
                volatility_forecast=ensemble_volatility,
                individual_predictions=individual_predictions,
                model_weights=model_weights,
                market_regime=market_regime,
                prediction_quality=prediction_quality,
                reasoning=reasoning
            )
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error calculating ensemble prediction: {str(e)}")
            # Return default prediction
            return EnsemblePrediction(
                symbol=symbol,
                target=target,
                timestamp=datetime.now(),
                prediction=0.0,
                confidence=0.5,
                lower_bound=-0.02,
                upper_bound=0.02,
                volatility_forecast=0.02,
                individual_predictions=individual_predictions,
                model_weights={},
                market_regime=market_regime,
                prediction_quality=0.5,
                reasoning=["Error in ensemble calculation"]
            )
    
    async def _calculate_model_weights(self, individual_predictions: List[ModelPrediction],
                                     market_regime: str) -> Dict[str, float]:
        """Calculate weights for ensemble based on model performance"""
        try:
            weights = {}
            
            for pred in individual_predictions:
                model_name = pred.model_name
                
                # Base weight from model performance
                if model_name in self.model_performance:
                    perf = self.model_performance[model_name]
                    
                    # Combine multiple metrics for weight calculation
                    accuracy_weight = perf.directional_accuracy
                    confidence_weight = 1.0 - perf.calibration_error
                    sharpe_weight = max(0, perf.sharpe_ratio) / 3.0  # Normalize Sharpe
                    
                    performance_weight = (accuracy_weight + confidence_weight + sharpe_weight) / 3.0
                else:
                    performance_weight = 0.5  # Default weight
                
                # Regime-specific adjustment
                if market_regime in model_name:
                    regime_bonus = 1.2  # 20% bonus for regime-specific models
                else:
                    regime_bonus = 1.0
                
                # Confidence-based adjustment
                confidence_adjustment = pred.confidence
                
                # Final weight
                final_weight = performance_weight * regime_bonus * confidence_adjustment
                weights[model_name] = final_weight
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating model weights: {str(e)}")
            return {pred.model_name: 1.0 for pred in individual_predictions}
    
    async def _calculate_prediction_quality(self, individual_predictions: List[ModelPrediction],
                                          ensemble_confidence: float, 
                                          market_regime: str) -> float:
        """Calculate overall prediction quality score"""
        try:
            quality_factors = []
            
            # Model agreement
            predictions = [p.prediction for p in individual_predictions]
            prediction_std = np.std(predictions)
            agreement_score = 1.0 / (1.0 + prediction_std)  # Higher agreement = higher quality
            quality_factors.append(agreement_score)
            
            # Average confidence
            quality_factors.append(ensemble_confidence)
            
            # Number of models (more models = higher quality)
            model_count_score = min(len(individual_predictions) / 5.0, 1.0)  # Max at 5 models
            quality_factors.append(model_count_score)
            
            # Regime-specific model availability
            regime_model_available = any(market_regime in p.model_name for p in individual_predictions)
            regime_score = 1.0 if regime_model_available else 0.8
            quality_factors.append(regime_score)
            
            # Overall quality score
            quality_score = np.mean(quality_factors)
            
            return min(max(quality_score, 0.0), 1.0)  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating prediction quality: {str(e)}")
            return 0.5
    
    async def _generate_prediction_reasoning(self, individual_predictions: List[ModelPrediction],
                                           model_weights: Dict[str, float],
                                           market_regime: str) -> List[str]:
        """Generate human-readable reasoning for the prediction"""
        try:
            reasoning = []
            
            # Model consensus
            predictions = [p.prediction for p in individual_predictions]
            avg_prediction = np.mean(predictions)
            prediction_std = np.std(predictions)
            
            if prediction_std < 0.005:  # Low disagreement
                reasoning.append(f"Strong model consensus with {len(predictions)} models agreeing")
            elif prediction_std > 0.02:  # High disagreement
                reasoning.append(f"Model disagreement detected (std: {prediction_std:.4f})")
            
            # Direction and magnitude
            if avg_prediction > 0.01:
                reasoning.append(f"Bullish prediction: {avg_prediction:.2%} expected return")
            elif avg_prediction < -0.01:
                reasoning.append(f"Bearish prediction: {avg_prediction:.2%} expected return")
            else:
                reasoning.append("Neutral prediction with limited directional bias")
            
            # Market regime influence
            reasoning.append(f"Market regime: {market_regime}")
            
            # Best performing model
            if model_weights:
                best_model = max(model_weights, key=model_weights.get)
                best_weight = model_weights[best_model]
                reasoning.append(f"Highest weight model: {best_model} ({best_weight:.2f})")
            
            # Confidence level
            avg_confidence = np.mean([p.confidence for p in individual_predictions])
            if avg_confidence > 0.8:
                reasoning.append("High confidence prediction")
            elif avg_confidence < 0.6:
                reasoning.append("Low confidence prediction - use caution")
            
            return reasoning
            
        except Exception as e:
            logger.error(f"Error generating prediction reasoning: {str(e)}")
            return ["Prediction generated with ensemble methods"]
    
    async def _store_prediction(self, prediction: EnsemblePrediction):
        """Store prediction in database for performance tracking"""
        try:
            insert_query = """
                INSERT INTO predictions (symbol, target_name, prediction_value, confidence,
                                       lower_bound, upper_bound, volatility_forecast,
                                       market_regime, prediction_quality, model_count,
                                       timestamp, horizon, prediction_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                prediction.symbol,
                prediction.target.name,
                prediction.prediction,
                prediction.confidence,
                prediction.lower_bound,
                prediction.upper_bound,
                prediction.volatility_forecast,
                prediction.market_regime,
                prediction.prediction_quality,
                len(prediction.individual_predictions),
                prediction.timestamp,
                prediction.target.horizon.value,
                prediction.target.type.value
            )
            
            await self.db_manager.execute_query(insert_query, values)
            
        except Exception as e:
            logger.error(f"Error storing prediction: {str(e)}")
    
    async def batch_predict(self, symbols: List[str], 
                           market_data: Dict[str, pd.DataFrame],
                           target: Optional[PredictionTarget] = None) -> Dict[str, EnsemblePrediction]:
        """Generate predictions for multiple symbols in batch"""
        try:
            predictions = {}
            
            # Create prediction tasks
            tasks = []
            for symbol in symbols:
                if symbol in market_data:
                    data = market_data[symbol]
                    task = self.predict(symbol, data, target)
                    tasks.append((symbol, task))
            
            # Execute tasks concurrently
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (symbol, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    logger.error(f"Error predicting {symbol}: {str(result)}")
                elif result is not None:
                    predictions[symbol] = result
            
            logger.info(f"Generated predictions for {len(predictions)} symbols")
            return predictions
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            return {}
    
    async def evaluate_predictions(self, symbol: str, 
                                 lookback_days: int = 30) -> Optional[Dict[str, Any]]:
        """Evaluate historical prediction performance"""
        try:
            # Get historical predictions
            query = """
                SELECT p.*, m.close as actual_price
                FROM predictions p
                JOIN market_data m ON p.symbol = m.symbol 
                    AND DATE(p.timestamp + INTERVAL p.horizon) = DATE(m.timestamp)
                WHERE p.symbol = %s 
                    AND p.timestamp >= %s
                ORDER BY p.timestamp DESC
            """
            
            lookback_date = datetime.now() - timedelta(days=lookback_days)
            results = await self.db_manager.execute_query(query, (symbol, lookback_date))
            
            if not results or len(results) < 5:
                logger.warning(f"Insufficient prediction history for evaluation: {len(results) if results else 0}")
                return None
            
            # Calculate metrics
            predictions = [r['prediction_value'] for r in results]
            actual_values = [r['actual_price'] for r in results]
            confidences = [r['confidence'] for r in results]
            
            # Performance metrics
            mse = mean_squared_error(actual_values, predictions)
            mae = mean_absolute_error(actual_values, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(actual_values, predictions)
            
            # Directional accuracy
            pred_directions = np.sign(predictions)
            actual_directions = np.sign(actual_values)
            directional_accuracy = np.mean(pred_directions == actual_directions)
            
            # Calibration (simplified)
            calibration_error = np.mean(np.abs(np.array(confidences) - directional_accuracy))
            
            evaluation = {
                'symbol': symbol,
                'evaluation_period': lookback_days,
                'sample_size': len(results),
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2,
                'directional_accuracy': directional_accuracy,
                'mean_confidence': np.mean(confidences),
                'calibration_error': calibration_error,
                'last_updated': datetime.now()
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating predictions for {symbol}: {str(e)}")
            return None
    
    async def get_prediction_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Get summary of recent predictions"""
        try:
            summary = {
                'total_symbols': len(symbols),
                'predictions_generated': 0,
                'avg_confidence': 0.0,
                'bullish_predictions': 0,
                'bearish_predictions': 0,
                'neutral_predictions': 0,
                'high_quality_predictions': 0,
                'by_regime': {},
                'model_usage': {}
            }
            
            # Get recent predictions
            if symbols:
                placeholders = ','.join(['%s'] * len(symbols))
                query = f"""
                    SELECT symbol, prediction_value, confidence, prediction_quality,
                           market_regime, model_count
                    FROM predictions 
                    WHERE symbol IN ({placeholders})
                        AND timestamp >= %s
                    ORDER BY timestamp DESC
                """
                
                recent_date = datetime.now() - timedelta(hours=24)
                params = symbols + [recent_date]
                results = await self.db_manager.execute_query(query, params)
                
                if results:
                    summary['predictions_generated'] = len(results)
                    
                    predictions = [r['prediction_value'] for r in results]
                    confidences = [r['confidence'] for r in results]
                    qualities = [r['prediction_quality'] for r in results]
                    regimes = [r['market_regime'] for r in results]
                    
                    summary['avg_confidence'] = np.mean(confidences)
                    summary['bullish_predictions'] = sum(1 for p in predictions if p > 0.005)
                    summary['bearish_predictions'] = sum(1 for p in predictions if p < -0.005)
                    summary['neutral_predictions'] = len(predictions) - summary['bullish_predictions'] - summary['bearish_predictions']
                    summary['high_quality_predictions'] = sum(1 for q in qualities if q > 0.8)
                    
                    # Regime breakdown
                    regime_counts = pd.Series(regimes).value_counts().to_dict()
                    summary['by_regime'] = regime_counts
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting prediction summary: {str(e)}")
            return summary
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'total_models': len(self.models),
            'model_types': list(set(key.split('_')[0] for key in self.models.keys())),
            'regimes_supported': list(set(key.split('_')[-1] for key in self.models.keys())),
            'prediction_targets': [target.name for target in self.prediction_targets],
            'sequence_length': self.sequence_length,
            'feature_count': self.feature_engineer.get_feature_count(),
            'performance_metrics_available': len(self.model_performance),
            'initialization_status': self.is_initialized
        }
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Clear models from memory
            for model in self.models.values():
                del model
            self.models.clear()
            
            # Clear scalers
            self.scalers.clear()
            
            # Clear cache
            self.prediction_cache.clear()
            
            logger.info("MarketPredictor cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
