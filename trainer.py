#!/usr/bin/env python3
"""
Model Training Script for Schwab AI Trading System
Handles training of all AI models including LSTM, BiConNet, Transformer, and Random Forest
"""

import os
import sys
import argparse
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel(logging.ERROR)

from config import Config
from database import Database
from utils.data_preprocessing import DataPreprocessor
from utils.feature_engineering import FeatureEngineer
from models.lstm_model import LSTMModel
from models.biconnet_model import BiConNetModel
from models.transformer_model import TransformerModel
from models.ensemble_model import EnsembleModel
from utils.model_evaluation import ModelEvaluator
from utils.logger import setup_logger

class ModelTrainer:
    """
    Comprehensive model training system for the Schwab AI Trading System
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the model trainer"""
        self.config = Config(config_path)
        self.db = Database(self.config.DATABASE_URL)
        self.logger = setup_logger('trainer', self.config.LOG_LEVEL)
        
        # Initialize components
        self.data_preprocessor = DataPreprocessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.model_evaluator = ModelEvaluator(self.config)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.model_metrics = {}
        
        # Training parameters
        self.sequence_length = self.config.SEQUENCE_LENGTH
        self.prediction_horizon = self.config.PREDICTION_HORIZON
        self.validation_split = self.config.TRAINING_VALIDATION_SPLIT
        self.batch_size = self.config.TRAINING_BATCH_SIZE
        self.epochs = self.config.TRAINING_EPOCHS
        self.learning_rate = self.config.TRAINING_LEARNING_RATE
        
        self.logger.info("ModelTrainer initialized successfully")

    def prepare_training_data(self, symbols: List[str] = None, 
                            start_date: str = None, 
                            end_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare and preprocess data for training
        """
        self.logger.info("Preparing training data...")
        
        try:
            # Set default parameters
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=self.config.TRAINING_DATA_DAYS)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if symbols is None:
                symbols = self.config.DEFAULT_SYMBOLS
            
            # Fetch market data
            market_data = self.db.get_market_data(symbols, start_date, end_date)
            if market_data.empty:
                raise ValueError("No market data available for training")
            
            # Fetch additional data sources
            news_data = self.db.get_news_data(symbols, start_date, end_date)
            economic_data = self.db.get_economic_data(start_date, end_date)
            
            # Preprocess data
            processed_data = self.data_preprocessor.preprocess_market_data(market_data)
            
            # Feature engineering
            features_df = self.feature_engineer.create_features(
                market_data=processed_data,
                news_data=news_data,
                economic_data=economic_data
            )
            
            # Remove NaN values and ensure data quality
            features_df = features_df.dropna()
            
            if features_df.empty:
                raise ValueError("No valid data after preprocessing")
            
            # Create target variables
            targets_df = self.create_target_variables(features_df)
            
            self.logger.info(f"Training data prepared: {len(features_df)} samples, {len(features_df.columns)} features")
            return features_df, targets_df
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            raise

    def create_target_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for different prediction tasks
        """
        targets = pd.DataFrame(index=data.index)
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            
            # Price prediction targets
            targets[f'{symbol}_price_1d'] = symbol_data['close'].shift(-1)
            targets[f'{symbol}_price_5d'] = symbol_data['close'].shift(-5)
            targets[f'{symbol}_price_10d'] = symbol_data['close'].shift(-10)
            
            # Return prediction targets
            targets[f'{symbol}_return_1d'] = symbol_data['close'].pct_change(1).shift(-1)
            targets[f'{symbol}_return_5d'] = symbol_data['close'].pct_change(5).shift(-5)
            targets[f'{symbol}_return_10d'] = symbol_data['close'].pct_change(10).shift(-10)
            
            # Direction prediction targets (classification)
            targets[f'{symbol}_direction_1d'] = (symbol_data['close'].shift(-1) > symbol_data['close']).astype(int)
            targets[f'{symbol}_direction_5d'] = (symbol_data['close'].shift(-5) > symbol_data['close']).astype(int)
            
            # Volatility prediction targets
            returns = symbol_data['close'].pct_change()
            targets[f'{symbol}_volatility_5d'] = returns.rolling(5).std().shift(-5)
            targets[f'{symbol}_volatility_10d'] = returns.rolling(10).std().shift(-10)
        
        return targets.dropna()

    def prepare_sequences(self, features: pd.DataFrame, targets: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for time series models
        """
        X_sequences = []
        y_sequences = []
        
        # Group by symbol to maintain temporal order
        for symbol in features['symbol'].unique():
            symbol_features = features[features['symbol'] == symbol].drop('symbol', axis=1)
            symbol_targets = targets[targets.index.isin(symbol_features.index)]
            
            # Create sequences
            for i in range(len(symbol_features) - self.sequence_length - self.prediction_horizon + 1):
                X_sequences.append(symbol_features.iloc[i:i+self.sequence_length].values)
                y_sequences.append(symbol_targets.iloc[i+self.sequence_length+self.prediction_horizon-1].values)
        
        return np.array(X_sequences), np.array(y_sequences)

    def train_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.Model:
        """
        Train LSTM model for time series prediction
        """
        self.logger.info("Training LSTM model...")
        
        try:
            # Build LSTM model
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]),
                     kernel_regularizer=l2(0.01)),
                Dropout(0.3),
                LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)),
                Dropout(0.3),
                LSTM(32, kernel_regularizer=l2(0.01)),
                Dropout(0.2),
                Dense(y_train.shape[1], activation='linear')
            ])
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7),
                ModelCheckpoint(
                    filepath=os.path.join(self.config.MODEL_STORAGE_PATH, 'lstm_best.h5'),
                    save_best_only=True,
                    monitor='val_loss'
                )
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            train_loss = model.evaluate(X_train, y_train, verbose=0)
            val_loss = model.evaluate(X_val, y_val, verbose=0)
            
            self.model_metrics['lstm'] = {
                'train_loss': float(train_loss[0]),
                'train_mae': float(train_loss[1]),
                'val_loss': float(val_loss[0]),
                'val_mae': float(val_loss[1]),
                'epochs_trained': len(history.history['loss'])
            }
            
            self.logger.info(f"LSTM model trained - Val Loss: {val_loss[0]:.6f}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {str(e)}")
            raise

    def train_biconnet_model(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.Model:
        """
        Train BiConNet model (Bidirectional Convolutional Network)
        """
        self.logger.info("Training BiConNet model...")
        
        try:
            # Import BiConNet architecture
            biconnet = BiConNetModel(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                output_dim=y_train.shape[1]
            )
            
            model = biconnet.build_model()
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(patience=7, factor=0.3, min_lr=1e-7),
                ModelCheckpoint(
                    filepath=os.path.join(self.config.MODEL_STORAGE_PATH, 'biconnet_best.h5'),
                    save_best_only=True,
                    monitor='val_loss'
                )
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            train_loss = model.evaluate(X_train, y_train, verbose=0)
            val_loss = model.evaluate(X_val, y_val, verbose=0)
            
            self.model_metrics['biconnet'] = {
                'train_loss': float(train_loss[0]),
                'train_mae': float(train_loss[1]),
                'val_loss': float(val_loss[0]),
                'val_mae': float(val_loss[1]),
                'epochs_trained': len(history.history['loss'])
            }
            
            self.logger.info(f"BiConNet model trained - Val Loss: {val_loss[0]:.6f}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error training BiConNet model: {str(e)}")
            raise

    def train_transformer_model(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.Model:
        """
        Train Transformer model for sequence prediction
        """
        self.logger.info("Training Transformer model...")
        
        try:
            # Import Transformer architecture
            transformer = TransformerModel(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                output_dim=y_train.shape[1],
                num_heads=8,
                ff_dim=256,
                num_layers=4
            )
            
            model = transformer.build_model()
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=12, restore_best_weights=True),
                ReduceLROnPlateau(patience=6, factor=0.4, min_lr=1e-7),
                ModelCheckpoint(
                    filepath=os.path.join(self.config.MODEL_STORAGE_PATH, 'transformer_best.h5'),
                    save_best_only=True,
                    monitor='val_loss'
                )
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            train_loss = model.evaluate(X_train, y_train, verbose=0)
            val_loss = model.evaluate(X_val, y_val, verbose=0)
            
            self.model_metrics['transformer'] = {
                'train_loss': float(train_loss[0]),
                'train_mae': float(train_loss[1]),
                'val_loss': float(val_loss[0]),
                'val_mae': float(val_loss[1]),
                'epochs_trained': len(history.history['loss'])
            }
            
            self.logger.info(f"Transformer model trained - Val Loss: {val_loss[0]:.6f}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error training Transformer model: {str(e)}")
            raise

    def train_random_forest_model(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray) -> RandomForestRegressor:
        """
        Train Random Forest model
        """
        self.logger.info("Training Random Forest model...")
        
        try:
            # Flatten sequences for Random Forest
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            
            # Train separate models for each target
            models = {}
            metrics = {}
            
            for i in range(y_train.shape[1]):
                self.logger.info(f"Training Random Forest for target {i+1}/{y_train.shape[1]}")
                
                # Create and train model
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train_flat, y_train[:, i])
                
                # Evaluate
                train_pred = model.predict(X_train_flat)
                val_pred = model.predict(X_val_flat)
                
                train_mse = mean_squared_error(y_train[:, i], train_pred)
                val_mse = mean_squared_error(y_val[:, i], val_pred)
                
                models[f'target_{i}'] = model
                metrics[f'target_{i}'] = {
                    'train_mse': float(train_mse),
                    'val_mse': float(val_mse),
                    'feature_importance': model.feature_importances_.tolist()
                }
            
            self.model_metrics['random_forest'] = metrics
            self.logger.info("Random Forest models trained successfully")
            return models
            
        except Exception as e:
            self.logger.error(f"Error training Random Forest model: {str(e)}")
            raise

    def train_ensemble_model(self) -> EnsembleModel:
        """
        Train ensemble model combining all individual models
        """
        self.logger.info("Training ensemble model...")
        
        try:
            # Create ensemble model
            ensemble = EnsembleModel(
                models=self.models,
                weights={
                    'lstm': self.config.LSTM_MODEL_WEIGHT,
                    'biconnet': self.config.BICONNET_MODEL_WEIGHT,
                    'transformer': self.config.TRANSFORMER_MODEL_WEIGHT,
                    'random_forest': self.config.RANDOM_FOREST_MODEL_WEIGHT
                }
            )
            
            # Train ensemble (meta-learning)
            ensemble.train_meta_learner()
            
            self.logger.info("Ensemble model trained successfully")
            return ensemble
            
        except Exception as e:
            self.logger.error(f"Error training ensemble model: {str(e)}")
            raise

    def save_models(self):
        """
        Save all trained models and scalers
        """
        self.logger.info("Saving models...")
        
        try:
            os.makedirs(self.config.MODEL_STORAGE_PATH, exist_ok=True)
            
            # Save Keras models
            for name, model in self.models.items():
                if hasattr(model, 'save'):
                    model_path = os.path.join(self.config.MODEL_STORAGE_PATH, f'{name}_model.h5')
                    model.save(model_path)
                    self.logger.info(f"Saved {name} model to {model_path}")
            
            # Save Random Forest models
            if 'random_forest' in self.models:
                rf_path = os.path.join(self.config.MODEL_STORAGE_PATH, 'random_forest_models.joblib')
                joblib.dump(self.models['random_forest'], rf_path)
                self.logger.info(f"Saved Random Forest models to {rf_path}")
            
            # Save scalers
            scalers_path = os.path.join(self.config.MODEL_STORAGE_PATH, 'scalers.joblib')
            joblib.dump(self.scalers, scalers_path)
            
            # Save model metrics
            metrics_path = os.path.join(self.config.MODEL_STORAGE_PATH, 'model_metrics.json')
            import json
            with open(metrics_path, 'w') as f:
                json.dump(self.model_metrics, f, indent=2)
            
            # Save training metadata
            metadata = {
                'training_date': datetime.now().isoformat(),
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'model_versions': {name: '1.0' for name in self.models.keys()}
            }
            
            metadata_path = os.path.join(self.config.MODEL_STORAGE_PATH, 'training_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info("All models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise

    def run_training_pipeline(self, symbols: List[str] = None, 
                            retrain: bool = False) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        """
        self.logger.info("Starting training pipeline...")
        
        try:
            # Check if retraining is needed
            if not retrain and self.should_skip_training():
                self.logger.info("Skipping training - models are up to date")
                return {'status': 'skipped', 'reason': 'models_up_to_date'}
            
            # Prepare training data
            features_df, targets_df = self.prepare_training_data(symbols)
            
            # Prepare sequences for time series models
            X_sequences, y_sequences = self.prepare_sequences(features_df, targets_df)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_sequences.reshape(-1, X_sequences.shape[-1]))
            X_scaled = X_scaled.reshape(X_sequences.shape)
            self.scalers['features'] = scaler
            
            # Scale targets
            target_scaler = StandardScaler()
            y_scaled = target_scaler.fit_transform(y_sequences)
            self.scalers['targets'] = target_scaler
            
            # Split data
            split_idx = int(len(X_scaled) * (1 - self.validation_split))
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
            
            self.logger.info(f"Training set: {X_train.shape[0]} samples")
            self.logger.info(f"Validation set: {X_val.shape[0]} samples")
            
            # Train individual models
            training_results = {}
            
            # Train LSTM
            try:
                lstm_model = self.train_lstm_model(X_train, y_train, X_val, y_val)
                self.models['lstm'] = lstm_model
                training_results['lstm'] = 'success'
            except Exception as e:
                self.logger.error(f"LSTM training failed: {str(e)}")
                training_results['lstm'] = f'failed: {str(e)}'
            
            # Train BiConNet
            try:
                biconnet_model = self.train_biconnet_model(X_train, y_train, X_val, y_val)
                self.models['biconnet'] = biconnet_model
                training_results['biconnet'] = 'success'
            except Exception as e:
                self.logger.error(f"BiConNet training failed: {str(e)}")
                training_results['biconnet'] = f'failed: {str(e)}'
            
            # Train Transformer
            try:
                transformer_model = self.train_transformer_model(X_train, y_train, X_val, y_val)
                self.models['transformer'] = transformer_model
                training_results['transformer'] = 'success'
            except Exception as e:
                self.logger.error(f"Transformer training failed: {str(e)}")
                training_results['transformer'] = f'failed: {str(e)}'
            
            # Train Random Forest
            try:
                rf_models = self.train_random_forest_model(X_train, y_train, X_val, y_val)
                self.models['random_forest'] = rf_models
                training_results['random_forest'] = 'success'
            except Exception as e:
                self.logger.error(f"Random Forest training failed: {str(e)}")
                training_results['random_forest'] = f'failed: {str(e)}'
            
            # Train ensemble if we have at least 2 successful models
            successful_models = [k for k, v in training_results.items() if v == 'success']
            if len(successful_models) >= 2:
                try:
                    ensemble_model = self.train_ensemble_model()
                    self.models['ensemble'] = ensemble_model
                    training_results['ensemble'] = 'success'
                except Exception as e:
                    self.logger.error(f"Ensemble training failed: {str(e)}")
                    training_results['ensemble'] = f'failed: {str(e)}'
            
            # Save models
            self.save_models()
            
            # Update database with training results
            self.update_training_results(training_results)
            
            result = {
                'status': 'completed',
                'models_trained': len(successful_models),
                'training_results': training_results,
                'model_metrics': self.model_metrics,
                'training_duration': datetime.now().isoformat()
            }
            
            self.logger.info(f"Training pipeline completed successfully - {len(successful_models)} models trained")
            return result
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def should_skip_training(self) -> bool:
        """
        Check if training should be skipped based on model freshness
        """
        try:
            metadata_path = os.path.join(self.config.MODEL_STORAGE_PATH, 'training_metadata.json')
            if not os.path.exists(metadata_path):
                return False
            
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            last_training = datetime.fromisoformat(metadata['training_date'])
            hours_since_training = (datetime.now() - last_training).total_seconds() / 3600
            
            # Skip if trained within the last 24 hours
            return hours_since_training < 24
            
        except Exception:
            return False

    def update_training_results(self, results: Dict[str, str]):
        """
        Update database with training results
        """
        try:
            training_record = {
                'timestamp': datetime.now(),
                'results': results,
                'metrics': self.model_metrics,
                'status': 'completed'
            }
            
            self.db.insert_training_record(training_record)
            self.logger.info("Training results updated in database")
            
        except Exception as e:
            self.logger.error(f"Failed to update training results: {str(e)}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train AI models for Schwab Trading System')
    parser.add_argument('--symbols', nargs='+', help='Symbols to train on')
    parser.add_argument('--retrain', action='store_true', help='Force retraining even if models are recent')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(args.config)
        trainer.logger.setLevel(args.log_level)
        
        # Run training
        results = trainer.run_training_pipeline(
            symbols=args.symbols,
            retrain=args.retrain
        )
        
        print(f"Training completed: {results['status']}")
        print(f"Models trained: {results.get('models_trained', 0)}")
        
        if results['status'] == 'completed':
            print("\nModel Performance:")
            for model_name, metrics in results.get('model_metrics', {}).items():
                if isinstance(metrics, dict) and 'val_loss' in metrics:
                    print(f"  {model_name}: Validation Loss = {metrics['val_loss']:.6f}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())