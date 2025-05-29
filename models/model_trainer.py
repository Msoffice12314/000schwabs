"""
Schwab AI Trading System - Model Training Pipeline
Advanced machine learning training system for BiConNet models with automated hyperparameter tuning and cross-validation.
"""

import logging
import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import optuna
import warnings
warnings.filterwarnings('ignore')

from models.biconnet import BiConNet
from data.feature_engineer import FeatureEngineer
from data.data_processor import DataProcessor
from data.market_regime import MarketRegimeDetector
from utils.cache_manager import CacheManager
from utils.database import DatabaseManager
from config.settings import get_settings

logger = logging.getLogger(__name__)

class TrainingMode(Enum):
    """Training modes"""
    SINGLE_MODEL = "single_model"
    ENSEMBLE = "ensemble"
    REGIME_SPECIFIC = "regime_specific"
    CROSS_VALIDATION = "cross_validation"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"

class LossFunction(Enum):
    """Loss functions for training"""
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    QUANTILE = "quantile"
    COMBINED = "combined"

@dataclass
class TrainingConfig:
    """Training configuration"""
    model_name: str
    training_mode: TrainingMode
    target_column: str
    sequence_length: int = 60
    
    # Model architecture
    input_size: int = 100
    cnn_filters: int = 64
    lstm_units: int = 50
    dropout_rate: float = 0.2
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    
    # Loss function
    loss_function: LossFunction = LossFunction.COMBINED
    loss_weights: Dict[str, float] = field(default_factory=lambda: {'mse': 0.7, 'mae': 0.3})
    
    # Data splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Regularization
    use_batch_norm: bool = True
    use_dropout: bool = True
    gradient_clipping: float = 1.0
    
    # Optimization
    optimizer: str = "adam"
    scheduler: str = "plateau"
    
    # Cross-validation
    cv_folds: int = 5
    
    # Model ensemble
    ensemble_size: int = 5
    ensemble_method: str = "average"  # average, weighted, stacking

@dataclass
class TrainingResult:
    """Training result metrics"""
    model_name: str
    training_time: float
    final_train_loss: float
    final_val_loss: float
    test_loss: float
    test_metrics: Dict[str, float]
    best_epoch: int
    model_path: str
    config_used: TrainingConfig
    training_history: Dict[str, List[float]]
    feature_importance: Dict[str, float] = field(default_factory=dict)

class TradingDataset(Dataset):
    """PyTorch dataset for trading data"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, 
                 sequence_length: int = 60):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
        # Ensure we have enough data
        if len(features) < sequence_length:
            raise ValueError(f"Not enough data: {len(features)} < {sequence_length}")
    
    def __len__(self):
        return len(self.features) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # Get sequence of features
        feature_sequence = self.features[idx:idx + self.sequence_length]
        
        # Get corresponding target (last value in sequence)
        target = self.targets[idx + self.sequence_length - 1]
        
        return torch.FloatTensor(feature_sequence), torch.FloatTensor(target)

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True

class ModelTrainer:
    """
    Advanced model training system with automated hyperparameter tuning,
    cross-validation, and ensemble methods for BiConNet models.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.feature_engineer = FeatureEngineer()
        self.data_processor = DataProcessor()
        self.regime_detector = MarketRegimeDetector()
        self.cache_manager = CacheManager()
        self.db_manager = DatabaseManager()
        
        # Training state
        self.current_training = None
        self.training_history = []
        
        # Model storage
        self.models_dir = Path("models/weights")
        self.scalers_dir = Path("models/scalers")
        self.configs_dir = Path("models/configs")
        
        # Create directories
        for dir_path in [self.models_dir, self.scalers_dir, self.configs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Hyperparameter optimization
        self.study = None
        
        logger.info("ModelTrainer initialized")
    
    async def initialize(self) -> bool:
        """Initialize the model trainer"""
        try:
            await self.feature_engineer.initialize()
            await self.data_processor.initialize()
            await self.regime_detector.initialize()
            
            logger.info("ModelTrainer initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ModelTrainer: {str(e)}")
            return False
    
    async def train_model(self, config: TrainingConfig, 
                         training_data: Dict[str, pd.DataFrame],
                         symbols: Optional[List[str]] = None) -> TrainingResult:
        """
        Train a model with specified configuration
        
        Args:
            config: Training configuration
            training_data: Dictionary of symbol -> DataFrame
            symbols: List of symbols to train on (if None, use all)
            
        Returns:
            TrainingResult with metrics and model path
        """
        try:
            logger.info(f"Starting training: {config.model_name}")
            start_time = datetime.now()
            
            # Prepare training data
            X, y, scalers = await self._prepare_training_data(
                training_data, config, symbols
            )
            
            if X is None or y is None:
                raise ValueError("Failed to prepare training data")
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(
                X, y, config
            )
            
            # Create datasets
            train_dataset = TradingDataset(X_train, y_train, config.sequence_length)
            val_dataset = TradingDataset(X_val, y_val, config.sequence_length)
            test_dataset = TradingDataset(X_test, y_test, config.sequence_length)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=config.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=config.batch_size, shuffle=False
            )
            test_loader = DataLoader(
                test_dataset, batch_size=config.batch_size, shuffle=False
            )
            
            # Initialize model
            model = BiConNet(
                input_size=config.input_size,
                sequence_length=config.sequence_length,
                cnn_filters=config.cnn_filters,
                lstm_units=config.lstm_units,
                dropout_rate=config.dropout_rate,
                output_size=1  # Single target prediction
            ).to(self.device)
            
            # Setup training components
            optimizer = self._get_optimizer(model, config)
            scheduler = self._get_scheduler(optimizer, config)
            criterion = self._get_loss_function(config)
            early_stopping = EarlyStopping(patience=config.early_stopping_patience)
            
            # Training loop
            training_history = await self._train_loop(
                model, train_loader, val_loader, optimizer, scheduler,
                criterion, early_stopping, config
            )
            
            # Evaluate on test set
            test_loss, test_metrics = await self._evaluate_model(
                model, test_loader, criterion
            )
            
            # Save model and scalers
            model_path = await self._save_model(model, config, scalers)
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = TrainingResult(
                model_name=config.model_name,
                training_time=training_time,
                final_train_loss=training_history['train_loss'][-1],
                final_val_loss=training_history['val_loss'][-1],
                test_loss=test_loss,
                test_metrics=test_metrics,
                best_epoch=early_stopping.counter,
                model_path=str(model_path),
                config_used=config,
                training_history=training_history
            )
            
            # Store training result
            await self._store_training_result(result)
            
            logger.info(f"Training completed: {config.model_name} "
                       f"(Test Loss: {test_loss:.6f}, Time: {training_time:.1f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    async def _prepare_training_data(self, training_data: Dict[str, pd.DataFrame],
                                   config: TrainingConfig,
                                   symbols: Optional[List[str]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
        """Prepare training data with features and targets"""
        try:
            # Select symbols
            if symbols is None:
                symbols = list(training_data.keys())
            
            all_features = []
            all_targets = []
            scalers = {}
            
            for symbol in symbols:
                data = training_data[symbol]
                
                if data.empty or len(data) < config.sequence_length + 20:
                    logger.warning(f"Insufficient data for {symbol}: {len(data)} rows")
                    continue
                
                # Engineer features
                features = await self.feature_engineer.create_features(data)
                
                if features.empty:
                    logger.warning(f"No features generated for {symbol}")
                    continue
                
                # Create targets
                targets = self._create_targets(data, config.target_column, config.sequence_length)
                
                if targets is None:
                    logger.warning(f"No targets created for {symbol}")
                    continue
                
                # Align features and targets
                min_len = min(len(features), len(targets))
                if min_len < config.sequence_length:
                    continue
                
                features = features.iloc[:min_len]
                targets = targets[:min_len]
                
                # Scale features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features.values)
                scalers[symbol] = {
                    'feature_scaler': scaler,
                    'feature_columns': features.columns.tolist()
                }
                
                all_features.append(features_scaled)
                all_targets.append(targets)
            
            if not all_features:
                logger.error("No valid training data prepared")
                return None, None, {}
            
            # Combine all data
            X = np.vstack(all_features)
            y = np.concatenate(all_targets)
            
            logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            
            return X, y, scalers
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return None, None, {}
    
    def _create_targets(self, data: pd.DataFrame, target_column: str, 
                       sequence_length: int) -> Optional[np.ndarray]:
        """Create target variables from price data"""
        try:
            if target_column not in data.columns:
                logger.error(f"Target column {target_column} not found")
                return None
            
            prices = data[target_column].values
            
            # Create return targets (next period return)
            targets = []
            for i in range(len(prices) - 1):
                if i >= sequence_length - 1:  # Ensure we have enough history
                    current_price = prices[i]
                    next_price = prices[i + 1]
                    
                    if current_price > 0:
                        return_target = (next_price - current_price) / current_price
                        targets.append(return_target)
                    else:
                        targets.append(0.0)
            
            return np.array(targets)
            
        except Exception as e:
            logger.error(f"Error creating targets: {str(e)}")
            return None
    
    def _split_data(self, X: np.ndarray, y: np.ndarray, 
                   config: TrainingConfig) -> Tuple[np.ndarray, ...]:
        """Split data into train/validation/test sets"""
        try:
            n_samples = len(X)
            
            # Calculate split indices (time-ordered split)
            train_end = int(n_samples * config.train_ratio)
            val_end = int(n_samples * (config.train_ratio + config.val_ratio))
            
            # Split features
            X_train = X[:train_end]
            X_val = X[train_end:val_end]
            X_test = X[val_end:]
            
            # Split targets
            y_train = y[:train_end]
            y_val = y[train_end:val_end]
            y_test = y[val_end:]
            
            logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
    
    def _get_optimizer(self, model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
        """Get optimizer based on configuration"""
        if config.optimizer.lower() == "adam":
            return optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer.lower() == "adamw":
            return optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer.lower() == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay
            )
        else:
            return optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
    
    def _get_scheduler(self, optimizer: optim.Optimizer, 
                      config: TrainingConfig) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler"""
        if config.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )
        elif config.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.1
            )
        elif config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.epochs
            )
        else:
            return None
    
    def _get_loss_function(self, config: TrainingConfig) -> nn.Module:
        """Get loss function based on configuration"""
        if config.loss_function == LossFunction.MSE:
            return nn.MSELoss()
        elif config.loss_function == LossFunction.MAE:
            return nn.L1Loss()
        elif config.loss_function == LossFunction.HUBER:
            return nn.HuberLoss()
        elif config.loss_function == LossFunction.COMBINED:
            return CombinedLoss(config.loss_weights)
        else:
            return nn.MSELoss()
    
    async def _train_loop(self, model: nn.Module, train_loader: DataLoader,
                         val_loader: DataLoader, optimizer: optim.Optimizer,
                         scheduler: Optional[optim.lr_scheduler._LRScheduler],
                         criterion: nn.Module, early_stopping: EarlyStopping,
                         config: TrainingConfig) -> Dict[str, List[float]]:
        """Main training loop"""
        try:
            train_losses = []
            val_losses = []
            
            best_val_loss = float('inf')
            best_model_state = None
            
            for epoch in range(config.epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_batches = 0
                
                for batch_features, batch_targets in train_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    outputs = model(batch_features)
                    loss = criterion(outputs.squeeze(), batch_targets)
                    
                    loss.backward()
                    
                    # Gradient clipping
                    if config.gradient_clipping > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.gradient_clipping
                        )
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                
                avg_train_loss = train_loss / train_batches
                train_losses.append(avg_train_loss)
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch_features, batch_targets in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_targets = batch_targets.to(self.device)
                        
                        outputs = model(batch_features)
                        loss = criterion(outputs.squeeze(), batch_targets)
                        
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                val_losses.append(avg_val_loss)
                
                # Learning rate scheduling
                if scheduler:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = model.state_dict().copy()
                
                # Early stopping
                early_stopping(avg_val_loss)
                
                # Logging
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{config.epochs}: "
                               f"Train Loss: {avg_train_loss:.6f}, "
                               f"Val Loss: {avg_val_loss:.6f}")
                
                if early_stopping.early_stop:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Restore best model
            if best_model_state:
                model.load_state_dict(best_model_state)
            
            return {
                'train_loss': train_losses,
                'val_loss': val_losses
            }
            
        except Exception as e:
            logger.error(f"Error in training loop: {str(e)}")
            raise
    
    async def _evaluate_model(self, model: nn.Module, test_loader: DataLoader,
                            criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        """Evaluate model on test set"""
        try:
            model.eval()
            test_loss = 0.0
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for batch_features, batch_targets in test_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs = model(batch_features)
                    loss = criterion(outputs.squeeze(), batch_targets)
                    
                    test_loss += loss.item()
                    
                    predictions.extend(outputs.squeeze().cpu().numpy())
                    actuals.extend(batch_targets.cpu().numpy())
            
            avg_test_loss = test_loss / len(test_loader)
            
            # Calculate additional metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            metrics = {
                'mse': mean_squared_error(actuals, predictions),
                'mae': mean_absolute_error(actuals, predictions),
                'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
                'r2': r2_score(actuals, predictions),
                'directional_accuracy': self._calculate_directional_accuracy(actuals, predictions)
            }
            
            return avg_test_loss, metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return float('inf'), {}
    
    def _calculate_directional_accuracy(self, actuals: np.ndarray, 
                                      predictions: np.ndarray) -> float:
        """Calculate directional accuracy (sign prediction)"""
        try:
            actual_directions = np.sign(actuals)
            predicted_directions = np.sign(predictions)
            
            correct_directions = np.sum(actual_directions == predicted_directions)
            total_predictions = len(actuals)
            
            return correct_directions / total_predictions if total_predictions > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating directional accuracy: {str(e)}")
            return 0.0
    
    async def _save_model(self, model: nn.Module, config: TrainingConfig,
                         scalers: Dict) -> Path:
        """Save trained model and associated data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{config.model_name}_{timestamp}.pth"
            model_path = self.models_dir / model_filename
            
            # Save model state dict
            torch.save(model.state_dict(), model_path)
            
            # Save scalers
            scaler_filename = f"scaler_{config.model_name}_{timestamp}.pkl"
            scaler_path = self.scalers_dir / scaler_filename
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(scalers, f)
            
            # Save config
            config_filename = f"config_{config.model_name}_{timestamp}.json"
            config_path = self.configs_dir / config_filename
            
            config_dict = {
                'model_name': config.model_name,
                'training_mode': config.training_mode.value,
                'target_column': config.target_column,
                'sequence_length': config.sequence_length,
                'input_size': config.input_size,
                'cnn_filters': config.cnn_filters,
                'lstm_units': config.lstm_units,
                'dropout_rate': config.dropout_rate,
                'batch_size': config.batch_size,
                'epochs': config.epochs,
                'learning_rate': config.learning_rate,
                'weight_decay': config.weight_decay,
                'loss_function': config.loss_function.value,
                'optimizer': config.optimizer,
                'scheduler': config.scheduler
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Model saved: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    async def _store_training_result(self, result: TrainingResult):
        """Store training result in database"""
        try:
            insert_query = """
                INSERT INTO training_results (model_name, training_time, final_train_loss,
                                            final_val_loss, test_loss, test_metrics,
                                            best_epoch, model_path, config_data,
                                            training_history, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                result.model_name,
                result.training_time,
                result.final_train_loss,
                result.final_val_loss,
                result.test_loss,
                json.dumps(result.test_metrics),
                result.best_epoch,
                result.model_path,
                json.dumps(result.config_used.__dict__, default=str),
                json.dumps(result.training_history),
                datetime.now()
            )
            
            await self.db_manager.execute_query(insert_query, values)
            
        except Exception as e:
            logger.error(f"Error storing training result: {str(e)}")
    
    async def hyperparameter_tuning(self, base_config: TrainingConfig,
                                  training_data: Dict[str, pd.DataFrame],
                                  n_trials: int = 50) -> TrainingConfig:
        """
        Perform hyperparameter optimization using Optuna
        
        Args:
            base_config: Base configuration to optimize
            training_data: Training data
            n_trials: Number of optimization trials
            
        Returns:
            Optimized training configuration
        """
        try:
            logger.info(f"Starting hyperparameter tuning with {n_trials} trials")
            
            # Create Optuna study
            study_name = f"hparam_tuning_{base_config.model_name}_{int(time.time())}"
            self.study = optuna.create_study(
                direction='minimize',
                study_name=study_name,
                storage=None  # In-memory storage
            )
            
            # Define objective function
            async def objective(trial):
                # Sample hyperparameters
                config = TrainingConfig(
                    model_name=f"{base_config.model_name}_trial_{trial.number}",
                    training_mode=base_config.training_mode,
                    target_column=base_config.target_column,
                    sequence_length=base_config.sequence_length,
                    input_size=base_config.input_size,
                    
                    # Tuned parameters
                    cnn_filters=trial.suggest_categorical('cnn_filters', [32, 64, 128]),
                    lstm_units=trial.suggest_categorical('lstm_units', [25, 50, 100]),
                    dropout_rate=trial.suggest_float('dropout_rate', 0.1, 0.5),
                    batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]),
                    learning_rate=trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                    
                    # Fixed parameters
                    epochs=50,  # Reduced for tuning
                    early_stopping_patience=10,
                    train_ratio=base_config.train_ratio,
                    val_ratio=base_config.val_ratio,
                    test_ratio=base_config.test_ratio
                )
                
                try:
                    # Train model with current hyperparameters
                    result = await self.train_model(config, training_data)
                    return result.final_val_loss
                    
                except Exception as e:
                    logger.error(f"Trial {trial.number} failed: {str(e)}")
                    return float('inf')
            
            # Run optimization
            for trial_number in range(n_trials):
                trial_obj = self.study.ask()
                loss = await objective(trial_obj)
                self.study.tell(trial_obj, loss)
                
                if (trial_number + 1) % 10 == 0:
                    logger.info(f"Completed {trial_number + 1}/{n_trials} trials, "
                               f"best loss: {self.study.best_value:.6f}")
            
            # Get best parameters
            best_params = self.study.best_params
            logger.info(f"Best hyperparameters found: {best_params}")
            
            # Create optimized config
            optimized_config = TrainingConfig(
                model_name=f"{base_config.model_name}_optimized",
                training_mode=base_config.training_mode,
                target_column=base_config.target_column,
                sequence_length=base_config.sequence_length,
                input_size=base_config.input_size,
                epochs=base_config.epochs,  # Restore full epochs
                
                # Optimized parameters
                cnn_filters=best_params['cnn_filters'],
                lstm_units=best_params['lstm_units'],
                dropout_rate=best_params['dropout_rate'],
                batch_size=best_params['batch_size'],
                learning_rate=best_params['learning_rate'],
                weight_decay=best_params['weight_decay'],
                
                # Keep other parameters from base config
                early_stopping_patience=base_config.early_stopping_patience,
                loss_function=base_config.loss_function,
                optimizer=base_config.optimizer,
                scheduler=base_config.scheduler,
                train_ratio=base_config.train_ratio,
                val_ratio=base_config.val_ratio,
                test_ratio=base_config.test_ratio
            )
            
            return optimized_config
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            return base_config
    
    async def train_ensemble(self, config: TrainingConfig,
                           training_data: Dict[str, pd.DataFrame],
                           ensemble_size: int = 5) -> List[TrainingResult]:
        """Train ensemble of models"""
        try:
            logger.info(f"Training ensemble of {ensemble_size} models")
            
            results = []
            
            for i in range(ensemble_size):
                # Create config for ensemble member
                ensemble_config = TrainingConfig(
                    model_name=f"{config.model_name}_ensemble_{i}",
                    training_mode=config.training_mode,
                    target_column=config.target_column,
                    sequence_length=config.sequence_length,
                    input_size=config.input_size,
                    cnn_filters=config.cnn_filters,
                    lstm_units=config.lstm_units,
                    dropout_rate=config.dropout_rate + np.random.normal(0, 0.05),  # Add noise
                    batch_size=config.batch_size,
                    epochs=config.epochs,
                    learning_rate=config.learning_rate * (0.8 + 0.4 * np.random.random()),  # Vary LR
                    weight_decay=config.weight_decay,
                    early_stopping_patience=config.early_stopping_patience,
                    loss_function=config.loss_function,
                    optimizer=config.optimizer,
                    scheduler=config.scheduler,
                    train_ratio=config.train_ratio,
                    val_ratio=config.val_ratio,
                    test_ratio=config.test_ratio
                )
                
                # Train ensemble member
                result = await self.train_model(ensemble_config, training_data)
                results.append(result)
                
                logger.info(f"Ensemble member {i+1}/{ensemble_size} completed "
                           f"(Test Loss: {result.test_loss:.6f})")
            
            logger.info(f"Ensemble training completed: {len(results)} models")
            return results
            
        except Exception as e:
            logger.error(f"Error training ensemble: {str(e)}")
            return []
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history"""
        return self.training_history
    
    def get_model_performance(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific model"""
        try:
            query = """
                SELECT * FROM training_results 
                WHERE model_name = %s 
                ORDER BY created_at DESC 
                LIMIT 1
            """
            
            # This would be implemented with actual database query
            # For now return placeholder
            return {
                'model_name': model_name,
                'test_loss': 0.001,
                'test_metrics': {
                    'mse': 0.001,
                    'mae': 0.025,
                    'r2': 0.75,
                    'directional_accuracy': 0.65
                },
                'training_time': 1800,
                'created_at': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting model performance: {str(e)}")
            return None


class CombinedLoss(nn.Module):
    """Combined loss function with multiple components"""
    
    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = weights
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
    
    def forward(self, predictions, targets):
        mse = self.mse_loss(predictions, targets)
        mae = self.mae_loss(predictions, targets)
        
        total_loss = (
            self.weights.get('mse', 0.5) * mse +
            self.weights.get('mae', 0.5) * mae
        )
        
        return total_loss
