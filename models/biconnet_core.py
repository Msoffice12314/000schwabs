"""
BiConNet: CNN-BiLSTM Hybrid Deep Learning Model for Stock Price Prediction
Time Delayed Hybrid Architecture with Convolutional and Bidirectional LSTM layers
Based on research paper implementation for equity price forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings

logger = logging.getLogger(__name__)

class TimeDelayEmbedding:
    """Time Delay Embedding for enhanced temporal pattern capture"""
    
    def __init__(self, embedding_dimension: int = 3, time_delay: int = 1):
        self.embedding_dimension = embedding_dimension
        self.time_delay = time_delay
    
    def embed(self, time_series: np.ndarray) -> np.ndarray:
        """
        Apply time delay embedding to time series data
        
        Args:
            time_series: Input time series data (N, features)
            
        Returns:
            Embedded time series with additional temporal dimensions
        """
        if len(time_series.shape) == 1:
            time_series = time_series.reshape(-1, 1)
        
        n_samples, n_features = time_series.shape
        embedded_length = n_samples - (self.embedding_dimension - 1) * self.time_delay
        
        if embedded_length <= 0:
            raise ValueError("Time series too short for specified embedding parameters")
        
        embedded_data = np.zeros((embedded_length, n_features * self.embedding_dimension))
        
        for i in range(self.embedding_dimension):
            start_idx = i * self.time_delay
            end_idx = start_idx + embedded_length
            embedded_data[:, i*n_features:(i+1)*n_features] = time_series[start_idx:end_idx]
        
        return embedded_data

class StockDataset(Dataset):
    """PyTorch Dataset for stock price time series data"""
    
    def __init__(self, data: np.ndarray, sequence_length: int = 60, 
                 prediction_horizon: int = 1, use_time_delay: bool = True):
        """
        Initialize stock dataset
        
        Args:
            data: Stock price data (samples, features)
            sequence_length: Length of input sequences
            prediction_horizon: Number of future steps to predict
            use_time_delay: Whether to apply time delay embedding
        """
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.use_time_delay = use_time_delay
        
        if use_time_delay:
            self.time_delay_embedding = TimeDelayEmbedding(embedding_dimension=3, time_delay=1)
            self.embedded_data = self.time_delay_embedding.embed(data)
        else:
            self.embedded_data = data
        
        self.n_samples = len(self.embedded_data) - sequence_length - prediction_horizon + 1
        
        if self.n_samples <= 0:
            raise ValueError("Dataset too small for specified sequence length and prediction horizon")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input sequence
        start_idx = idx
        end_idx = idx + self.sequence_length
        x = torch.tensor(self.embedded_data[start_idx:end_idx], dtype=torch.float32)
        
        # Target (future price)
        target_idx = end_idx + self.prediction_horizon - 1
        if self.use_time_delay:
            # For time delay embedded data, use original data for target
            original_target_idx = min(target_idx, len(self.data) - 1)
            y = torch.tensor(self.data[original_target_idx, 0], dtype=torch.float32)  # Assuming price is first feature
        else:
            y = torch.tensor(self.embedded_data[target_idx, 0], dtype=torch.float32)
        
        return x, y

class ConvolutionalBlock(nn.Module):
    """1D Convolutional block for local pattern extraction"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dropout_rate: float = 0.2):
        super(ConvolutionalBlock, self).__init__()
        
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, features)
        # Conv1d expects: (batch_size, features, sequence_length)
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        # Convert back to (batch_size, sequence_length, features)
        x = x.transpose(1, 2)
        return x

class BiLSTMBlock(nn.Module):
    """Bidirectional LSTM block for long-term dependency capture"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropout_rate: float = 0.2):
        super(BiLSTMBlock, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, sequence_length, features)
        output, (hidden, cell) = self.bilstm(x)
        
        # output shape: (batch_size, sequence_length, hidden_size * 2)
        output = self.dropout(output)
        
        # hidden shape: (num_layers * 2, batch_size, hidden_size)
        # Concatenate forward and backward final hidden states
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        return output, final_hidden

class AttentionMechanism(nn.Module):
    """Attention mechanism for focusing on important time steps"""
    
    def __init__(self, hidden_size: int):
        super(AttentionMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_output shape: (batch_size, sequence_length, hidden_size)
        
        # Calculate attention weights
        attention_weights = self.attention(lstm_output)  # (batch_size, sequence_length, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # Normalize over sequence length
        
        # Apply attention weights
        context_vector = torch.sum(lstm_output * attention_weights, dim=1)  # (batch_size, hidden_size)
        
        return context_vector, attention_weights.squeeze(-1)

class BiConNet(nn.Module):
    """
    BiConNet: CNN-BiLSTM Hybrid Deep Learning Model
    Combines CNN for local pattern recognition with BiLSTM for long-term dependencies
    """
    
    def __init__(self, input_features: int, sequence_length: int = 60,
                 cnn_filters: int = 64, lstm_units: int = 50,
                 cnn_kernel_size: int = 3, lstm_layers: int = 2,
                 dropout_rate: float = 0.2, use_attention: bool = True,
                 prediction_horizon: int = 1):
        """
        Initialize BiConNet model
        
        Args:
            input_features: Number of input features
            sequence_length: Length of input sequences
            cnn_filters: Number of CNN filters
            lstm_units: Number of LSTM hidden units
            cnn_kernel_size: CNN kernel size
            lstm_layers: Number of LSTM layers
            dropout_rate: Dropout rate
            use_attention: Whether to use attention mechanism
            prediction_horizon: Number of future steps to predict
        """
        super(BiConNet, self).__init__()
        
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.use_attention = use_attention
        self.prediction_horizon = prediction_horizon
        
        # CNN layers for local pattern extraction
        self.conv_block1 = ConvolutionalBlock(input_features, cnn_filters, cnn_kernel_size, dropout_rate)
        self.conv_block2 = ConvolutionalBlock(cnn_filters, cnn_filters * 2, cnn_kernel_size, dropout_rate)
        
        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        
        # BiLSTM layers for long-term dependencies
        self.bilstm = BiLSTMBlock(cnn_filters * 2, lstm_units, lstm_layers, dropout_rate)
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionMechanism(lstm_units * 2)  # *2 for bidirectional
            final_features = lstm_units * 2
        else:
            final_features = lstm_units * 2
        
        # Output layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(final_features, final_features // 2)
        self.fc2 = nn.Linear(final_features // 2, final_features // 4)
        self.output = nn.Linear(final_features // 4, prediction_horizon)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through BiConNet
        
        Args:
            x: Input tensor (batch_size, sequence_length, features)
            
        Returns:
            Dictionary containing predictions and attention weights
        """
        batch_size = x.size(0)
        
        # CNN feature extraction
        cnn_output = self.conv_block1(x)
        cnn_output = self.conv_block2(cnn_output)
        
        # Optional pooling (transpose for pooling, then transpose back)
        cnn_output = cnn_output.transpose(1, 2)
        cnn_output = self.pool(cnn_output)
        cnn_output = cnn_output.transpose(1, 2)
        
        # BiLSTM processing
        lstm_output, final_hidden = self.bilstm(cnn_output)
        
        # Attention mechanism or use final hidden state
        if self.use_attention:
            context_vector, attention_weights = self.attention(lstm_output)
        else:
            context_vector = final_hidden
            attention_weights = None
        
        # Output prediction
        output = self.dropout(context_vector)
        output = F.relu(self.fc1(output))
        output = self.dropout(output)
        output = F.relu(self.fc2(output))
        output = self.dropout(output)
        predictions = self.output(output)
        
        result = {'predictions': predictions}
        if attention_weights is not None:
            result['attention_weights'] = attention_weights
        
        return result
    
    def predict_with_confidence(self, x: torch.Tensor, n_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        Make predictions with confidence intervals using Monte Carlo Dropout
        
        Args:
            x: Input tensor
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with mean predictions, std, and confidence intervals
        """
        self.train()  # Enable dropout for MC sampling
        
        predictions = []
        attention_weights_list = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.forward(x)
                predictions.append(output['predictions'])
                if 'attention_weights' in output:
                    attention_weights_list.append(output['attention_weights'])
        
        predictions = torch.stack(predictions, dim=0)  # (n_samples, batch_size, prediction_horizon)
        
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        # 95% confidence intervals
        confidence_lower = mean_pred - 1.96 * std_pred
        confidence_upper = mean_pred + 1.96 * std_pred
        
        result = {
            'mean_predictions': mean_pred,
            'std_predictions': std_pred,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'all_predictions': predictions
        }
        
        if attention_weights_list:
            mean_attention = torch.mean(torch.stack(attention_weights_list, dim=0), dim=0)
            result['mean_attention_weights'] = mean_attention
        
        self.eval()  # Return to eval mode
        return result

class BiConNetTrainer:
    """Training manager for BiConNet model"""
    
    def __init__(self, model: BiConNet, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        logger.info(f"BiConNet trainer initialized on device: {self.device}")
    
    def setup_optimizer(self, learning_rate: float = 0.001, weight_decay: float = 1e-4,
                       optimizer_type: str = 'adam'):
        """Setup optimizer and learning rate scheduler"""
        if optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=True
        )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(data)
            predictions = output['predictions'].squeeze()
            
            if len(predictions.shape) == 0:
                predictions = predictions.unsqueeze(0)
            if len(target.shape) == 0:
                target = target.unsqueeze(0)
            
            loss = self.criterion(predictions, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.debug(f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}")
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                predictions = output['predictions'].squeeze()
                
                if len(predictions.shape) == 0:
                    predictions = predictions.unsqueeze(0)
                if len(target.shape) == 0:
                    target = target.unsqueeze(0)
                
                loss = self.criterion(predictions, target)
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None,
              epochs: int = 100, early_stopping_patience: int = 15,
              save_best_model: bool = True, model_save_path: str = None) -> Dict[str, List[float]]:
        """
        Train the BiConNet model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            save_best_model: Whether to save the best model
            model_save_path: Path to save the best model
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting BiConNet training for {epochs} epochs")
        
        if model_save_path is None:
            model_save_path = f"{settings.model.model_save_path}/biconnet_best.pth"
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = None
            if val_loader:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    
                    if save_best_model:
                        torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'epoch': epoch,
                            'val_loss': val_loss,
                            'train_loss': train_loss,
                            'model_config': {
                                'input_features': self.model.input_features,
                                'sequence_length': self.model.sequence_length,
                                'cnn_filters': self.model.cnn_filters,
                                'lstm_units': self.model.lstm_units,
                                'prediction_horizon': self.model.prediction_horizon
                            }
                        }, model_save_path)
                        logger.info(f"Best model saved at epoch {epoch}")
                else:
                    self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Logging
            if epoch % 10 == 0 or epoch == epochs - 1:
                log_msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    log_msg += f", Val Loss: {val_loss:.6f}"
                logger.info(log_msg)
        
        logger.info("Training completed")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
    
    def load_best_model(self, model_path: str):
        """Load the best saved model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Best model loaded from {model_path}")
        return checkpoint

def create_biconnet_model(input_features: int, **kwargs) -> BiConNet:
    """
    Factory function to create BiConNet model with default parameters
    
    Args:
        input_features: Number of input features
        **kwargs: Additional model parameters
        
    Returns:
        Configured BiConNet model
    """
    default_params = {
        'sequence_length': settings.model.sequence_length,
        'cnn_filters': settings.model.cnn_filters,
        'lstm_units': settings.model.lstm_units,
        'dropout_rate': settings.model.dropout_rate,
        'use_attention': True,
        'prediction_horizon': 1
    }
    
    default_params.update(kwargs)
    
    model = BiConNet(input_features=input_features, **default_params)
    
    logger.info(f"Created BiConNet model with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model

# Export key classes and functions
__all__ = [
    'BiConNet',
    'BiConNetTrainer',
    'StockDataset',
    'TimeDelayEmbedding',
    'create_biconnet_model'
]