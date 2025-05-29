"""
Model parameters configuration for BiConNet and other AI models
"""

from config.settings import settings

# BiConNet Model Parameters
BICONNET_PARAMS = {
    'sequence_length': settings.model.sequence_length,
    'cnn_filters': settings.model.cnn_filters,
    'lstm_units': settings.model.lstm_units,
    'dropout_rate': settings.model.dropout_rate,
    'learning_rate': settings.model.learning_rate,
    'batch_size': settings.model.batch_size,
    'epochs': settings.model.epochs,
    'validation_split': settings.model.validation_split,
    'early_stopping_patience': settings.model.early_stopping_patience,
    'use_attention': True,
    'prediction_horizon': 1
}

# Time Delay Embedding Parameters
TIME_DELAY_PARAMS = {
    'embedding_dimension': 3,
    'time_delay': 1
}

# Market Regime Detection Parameters
REGIME_PARAMS = {
    'lookback_period': 252,  # 1 year of trading days
    'regime_threshold': 0.02,
    'smoothing_window': 5
}

# Feature Engineering Parameters
FEATURE_PARAMS = {
    'technical_indicators': [
        'sma_10', 'sma_20', 'sma_50',
        'ema_10', 'ema_20',
        'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'bb_middle',
        'volume_sma', 'volume_ratio',
        'volatility'
    ],
    'normalization_method': 'minmax',
    'outlier_threshold': 3.0
}
