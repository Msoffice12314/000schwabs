"""
Schwab AI Trading System - Data Processing Pipeline
Advanced data cleaning, preprocessing, and quality management system for financial time series data.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

from utils.cache_manager import CacheManager
from utils.database import DatabaseManager
from config.settings import get_settings

logger = logging.getLogger(__name__)

class ProcessingStep(Enum):
    """Data processing steps"""
    VALIDATION = "validation"
    CLEANING = "cleaning"
    OUTLIER_DETECTION = "outlier_detection"
    MISSING_DATA = "missing_data"
    NORMALIZATION = "normalization"
    FEATURE_SCALING = "feature_scaling"
    SMOOTHING = "smoothing"
    RESAMPLING = "resampling"

class OutlierMethod(Enum):
    """Outlier detection methods"""
    Z_SCORE = "z_score"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    MODIFIED_Z_SCORE = "modified_z_score"

class ImputationMethod(Enum):
    """Missing data imputation methods"""
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    LINEAR_INTERPOLATION = "linear"
    SPLINE_INTERPOLATION = "spline"
    MEDIAN = "median"
    MEAN = "mean"
    KNN = "knn"
    SEASONAL = "seasonal"

class ScalingMethod(Enum):
    """Feature scaling methods"""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    QUANTILE = "quantile"
    NONE = "none"

@dataclass
class ProcessingConfig:
    """Data processing configuration"""
    # Validation
    validate_ohlc: bool = True
    validate_volume: bool = True
    validate_timestamps: bool = True
    
    # Cleaning
    remove_duplicates: bool = True
    handle_gaps: bool = True
    fix_split_adjustments: bool = True
    
    # Outlier detection
    outlier_method: OutlierMethod = OutlierMethod.MODIFIED_Z_SCORE
    outlier_threshold: float = 3.5
    outlier_action: str = "cap"  # remove, cap, interpolate
    
    # Missing data
    imputation_method: ImputationMethod = ImputationMethod.LINEAR_INTERPOLATION
    max_consecutive_missing: int = 5
    max_missing_ratio: float = 0.1
    
    # Scaling
    scaling_method: ScalingMethod = ScalingMethod.ROBUST
    scaling_columns: List[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close'])
    
    # Smoothing
    apply_smoothing: bool = False
    smoothing_window: int = 5
    smoothing_method: str = "savgol"  # savgol, rolling_mean, ewm
    
    # Resampling
    target_frequency: Optional[str] = None  # '1T', '5T', '1H', '1D'
    aggregation_rules: Dict[str, str] = field(default_factory=lambda: {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

@dataclass
class QualityMetrics:
    """Data quality metrics"""
    total_records: int
    valid_records: int
    missing_values: int
    outliers_detected: int
    duplicates_removed: int
    gaps_filled: int
    quality_score: float
    processing_time: float
    issues_found: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class ProcessingResult:
    """Data processing result"""
    original_data: pd.DataFrame
    processed_data: pd.DataFrame
    config_used: ProcessingConfig
    quality_metrics: QualityMetrics
    scalers: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataProcessor:
    """
    Advanced data processing pipeline for financial time series data
    with comprehensive cleaning, validation, and quality management.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache_manager = CacheManager()
        self.db_manager = DatabaseManager()
        
        # Processing state
        self.scalers = {}
        self.processing_history = []
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_quality_score': 0.7,
            'max_missing_ratio': 0.15,
            'max_outlier_ratio': 0.05,
            'min_data_points': 100
        }
        
        logger.info("DataProcessor initialized")
    
    async def initialize(self) -> bool:
        """Initialize the data processor"""
        try:
            # Load any cached scalers or configurations
            await self._load_cached_scalers()
            
            logger.info("DataProcessor initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DataProcessor: {str(e)}")
            return False
    
    async def process_data(self, data: pd.DataFrame, symbol: str,
                          config: Optional[ProcessingConfig] = None) -> ProcessingResult:
        """
        Process financial time series data with comprehensive cleaning and validation
        
        Args:
            data: Raw OHLCV data
            symbol: Symbol identifier
            config: Processing configuration
            
        Returns:
            ProcessingResult with processed data and quality metrics
        """
        try:
            start_time = datetime.now()
            
            if config is None:
                config = ProcessingConfig()
            
            # Make a copy to avoid modifying original
            processed_data = data.copy()
            issues_found = []
            warnings_list = []
            
            logger.info(f"Processing data for {symbol}: {len(data)} records")
            
            # Step 1: Validation
            validation_issues = await self._validate_data(processed_data, symbol)
            issues_found.extend(validation_issues)
            
            # Step 2: Basic cleaning
            processed_data, cleaning_stats = await self._clean_data(processed_data, config)
            
            # Step 3: Handle missing data
            processed_data, missing_stats = await self._handle_missing_data(
                processed_data, config
            )
            
            # Step 4: Detect and handle outliers
            processed_data, outlier_stats = await self._handle_outliers(
                processed_data, config
            )
            
            # Step 5: Apply scaling if requested
            scalers = {}
            if config.scaling_method != ScalingMethod.NONE:
                processed_data, scalers = await self._apply_scaling(
                    processed_data, config, symbol
                )
            
            # Step 6: Apply smoothing if requested
            if config.apply_smoothing:
                processed_data = await self._apply_smoothing(processed_data, config)
            
            # Step 7: Resample if requested
            if config.target_frequency:
                processed_data = await self._resample_data(processed_data, config)
            
            # Step 8: Final validation
            final_issues = await self._validate_processed_data(processed_data)
            issues_found.extend(final_issues)
            
            # Calculate quality metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            quality_metrics = await self._calculate_quality_metrics(
                data, processed_data, processing_time, cleaning_stats,
                missing_stats, outlier_stats, issues_found, warnings_list
            )
            
            # Create result
            result = ProcessingResult(
                original_data=data,
                processed_data=processed_data,
                config_used=config,
                quality_metrics=quality_metrics,
                scalers=scalers,
                metadata={
                    'symbol': symbol,
                    'processing_timestamp': datetime.now(),
                    'original_shape': data.shape,
                    'processed_shape': processed_data.shape
                }
            )
            
            # Store processing result
            await self._store_processing_result(result, symbol)
            
            logger.info(f"Data processing completed for {symbol}: "
                       f"Quality score {quality_metrics.quality_score:.3f}, "
                       f"Time {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {str(e)}")
            raise
    
    async def _validate_data(self, data: pd.DataFrame, symbol: str) -> List[str]:
        """Validate data structure and basic constraints"""
        issues = []
        
        try:
            # Check required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                issues.append(f"Missing required columns: {missing_cols}")
            
            # Check data types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                    issues.append(f"Column {col} is not numeric")
            
            # Check OHLC relationships
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                # High should be >= Open, Close, Low
                invalid_high = (data['high'] < data[['open', 'close', 'low']].max(axis=1)).sum()
                if invalid_high > 0:
                    issues.append(f"Invalid high values: {invalid_high} records")
                
                # Low should be <= Open, Close, High
                invalid_low = (data['low'] > data[['open', 'close', 'high']].min(axis=1)).sum()
                if invalid_low > 0:
                    issues.append(f"Invalid low values: {invalid_low} records")
            
            # Check for negative prices
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in data.columns:
                    negative_prices = (data[col] <= 0).sum()
                    if negative_prices > 0:
                        issues.append(f"Negative/zero prices in {col}: {negative_prices} records")
            
            # Check volume
            if 'volume' in data.columns:
                negative_volume = (data['volume'] < 0).sum()
                if negative_volume > 0:
                    issues.append(f"Negative volume: {negative_volume} records")
            
            # Check timestamp consistency
            if isinstance(data.index, pd.DatetimeIndex):
                # Check for duplicated timestamps
                duplicated_times = data.index.duplicated().sum()
                if duplicated_times > 0:
                    issues.append(f"Duplicated timestamps: {duplicated_times}")
                
                # Check chronological order
                if not data.index.is_monotonic_increasing:
                    issues.append("Timestamps not in chronological order")
            
            # Check data completeness
            total_missing = data.isnull().sum().sum()
            missing_ratio = total_missing / (len(data) * len(data.columns))
            if missing_ratio > 0.2:
                issues.append(f"High missing data ratio: {missing_ratio:.1%}")
            
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
        
        return issues
    
    async def _clean_data(self, data: pd.DataFrame, 
                         config: ProcessingConfig) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Clean data by removing duplicates and fixing basic issues"""
        stats = {'duplicates_removed': 0, 'gaps_filled': 0, 'adjustments_made': 0}
        
        try:
            # Remove duplicated rows
            if config.remove_duplicates:
                initial_length = len(data)
                data = data.drop_duplicates()
                stats['duplicates_removed'] = initial_length - len(data)
            
            # Sort by timestamp if datetime index
            if isinstance(data.index, pd.DatetimeIndex):
                data = data.sort_index()
            
            # Fix obvious data entry errors
            price_columns = ['open', 'high', 'low', 'close']
            
            # Cap extreme values (likely errors)
            for col in price_columns:
                if col in data.columns:
                    # Remove values that are 100x or 1/100x of median (likely decimal point errors)
                    median_price = data[col].median()
                    if median_price > 0:
                        extreme_high = data[col] > median_price * 100
                        extreme_low = data[col] < median_price / 100
                        
                        if extreme_high.any() or extreme_low.any():
                            # Try to fix decimal point errors
                            data.loc[extreme_high, col] = data.loc[extreme_high, col] / 100
                            data.loc[extreme_low, col] = data.loc[extreme_low, col] * 100
                            stats['adjustments_made'] += extreme_high.sum() + extreme_low.sum()
            
            # Handle stock splits (simplified detection)
            if config.fix_split_adjustments:
                stats['adjustments_made'] += await self._detect_and_fix_splits(data)
            
            # Fill small gaps in data
            if config.handle_gaps:
                gaps_filled = await self._fill_small_gaps(data)
                stats['gaps_filled'] = gaps_filled
            
        except Exception as e:
            logger.error(f"Error in data cleaning: {str(e)}")
        
        return data, stats
    
    async def _detect_and_fix_splits(self, data: pd.DataFrame) -> int:
        """Detect and fix stock splits (simplified approach)"""
        try:
            adjustments = 0
            
            if 'close' not in data.columns or len(data) < 10:
                return adjustments
            
            # Calculate daily returns
            returns = data['close'].pct_change()
            
            # Look for extreme negative returns (potential splits)
            split_threshold = -0.4  # 40% drop might indicate 2:1 split
            potential_splits = returns < split_threshold
            
            if potential_splits.any():
                split_dates = data.index[potential_splits]
                
                for split_date in split_dates:
                    # Check if this looks like a split
                    split_idx = data.index.get_loc(split_date)
                    
                    if split_idx > 0:
                        prev_close = data['close'].iloc[split_idx - 1]
                        curr_close = data['close'].iloc[split_idx]
                        
                        # Estimate split ratio
                        ratio = prev_close / curr_close
                        
                        # Common split ratios
                        common_ratios = [2.0, 3.0, 1.5, 4.0, 0.5, 0.33, 0.25]
                        
                        # Find closest common ratio
                        closest_ratio = min(common_ratios, key=lambda x: abs(x - ratio))
                        
                        if abs(closest_ratio - ratio) < 0.1:  # Close enough to common ratio
                            # Adjust prices before split date
                            price_cols = ['open', 'high', 'low', 'close']
                            for col in price_cols:
                                if col in data.columns:
                                    data.iloc[:split_idx, data.columns.get_loc(col)] /= closest_ratio
                            
                            # Adjust volume
                            if 'volume' in data.columns:
                                data.iloc[:split_idx, data.columns.get_loc('volume')] *= closest_ratio
                            
                            adjustments += 1
                            logger.info(f"Applied split adjustment: ratio {closest_ratio} at {split_date}")
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Error detecting splits: {str(e)}")
            return 0
    
    async def _fill_small_gaps(self, data: pd.DataFrame) -> int:
        """Fill small gaps in time series data"""
        try:
            gaps_filled = 0
            
            if not isinstance(data.index, pd.DatetimeIndex):
                return gaps_filled
            
            # Detect expected frequency
            freq = pd.infer_freq(data.index)
            if freq is None:
                # Try to infer from most common interval
                intervals = data.index.to_series().diff().dropna()
                most_common_interval = intervals.mode().iloc[0] if not intervals.empty else None
                
                if most_common_interval:
                    # Create complete time range
                    full_range = pd.date_range(
                        start=data.index.min(),
                        end=data.index.max(),
                        freq=most_common_interval
                    )
                    
                    # Reindex to fill gaps
                    original_length = len(data)
                    data_reindexed = data.reindex(full_range)
                    
                    # Forward fill small gaps (up to 3 periods)
                    data_filled = data_reindexed.fillna(method='ffill', limit=3)
                    
                    # Update original data
                    data.update(data_filled)
                    gaps_filled = len(data_filled) - original_length
            
            return gaps_filled
            
        except Exception as e:
            logger.error(f"Error filling gaps: {str(e)}")
            return 0
    
    async def _handle_missing_data(self, data: pd.DataFrame,
                                 config: ProcessingConfig) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Handle missing data using specified imputation method"""
        stats = {'missing_before': 0, 'missing_after': 0, 'imputed': 0}
        
        try:
            stats['missing_before'] = data.isnull().sum().sum()
            
            if stats['missing_before'] == 0:
                return data, stats
            
            # Check if missing data ratio is acceptable
            missing_ratio = stats['missing_before'] / (len(data) * len(data.columns))
            if missing_ratio > config.max_missing_ratio:
                logger.warning(f"High missing data ratio: {missing_ratio:.1%}")
            
            # Apply imputation method
            if config.imputation_method == ImputationMethod.FORWARD_FILL:
                data = data.fillna(method='ffill', limit=config.max_consecutive_missing)
                
            elif config.imputation_method == ImputationMethod.BACKWARD_FILL:
                data = data.fillna(method='bfill', limit=config.max_consecutive_missing)
                
            elif config.imputation_method == ImputationMethod.LINEAR_INTERPOLATION:
                data = data.interpolate(method='linear', limit=config.max_consecutive_missing)
                
            elif config.imputation_method == ImputationMethod.SPLINE_INTERPOLATION:
                data = data.interpolate(method='spline', order=2, limit=config.max_consecutive_missing)
                
            elif config.imputation_method == ImputationMethod.MEDIAN:
                imputer = SimpleImputer(strategy='median')
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
                
            elif config.imputation_method == ImputationMethod.MEAN:
                imputer = SimpleImputer(strategy='mean')
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
                
            elif config.imputation_method == ImputationMethod.KNN:
                imputer = KNNImputer(n_neighbors=5)
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
            
            stats['missing_after'] = data.isnull().sum().sum()
            stats['imputed'] = stats['missing_before'] - stats['missing_after']
            
        except Exception as e:
            logger.error(f"Error handling missing data: {str(e)}")
        
        return data, stats
    
    async def _handle_outliers(self, data: pd.DataFrame,
                             config: ProcessingConfig) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Detect and handle outliers"""
        stats = {'outliers_detected': 0, 'outliers_handled': 0}
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col in data.columns:
                    outliers = await self._detect_outliers(
                        data[col], config.outlier_method, config.outlier_threshold
                    )
                    
                    stats['outliers_detected'] += outliers.sum()
                    
                    if outliers.any():
                        if config.outlier_action == "remove":
                            data = data[~outliers]
                            stats['outliers_handled'] += outliers.sum()
                            
                        elif config.outlier_action == "cap":
                            # Cap at percentiles
                            lower_bound = data[col].quantile(0.01)
                            upper_bound = data[col].quantile(0.99)
                            
                            data.loc[outliers & (data[col] < lower_bound), col] = lower_bound
                            data.loc[outliers & (data[col] > upper_bound), col] = upper_bound
                            stats['outliers_handled'] += outliers.sum()
                            
                        elif config.outlier_action == "interpolate":
                            data.loc[outliers, col] = np.nan
                            data[col] = data[col].interpolate(method='linear')
                            stats['outliers_handled'] += outliers.sum()
            
        except Exception as e:
            logger.error(f"Error handling outliers: {str(e)}")
        
        return data, stats
    
    async def _detect_outliers(self, series: pd.Series, method: OutlierMethod,
                             threshold: float) -> pd.Series:
        """Detect outliers using specified method"""
        try:
            if method == OutlierMethod.Z_SCORE:
                z_scores = np.abs(stats.zscore(series.dropna()))
                return pd.Series(z_scores > threshold, index=series.index).fillna(False)
                
            elif method == OutlierMethod.MODIFIED_Z_SCORE:
                median = series.median()
                mad = np.median(np.abs(series - median))
                modified_z_scores = 0.6745 * (series - median) / mad
                return np.abs(modified_z_scores) > threshold
                
            elif method == OutlierMethod.IQR:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                return (series < lower_bound) | (series > upper_bound)
                
            else:
                # Default to modified z-score
                return await self._detect_outliers(series, OutlierMethod.MODIFIED_Z_SCORE, threshold)
                
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            return pd.Series(False, index=series.index)
    
    async def _apply_scaling(self, data: pd.DataFrame, config: ProcessingConfig,
                           symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply feature scaling"""
        scalers = {}
        
        try:
            # Select columns to scale
            cols_to_scale = [col for col in config.scaling_columns if col in data.columns]
            
            if not cols_to_scale:
                return data, scalers
            
            # Choose scaler
            if config.scaling_method == ScalingMethod.STANDARD:
                scaler = StandardScaler()
            elif config.scaling_method == ScalingMethod.MINMAX:
                scaler = MinMaxScaler()
            elif config.scaling_method == ScalingMethod.ROBUST:
                scaler = RobustScaler()
            else:
                return data, scalers
            
            # Fit and transform
            scaled_values = scaler.fit_transform(data[cols_to_scale])
            
            # Update data
            for i, col in enumerate(cols_to_scale):
                data[col] = scaled_values[:, i]
            
            # Store scaler
            scalers[f'{symbol}_scaler'] = {
                'scaler': scaler,
                'columns': cols_to_scale,
                'method': config.scaling_method.value
            }
            
        except Exception as e:
            logger.error(f"Error applying scaling: {str(e)}")
        
        return data, scalers
    
    async def _apply_smoothing(self, data: pd.DataFrame, 
                             config: ProcessingConfig) -> pd.DataFrame:
        """Apply smoothing to data"""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if len(data[col].dropna()) > config.smoothing_window:
                    if config.smoothing_method == "savgol":
                        # Savitzky-Golay filter
                        window_length = min(config.smoothing_window, len(data[col]) // 2)
                        if window_length % 2 == 0:
                            window_length += 1  # Must be odd
                        
                        if window_length >= 3:
                            data[col] = savgol_filter(
                                data[col].fillna(method='ffill'), 
                                window_length, 
                                polyorder=2
                            )
                    
                    elif config.smoothing_method == "rolling_mean":
                        data[col] = data[col].rolling(
                            window=config.smoothing_window, 
                            min_periods=1
                        ).mean()
                    
                    elif config.smoothing_method == "ewm":
                        data[col] = data[col].ewm(
                            span=config.smoothing_window
                        ).mean()
            
        except Exception as e:
            logger.error(f"Error applying smoothing: {str(e)}")
        
        return data
    
    async def _resample_data(self, data: pd.DataFrame, 
                           config: ProcessingConfig) -> pd.DataFrame:
        """Resample data to target frequency"""
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                logger.warning("Cannot resample data without datetime index")
                return data
            
            # Apply resampling with aggregation rules
            resampled = data.resample(config.target_frequency).agg(config.aggregation_rules)
            
            # Remove any rows with all NaN values
            resampled = resampled.dropna(how='all')
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling data: {str(e)}")
            return data
    
    async def _validate_processed_data(self, data: pd.DataFrame) -> List[str]:
        """Final validation of processed data"""
        issues = []
        
        try:
            # Check for remaining issues
            if data.empty:
                issues.append("Processed data is empty")
                return issues
            
            # Check for infinite values
            inf_values = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
            if inf_values > 0:
                issues.append(f"Infinite values found: {inf_values}")
            
            # Check for excessive missing data
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_ratio > 0.1:
                issues.append(f"High remaining missing data: {missing_ratio:.1%}")
            
            # Check data integrity
            if 'close' in data.columns:
                if (data['close'] <= 0).any():
                    issues.append("Zero or negative closing prices remain")
            
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
        
        return issues
    
    async def _calculate_quality_metrics(self, original_data: pd.DataFrame,
                                       processed_data: pd.DataFrame,
                                       processing_time: float,
                                       cleaning_stats: Dict[str, int],
                                       missing_stats: Dict[str, int],
                                       outlier_stats: Dict[str, int],
                                       issues_found: List[str],
                                       warnings_list: List[str]) -> QualityMetrics:
        """Calculate comprehensive quality metrics"""
        try:
            total_records = len(original_data)
            valid_records = len(processed_data)
            
            # Calculate quality score
            quality_factors = []
            
            # Completeness factor
            completeness = valid_records / total_records if total_records > 0 else 0
            quality_factors.append(completeness)
            
            # Missing data factor
            missing_ratio = missing_stats['missing_after'] / (len(processed_data) * len(processed_data.columns)) if len(processed_data) > 0 else 1
            missing_factor = 1 - missing_ratio
            quality_factors.append(missing_factor)
            
            # Outlier factor
            outlier_ratio = outlier_stats['outliers_detected'] / total_records if total_records > 0 else 0
            outlier_factor = 1 - min(outlier_ratio, 0.1) / 0.1  # Cap impact at 10% outliers
            quality_factors.append(outlier_factor)
            
            # Issues factor
            issues_factor = max(0, 1 - len(issues_found) / 10)  # Each issue reduces score
            quality_factors.append(issues_factor)
            
            # Overall quality score
            quality_score = np.mean(quality_factors)
            
            return QualityMetrics(
                total_records=total_records,
                valid_records=valid_records,
                missing_values=missing_stats['missing_after'],
                outliers_detected=outlier_stats['outliers_detected'],
                duplicates_removed=cleaning_stats['duplicates_removed'],
                gaps_filled=cleaning_stats['gaps_filled'],
                quality_score=quality_score,
                processing_time=processing_time,
                issues_found=issues_found,
                warnings=warnings_list
            )
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {str(e)}")
            return QualityMetrics(
                total_records=len(original_data),
                valid_records=len(processed_data),
                missing_values=0,
                outliers_detected=0,
                duplicates_removed=0,
                gaps_filled=0,
                quality_score=0.5,
                processing_time=processing_time,
                issues_found=issues_found,
                warnings=warnings_list
            )
    
    async def _store_processing_result(self, result: ProcessingResult, symbol: str):
        """Store processing result in database"""
        try:
            insert_query = """
                INSERT INTO data_processing_results (symbol, total_records, valid_records,
                                                   missing_values, outliers_detected,
                                                   duplicates_removed, quality_score,
                                                   processing_time, issues_found, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                symbol,
                result.quality_metrics.total_records,
                result.quality_metrics.valid_records,
                result.quality_metrics.missing_values,
                result.quality_metrics.outliers_detected,
                result.quality_metrics.duplicates_removed,
                result.quality_metrics.quality_score,
                result.quality_metrics.processing_time,
                json.dumps(result.quality_metrics.issues_found),
                datetime.now()
            )
            
            await self.db_manager.execute_query(insert_query, values)
            
        except Exception as e:
            logger.error(f"Error storing processing result: {str(e)}")
    
    async def _load_cached_scalers(self):
        """Load cached scalers from previous processing"""
        try:
            # This would load scalers from cache or database
            # For now, initialize empty
            self.scalers = {}
            
        except Exception as e:
            logger.error(f"Error loading cached scalers: {str(e)}")
    
    async def batch_process(self, data_dict: Dict[str, pd.DataFrame],
                           config: Optional[ProcessingConfig] = None) -> Dict[str, ProcessingResult]:
        """Process multiple symbols in batch"""
        try:
            results = {}
            
            for symbol, data in data_dict.items():
                try:
                    result = await self.process_data(data, symbol, config)
                    results[symbol] = result
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
            
            logger.info(f"Batch processing completed: {len(results)} symbols")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return {}
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        try:
            # Query recent processing results
            stats = {
                'total_processed': len(self.processing_history),
                'scalers_cached': len(self.scalers),
                'quality_thresholds': self.quality_thresholds,
                'processing_history': self.processing_history[-10:]  # Last 10
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting processing statistics: {str(e)}")
            return {}
    
    def create_processing_config(self, **kwargs) -> ProcessingConfig:
        """Create processing configuration with custom parameters"""
        return ProcessingConfig(**kwargs)
