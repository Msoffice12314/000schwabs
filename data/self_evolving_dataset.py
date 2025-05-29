import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import asyncio
import pickle
import json
from pathlib import Path
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
import threading
import time
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

Base = declarative_base()

class DatasetMetadata(Base):
    """SQLAlchemy model for dataset metadata"""
    __tablename__ = 'dataset_metadata'
    
    id = Column(Integer, primary_key=True)
    dataset_name = Column(String(100), nullable=False)
    version = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    num_samples = Column(Integer, nullable=False)
    num_features = Column(Integer, nullable=False)
    performance_score = Column(Float, nullable=True)
    is_active = Column(Boolean, default=True)
    metadata_json = Column(Text, nullable=True)

class FeatureImportance(Base):
    """SQLAlchemy model for feature importance tracking"""
    __tablename__ = 'feature_importance'
    
    id = Column(Integer, primary_key=True)
    dataset_version = Column(Integer, nullable=False)
    feature_name = Column(String(100), nullable=False)
    importance_score = Column(Float, nullable=False)
    stability_score = Column(Float, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow)

class DataDrift(Base):
    """SQLAlchemy model for data drift detection"""
    __tablename__ = 'data_drift'
    
    id = Column(Integer, primary_key=True)
    feature_name = Column(String(100), nullable=False)
    drift_score = Column(Float, nullable=False)
    drift_type = Column(String(50), nullable=False)  # 'statistical', 'distribution', 'concept'
    detection_date = Column(DateTime, default=datetime.utcnow)
    severity = Column(String(20), nullable=False)  # 'LOW', 'MEDIUM', 'HIGH'
    metadata_json = Column(Text, nullable=True)

class SelfEvolvingDataset:
    """Advanced self-evolving dataset management system"""
    
    def __init__(self, base_path: str = "datasets", db_path: str = "dataset.db"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Database setup
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Dataset storage
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.feature_metadata: Dict[str, Dict] = {}
        self.performance_history: Dict[str, List] = defaultdict(list)
        
        # Evolution parameters
        self.evolution_config = {
            'min_samples_for_evolution': 1000,
            'feature_stability_threshold': 0.7,
            'drift_detection_threshold': 0.3,
            'performance_improvement_threshold': 0.02,
            'max_features': 500,
            'feature_selection_method': 'importance_stability'
        }
        
        # Real-time processing
        self.incoming_data_buffer = deque(maxlen=10000)
        self.processing_thread = None
        self.stop_processing = threading.Event()
        
        # Feature engineering components
        self.scalers: Dict[str, Any] = {}
        self.feature_generators: List[callable] = []
        self.outlier_detectors: Dict[str, Any] = {}
        
        # Drift detection
        self.drift_detectors: Dict[str, Any] = {}
        self.reference_distributions: Dict[str, Dict] = {}
        
        self.start_background_processing()
    
    def start_background_processing(self):
        """Start background thread for data processing"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_processing.clear()
            self.processing_thread = threading.Thread(
                target=self._background_processing_loop,
                daemon=True
            )
            self.processing_thread.start()
    
    def stop_background_processing(self):
        """Stop background processing"""
        self.stop_processing.set()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
    
    def _background_processing_loop(self):
        """Background loop for continuous data processing"""
        while not self.stop_processing.is_set():
            try:
                self._process_buffered_data()
                self._detect_drift()
                self._evaluate_evolution_triggers()
                time.sleep(300)  # Process every 5 minutes
            except Exception as e:
                self.logger.error(f"Error in background processing: {e}")
                time.sleep(60)
    
    def add_data_point(self, data: Dict[str, Any], target: Optional[float] = None,
                      dataset_name: str = "main"):
        """Add a single data point to the buffer for processing"""
        try:
            data_point = {
                'timestamp': datetime.now(),
                'data': data,
                'target': target,
                'dataset_name': dataset_name
            }
            self.incoming_data_buffer.append(data_point)
            
        except Exception as e:
            self.logger.error(f"Error adding data point: {e}")
    
    def add_batch_data(self, df: pd.DataFrame, target_column: Optional[str] = None,
                      dataset_name: str = "main"):
        """Add batch data to the dataset"""
        try:
            # Validate and clean data
            df_clean = self._clean_and_validate_data(df)
            
            # Generate features
            df_enhanced = self._generate_features(df_clean)
            
            # Update or create dataset
            if dataset_name in self.datasets:
                self.datasets[dataset_name] = pd.concat([
                    self.datasets[dataset_name], df_enhanced
                ], ignore_index=True)
            else:
                self.datasets[dataset_name] = df_enhanced
            
            # Update metadata
            self._update_dataset_metadata(dataset_name, df_enhanced)
            
            # Trigger evolution if needed
            if len(df_enhanced) >= self.evolution_config['min_samples_for_evolution']:
                self._trigger_evolution(dataset_name)
            
            self.logger.info(f"Added {len(df)} samples to dataset '{dataset_name}'")
            
        except Exception as e:
            self.logger.error(f"Error adding batch data: {e}")
    
    def _process_buffered_data(self):
        """Process data points from the buffer"""
        if not self.incoming_data_buffer:
            return
        
        try:
            # Extract data points
            data_points = []
            while self.incoming_data_buffer and len(data_points) < 100:
                data_points.append(self.incoming_data_buffer.popleft())
            
            # Group by dataset
            dataset_groups = defaultdict(list)
            for dp in data_points:
                dataset_groups[dp['dataset_name']].append(dp)
            
            # Process each dataset group
            for dataset_name, points in dataset_groups.items():
                df_data = []
                for point in points:
                    row = point['data'].copy()
                    row['timestamp'] = point['timestamp']
                    if point['target'] is not None:
                        row['target'] = point['target']
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                self.add_batch_data(df, 'target' if 'target' in df.columns else None, dataset_name)
            
        except Exception as e:
            self.logger.error(f"Error processing buffered data: {e}")
    
    def _clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate incoming data"""
        try:
            df_clean = df.copy()
            
            # Remove completely empty rows
            df_clean = df_clean.dropna(how='all')
            
            # Handle missing values
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            categorical_columns = df_clean.select_dtypes(include=['object']).columns
            
            # Fill numeric missing values with median
            for col in numeric_columns:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            
            # Fill categorical missing values with mode
            for col in categorical_columns:
                if df_clean[col].isnull().sum() > 0:
                    mode_value = df_clean[col].mode().iloc[0] if len(df_clean[col].mode()) > 0 else 'unknown'
                    df_clean[col].fillna(mode_value, inplace=True)
            
            # Remove outliers using IQR method for numeric columns
            for col in numeric_columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                ]
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error cleaning data: {e}")
            return df
    
    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate additional features from raw data"""
        try:
            df_enhanced = df.copy()
            
            # Technical indicators for financial data
            if self._is_financial_data(df):
                df_enhanced = self._add_technical_indicators(df_enhanced)
            
            # Time-based features
            if 'timestamp' in df.columns:
                df_enhanced = self._add_time_features(df_enhanced)
            
            # Statistical features
            df_enhanced = self._add_statistical_features(df_enhanced)
            
            # Polynomial features for key numeric columns
            df_enhanced = self._add_polynomial_features(df_enhanced)
            
            # Apply custom feature generators
            for generator in self.feature_generators:
                try:
                    df_enhanced = generator(df_enhanced)
                except Exception as e:
                    self.logger.warning(f"Feature generator failed: {e}")
            
            return df_enhanced
            
        except Exception as e:
            self.logger.error(f"Error generating features: {e}")
            return df
    
    def _is_financial_data(self, df: pd.DataFrame) -> bool:
        """Check if data appears to be financial/market data"""
        financial_indicators = ['open', 'high', 'low', 'close', 'volume', 'price']
        return any(col.lower() in financial_indicators for col in df.columns)
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for financial data"""
        try:
            # Assume we have price data
            price_columns = ['close', 'price']
            price_col = None
            
            for col in price_columns:
                if col in df.columns:
                    price_col = col
                    break
            
            if price_col is None:
                return df
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                df[f'ma_{window}'] = df[price_col].rolling(window=window).mean()
                df[f'ma_{window}_ratio'] = df[price_col] / df[f'ma_{window}']
            
            # RSI
            delta = df[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df[price_col].rolling(window=20).mean()
            bb_std = df[price_col].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volatility
            df['volatility'] = df[price_col].rolling(window=20).std()
            
            # Price momentum
            for period in [1, 5, 10]:
                df[f'momentum_{period}'] = df[price_col].pct_change(periods=period)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            timestamp_col = 'timestamp'
            if timestamp_col not in df.columns:
                return df
            
            # Convert to datetime if needed
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            # Extract time components
            df['hour'] = df[timestamp_col].dt.hour
            df['day_of_week'] = df[timestamp_col].dt.dayofweek
            df['day_of_month'] = df[timestamp_col].dt.day
            df['month'] = df[timestamp_col].dt.month
            df['quarter'] = df[timestamp_col].dt.quarter
            df['year'] = df[timestamp_col].dt.year
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Market session indicators (assuming US market hours)
            df['is_market_hours'] = ((df['hour'] >= 9) & (df['hour'] < 16) & 
                                   (df['day_of_week'] < 5)).astype(int)
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding time features: {e}")
            return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col.endswith(('_sin', '_cos', '_ratio')):  # Skip derived features
                    continue
                
                # Rolling statistics
                for window in [5, 10, 20]:
                    df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
                    df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
                    df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window).min()
                    df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window).max()
                
                # Z-score (standardization)
                df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
                
                # Percentile rank
                df[f'{col}_pct_rank'] = df[col].rank(pct=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding statistical features: {e}")
            return df
    
    def _add_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial features for key numeric columns"""
        try:
            # Select top numeric columns by variance
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return df
            
            # Calculate variance and select top features
            variances = df[numeric_columns].var().sort_values(ascending=False)
            top_features = variances.head(5).index.tolist()
            
            # Add polynomial features
            for col in top_features:
                if not col.endswith(('_sin', '_cos', '_ratio', '_zscore')):
                    df[f'{col}_squared'] = df[col] ** 2
                    df[f'{col}_cubed'] = df[col] ** 3
                    df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
            
            # Add interaction features for top 3 features
            top_3 = top_features[:3]
            for i, col1 in enumerate(top_3):
                for col2 in top_3[i+1:]:
                    df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding polynomial features: {e}")
            return df
    
    def _update_dataset_metadata(self, dataset_name: str, df: pd.DataFrame):
        """Update dataset metadata in database"""
        try:
            with self.SessionLocal() as session:
                # Get current version
                latest = session.query(DatasetMetadata).filter(
                    DatasetMetadata.dataset_name == dataset_name
                ).order_by(DatasetMetadata.version.desc()).first()
                
                new_version = 1 if latest is None else latest.version + 1
                
                # Create metadata
                metadata = {
                    'columns': df.columns.tolist(),
                    'dtypes': df.dtypes.to_dict(),
                    'memory_usage': df.memory_usage().sum(),
                    'null_counts': df.isnull().sum().to_dict(),
                    'unique_counts': df.nunique().to_dict()
                }
                
                # Store metadata
                db_metadata = DatasetMetadata(
                    dataset_name=dataset_name,
                    version=new_version,
                    num_samples=len(df),
                    num_features=len(df.columns),
                    metadata_json=json.dumps(metadata, default=str)
                )
                
                session.add(db_metadata)
                session.commit()
                
                self.feature_metadata[dataset_name] = metadata
                
        except Exception as e:
            self.logger.error(f"Error updating dataset metadata: {e}")
    
    def _detect_drift(self):
        """Detect data drift in features"""
        try:
            for dataset_name, df in self.datasets.items():
                if len(df) < 100:  # Need minimum samples
                    continue
                
                # Split into reference and current data
                split_point = int(len(df) * 0.7)
                reference_data = df.iloc[:split_point]
                current_data = df.iloc[split_point:]
                
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_columns:
                    drift_score = self._calculate_drift_score(
                        reference_data[col], current_data[col]
                    )
                    
                    if drift_score > self.evolution_config['drift_detection_threshold']:
                        self._record_drift_detection(col, drift_score, 'statistical')
                        self.logger.warning(f"Drift detected in feature {col}: {drift_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error detecting drift: {e}")
    
    def _calculate_drift_score(self, reference: pd.Series, current: pd.Series) -> float:
        """Calculate drift score between two distributions"""
        try:
            from scipy import stats
            
            # Remove NaN values
            ref_clean = reference.dropna()
            curr_clean = current.dropna()
            
            if len(ref_clean) == 0 or len(curr_clean) == 0:
                return 0.0
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.ks_2samp(ref_clean, curr_clean)
            
            # Mean and variance drift
            mean_drift = abs(ref_clean.mean() - curr_clean.mean()) / (ref_clean.std() + 1e-8)
            var_drift = abs(ref_clean.var() - curr_clean.var()) / (ref_clean.var() + 1e-8)
            
            # Combined drift score
            drift_score = (ks_stat + mean_drift + var_drift) / 3
            
            return min(drift_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating drift score: {e}")
            return 0.0
    
    def _record_drift_detection(self, feature_name: str, drift_score: float, drift_type: str):
        """Record drift detection in database"""
        try:
            severity = 'HIGH' if drift_score > 0.7 else 'MEDIUM' if drift_score > 0.5 else 'LOW'
            
            with self.SessionLocal() as session:
                drift_record = DataDrift(
                    feature_name=feature_name,
                    drift_score=drift_score,
                    drift_type=drift_type,
                    severity=severity,
                    metadata_json=json.dumps({'threshold': self.evolution_config['drift_detection_threshold']})
                )
                session.add(drift_record)
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Error recording drift detection: {e}")
    
    def _evaluate_evolution_triggers(self):
        """Evaluate if dataset evolution should be triggered"""
        try:
            for dataset_name in self.datasets.keys():
                should_evolve = False
                reasons = []
                
                # Check for significant drift
                with self.SessionLocal() as session:
                    recent_drift = session.query(DataDrift).filter(
                        DataDrift.detection_date > datetime.now() - timedelta(days=7),
                        DataDrift.severity.in_(['MEDIUM', 'HIGH'])
                    ).count()
                    
                    if recent_drift > 5:
                        should_evolve = True
                        reasons.append(f"High drift activity: {recent_drift} detections")
                
                # Check performance degradation
                if len(self.performance_history[dataset_name]) > 5:
                    recent_scores = self.performance_history[dataset_name][-5:]
                    older_scores = self.performance_history[dataset_name][-10:-5] if len(self.performance_history[dataset_name]) >= 10 else []
                    
                    if older_scores and np.mean(recent_scores) < np.mean(older_scores) - self.evolution_config['performance_improvement_threshold']:
                        should_evolve = True
                        reasons.append("Performance degradation detected")
                
                # Check dataset size growth
                if len(self.datasets[dataset_name]) > self.evolution_config['min_samples_for_evolution'] * 2:
                    should_evolve = True
                    reasons.append("Dataset size threshold exceeded")
                
                if should_evolve:
                    self.logger.info(f"Triggering evolution for {dataset_name}: {', '.join(reasons)}")
                    self._trigger_evolution(dataset_name)
            
        except Exception as e:
            self.logger.error(f"Error evaluating evolution triggers: {e}")
    
    def _trigger_evolution(self, dataset_name: str):
        """Trigger dataset evolution process"""
        try:
            if dataset_name not in self.datasets:
                return
            
            df = self.datasets[dataset_name]
            
            # Feature selection
            selected_features = self._select_features(df, dataset_name)
            
            # Create evolved dataset
            evolved_df = df[selected_features].copy()
            
            # Feature scaling
            evolved_df = self._scale_features(evolved_df, dataset_name)
            
            # Store evolved dataset
            evolved_name = f"{dataset_name}_evolved_{int(time.time())}"
            self.datasets[evolved_name] = evolved_df
            
            # Update metadata
            self._update_dataset_metadata(evolved_name, evolved_df)
            
            self.logger.info(f"Dataset evolution completed: {dataset_name} -> {evolved_name}")
            
        except Exception as e:
            self.logger.error(f"Error in dataset evolution: {e}")
    
    def _select_features(self, df: pd.DataFrame, dataset_name: str) -> List[str]:
        """Select optimal features for the dataset"""
        try:
            from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
            from sklearn.ensemble import RandomForestRegressor
            
            # Separate features and target
            feature_columns = [col for col in df.columns if col != 'target']
            
            if 'target' not in df.columns or len(feature_columns) == 0:
                return feature_columns[:self.evolution_config['max_features']]
            
            X = df[feature_columns].fillna(0)
            y = df['target'].fillna(0)
            
            # Method 1: Statistical selection
            selector_stats = SelectKBest(f_regression, k=min(100, len(feature_columns)))
            selector_stats.fit(X, y)
            stats_features = [feature_columns[i] for i in selector_stats.get_support(indices=True)]
            
            # Method 2: Random Forest importance
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            rf_features = feature_importance.head(100)['feature'].tolist()
            
            # Method 3: Mutual information
            mi_scores = mutual_info_regression(X, y)
            mi_features = pd.DataFrame({
                'feature': feature_columns,
                'mi_score': mi_scores
            }).sort_values('mi_score', ascending=False).head(100)['feature'].tolist()
            
            # Combine methods
            feature_scores = defaultdict(float)
            for feature in stats_features:
                feature_scores[feature] += 1.0
            for feature in rf_features:
                feature_scores[feature] += 1.0
            for feature in mi_features:
                feature_scores[feature] += 1.0
            
            # Select top features
            top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            selected = [feat for feat, score in top_features[:self.evolution_config['max_features']]]
            
            # Store feature importance in database
            self._store_feature_importance(dataset_name, feature_importance)
            
            return selected
            
        except Exception as e:
            self.logger.error(f"Error selecting features: {e}")
            return df.columns.tolist()[:self.evolution_config['max_features']]
    
    def _scale_features(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Scale features for optimal model performance"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                return df
            
            # Use RobustScaler for better outlier handling
            scaler = RobustScaler()
            df_scaled = df.copy()
            df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])
            
            # Store scaler for future use
            self.scalers[dataset_name] = scaler
            
            return df_scaled
            
        except Exception as e:
            self.logger.error(f"Error scaling features: {e}")
            return df
    
    def _store_feature_importance(self, dataset_name: str, importance_df: pd.DataFrame):
        """Store feature importance in database"""
        try:
            with self.SessionLocal() as session:
                # Get current version
                latest = session.query(DatasetMetadata).filter(
                    DatasetMetadata.dataset_name == dataset_name
                ).order_by(DatasetMetadata.version.desc()).first()
                
                version = latest.version if latest else 1
                
                # Store importance scores
                for _, row in importance_df.iterrows():
                    importance_record = FeatureImportance(
                        dataset_version=version,
                        feature_name=row['feature'],
                        importance_score=row['importance'],
                        stability_score=1.0  # Placeholder
                    )
                    session.add(importance_record)
                
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing feature importance: {e}")
    
    def get_dataset(self, dataset_name: str = "main", version: str = "latest") -> pd.DataFrame:
        """Get dataset by name and version"""
        try:
            if version == "latest":
                return self.datasets.get(dataset_name, pd.DataFrame())
            else:
                # Load specific version from storage
                file_path = self.base_path / f"{dataset_name}_v{version}.pkl"
                if file_path.exists():
                    return pd.read_pickle(file_path)
                else:
                    return pd.DataFrame()
                    
        except Exception as e:
            self.logger.error(f"Error getting dataset: {e}")
            return pd.DataFrame()
    
    def save_dataset(self, dataset_name: str = "main"):
        """Save dataset to persistent storage"""
        try:
            if dataset_name not in self.datasets:
                return
            
            df = self.datasets[dataset_name]
            timestamp = int(time.time())
            file_path = self.base_path / f"{dataset_name}_{timestamp}.pkl"
            
            df.to_pickle(file_path)
            
            # Also save as CSV for readability
            csv_path = self.base_path / f"{dataset_name}_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            
            self.logger.info(f"Dataset saved: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving dataset: {e}")
    
    def add_feature_generator(self, generator: callable):
        """Add custom feature generator function"""
        self.feature_generators.append(generator)
    
    def get_dataset_info(self, dataset_name: str = "main") -> Dict[str, Any]:
        """Get comprehensive dataset information"""
        try:
            if dataset_name not in self.datasets:
                return {}
            
            df = self.datasets[dataset_name]
            
            return {
                'name': dataset_name,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': f"{df.memory_usage().sum() / 1024 / 1024:.2f} MB",
                'null_counts': df.isnull().sum().to_dict(),
                'unique_counts': df.nunique().to_dict(),
                'numeric_summary': df.describe().to_dict(),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting dataset info: {e}")
            return {}
    
    def get_drift_report(self, days_back: int = 30) -> Dict[str, Any]:
        """Get drift detection report"""
        try:
            with self.SessionLocal() as session:
                start_date = datetime.now() - timedelta(days=days_back)
                
                drift_records = session.query(DataDrift).filter(
                    DataDrift.detection_date >= start_date
                ).all()
                
                # Aggregate by feature
                feature_drift = defaultdict(list)
                for record in drift_records:
                    feature_drift[record.feature_name].append({
                        'score': record.drift_score,
                        'type': record.drift_type,
                        'severity': record.severity,
                        'date': record.detection_date.isoformat()
                    })
                
                # Summary statistics
                total_detections = len(drift_records)
                high_severity = sum(1 for r in drift_records if r.severity == 'HIGH')
                affected_features = len(feature_drift)
                
                return {
                    'period_days': days_back,
                    'total_detections': total_detections,
                    'high_severity_detections': high_severity,
                    'affected_features': affected_features,
                    'feature_drift_details': dict(feature_drift),
                    'generated_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error generating drift report: {e}")
            return {}
    
    def export_dataset(self, dataset_name: str = "main", format: str = "csv", 
                      file_path: Optional[str] = None) -> str:
        """Export dataset to various formats"""
        try:
            if dataset_name not in self.datasets:
                raise ValueError(f"Dataset '{dataset_name}' not found")
            
            df = self.datasets[dataset_name]
            
            if file_path is None:
                timestamp = int(time.time())
                file_path = str(self.base_path / f"{dataset_name}_{timestamp}.{format}")
            
            if format.lower() == 'csv':
                df.to_csv(file_path, index=False)
            elif format.lower() == 'json':
                df.to_json(file_path, orient='records', indent=2)
            elif format.lower() == 'parquet':
                df.to_parquet(file_path)
            elif format.lower() == 'pickle':
                df.to_pickle(file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Dataset exported to: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error exporting dataset: {e}")
            return ""
    
    def update_evolution_config(self, config: Dict[str, Any]):
        """Update evolution configuration"""
        self.evolution_config.update(config)
        self.logger.info("Evolution configuration updated")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_background_processing()
