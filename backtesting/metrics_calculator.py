import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MetricsCalculator:
    """Advanced performance metrics calculator for backtesting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    def calculate_all_metrics(self, portfolio_df: pd.DataFrame, 
                            benchmark_df: Optional[pd.DataFrame] = None,
                            initial_capital: float = 100000.0) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        try:
            metrics = {}
            
            # Basic return metrics
            metrics.update(self._calculate_return_metrics(portfolio_df, initial_capital))
            
            # Risk metrics
            metrics.update(self._calculate_risk_metrics(portfolio_df))
            
            # Risk-adjusted metrics
            metrics.update(self._calculate_risk_adjusted_metrics(portfolio_df))
            
            # Drawdown metrics
            metrics.update(self._calculate_drawdown_metrics(portfolio_df))
            
            # Distribution metrics
            metrics.update(self._calculate_distribution_metrics(portfolio_df))
            
            # Benchmark comparison (if benchmark provided)
            if benchmark_df is not None:
                metrics.update(self._calculate_benchmark_metrics(portfolio_df, benchmark_df))
            
            # Time-based metrics
            metrics.update(self._calculate_time_metrics(portfolio_df))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def _calculate_return_metrics(self, portfolio_df: pd.DataFrame, initial_capital: float) -> Dict[str, float]:
        """Calculate basic return metrics"""
        try:
            returns = portfolio_df['returns'].dropna()
            total_value = portfolio_df['total_value']
            
            # Total return
            total_return = (total_value.iloc[-1] - initial_capital) / initial_capital
            
            # Annualized return
            days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
            years = days / 365.25
            annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
            
            # Cumulative returns
            cumulative_returns = (1 + returns).cumprod() - 1
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'average_daily_return': returns.mean(),
                'average_monthly_return': returns.mean() * 21,  # Approx 21 trading days per month
                'cumulative_return_final': cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 0,
                'geometric_mean_return': stats.gmean(1 + returns) - 1 if len(returns) > 0 and all(1 + returns > 0) else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating return metrics: {e}")
            return {}
    
    def _calculate_risk_metrics(self, portfolio_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk metrics"""
        try:
            returns = portfolio_df['returns'].dropna()
            
            if len(returns) == 0:
                return {}
            
            # Volatility metrics
            daily_volatility = returns.std()
            annualized_volatility = daily_volatility * np.sqrt(252)  # 252 trading days
            
            # Downside deviation
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std()
            annualized_downside_deviation = downside_deviation * np.sqrt(252)
            
            # Value at Risk (VaR)
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)
            
            # Conditional Value at Risk (CVaR/Expected Shortfall)
            cvar_95 = returns[returns <= var_95].mean()
            cvar_99 = returns[returns <= var_99].mean()
            
            return {
                'daily_volatility': daily_volatility,
                'annualized_volatility': annualized_volatility,
                'downside_deviation': downside_deviation,
                'annualized_downside_deviation': annualized_downside_deviation,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'semi_deviation': np.sqrt(np.mean(np.minimum(returns, 0)**2))
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_risk_adjusted_metrics(self, portfolio_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        try:
            returns = portfolio_df['returns'].dropna()
            
            if len(returns) == 0:
                return {}
            
            # Annualized metrics
            annualized_return = returns.mean() * 252
            annualized_volatility = returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            excess_return = annualized_return - self.risk_free_rate
            sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0
            
            # Sortino ratio
            downside_returns = returns[returns < self.risk_free_rate/252]
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
            
            # Calmar ratio
            max_drawdown = self._calculate_max_drawdown(portfolio_df)
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Information ratio (assuming benchmark is risk-free rate)
            tracking_error = returns.std() * np.sqrt(252)
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
            
            # Omega ratio
            omega_ratio = self._calculate_omega_ratio(returns)
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'information_ratio': information_ratio,
                'omega_ratio': omega_ratio,
                'treynor_ratio': self._calculate_treynor_ratio(returns)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted metrics: {e}")
            return {}
    
    def _calculate_drawdown_metrics(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate drawdown metrics"""
        try:
            total_value = portfolio_df['total_value']
            
            # Calculate running maximum
            running_max = total_value.expanding().max()
            
            # Calculate drawdown
            drawdown = (total_value - running_max) / running_max
            
            # Maximum drawdown
            max_drawdown = drawdown.min()
            
            # Drawdown duration
            drawdown_periods = self._calculate_drawdown_periods(drawdown)
            
            # Current drawdown
            current_drawdown = drawdown.iloc[-1]
            
            # Recovery metrics
            recovery_metrics = self._calculate_recovery_metrics(drawdown, total_value)
            
            return {
                'max_drawdown': max_drawdown,
                'current_drawdown': current_drawdown,
                'max_drawdown_duration': max(drawdown_periods) if drawdown_periods else 0,
                'average_drawdown_duration': np.mean(drawdown_periods) if drawdown_periods else 0,
                'number_of_drawdowns': len(drawdown_periods),
                'recovery_factor': recovery_metrics['recovery_factor'],
                'pain_index': self._calculate_pain_index(drawdown),
                'ulcer_index': self._calculate_ulcer_index(drawdown)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdown metrics: {e}")
            return {}
    
    def _calculate_distribution_metrics(self, portfolio_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate return distribution metrics"""
        try:
            returns = portfolio_df['returns'].dropna()
            
            if len(returns) == 0:
                return {}
            
            # Skewness and kurtosis
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            # Normality test
            shapiro_stat, shapiro_p = stats.shapiro(returns[:5000])  # Limit sample size for Shapiro-Wilk
            
            # Tail ratio
            tail_ratio = self._calculate_tail_ratio(returns)
            
            return {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'excess_kurtosis': kurtosis,  # scipy already returns excess kurtosis
                'is_normal_shapiro': shapiro_p > 0.05,
                'shapiro_p_value': shapiro_p,
                'tail_ratio': tail_ratio,
                'return_percentiles': {
                    '1%': returns.quantile(0.01),
                    '5%': returns.quantile(0.05),
                    '25%': returns.quantile(0.25),
                    '50%': returns.quantile(0.50),
                    '75%': returns.quantile(0.75),
                    '95%': returns.quantile(0.95),
                    '99%': returns.quantile(0.99)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating distribution metrics: {e}")
            return {}
    
    def _calculate_benchmark_metrics(self, portfolio_df: pd.DataFrame, 
                                   benchmark_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate benchmark comparison metrics"""
        try:
            portfolio_returns = portfolio_df['returns'].dropna()
            
            # Align benchmark data with portfolio dates
            benchmark_aligned = benchmark_df.reindex(portfolio_df.index, method='ffill')
            benchmark_returns = benchmark_aligned.pct_change().dropna()
            
            # Align the series
            min_length = min(len(portfolio_returns), len(benchmark_returns))
            portfolio_returns = portfolio_returns.iloc[-min_length:]
            benchmark_returns = benchmark_returns.iloc[-min_length:]
            
            if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
                return {}
            
            # Alpha and Beta
            beta, alpha = self._calculate_alpha_beta(portfolio_returns, benchmark_returns)
            
            # Tracking error
            excess_returns = portfolio_returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            
            # Information ratio
            information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
            
            # Up/Down capture ratios
            up_capture, down_capture = self._calculate_capture_ratios(portfolio_returns, benchmark_returns)
            
            # Correlation
            correlation = portfolio_returns.corr(benchmark_returns)
            
            return {
                'alpha': alpha,
                'beta': beta,
                'correlation_with_benchmark': correlation,
                'tracking_error': tracking_error,
                'information_ratio_vs_benchmark': information_ratio,
                'up_capture_ratio': up_capture,
                'down_capture_ratio': down_capture,
                'excess_return_annualized': excess_returns.mean() * 252,
                'batting_average': (excess_returns > 0).mean()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating benchmark metrics: {e}")
            return {}
    
    def _calculate_time_metrics(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate time-based metrics"""
        try:
            returns = portfolio_df['returns'].dropna()
            
            if len(returns) == 0:
                return {}
            
            # Best and worst days
            best_day = returns.max()
            worst_day = returns.min()
            
            # Winning/losing streaks
            win_streak, loss_streak = self._calculate_streaks(returns)
            
            # Monthly/yearly performance
            monthly_returns = self._calculate_monthly_returns(portfolio_df)
            yearly_returns = self._calculate_yearly_returns(portfolio_df)
            
            return {
                'best_day': best_day,
                'worst_day': worst_day,
                'best_month': monthly_returns.max() if len(monthly_returns) > 0 else 0,
                'worst_month': monthly_returns.min() if len(monthly_returns) > 0 else 0,
                'best_year': yearly_returns.max() if len(yearly_returns) > 0 else 0,
                'worst_year': yearly_returns.min() if len(yearly_returns) > 0 else 0,
                'longest_winning_streak': win_streak,
                'longest_losing_streak': loss_streak,
                'positive_days_ratio': (returns > 0).mean(),
                'monthly_win_rate': (monthly_returns > 0).mean() if len(monthly_returns) > 0 else 0,
                'yearly_win_rate': (yearly_returns > 0).mean() if len(yearly_returns) > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating time metrics: {e}")
            return {}
    
    def _calculate_max_drawdown(self, portfolio_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        total_value = portfolio_df['total_value']
        running_max = total_value.expanding().max()
        drawdown = (total_value - running_max) / running_max
        return drawdown.min()
    
    def _calculate_drawdown_periods(self, drawdown: pd.Series) -> List[int]:
        """Calculate drawdown periods"""
        is_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for in_drawdown in is_drawdown:
            if in_drawdown:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        # Add the last period if it ends in drawdown
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return drawdown_periods
    
    def _calculate_recovery_metrics(self, drawdown: pd.Series, total_value: pd.Series) -> Dict[str, float]:
        """Calculate recovery metrics"""
        max_drawdown = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # Find recovery (when drawdown returns to 0)
        recovery_idx = None
        for idx in drawdown.index[drawdown.index > max_dd_idx]:
            if drawdown[idx] >= 0:
                recovery_idx = idx
                break
        
        recovery_factor = 0
        if recovery_idx is not None:
            peak_value = total_value[max_dd_idx] / (1 + max_drawdown)
            recovery_value = total_value[recovery_idx]
            recovery_factor = (recovery_value - peak_value) / abs(peak_value * max_drawdown)
        
        return {'recovery_factor': recovery_factor}
    
    def _calculate_pain_index(self, drawdown: pd.Series) -> float:
        """Calculate pain index (average drawdown)"""
        return abs(drawdown.mean())
    
    def _calculate_ulcer_index(self, drawdown: pd.Series) -> float:
        """Calculate ulcer index"""
        return np.sqrt(np.mean(drawdown**2))
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        p95 = returns.quantile(0.95)
        p5 = returns.quantile(0.05)
        return abs(p95 / p5) if p5 != 0 else float('inf')
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio"""
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0].sum())
        return gains / losses if losses > 0 else float('inf')
    
    def _calculate_treynor_ratio(self, returns: pd.Series, beta: float = 1.0) -> float:
        """Calculate Treynor ratio"""
        annualized_return = returns.mean() * 252
        excess_return = annualized_return - self.risk_free_rate
        return excess_return / beta if beta != 0 else 0
    
    def _calculate_alpha_beta(self, portfolio_returns: pd.Series, 
                            benchmark_returns: pd.Series) -> Tuple[float, float]:
        """Calculate alpha and beta using linear regression"""
        try:
            if len(portfolio_returns) != len(benchmark_returns):
                min_len = min(len(portfolio_returns), len(benchmark_returns))
                portfolio_returns = portfolio_returns.iloc[-min_len:]
                benchmark_returns = benchmark_returns.iloc[-min_len:]
            
            # Remove any NaN values
            mask = ~(np.isnan(portfolio_returns) | np.isnan(benchmark_returns))
            portfolio_clean = portfolio_returns[mask]
            benchmark_clean = benchmark_returns[mask]
            
            if len(portfolio_clean) < 2 or len(benchmark_clean) < 2:
                return 0.0, 1.0
            
            # Linear regression
            beta, alpha, r_value, p_value, std_err = stats.linregress(benchmark_clean, portfolio_clean)
            
            # Annualize alpha
            alpha_annualized = alpha * 252
            
            return alpha_annualized, beta
            
        except Exception as e:
            self.logger.error(f"Error calculating alpha/beta: {e}")
            return 0.0, 1.0
    
    def _calculate_capture_ratios(self, portfolio_returns: pd.Series, 
                                benchmark_returns: pd.Series) -> Tuple[float, float]:
        """Calculate up and down capture ratios"""
        try:
            # Up capture ratio
            up_market = benchmark_returns > 0
            if up_market.sum() > 0:
                portfolio_up = portfolio_returns[up_market].mean()
                benchmark_up = benchmark_returns[up_market].mean()
                up_capture = portfolio_up / benchmark_up if benchmark_up != 0 else 0
            else:
                up_capture = 0
            
            # Down capture ratio
            down_market = benchmark_returns < 0
            if down_market.sum() > 0:
                portfolio_down = portfolio_returns[down_market].mean()
                benchmark_down = benchmark_returns[down_market].mean()
                down_capture = portfolio_down / benchmark_down if benchmark_down != 0 else 0
            else:
                down_capture = 0
            
            return up_capture, down_capture
            
        except Exception as e:
            self.logger.error(f"Error calculating capture ratios: {e}")
            return 1.0, 1.0
    
    def _calculate_streaks(self, returns: pd.Series) -> Tuple[int, int]:
        """Calculate longest winning and losing streaks"""
        if len(returns) == 0:
            return 0, 0
        
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for ret in returns:
            if ret > 0:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            elif ret < 0:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)
            else:
                win_streak = 0
                loss_streak = 0
        
        return max_win_streak, max_loss_streak
    
    def _calculate_monthly_returns(self, portfolio_df: pd.DataFrame) -> pd.Series:
        """Calculate monthly returns"""
        try:
            monthly_data = portfolio_df['total_value'].resample('M').last()
            return monthly_data.pct_change().dropna()
        except Exception:
            return pd.Series()
    
    def _calculate_yearly_returns(self, portfolio_df: pd.DataFrame) -> pd.Series:
        """Calculate yearly returns"""
        try:
            yearly_data = portfolio_df['total_value'].resample('Y').last()
            return yearly_data.pct_change().dropna()
        except Exception:
            return pd.Series()

class RiskMetrics:
    """Additional risk calculation utilities"""
    
    @staticmethod
    def calculate_var_parametric(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate parametric VaR assuming normal distribution"""
        mean = returns.mean()
        std = returns.std()
        var = stats.norm.ppf(confidence_level, mean, std)
        return var
    
    @staticmethod
    def calculate_var_historical(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate historical VaR"""
        return returns.quantile(confidence_level)
    
    @staticmethod
    def calculate_var_monte_carlo(returns: pd.Series, confidence_level: float = 0.05, 
                                num_simulations: int = 10000) -> float:
        """Calculate Monte Carlo VaR"""
        mean = returns.mean()
        std = returns.std()
        simulated_returns = np.random.normal(mean, std, num_simulations)
        var = np.percentile(simulated_returns, confidence_level * 100)
        return var
    
    @staticmethod
    def calculate_expected_shortfall(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        var = returns.quantile(confidence_level)
        es = returns[returns <= var].mean()
        return es
    
    @staticmethod
    def calculate_maximum_entropy_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Maximum Entropy VaR"""
        # This is a simplified implementation
        # In practice, you would use more sophisticated maximum entropy methods
        sorted_returns = returns.sort_values()
        index = int(confidence_level * len(sorted_returns))
        return sorted_returns.iloc[index] if index < len(sorted_returns) else sorted_returns.iloc[-1]

class PerformanceAttribution:
    """Performance attribution analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def brinson_attribution(self, portfolio_weights: pd.DataFrame, 
                          portfolio_returns: pd.DataFrame,
                          benchmark_weights: pd.DataFrame,
                          benchmark_returns: pd.DataFrame) -> Dict[str, pd.Series]:
        """Brinson-Fachler performance attribution"""
        try:
            # Allocation effect
            allocation_effect = (portfolio_weights - benchmark_weights) * benchmark_returns
            
            # Selection effect  
            selection_effect = benchmark_weights * (portfolio_returns - benchmark_returns)
            
            # Interaction effect
            interaction_effect = (portfolio_weights - benchmark_weights) * (portfolio_returns - benchmark_returns)
            
            return {
                'allocation_effect': allocation_effect.sum(axis=1),
                'selection_effect': selection_effect.sum(axis=1),
                'interaction_effect': interaction_effect.sum(axis=1),
                'total_effect': (allocation_effect + selection_effect + interaction_effect).sum(axis=1)
            }
            
        except Exception as e:
            self.logger.error(f"Error in Brinson attribution: {e}")
            return {}
