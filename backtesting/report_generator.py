import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import json
from pathlib import Path
import base64
from io import BytesIO

class ReportGenerator:
    """Comprehensive backtesting report generator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Setup matplotlib style for professional reports"""
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Set global parameters
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    
    def generate_report(self, portfolio_df: pd.DataFrame, metrics: Dict[str, Any], 
                       trade_analysis: Dict[str, Any], config: Any,
                       output_path: str = "backtest_report.pdf") -> Dict[str, Any]:
        """Generate comprehensive backtesting report"""
        try:
            self.logger.info("Generating backtesting report")
            
            # Generate all report components
            report_data = {
                'executive_summary': self._generate_executive_summary(metrics, trade_analysis, config),
                'performance_overview': self._generate_performance_overview(portfolio_df, metrics),
                'risk_analysis': self._generate_risk_analysis(metrics),
                'trade_analysis': self._generate_trade_analysis_report(trade_analysis),
                'drawdown_analysis': self._generate_drawdown_analysis(portfolio_df, metrics),
                'monthly_performance': self._generate_monthly_performance(portfolio_df),
                'charts': self._generate_charts(portfolio_df, metrics),
                'recommendations': self._generate_recommendations(metrics, trade_analysis)
            }
            
            # Generate PDF report
            self._generate_pdf_report(report_data, output_path)
            
            # Generate HTML report
            html_report = self._generate_html_report(report_data)
            
            # Generate JSON summary
            json_summary = self._generate_json_summary(report_data)
            
            self.logger.info(f"Report generated successfully: {output_path}")
            
            return {
                'report_data': report_data,
                'html_report': html_report,
                'json_summary': json_summary,
                'pdf_path': output_path
            }
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return {}
    
    def _generate_executive_summary(self, metrics: Dict[str, Any], 
                                  trade_analysis: Dict[str, Any], 
                                  config: Any) -> Dict[str, Any]:
        """Generate executive summary"""
        try:
            # Key performance indicators
            total_return = metrics.get('total_return', 0) * 100
            annualized_return = metrics.get('annualized_return', 0) * 100
            max_drawdown = metrics.get('max_drawdown', 0) * 100
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            
            # Trading statistics
            total_trades = trade_analysis.get('total_trades', 0)
            win_rate = trade_analysis.get('win_rate', 0) * 100
            profit_factor = trade_analysis.get('profit_factor', 0)
            
            # Risk assessment
            risk_level = self._assess_risk_level(metrics)
            
            # Performance grade
            performance_grade = self._calculate_performance_grade(metrics, trade_analysis)
            
            summary = {
                'period': f"{config.start_date.strftime('%Y-%m-%d')} to {config.end_date.strftime('%Y-%m-%d')}",
                'initial_capital': f"${config.initial_capital:,.2f}",
                'final_value': f"${metrics.get('final_value', config.initial_capital):,.2f}",
                'total_return': f"{total_return:.2f}%",
                'annualized_return': f"{annualized_return:.2f}%",
                'max_drawdown': f"{abs(max_drawdown):.2f}%",
                'sharpe_ratio': f"{sharpe_ratio:.2f}",
                'volatility': f"{metrics.get('annualized_volatility', 0) * 100:.2f}%",
                'total_trades': total_trades,
                'win_rate': f"{win_rate:.1f}%",
                'profit_factor': f"{profit_factor:.2f}",
                'risk_level': risk_level,
                'performance_grade': performance_grade,
                'key_strengths': self._identify_key_strengths(metrics, trade_analysis),
                'key_concerns': self._identify_key_concerns(metrics, trade_analysis)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {e}")
            return {}
    
    def _generate_performance_overview(self, portfolio_df: pd.DataFrame, 
                                     metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance overview section"""
        try:
            returns = portfolio_df['returns'].dropna()
            
            overview = {
                'return_metrics': {
                    'total_return': f"{metrics.get('total_return', 0) * 100:.2f}%",
                    'annualized_return': f"{metrics.get('annualized_return', 0) * 100:.2f}%",
                    'average_monthly_return': f"{metrics.get('average_monthly_return', 0) * 100:.2f}%",
                    'best_month': f"{metrics.get('best_month', 0) * 100:.2f}%",
                    'worst_month': f"{metrics.get('worst_month', 0) * 100:.2f}%",
                    'positive_months': f"{metrics.get('monthly_win_rate', 0) * 100:.1f}%"
                },
                'risk_metrics': {
                    'volatility': f"{metrics.get('annualized_volatility', 0) * 100:.2f}%",
                    'downside_deviation': f"{metrics.get('annualized_downside_deviation', 0) * 100:.2f}%",
                    'var_95': f"{metrics.get('var_95', 0) * 100:.2f}%",
                    'cvar_95': f"{metrics.get('cvar_95', 0) * 100:.2f}%",
                    'max_drawdown': f"{abs(metrics.get('max_drawdown', 0)) * 100:.2f}%",
                    'current_drawdown': f"{abs(metrics.get('current_drawdown', 0)) * 100:.2f}%"
                },
                'risk_adjusted_metrics': {
                    'sharpe_ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                    'sortino_ratio': f"{metrics.get('sortino_ratio', 0):.2f}",
                    'calmar_ratio': f"{metrics.get('calmar_ratio', 0):.2f}",
                    'omega_ratio': f"{metrics.get('omega_ratio', 0):.2f}"
                },
                'distribution_analysis': {
                    'skewness': f"{metrics.get('skewness', 0):.2f}",
                    'kurtosis': f"{metrics.get('kurtosis', 0):.2f}",
                    'tail_ratio': f"{metrics.get('tail_ratio', 0):.2f}",
                    'is_normal': metrics.get('is_normal_shapiro', False)
                }
            }
            
            return overview
            
        except Exception as e:
            self.logger.error(f"Error generating performance overview: {e}")
            return {}
    
    def _generate_risk_analysis(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed risk analysis"""
        try:
            risk_analysis = {
                'drawdown_analysis': {
                    'max_drawdown': f"{abs(metrics.get('max_drawdown', 0)) * 100:.2f}%",
                    'average_drawdown_duration': f"{metrics.get('average_drawdown_duration', 0):.0f} days",
                    'max_drawdown_duration': f"{metrics.get('max_drawdown_duration', 0):.0f} days",
                    'recovery_factor': f"{metrics.get('recovery_factor', 0):.2f}",
                    'pain_index': f"{metrics.get('pain_index', 0) * 100:.2f}%",
                    'ulcer_index': f"{metrics.get('ulcer_index', 0) * 100:.2f}%"
                },
                'var_analysis': {
                    'daily_var_95': f"{metrics.get('var_95', 0) * 100:.2f}%",
                    'daily_var_99': f"{metrics.get('var_99', 0) * 100:.2f}%",
                    'expected_shortfall_95': f"{metrics.get('cvar_95', 0) * 100:.2f}%",
                    'expected_shortfall_99': f"{metrics.get('cvar_99', 0) * 100:.2f}%"
                },
                'volatility_analysis': {
                    'daily_volatility': f"{metrics.get('daily_volatility', 0) * 100:.2f}%",
                    'annualized_volatility': f"{metrics.get('annualized_volatility', 0) * 100:.2f}%",
                    'downside_volatility': f"{metrics.get('annualized_downside_deviation', 0) * 100:.2f}%",
                    'semi_deviation': f"{metrics.get('semi_deviation', 0) * 100:.2f}%"
                },
                'tail_risk': {
                    'skewness': metrics.get('skewness', 0),
                    'excess_kurtosis': metrics.get('excess_kurtosis', 0),
                    'tail_ratio': metrics.get('tail_ratio', 0)
                },
                'risk_assessment': self._generate_risk_assessment(metrics)
            }
            
            return risk_analysis
            
        except Exception as e:
            self.logger.error(f"Error generating risk analysis: {e}")
            return {}
    
    def _generate_trade_analysis_report(self, trade_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed trade analysis report"""
        try:
            if not trade_analysis:
                return {'message': 'No trade data available for analysis'}
            
            report = {
                'trading_statistics': {
                    'total_trades': trade_analysis.get('total_trades', 0),
                    'winning_trades': trade_analysis.get('winning_trades', 0),
                    'losing_trades': trade_analysis.get('losing_trades', 0),
                    'win_rate': f"{trade_analysis.get('win_rate', 0) * 100:.1f}%"
                },
                'profit_loss_analysis': {
                    'total_pnl': f"${trade_analysis.get('total_pnl', 0):,.2f}",
                    'average_win': f"${trade_analysis.get('avg_win', 0):,.2f}",
                    'average_loss': f"${trade_analysis.get('avg_loss', 0):,.2f}",
                    'largest_win': f"${trade_analysis.get('largest_win', 0):,.2f}",
                    'largest_loss': f"${trade_analysis.get('largest_loss', 0):,.2f}",
                    'profit_factor': f"{trade_analysis.get('profit_factor', 0):.2f}"
                },
                'risk_reward': {
                    'avg_win_loss_ratio': f"{abs(trade_analysis.get('avg_win', 0) / trade_analysis.get('avg_loss', 1)):.2f}",
                    'expectancy': f"${self._calculate_expectancy(trade_analysis):,.2f}",
                    'kelly_criterion': f"{self._calculate_kelly_criterion(trade_analysis):.2f}%"
                },
                'trade_quality': self._assess_trade_quality(trade_analysis)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating trade analysis: {e}")
            return {}
    
    def _generate_drawdown_analysis(self, portfolio_df: pd.DataFrame, 
                                  metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed drawdown analysis"""
        try:
            total_value = portfolio_df['total_value']
            running_max = total_value.expanding().max()
            drawdown = (total_value - running_max) / running_max
            
            # Find all drawdown periods
            drawdown_periods = self._find_drawdown_periods(drawdown)
            
            analysis = {
                'current_status': {
                    'in_drawdown': drawdown.iloc[-1] < -0.001,  # More than 0.1% drawdown
                    'current_drawdown': f"{abs(drawdown.iloc[-1]) * 100:.2f}%",
                    'days_in_drawdown': self._days_in_current_drawdown(drawdown)
                },
                'historical_drawdowns': {
                    'number_of_drawdowns': len(drawdown_periods),
                    'average_depth': f"{np.mean([dd['depth'] for dd in drawdown_periods]) * 100:.2f}%" if drawdown_periods else "0.00%",
                    'average_duration': f"{np.mean([dd['duration'] for dd in drawdown_periods]):.0f} days" if drawdown_periods else "0 days",
                    'average_recovery': f"{np.mean([dd['recovery_time'] for dd in drawdown_periods if dd['recovery_time'] is not None]):.0f} days" if any(dd['recovery_time'] for dd in drawdown_periods) else "N/A"
                },
                'worst_drawdowns': sorted(drawdown_periods, key=lambda x: x['depth'])[:5] if drawdown_periods else [],
                'recovery_analysis': {
                    'recovery_factor': metrics.get('recovery_factor', 0),
                    'pain_index': f"{metrics.get('pain_index', 0) * 100:.2f}%",
                    'ulcer_index': f"{metrics.get('ulcer_index', 0) * 100:.2f}%"
                }
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating drawdown analysis: {e}")
            return {}
    
    def _generate_monthly_performance(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate monthly performance breakdown"""
        try:
            # Resample to monthly returns
            monthly_data = portfolio_df.resample('M').agg({
                'total_value': 'last',
                'cash': 'last',
                'positions_value': 'last'
            })
            
            monthly_returns = monthly_data['total_value'].pct_change().dropna()
            
            # Create monthly performance table
            monthly_table = []
            for date, ret in monthly_returns.items():
                monthly_table.append({
                    'month': date.strftime('%Y-%m'),
                    'return': f"{ret * 100:.2f}%",
                    'return_value': ret,
                    'cumulative_value': monthly_data.loc[date, 'total_value']
                })
            
            # Calculate monthly statistics
            monthly_stats = {
                'best_month': f"{monthly_returns.max() * 100:.2f}%",
                'worst_month': f"{monthly_returns.min() * 100:.2f}%",
                'average_month': f"{monthly_returns.mean() * 100:.2f}%",
                'positive_months': f"{(monthly_returns > 0).sum()}/{len(monthly_returns)}",
                'win_rate': f"{(monthly_returns > 0).mean() * 100:.1f}%",
                'volatility': f"{monthly_returns.std() * np.sqrt(12) * 100:.2f}%"
            }
            
            return {
                'monthly_table': monthly_table,
                'monthly_stats': monthly_stats,
                'monthly_returns_series': monthly_returns
            }
            
        except Exception as e:
            self.logger.error(f"Error generating monthly performance: {e}")
            return {}
    
    def _generate_charts(self, portfolio_df: pd.DataFrame, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Generate all charts and return as base64 encoded strings"""
        try:
            charts = {}
            
            # Portfolio value chart
            charts['portfolio_value'] = self._create_portfolio_value_chart(portfolio_df)
            
            # Drawdown chart
            charts['drawdown'] = self._create_drawdown_chart(portfolio_df)
            
            # Monthly returns heatmap
            charts['monthly_heatmap'] = self._create_monthly_heatmap(portfolio_df)
            
            # Return distribution
            charts['return_distribution'] = self._create_return_distribution_chart(portfolio_df)
            
            # Rolling metrics chart
            charts['rolling_metrics'] = self._create_rolling_metrics_chart(portfolio_df)
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error generating charts: {e}")
            return {}
    
    def _create_portfolio_value_chart(self, portfolio_df: pd.DataFrame) -> str:
        """Create portfolio value over time chart"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Portfolio value
            ax1.plot(portfolio_df.index, portfolio_df['total_value'], linewidth=2, label='Portfolio Value')
            ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Daily returns
            returns = portfolio_df['returns'].dropna()
            ax2.plot(returns.index, returns * 100, linewidth=1, alpha=0.7, color='orange')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.set_title('Daily Returns (%)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Daily Return (%)')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_base64
            
        except Exception as e:
            self.logger.error(f"Error creating portfolio value chart: {e}")
            return ""
    
    def _create_drawdown_chart(self, portfolio_df: pd.DataFrame) -> str:
        """Create drawdown chart"""
        try:
            total_value = portfolio_df['total_value']
            running_max = total_value.expanding().max()
            drawdown = (total_value - running_max) / running_max * 100
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.fill_between(drawdown.index, drawdown, 0, alpha=0.7, color='red', label='Drawdown')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax.set_title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
            ax.set_ylabel('Drawdown (%)')
            ax.set_xlabel('Date')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_base64
            
        except Exception as e:
            self.logger.error(f"Error creating drawdown chart: {e}")
            return ""
    
    def _create_monthly_heatmap(self, portfolio_df: pd.DataFrame) -> str:
        """Create monthly returns heatmap"""
        try:
            # Calculate monthly returns
            monthly_data = portfolio_df['total_value'].resample('M').last()
            monthly_returns = monthly_data.pct_change().dropna() * 100
            
            if len(monthly_returns) == 0:
                return ""
            
            # Create pivot table for heatmap
            monthly_returns.index = pd.to_datetime(monthly_returns.index)
            monthly_returns_df = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values
            })
            
            pivot_table = monthly_returns_df.pivot(index='Month', columns='Year', values='Return')
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                       ax=ax, cbar_kws={'label': 'Monthly Return (%)'})
            ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Month')
            ax.set_xlabel('Year')
            
            # Set month labels
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_yticklabels(month_labels[:len(pivot_table.index)])
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_base64
            
        except Exception as e:
            self.logger.error(f"Error creating monthly heatmap: {e}")
            return ""
    
    def _create_return_distribution_chart(self, portfolio_df: pd.DataFrame) -> str:
        """Create return distribution chart"""
        try:
            returns = portfolio_df['returns'].dropna() * 100
            
            if len(returns) == 0:
                return ""
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram
            ax1.hist(returns, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            ax1.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.2f}%')
            ax1.axvline(returns.quantile(0.05), color='orange', linestyle='--', label=f'5th Percentile: {returns.quantile(0.05):.2f}%')
            ax1.axvline(returns.quantile(0.95), color='orange', linestyle='--', label=f'95th Percentile: {returns.quantile(0.95):.2f}%')
            ax1.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Daily Return (%)')
            ax1.set_ylabel('Density')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Q-Q plot
            stats.probplot(returns, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot vs Normal Distribution', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_base64
            
        except Exception as e:
            self.logger.error(f"Error creating return distribution chart: {e}")
            return ""
    
    def _create_rolling_metrics_chart(self, portfolio_df: pd.DataFrame) -> str:
        """Create rolling metrics chart"""
        try:
            returns = portfolio_df['returns'].dropna()
            
            if len(returns) < 252:  # Need at least 1 year of data
                return ""
            
            # Calculate rolling metrics
            window = 252  # 1 year
            rolling_sharpe = returns.rolling(window).apply(
                lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
            )
            rolling_volatility = returns.rolling(window).std() * np.sqrt(252) * 100
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Rolling Sharpe ratio
            ax1.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2, color='blue')
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Sharpe = 1')
            ax1.set_title('Rolling Sharpe Ratio (1-Year Window)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Sharpe Ratio')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Rolling volatility
            ax2.plot(rolling_volatility.index, rolling_volatility, linewidth=2, color='orange')
            ax2.set_title('Rolling Volatility (1-Year Window)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Volatility (%)')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_base64
            
        except Exception as e:
            self.logger.error(f"Error creating rolling metrics chart: {e}")
            return ""
    
    def _generate_recommendations(self, metrics: Dict[str, Any], 
                                trade_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on performance"""
        recommendations = []
        
        try:
            # Performance-based recommendations
            if metrics.get('sharpe_ratio', 0) < 1.0:
                recommendations.append("Consider improving risk-adjusted returns. Current Sharpe ratio is below 1.0.")
            
            if abs(metrics.get('max_drawdown', 0)) > 0.2:
                recommendations.append("Maximum drawdown exceeds 20%. Consider implementing stronger risk management.")
            
            if metrics.get('annualized_volatility', 0) > 0.25:
                recommendations.append("High volatility detected. Consider diversification or position sizing adjustments.")
            
            # Trade-based recommendations
            if trade_analysis and trade_analysis.get('win_rate', 0) < 0.4:
                recommendations.append("Win rate is below 40%. Review entry signals and market conditions.")
            
            if trade_analysis and trade_analysis.get('profit_factor', 0) < 1.25:
                recommendations.append("Profit factor is low. Focus on improving average win size or reducing average loss size.")
            
            # Risk-based recommendations
            if metrics.get('tail_ratio', 0) < 1.0:
                recommendations.append("Negative skewness detected. Be cautious of tail risk.")
            
            if metrics.get('correlation_with_benchmark', 0) > 0.9:
                recommendations.append("High correlation with benchmark. Consider adding uncorrelated strategies.")
            
            # Recovery recommendations
            if metrics.get('recovery_factor', 0) < 1.0:
                recommendations.append("Slow recovery from drawdowns. Consider more adaptive position sizing.")
            
            # Default recommendations if none triggered
            if not recommendations:
                recommendations.append("Strategy shows solid performance. Continue monitoring key metrics.")
                recommendations.append("Consider stress testing under different market conditions.")
                recommendations.append("Regular rebalancing and parameter optimization may improve results.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations. Please review performance manually."]
    
    def _generate_pdf_report(self, report_data: Dict[str, Any], output_path: str):
        """Generate PDF version of the report"""
        try:
            with PdfPages(output_path) as pdf:
                # Title page
                self._create_title_page(pdf, report_data)
                
                # Executive summary
                self._create_executive_summary_page(pdf, report_data)
                
                # Performance charts
                self._create_charts_pages(pdf, report_data)
                
                # Detailed metrics
                self._create_metrics_pages(pdf, report_data)
                
        except Exception as e:
            self.logger.error(f"Error generating PDF report: {e}")
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML version of the report"""
        try:
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Backtesting Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }}
                    .section {{ margin: 30px 0; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; }}
                    .chart {{ text-align: center; margin: 20px 0; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Backtesting Performance Report</h1>
                    <p>Generated on {date}</p>
                </div>
                
                <div class="section">
                    <h2>Executive Summary</h2>
                    {executive_summary}
                </div>
                
                <div class="section">
                    <h2>Performance Charts</h2>
                    {charts}
                </div>
                
                <div class="section">
                    <h2>Detailed Metrics</h2>
                    {detailed_metrics}
                </div>
                
                <div class="section">
                    <h2>Recommendations</h2>
                    {recommendations}
                </div>
            </body>
            </html>
            """
            
            # Format the template with actual data
            html_content = html_template.format(
                date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                executive_summary=self._format_executive_summary_html(report_data.get('executive_summary', {})),
                charts=self._format_charts_html(report_data.get('charts', {})),
                detailed_metrics=self._format_metrics_html(report_data.get('performance_overview', {})),
                recommendations=self._format_recommendations_html(report_data.get('recommendations', []))
            )
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")
            return ""
    
    def _generate_json_summary(self, report_data: Dict[str, Any]) -> str:
        """Generate JSON summary of the report"""
        try:
            # Create a clean summary without charts (base64 strings are too large)
            summary = {
                'executive_summary': report_data.get('executive_summary', {}),
                'performance_overview': report_data.get('performance_overview', {}),
                'risk_analysis': report_data.get('risk_analysis', {}),
                'trade_analysis': report_data.get('trade_analysis', {}),
                'recommendations': report_data.get('recommendations', []),
                'generated_at': datetime.now().isoformat()
            }
            
            return json.dumps(summary, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Error generating JSON summary: {e}")
            return "{}"
    
    # Helper methods
    def _assess_risk_level(self, metrics: Dict[str, Any]) -> str:
        """Assess overall risk level"""
        volatility = metrics.get('annualized_volatility', 0)
        max_drawdown = abs(metrics.get('max_drawdown', 0))
        
        if volatility < 0.1 and max_drawdown < 0.05:
            return "Low"
        elif volatility < 0.2 and max_drawdown < 0.15:
            return "Moderate"
        elif volatility < 0.3 and max_drawdown < 0.25:
            return "High"
        else:
            return "Very High"
    
    def _calculate_performance_grade(self, metrics: Dict[str, Any], 
                                   trade_analysis: Dict[str, Any]) -> str:
        """Calculate overall performance grade"""
        score = 0
        
        # Sharpe ratio contribution (0-30 points)
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe >= 2.0:
            score += 30
        elif sharpe >= 1.5:
            score += 25
        elif sharpe >= 1.0:
            score += 20
        elif sharpe >= 0.5:
            score += 15
        elif sharpe >= 0:
            score += 10
        
        # Return contribution (0-25 points)
        annual_return = metrics.get('annualized_return', 0)
        if annual_return >= 0.20:
            score += 25
        elif annual_return >= 0.15:
            score += 20
        elif annual_return >= 0.10:
            score += 15
        elif annual_return >= 0.05:
            score += 10
        elif annual_return >= 0:
            score += 5
        
        # Drawdown contribution (0-25 points)
        max_dd = abs(metrics.get('max_drawdown', 0))
        if max_dd <= 0.05:
            score += 25
        elif max_dd <= 0.10:
            score += 20
        elif max_dd <= 0.15:
            score += 15
        elif max_dd <= 0.20:
            score += 10
        elif max_dd <= 0.30:
            score += 5
        
        # Trading efficiency (0-20 points)
        if trade_analysis:
            win_rate = trade_analysis.get('win_rate', 0)
            profit_factor = trade_analysis.get('profit_factor', 0)
            
            if win_rate >= 0.6 and profit_factor >= 2.0:
                score += 20
            elif win_rate >= 0.5 and profit_factor >= 1.5:
                score += 15
            elif win_rate >= 0.4 and profit_factor >= 1.25:
                score += 10
            elif profit_factor >= 1.0:
                score += 5
        
        # Convert score to grade
        if score >= 85:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 75:
            return "A-"
        elif score >= 70:
            return "B+"
        elif score >= 65:
            return "B"
        elif score >= 60:
            return "B-"
        elif score >= 55:
            return "C+"
        elif score >= 50:
            return "C"
        elif score >= 45:
            return "C-"
        elif score >= 40:
            return "D"
        else:
            return "F"
    
    def _identify_key_strengths(self, metrics: Dict[str, Any], 
                              trade_analysis: Dict[str, Any]) -> List[str]:
        """Identify key strengths of the strategy"""
        strengths = []
        
        if metrics.get('sharpe_ratio', 0) >= 1.5:
            strengths.append("Excellent risk-adjusted returns")
        
        if abs(metrics.get('max_drawdown', 0)) <= 0.1:
            strengths.append("Low maximum drawdown")
        
        if trade_analysis and trade_analysis.get('win_rate', 0) >= 0.6:
            strengths.append("High win rate")
        
        if trade_analysis and trade_analysis.get('profit_factor', 0) >= 2.0:
            strengths.append("Strong profit factor")
        
        if metrics.get('sortino_ratio', 0) >= 1.5:
            strengths.append("Good downside risk management")
        
        return strengths[:3]  # Return top 3 strengths
    
    def _identify_key_concerns(self, metrics: Dict[str, Any], 
                             trade_analysis: Dict[str, Any]) -> List[str]:
        """Identify key concerns with the strategy"""
        concerns = []
        
        if metrics.get('sharpe_ratio', 0) < 0.5:
            concerns.append("Low risk-adjusted returns")
        
        if abs(metrics.get('max_drawdown', 0)) > 0.2:
            concerns.append("High maximum drawdown")
        
        if trade_analysis and trade_analysis.get('win_rate', 0) < 0.4:
            concerns.append("Low win rate")
        
        if trade_analysis and trade_analysis.get('profit_factor', 0) < 1.25:
            concerns.append("Poor profit factor")
        
        if metrics.get('annualized_volatility', 0) > 0.3:
            concerns.append("High volatility")
        
        return concerns[:3]  # Return top 3 concerns
    
    def _calculate_expectancy(self, trade_analysis: Dict[str, Any]) -> float:
        """Calculate trading expectancy"""
        if not trade_analysis:
            return 0.0
        
        win_rate = trade_analysis.get('win_rate', 0)
        avg_win = trade_analysis.get('avg_win', 0)
        avg_loss = trade_analysis.get('avg_loss', 0)
        
        return (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    def _calculate_kelly_criterion(self, trade_analysis: Dict[str, Any]) -> float:
        """Calculate Kelly Criterion for optimal position sizing"""
        if not trade_analysis:
            return 0.0
        
        win_rate = trade_analysis.get('win_rate', 0)
        avg_win = trade_analysis.get('avg_win', 0)
        avg_loss = abs(trade_analysis.get('avg_loss', 1))
        
        if avg_loss == 0:
            return 0.0
        
        kelly = win_rate - ((1 - win_rate) * (avg_loss / avg_win))
        return max(0, min(kelly * 100, 25))  # Cap at 25% for safety
    
    def _assess_trade_quality(self, trade_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall trade quality"""
        if not trade_analysis:
            return {}
        
        win_rate = trade_analysis.get('win_rate', 0)
        profit_factor = trade_analysis.get('profit_factor', 0)
        avg_win = trade_analysis.get('avg_win', 0)
        avg_loss = abs(trade_analysis.get('avg_loss', 1))
        
        # Quality assessment
        quality_score = 0
        if win_rate >= 0.5:
            quality_score += 25
        if profit_factor >= 1.5:
            quality_score += 25
        if avg_win / avg_loss >= 1.5:
            quality_score += 25
        if trade_analysis.get('total_trades', 0) >= 50:
            quality_score += 25
        
        if quality_score >= 80:
            quality = "Excellent"
        elif quality_score >= 60:
            quality = "Good"
        elif quality_score >= 40:
            quality = "Fair"
        else:
            quality = "Poor"
        
        return {
            'overall_quality': quality,
            'quality_score': quality_score,
            'consistency': "High" if win_rate >= 0.5 and profit_factor >= 1.5 else "Low"
        }
    
    def _find_drawdown_periods(self, drawdown: pd.Series) -> List[Dict[str, Any]]:
        """Find all drawdown periods with details"""
        periods = []
        in_drawdown = False
        start_date = None
        peak_dd = 0
        
        for date, dd in drawdown.items():
            if dd < -0.001 and not in_drawdown:  # Start of drawdown
                in_drawdown = True
                start_date = date
                peak_dd = dd
            elif dd < -0.001 and in_drawdown:  # Continue drawdown
                peak_dd = min(peak_dd, dd)
            elif dd >= -0.001 and in_drawdown:  # End of drawdown
                in_drawdown = False
                duration = (date - start_date).days
                recovery_time = None  # Would need to calculate recovery to new high
                
                periods.append({
                    'start_date': start_date,
                    'end_date': date,
                    'duration': duration,
                    'depth': peak_dd,
                    'recovery_time': recovery_time
                })
        
        return periods
    
    def _days_in_current_drawdown(self, drawdown: pd.Series) -> int:
        """Calculate days in current drawdown"""
        if drawdown.iloc[-1] >= -0.001:
            return 0
        
        days = 0
        for i in range(len(drawdown) - 1, -1, -1):
            if drawdown.iloc[i] < -0.001:
                days += 1
            else:
                break
        
        return days
    
    # HTML formatting helper methods
    def _format_executive_summary_html(self, summary: Dict[str, Any]) -> str:
        """Format executive summary for HTML"""
        if not summary:
            return "<p>No summary available</p>"
        
        html = f"""
        <div class="summary-grid">
            <div class="metric">
                <h4>Period</h4>
                <p>{summary.get('period', 'N/A')}</p>
            </div>
            <div class="metric">
                <h4>Total Return</h4>
                <p>{summary.get('total_return', 'N/A')}</p>
            </div>
            <div class="metric">
                <h4>Sharpe Ratio</h4>
                <p>{summary.get('sharpe_ratio', 'N/A')}</p>
            </div>
            <div class="metric">
                <h4>Max Drawdown</h4>
                <p>{summary.get('max_drawdown', 'N/A')}</p>
            </div>
            <div class="metric">
                <h4>Win Rate</h4>
                <p>{summary.get('win_rate', 'N/A')}</p>
            </div>
            <div class="metric">
                <h4>Performance Grade</h4>
                <p>{summary.get('performance_grade', 'N/A')}</p>
            </div>
        </div>
        """
        
        return html
    
    def _format_charts_html(self, charts: Dict[str, str]) -> str:
        """Format charts for HTML"""
        html = ""
        for chart_name, chart_data in charts.items():
            if chart_data:
                html += f'<div class="chart"><img src="data:image/png;base64,{chart_data}" alt="{chart_name}"></div>'
        return html
    
    def _format_metrics_html(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for HTML"""
        # This would format the detailed metrics as HTML tables
        return "<p>Detailed metrics table would go here</p>"
    
    def _format_recommendations_html(self, recommendations: List[str]) -> str:
        """Format recommendations for HTML"""
        if not recommendations:
            return "<p>No recommendations available</p>"
        
        html = "<ul>"
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += "</ul>"
        
        return html
    
    def _create_title_page(self, pdf, report_data: Dict[str, Any]):
        """Create PDF title page"""
        # Implementation for PDF title page
        pass
    
    def _create_executive_summary_page(self, pdf, report_data: Dict[str, Any]):
        """Create PDF executive summary page"""
        # Implementation for PDF executive summary
        pass
    
    def _create_charts_pages(self, pdf, report_data: Dict[str, Any]):
        """Create PDF charts pages"""
        # Implementation for PDF charts
        pass
    
    def _create_metrics_pages(self, pdf, report_data: Dict[str, Any]):
        """Create PDF metrics pages"""
        # Implementation for PDF detailed metrics
        pass
