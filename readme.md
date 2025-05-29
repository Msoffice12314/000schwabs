# Schwab AI Trading System

🚀 **Advanced AI-powered trading system with BiConNet neural networks and real-time market analysis**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🌟 Features

### 🤖 AI-Powered Trading
- **BiConNet Architecture**: CNN-BiLSTM hybrid model for price prediction
- **Time Delay Embedding**: Enhanced temporal pattern recognition
- **Market Regime Detection**: Automatic adaptation to market conditions
- **Confidence Scoring**: AI predictions with uncertainty quantification

### 📊 Real-Time Market Data
- **Schwab API Integration**: Official Schwab API with OAuth2 authentication
- **WebSocket Streaming**: Real-time quotes and market data
- **Technical Indicators**: TA-Lib integration with 150+ indicators
- **Multi-Asset Support**: Stocks, options, ETFs, and indices

### 🎯 Advanced Trading Strategies
- **Multiple Strategy Modes**: Conservative, Moderate, Aggressive, Scalping
- **Risk Management**: Position sizing, stop-loss, portfolio limits
- **Portfolio Optimization**: Automatic rebalancing and allocation
- **Backtesting Engine**: Historical strategy validation

### 🌐 Modern Web Interface
- **Dark Theme UI**: Professional trading dashboard
- **Real-Time Updates**: Live portfolio and market data
- **Interactive Charts**: TradingView-style price charts
- **Mobile Responsive**: Trade from any device

### 🔒 Security & Compliance
- **Encrypted Credentials**: Secure storage of API keys and tokens
- **OAuth2 Authentication**: Secure Schwab API access
- **Audit Trail**: Comprehensive logging of all activities
- **Risk Controls**: Multiple layers of risk management

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Windows 10/11 (Intel i7 12700k, 32GB RAM, RTX 3090)
- Schwab Developer Account
- PostgreSQL (recommended) or SQLite

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/schwab-ai-trading.git
   cd schwab-ai-trading
   ```

2. **Set up Python environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

3. **Install TA-Lib** (Windows):
   ```bash
   pip install ta_lib-0.6.3-cp313-cp313-win_amd64.whl
   ```

4. **Configure environment**:
   ```bash
   copy .env.example .env
   # Edit .env with your Schwab API credentials
   ```

5. **Initialize database**:
   ```bash
   python -c "from database.models import create_tables; create_tables()"
   ```

6. **Authenticate with Schwab**:
   ```bash
   python main.py --authenticate
   ```

7. **Start the application**:
   ```bash
   python main.py --web
   ```

Visit `http://localhost:8000` to access the web interface.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │  Strategy Engine │    │   AI Models     │
│                 │    │                 │    │                 │
│ • Dashboard     │◄──►│ • Signal Gen.   │◄──►│ • BiConNet      │
│ • Portfolio     │    │ • Risk Mgmt     │    │ • Regime Det.   │
│ • Analysis      │    │ • Execution     │    │ • Predictors    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │   Schwab API    │    │   Monitoring    │
│                 │    │                 │    │                 │
│ • PostgreSQL    │    │ • Market Data   │    │ • Logging       │
│ • Redis Cache   │    │ • Trading       │    │ • Metrics       │
│ • File Storage  │    │ • Authentication│    │ • Alerts        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📈 Trading Strategies

### Strategy Modes

| Mode         | Risk Level | Max Position | Stop Loss | Take Profit | Max Daily Trades |
|--------------|------------|--------------|-----------|-------------|------------------|
| Conservative | Low        | 5%          | 1.5%      | 3%          | 3                |
| Moderate     | Medium     | 10%         | 2%        | 4%          | 10               |
| Aggressive   | High       | 15%         | 3%        | 6%          | 20               |
| Scalping     | Very High  | 8%          | 0.5%      | 1%          | 50               |

### AI Signal Generation

1. **Data Collection**: Real-time market data via Schwab API
2. **Feature Engineering**: Technical indicators + market microstructure
3. **BiConNet Prediction**: CNN-BiLSTM hybrid model inference
4. **Signal Fusion**: Combine AI predictions with technical analysis
5. **Risk Assessment**: Position sizing based on confidence and volatility
6. **Order Execution**: Smart order routing with slippage minimization

## 🧠 AI Models

### BiConNet Architecture
- **Input Layer**: Time-delay embedded price series
- **CNN Layers**: Local pattern extraction (64-128 filters)
- **BiLSTM Layers**: Long-term dependency modeling (50-100 units)
- **Attention Mechanism**: Focus on important time steps
- **Output Layer**: Price prediction with confidence intervals

### Training Pipeline
```bash
# Collect training data
python main.py --collect-data

# Train models
python main.py --train

# Validate performance
python main.py --backtest 2023-01-01 2023-12-31
```

## 🔧 Configuration

### Environment Variables
```env
# Schwab API
SCHWAB_CLIENT_ID=your_client_id
SCHWAB_CLIENT_SECRET=your_client_secret
SCHWAB_REDIRECT_URI=https://127.0.0.1:8182

# Security
MASTER_PASSWORD=secure_password
WEB_SECRET_KEY=random_secret_key

# Database
DB_HOST=localhost
DB_NAME=schwab_ai
DB_USERNAME=postgres
DB_PASSWORD=your_db_password

# Trading
MAX_PORTFOLIO_RISK=0.02
MAX_POSITION_SIZE=0.1
STOP_LOSS_PCT=0.02
TAKE_PROFIT_PCT=0.04

# Logging
LOG_LEVEL=INFO
LOG_FILE_PATH=./logs/schwab_ai.log
```

### Model Parameters
```python
MODEL_PARAMS = {
    'sequence_length': 60,
    'cnn_filters': 64,
    'lstm_units': 50,
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
}
```

## 🎮 Usage Examples

### Command Line Interface
```bash
# Start web interface
python main.py --web

# Run automated trading
python main.py --trade

# Data collection mode
python main.py --collect-data

# Train AI models
python main.py --train --retrain-all

# Run backtesting
python main.py --backtest 2023-01-01 2023-12-31 --symbols AAPL MSFT GOOGL

# Authentication
python main.py --authenticate
```

### Web Interface
- **Dashboard**: Real-time portfolio overview and market data
- **Portfolio**: Position management and performance analytics
- **Analysis**: AI predictions and technical analysis
- **Settings**: Configuration and risk parameters

### API Endpoints
```python
# Get real-time quote
GET /api/market/quote/AAPL

# Get AI predictions
GET /api/predictions/AAPL

# Portfolio summary
GET /api/portfolio/summary

# Risk analysis
GET /api/risk/analysis
```

## 📊 Performance Monitoring

### Key Metrics
- **Total Return**: Cumulative portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

### Logging
- **Trading Logs**: All trade executions and signals
- **Performance Logs**: Strategy metrics and AI model performance
- **Error Logs**: System errors and exceptions
- **Audit Logs**: Security and compliance events

## 🛡️ Risk Management

### Position Level
- **Position Sizing**: Kelly Criterion + volatility adjustment
- **Stop Loss**: Dynamic trailing stops
- **Take Profit**: Multiple exit levels
- **Correlation Limits**: Avoid concentrated risks

### Portfolio Level
- **Portfolio Heat**: Maximum risk exposure limits
- **Sector Limits**: Diversification requirements
- **Volatility Targeting**: Dynamic allocation based on market regime
- **Drawdown Controls**: Automatic position reduction

## 🔍 Backtesting

```python
from backtesting.backtest_engine import BacktestEngine

engine = BacktestEngine()
results = engine.run_backtest(
    start_date='2023-01-01',
    end_date='2023-12-31',
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    initial_capital=100000,
    strategy_mode='moderate'
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

## 🚨 Monitoring & Alerts

### System Health
- **API Status**: Schwab API connectivity
- **Model Performance**: AI prediction accuracy
- **Database Health**: Connection and performance
- **Memory Usage**: System resource monitoring

### Trading Alerts
- **Large Positions**: Significant position changes
- **Risk Breaches**: Risk limit violations
- **System Errors**: Critical system failures
- **Market Events**: Unusual market conditions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v --cov=schwab_ai

# Code formatting
black schwab_ai/
isort schwab_ai/

# Type checking
mypy schwab_ai/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. The authors and contributors are not responsible for any financial losses incurred through the use of this software.

## 🆘 Support

- **Documentation**: [Wiki](https://github.com/yourusername/schwab-ai-trading/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/schwab-ai-trading/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/schwab-ai-trading/discussions)
- **Discord**: [Join our Discord](https://discord.gg/your-discord-link)

## 🙏 Acknowledgments

- **Schwab Developer Platform** for API access
- **BiConNet Research** for the hybrid neural network architecture
- **TA-Lib** for technical analysis indicators
- **FastAPI** for the modern web framework
- **PyTorch** for deep learning capabilities

---

**Built with ❤️ for algorithmic traders**

*Happy Trading! 🚀*