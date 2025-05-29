# 🚀 Schwab AI Trading System

<div align="center">

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Schwab API](https://img.shields.io/badge/API-Schwab%20Official-green.svg)](https://developer.schwab.com/)
[![AI Powered](https://img.shields.io/badge/AI-BiConNet%20Neural%20Networks-red.svg)](#ai-models)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange.svg)](#development-status)

**Advanced AI-powered algorithmic trading system with BiConNet neural networks, real-time market analysis, and institutional-grade risk management.**

[Features](#-features) • [Installation](#-installation) • [Architecture](#-system-architecture) • [Development Status](#-development-status) • [Contributing](#-contributing)

</div>

---

## 🎯 **Project Overview**

The Schwab AI Trading System is an advanced algorithmic trading platform that integrates artificial intelligence with professional-grade risk management and execution capabilities. Built specifically for the Charles Schwab API, it provides sophisticated trading automation with comprehensive backtesting, real-time monitoring, and adaptive AI models.

### **🧠 Core AI Architecture**
- **BiConNet Neural Networks**: Hybrid CNN-BiLSTM architecture for market prediction
- **Time Delay Embedding**: Enhanced temporal pattern recognition  
- **Market Regime Detection**: Automatic adaptation to market conditions
- **Ensemble Methods**: Multiple model consensus for robust predictions
- **Self-Evolving Datasets**: Adaptive data management with drift detection

### **⚡ Advanced Trading Infrastructure**
- **Professional Backtesting Engine**: Comprehensive strategy validation with advanced metrics
- **Real-time Performance Tracking**: Live portfolio monitoring and attribution analysis
- **Smart Order Management**: Advanced execution algorithms and risk controls
- **Multi-level Caching**: Redis-based caching for optimal performance
- **Comprehensive Monitoring**: System health tracking and intelligent alerting

---

## 🌟 **Implemented Features**

### ✅ **Completed Components**

<table>
<tr>
<td width="50%">

**🔬 Backtesting System**
- Advanced backtesting engine with portfolio simulation
- Comprehensive performance metrics calculation
- Professional PDF/HTML report generation
- Risk-adjusted returns analysis
- Drawdown and recovery analytics
- Trade-by-trade performance attribution

**📊 Performance Analytics**
- Real-time portfolio performance tracking
- Multi-timeframe return calculations
- Risk metrics (Sharpe, Sortino, Calmar ratios)
- Benchmark comparison and alpha/beta analysis
- Rolling performance windows
- Attribution analysis by position/strategy

</td>
<td width="50%">

**🛡️ Risk Management**
- Dynamic position sizing with Kelly Criterion
- Portfolio-level risk controls and limits
- Real-time drawdown monitoring
- Correlation-based position limits
- Value-at-Risk (VaR) calculations
- Comprehensive risk reporting

**🔧 Infrastructure**
- Multi-level caching system (Redis/Memory)
- Advanced database management (PostgreSQL/SQLite)
- Real-time WebSocket streaming
- Comprehensive system monitoring
- Intelligent alert management
- Professional logging and debugging

</td>
</tr>
</table>

**🌐 Web Interface & API**
- RESTful API with comprehensive endpoints
- Real-time WebSocket communication
- JWT-based authentication system
- Rate limiting and security controls
- Multi-user support with role-based access
- Professional session management

**📈 Data Management**
- Self-evolving dataset with drift detection
- Multi-source market data integration
- Schwab API streaming client
- Intelligent data validation and cleaning
- Feature engineering and selection
- Historical data management

---

## 🏗️ **System Architecture**

```
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Web Interface │ │ Trading Engine  │ │   AI Models     │
│   [In Progress] │ │                 │ │                 │
│ • Dashboard     │◄┤ • Signal Gen.   │◄┤ • BiConNet      │
│ • Portfolio     │ │ • Risk Mgmt     │ │ • Regime Det.   │
│ • Analysis      │ │ • Execution     │ │ • Predictors    │
│ • Settings      │ │ • Performance   │ │ • Ensembles     │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Data Layer    │ │   Schwab API    │ │   Monitoring    │
│       ✅        │ │       ✅        │ │       ✅        │
│ • PostgreSQL    │ │ • Market Data   │ │ • System Health │
│ • Redis Cache   │ │ • Streaming     │ │ • Performance   │
│ • Time Series   │ │ • Authentication│ │ • Alerts        │
│ • File Storage  │ │ • Rate Limiting │ │ • Notifications │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## 📁 **Project Structure**

```
schwab_trading_system/
├── 📊 backtesting/              # Backtesting System ✅
│   ├── backtest_engine.py       # Advanced backtesting engine
│   ├── metrics_calculator.py    # Performance metrics
│   └── report_generator.py      # PDF/HTML reports
├── 🧠 ai/                       # AI Models [Existing]
│   ├── biconnet_model.py        # BiConNet neural network
│   ├── market_predictor.py      # Prediction engine
│   └── regime_detector.py       # Market regime detection
├── 📈 data/                     # Data Management ✅
│   └── self_evolving_dataset.py # Adaptive dataset management
├── 🗄️ database/                 # Database Models ✅
│   └── models.py                # SQLAlchemy models
├── 📡 schwab_api/               # Schwab Integration ✅
│   └── streaming_client.py      # WebSocket streaming
├── 💼 trading/                  # Trading Engine ✅
│   └── performance_tracker.py   # Real-time performance
├── 🛠️ utils/                    # Utilities ✅
│   ├── cache_manager.py         # Redis caching system
│   ├── database.py              # Database management
│   ├── notification.py          # Alert notifications
│   └── helpers.py               # Common utilities
├── 📊 monitoring/               # System Monitoring ✅
│   ├── system_monitor.py        # Health monitoring
│   └── alert_manager.py         # Alert management
├── 🌐 web_app/                  # Web Interface ✅
│   ├── api_routes.py            # REST API endpoints
│   ├── websocket_handler.py     # WebSocket server
│   └── auth.py                  # Authentication
├── 🎨 static/                   # Frontend Assets [Pending]
│   ├── js/                      # JavaScript files
│   └── css/                     # Stylesheets
├── 📄 templates/                # HTML Templates [Pending]
├── 📋 requirements.txt          # Python dependencies
├── 🐳 docker-compose.yml        # Docker configuration [Pending]
├── ⚙️ .env.example              # Environment template [Pending]
└── 🚀 main.py                   # Application entry point [Existing]
```

---

## 🚀 **Installation & Setup**

### **System Requirements**
- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.13+ with pip
- **Database**: PostgreSQL (recommended) or SQLite
- **Cache**: Redis (optional but recommended)
- **API**: Charles Schwab Developer Account

### **Quick Start**

```bash
# 1. Clone the repository
git clone https://github.com/Msoffice12314/000schwabs.git
cd 000schwabs

# 2. Set up Python environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your configuration

# 5. Initialize database
python -c "from utils.database import get_database_manager; get_database_manager()"

# 6. Start the application
python main.py
```

### **Environment Configuration**

Create a `.env` file with your configuration:

```bash
# Schwab API Configuration
SCHWAB_CLIENT_ID=your_client_id
SCHWAB_CLIENT_SECRET=your_client_secret
SCHWAB_REDIRECT_URI=https://127.0.0.1:8182

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/schwab_trading
# Or for SQLite: sqlite:///trading_system.db

# Redis Configuration (Optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=optional_password

# Security
SECRET_KEY=your_secret_key_32_chars_minimum
JWT_SECRET_KEY=your_jwt_secret_key

# Trading Parameters
MAX_PORTFOLIO_RISK=0.02
MAX_POSITION_SIZE=0.10
DEFAULT_STOP_LOSS=0.02

# Notification Settings
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
NOTIFICATION_EMAILS=alerts@yourdomain.com

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading_system.log
```

---

## 💻 **Usage Examples**

### **Backtesting System**

```python
from backtesting.backtest_engine import BacktestEngine, BacktestConfig
from datetime import datetime

# Configure backtesting
config = BacktestConfig(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    initial_capital=100000.0,
    commission_rate=0.001,
    max_positions=10
)

# Run backtest
engine = BacktestEngine(config)
await engine.load_market_data(['AAPL', 'MSFT', 'GOOGL'])

# Your strategy signals (DataFrame with buy/sell signals)
strategy_signals = generate_your_signals()
results = engine.run_backtest(strategy_signals)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
```

### **Performance Tracking**

```python
from trading.performance_tracker import PerformanceTracker

# Initialize performance tracker
tracker = PerformanceTracker(initial_capital=100000.0)

# Record trades
tracker.record_trade('AAPL', 'BUY', 100, 150.00, commission=1.00)
tracker.record_trade('MSFT', 'BUY', 50, 300.00, commission=1.00)

# Update portfolio with current positions
positions = {
    'AAPL': {'quantity': 100, 'current_price': 155.00, 'unrealized_pnl': 500.00},
    'MSFT': {'quantity': 50, 'current_price': 310.00, 'unrealized_pnl': 500.00}
}
tracker.update_portfolio_value(positions, cash=85000.00)

# Get performance metrics
metrics = tracker.get_performance_metrics(period_days=30)
print(f"Total Return: {metrics['returns']['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['risk_adjusted_metrics']['sharpe_ratio']:.2f}")
```

### **API Usage**

```python
import asyncio
from web_app.api_routes import app
from schwab_api.streaming_client import SchwabStreamingClient

# Start API server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# WebSocket streaming
async def stream_market_data():
    client = SchwabStreamingClient(access_token, refresh_token, account_info)
    
    await client.connect()
    await client.subscribe_quotes(['AAPL', 'MSFT', 'GOOGL'])
    
    # Handle real-time data
    def handle_quote(message):
        print(f"Quote: {message.data}")
    
    client.add_message_handler('QUOTE', handle_quote)

asyncio.run(stream_market_data())
```

---

## 🔧 **Development Status**

### ✅ **Completed (Available Now)**
- **Backtesting Engine**: Full-featured backtesting with advanced metrics
- **Performance Tracking**: Real-time portfolio performance analysis  
- **Database Management**: PostgreSQL/SQLite with comprehensive models
- **Caching System**: Multi-level Redis and memory caching
- **Streaming Client**: Schwab WebSocket integration
- **Alert System**: Comprehensive notification system
- **API Framework**: RESTful API with authentication
- **Risk Management**: Advanced risk controls and monitoring
- **System Monitoring**: Health checks and performance metrics

### 🔄 **In Progress**
- **Web Interface**: HTML templates and frontend JavaScript
- **AI Model Integration**: BiConNet model integration with backtesting
- **Order Execution**: Live trading with Schwab API
- **Configuration Files**: Docker and deployment configurations

### 📋 **Planned Features**
- **Mobile Interface**: Responsive mobile trading interface
- **Advanced Strategies**: Additional algorithmic trading strategies
- **Options Trading**: Options strategies and Greeks calculations
- **Tax Optimization**: Tax-loss harvesting and reporting
- **Cloud Deployment**: AWS/Azure deployment configurations

---

## 🧠 **AI Models & Architecture**

### **BiConNet Implementation**
The system supports the BiConNet (Bi-directional Convolutional Network) architecture:

```python
# Model configuration
BICONNET_CONFIG = {
    'sequence_length': 60,
    'cnn_filters': 64,
    'lstm_units': 50,
    'dropout_rate': 0.2,
    'attention_heads': 8,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
}
```

### **Self-Evolving Datasets**
Advanced data management with automatic adaptation:
- **Drift Detection**: Statistical tests for data distribution changes
- **Feature Evolution**: Automatic feature selection and engineering
- **Model Retraining**: Triggered by performance degradation
- **Data Quality**: Validation and cleaning pipelines

---

## 📊 **Performance Metrics**

The system calculates comprehensive performance metrics:

### **Return Metrics**
- Total Return, Annualized Return
- Risk-adjusted Returns (Sharpe, Sortino, Calmar)
- Rolling Performance Windows
- Benchmark Comparison (Alpha, Beta)

### **Risk Metrics**  
- Maximum Drawdown and Recovery
- Value-at-Risk (VaR) and Conditional VaR
- Volatility and Downside Deviation
- Correlation Analysis

### **Trading Metrics**
- Win Rate and Profit Factor
- Average Win/Loss Ratios
- Trade Frequency and Turnover
- Transaction Cost Analysis

---

## 🛡️ **Risk Management**

### **Multi-Layer Risk Controls**

```python
# Risk configuration example
RISK_CONFIG = {
    'max_portfolio_risk': 0.02,        # 2% portfolio VaR
    'max_position_size': 0.10,         # 10% maximum position
    'stop_loss_pct': 0.02,            # 2% stop loss
    'correlation_limit': 0.7,          # Maximum correlation
    'sector_limits': {
        'Technology': 0.40,            # 40% sector limit
        'Healthcare': 0.25,
        'Finance': 0.20
    },
    'daily_loss_limit': 0.05,         # 5% daily loss limit
    'max_drawdown_limit': 0.15        # 15% maximum drawdown
}
```

---

## 📡 **API Reference**

### **REST API Endpoints**

```bash
# Authentication
POST /api/auth/login              # User login
POST /api/auth/logout            # User logout

# Portfolio Management  
GET  /api/portfolio              # Portfolio summary
GET  /api/portfolio/{id}         # Specific portfolio
POST /api/portfolio              # Create portfolio
PUT  /api/portfolio/{id}         # Update portfolio

# Performance Analytics
GET  /api/performance/{portfolio_id}    # Performance metrics
GET  /api/performance/attribution       # Performance attribution

# Market Data
GET  /api/market-data/{symbol}          # Real-time quotes
GET  /api/predictions/{symbol}          # AI predictions

# Risk Management
GET  /api/risk/analysis                 # Risk assessment
GET  /api/alerts                        # Active alerts

# System Status
GET  /api/system/status                 # System health
GET  /api/system/metrics                # Performance metrics
```

### **WebSocket Streaming**

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8765');

// Subscribe to real-time data
ws.send(JSON.stringify({
    type: 'subscribe',
    topic: 'market_data',
    filters: { symbols: ['AAPL', 'MSFT'] }
}));

// Handle incoming data
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Real-time data:', data);
};
```

---

## 🔧 **Configuration**

### **Database Configuration**

```python
# Database settings
DATABASE_CONFIG = {
    'postgresql': {
        'host': 'localhost',
        'port': 5432,
        'database': 'schwab_trading',
        'pool_size': 10,
        'max_overflow': 20
    },
    'sqlite': {
        'path': 'trading_system.db',
        'pool_size': 5
    }
}
```

### **Caching Configuration**

```python
# Cache settings
CACHE_CONFIG = {
    'redis': {
        'host': 'localhost',
        'port': 6379,
        'db': 0,
        'max_connections': 20
    },
    'memory': {
        'max_size': 1000,
        'max_memory_mb': 100
    }
}
```

---

## 🤝 **Contributing**

We welcome contributions! Here's how to get started:

### **Development Setup**

```bash
# Clone and setup
git clone https://github.com/Msoffice12314/000schwabs.git
cd 000schwabs

# Create development environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v --cov

# Code formatting
black .
isort .
flake8 .
```

### **Contribution Areas**
- 🧠 **AI Models**: Improve BiConNet architecture and add new models
- 📊 **Strategies**: Implement additional trading algorithms
- 🌐 **Frontend**: Complete the web interface development
- 📱 **Mobile**: Create mobile-responsive interface
- 🧪 **Testing**: Add comprehensive test coverage
- 📚 **Documentation**: Improve documentation and tutorials

---

## ⚖️ **Legal & Compliance**

### **Important Disclaimers**

> **⚠️ INVESTMENT RISK NOTICE**
> 
> This software is for educational and research purposes. Trading involves substantial risk and is not suitable for all investors.
>
> - **No Financial Advice**: This system does not provide investment advice
> - **Use at Your Own Risk**: All trading decisions are your responsibility
> - **Past Performance**: Does not guarantee future results
> - **Regulatory Compliance**: Ensure compliance with local regulations

### **License**
Licensed under the MIT License - see [LICENSE](LICENSE) for details.

### **Trademark Notice**
Charles Schwab® is a registered trademark of The Charles Schwab Corporation. This project is not affiliated with or endorsed by Charles Schwab.

---

## 🌟 **Acknowledgments**

- **Charles Schwab & Co.** for the official API platform
- **BiConNet Research** for the neural network architecture
- **Open Source Community** for excellent libraries and tools
- **Contributors** who help improve this project

---

## 📈 **Roadmap**

### **Phase 1: Core Completion** (Current)
- [x] Backtesting system
- [x] Performance tracking
- [x] Database management
- [x] API framework
- [ ] Web interface completion
- [ ] Live trading integration

### **Phase 2: Advanced Features** (Q3 2024)
- [ ] Options trading strategies
- [ ] Advanced order types
- [ ] Tax optimization features
- [ ] Mobile interface

### **Phase 3: Enterprise Features** (Q4 2024)
- [ ] Multi-broker support
- [ ] Institutional features
- [ ] Advanced compliance tools
- [ ] Cloud deployment

---

<div align="center">

**Built with ❤️ for algorithmic traders**

[![GitHub stars](https://img.shields.io/github/stars/Msoffice12314/000schwabs?style=social)](https://github.com/Msoffice12314/000schwabs/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Msoffice12314/000schwabs?style=social)](https://github.com/Msoffice12314/000schwabs/network)

**[⭐ Star this repo](https://github.com/Msoffice12314/000schwabs) | [🍴 Fork it](https://github.com/Msoffice12314/000schwabs/fork) | [🐛 Report issues](https://github.com/Msoffice12314/000schwabs/issues)**

---

*Building the future of algorithmic trading, one commit at a time* 🚀

</div>
