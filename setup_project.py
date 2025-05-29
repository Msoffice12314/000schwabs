#!/usr/bin/env python3
"""
Schwab AI Trading System - Automated Project Setup
Creates directory structure and generates template files
"""

import os
import sys
from pathlib import Path
import shutil
import subprocess
from typing import List, Dict

class ProjectSetup:
    """Automated setup for Schwab AI Trading System"""
    
    def __init__(self, project_path: str = "G:\\000Schwab_ai"):
        self.project_path = Path(project_path)
        self.directories = [
            "config",
            "schwab_api", 
            "data",
            "models",
            "training",
            "trading",
            "backtesting",
            "web_app",
            "templates",
            "static/css",
            "static/js", 
            "static/images/logos",
            "utils",
            "database",
            "monitoring",
            "logs",
            "data_storage/raw_data",
            "data_storage/processed_data", 
            "data_storage/models",
            "data_storage/backups"
        ]
        
        self.init_files = [
            "config/__init__.py",
            "schwab_api/__init__.py",
            "data/__init__.py", 
            "models/__init__.py",
            "training/__init__.py",
            "trading/__init__.py",
            "backtesting/__init__.py",
            "web_app/__init__.py",
            "utils/__init__.py",
            "database/__init__.py",
            "monitoring/__init__.py"
        ]

    def create_directories(self):
        """Create project directory structure"""
        print("🏗️  Creating project directories...")
        
        # Create main project directory
        self.project_path.mkdir(parents=True, exist_ok=True)
        os.chdir(self.project_path)
        
        # Create subdirectories
        for directory in self.directories:
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {directory}")
    
    def create_init_files(self):
        """Create __init__.py files for Python packages"""
        print("\n📦 Creating Python package files...")
        
        for init_file in self.init_files:
            init_path = Path(init_file)
            init_path.touch()
            print(f"   ✅ Created: {init_file}")
    
    def create_env_file(self):
        """Create environment variables template"""
        print("\n🔧 Creating environment configuration...")
        
        env_content = """# Schwab AI Trading System Environment Variables

# Schwab API Configuration
SCHWAB_CLIENT_ID=your_client_id_here
SCHWAB_CLIENT_SECRET=your_client_secret_here
SCHWAB_REDIRECT_URI=https://127.0.0.1:8182

# Security
MASTER_PASSWORD=create_a_secure_master_password
WEB_SECRET_KEY=generate_a_random_secret_key_here

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USERNAME=postgres
DB_PASSWORD=your_database_password
DB_NAME=schwab_ai

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# Trading Configuration
MAX_PORTFOLIO_RISK=0.02
MAX_POSITION_SIZE=0.1
STOP_LOSS_PCT=0.02
TAKE_PROFIT_PCT=0.04
MIN_CONFIDENCE_SCORE=0.7

# Model Configuration
MODEL_SEQUENCE_LENGTH=60
MODEL_CNN_FILTERS=64
MODEL_LSTM_UNITS=50
MODEL_DROPOUT_RATE=0.2
MODEL_LEARNING_RATE=0.001
MODEL_BATCH_SIZE=32
MODEL_EPOCHS=100

# Web Application
WEB_HOST=0.0.0.0
WEB_PORT=8000
WEB_DEBUG=False

# Logging
LOG_LEVEL=INFO
LOG_FILE_PATH=./logs/schwab_ai.log
LOG_MAX_FILE_SIZE=10485760
LOG_BACKUP_COUNT=5
LOG_ENABLE_CONSOLE=True

# Environment
ENVIRONMENT=development
"""
        
        with open(".env", "w") as f:
            f.write(env_content)
        print("   ✅ Created: .env")
    
    def create_gitignore(self):
        """Create .gitignore file"""
        print("\n🚫 Creating .gitignore...")
        
        gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.env.local
.env.production
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# Schwab AI Specific
credentials.enc
.key
localhost.pem
schwab_token.json
*.db
*.sqlite3
logs/
*.log
data_storage/raw_data/
data_storage/processed_data/
data_storage/backups/
data_storage/models/*.pth
data_storage/models/*.pkl

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Documentation
docs/_build/
"""
        
        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
        print("   ✅ Created: .gitignore")
    
    def create_dockerfile(self):
        """Create Dockerfile for containerization"""
        print("\n🐳 Creating Docker configuration...")
        
        dockerfile_content = """FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data_storage/raw_data data_storage/processed_data data_storage/models data_storage/backups

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py", "--web"]
"""
        
        with open("Dockerfile", "w") as f:
            f.write(dockerfile_content)
        print("   ✅ Created: Dockerfile")
        
        # Docker Compose
        docker_compose_content = """version: '3.8'

services:
  schwab-ai:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    volumes:
      - ./logs:/app/logs
      - ./data_storage:/app/data_storage
    depends_on:
      - postgres
      - redis
    
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: schwab_ai
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
"""
        
        with open("docker-compose.yml", "w") as f:
            f.write(docker_compose_content)
        print("   ✅ Created: docker-compose.yml")
    
    def create_scripts(self):
        """Create utility scripts"""
        print("\n📜 Creating utility scripts...")
        
        # Windows batch script
        batch_script = """@echo off
echo Starting Schwab AI Trading System...

REM Activate virtual environment
call venv\\Scripts\\activate

REM Check if authenticated
python -c "from schwab_api.auth_manager import is_authenticated; exit(0 if is_authenticated() else 1)"
if %errorlevel% neq 0 (
    echo Authentication required. Please run: python main.py --authenticate
    pause
    exit /b 1
)

REM Start the application
python main.py --web

pause
"""
        
        with open("start.bat", "w") as f:
            f.write(batch_script)
        print("   ✅ Created: start.bat")
        
        # Shell script for Linux/Mac
        shell_script = """#!/bin/bash
echo "Starting Schwab AI Trading System..."

# Activate virtual environment
source venv/bin/activate

# Check if authenticated
python -c "from schwab_api.auth_manager import is_authenticated; exit(0 if is_authenticated() else 1)"
if [ $? -ne 0 ]; then
    echo "Authentication required. Please run: python main.py --authenticate"
    exit 1
fi

# Start the application
python main.py --web
"""
        
        with open("start.sh", "w") as f:
            f.write(shell_script)
        
        # Make shell script executable
        os.chmod("start.sh", 0o755)
        print("   ✅ Created: start.sh")
    
    def create_config_templates(self):
        """Create configuration template files"""
        print("\n⚙️  Creating configuration templates...")
        
        # Model parameters template
        model_params_content = """\"\"\"
Model parameters configuration for BiConNet and other AI models
\"\"\"

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
"""
        
        with open("config/model_params.py", "w") as f:
            f.write(model_params_content)
        print("   ✅ Created: config/model_params.py")
        
        # Trading parameters template
        trading_params_content = """\"\"\"
Trading strategy parameters and configuration
\"\"\"

from config.settings import settings
from trading.strategy_engine import StrategyMode

# Strategy Mode Configurations
STRATEGY_CONFIGS = {
    StrategyMode.CONSERVATIVE: {
        'min_confidence_threshold': 0.8,
        'max_position_size': 0.05,
        'stop_loss_pct': 0.015,
        'take_profit_pct': 0.03,
        'max_daily_trades': 3,
        'max_positions': 3,
        'rebalance_frequency': 'daily'
    },
    StrategyMode.MODERATE: {
        'min_confidence_threshold': 0.7,
        'max_position_size': 0.1,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04,
        'max_daily_trades': 10,
        'max_positions': 5,
        'rebalance_frequency': 'daily'
    },
    StrategyMode.AGGRESSIVE: {
        'min_confidence_threshold': 0.6,
        'max_position_size': 0.15,
        'stop_loss_pct': 0.03,
        'take_profit_pct': 0.06,
        'max_daily_trades': 20,
        'max_positions': 8,
        'rebalance_frequency': 'daily'
    },
    StrategyMode.SCALPING: {
        'min_confidence_threshold': 0.65,
        'max_position_size': 0.08,
        'stop_loss_pct': 0.005,
        'take_profit_pct': 0.01,
        'max_daily_trades': 50,
        'max_positions': 10,
        'rebalance_frequency': 'hourly'
    }
}

# Risk Management Parameters
RISK_PARAMS = {
    'max_portfolio_risk': settings.trading.max_portfolio_risk,
    'max_sector_allocation': 0.3,
    'max_correlation_threshold': 0.7,
    'volatility_lookback': 20,
    'var_confidence_level': 0.05,
    'max_drawdown_limit': 0.15
}

# Default Watchlists by Strategy Mode
DEFAULT_WATCHLISTS = {
    StrategyMode.CONSERVATIVE: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
    StrategyMode.MODERATE: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'JPM', 'V', 'JNJ', 'PG'],
    StrategyMode.AGGRESSIVE: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'CRM', 'ADBE', 'PYPL', 'SHOP', 'SQ', 'ROKU', 'ZOOM'],
    StrategyMode.SCALPING: ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'SQQQ', 'TQQQ']
}
"""
        
        with open("config/trading_params.py", "w") as f:
            f.write(trading_params_content)
        print("   ✅ Created: config/trading_params.py")
    
    def setup_git_repository(self):
        """Initialize Git repository"""
        print("\n🔧 Setting up Git repository...")
        
        try:
            # Check if git is available
            subprocess.run(["git", "--version"], check=True, capture_output=True)
            
            # Initialize repository
            subprocess.run(["git", "init"], check=True)
            print("   ✅ Initialized Git repository")
            
            # Add all files
            subprocess.run(["git", "add", "."], check=True)
            print("   ✅ Added files to Git")
            
            # Create initial commit
            commit_message = """Initial commit: Schwab AI Trading System

- Core configuration management
- Schwab API integration with OAuth2
- BiConNet AI model implementation  
- Trading strategy engine
- Modern dark theme web interface
- Comprehensive logging system
- Complete project structure"""
            
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            print("   ✅ Created initial commit")
            
            print("\n📌 Next steps:")
            print("   1. Add your remote repository: git remote add origin <your-repo-url>")
            print("   2. Push to remote: git push -u origin main")
            
        except subprocess.CalledProcessError:
            print("   ⚠️  Git not found or error occurred. Please initialize Git manually.")
        except FileNotFoundError:
            print("   ⚠️  Git not installed. Please install Git and run setup again.")
    
    def create_readme_files(self):
        """Create additional README files for subdirectories"""
        print("\n📚 Creating documentation...")
        
        readme_files = {
            "config/README.md": "# Configuration\n\nThis directory contains all configuration files for the Schwab AI Trading System.",
            "models/README.md": "# AI Models\n\nThis directory contains AI model implementations including BiConNet neural networks.",
            "trading/README.md": "# Trading Engine\n\nThis directory contains trading strategy implementations and portfolio management.",
            "data/README.md": "# Data Management\n\nThis directory contains data collection, processing, and feature engineering modules.",
            "web_app/README.md": "# Web Application\n\nThis directory contains the FastAPI web application and user interface.",
            "utils/README.md": "# Utilities\n\nThis directory contains utility functions and helper modules."
        }
        
        for file_path, content in readme_files.items():
            with open(file_path, "w") as f:
                f.write(content)
            print(f"   ✅ Created: {file_path}")
    
    def display_summary(self):
        """Display setup summary and next steps"""
        print("\n" + "="*60)
        print("🎉 SCHWAB AI TRADING SYSTEM SETUP COMPLETE!")
        print("="*60)
        
        print("\n📁 Project Structure Created:")
        for directory in self.directories:
            print(f"   📂 {directory}")
        
        print("\n📋 Next Steps:")
        print("   1. 🔐 Edit .env file with your Schwab API credentials")
        print("   2. 🐍 Create virtual environment: python -m venv venv")
        print("   3. 📦 Activate environment: venv\\Scripts\\activate (Windows)")
        print("   4. 📚 Install dependencies: pip install -r requirements.txt")
        print("   5. 🔧 Install TA-Lib: pip install ta_lib-0.6.3-cp313-cp313-win_amd64.whl")
        print("   6. 🏢 Set up database (PostgreSQL recommended)")
        print("   7. 🔑 Authenticate: python main.py --authenticate")
        print("   8. 🚀 Start application: python main.py --web")
        
        print("\n🌐 Web Interface will be available at: http://localhost:8000")
        print("\n💡 Tip: Use 'python main.py --help' to see all available commands")
        
        print("\n" + "="*60)
    
    def run_setup(self):
        """Run the complete setup process"""
        print("🚀 Setting up Schwab AI Trading System...")
        print(f"📍 Project path: {self.project_path}")
        
        try:
            self.create_directories()
            self.create_init_files()
            self.create_env_file()
            self.create_gitignore()
            self.create_dockerfile()
            self.create_scripts()
            self.create_config_templates()
            self.create_readme_files()
            self.setup_git_repository()
            self.display_summary()
            
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            sys.exit(1)

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Schwab AI Trading System Setup")
    parser.add_argument("--path", default="G:\\000Schwab_ai", 
                       help="Project directory path")
    parser.add_argument("--no-git", action="store_true",
                       help="Skip Git repository initialization")
    
    args = parser.parse_args()
    
    setup = ProjectSetup(args.path)
    setup.run_setup()

if __name__ == "__main__":
    main()
