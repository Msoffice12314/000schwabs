# Schwab AI Trading System - Setup Guide

## Quick Setup Instructions

### 1. Create Project Structure

```bash
# Navigate to your project directory
cd G:\000Schwab_ai

# Create all necessary directories
mkdir -p config schwab_api data models training trading backtesting web_app templates static\css static\js static\images\logos utils database monitoring logs data_storage\raw_data data_storage\processed_data data_storage\models data_storage\backups
```

### 2. Copy Files to Project Structure

Copy each file from the artifacts to its corresponding location:

```
G:\000Schwab_ai\
├── main.py
├── requirements.txt
├── README.md
├── .env (create this file with your credentials)
├── config\
│   ├── __init__.py (create empty file)
│   ├── settings.py
│   ├── credentials.py
│   ├── model_params.py (create based on settings)
│   └── trading_params.py (create based on settings)
├── schwab_api\
│   ├── __init__.py (create empty file)
│   ├── auth_manager.py
│   ├── market_data.py
│   └── (other API files as needed)
├── models\
│   ├── __init__.py (create empty file)
│   └── biconnet_core.py
├── trading\
│   ├── __init__.py (create empty file)
│   └── strategy_engine.py
├── web_app\
│   ├── __init__.py (create empty file)
│   └── app.py
├── templates\
│   └── dashboard.html
├── static\
│   ├── css\
│   │   └── dark_theme.css
│   └── js\
│       └── dashboard.js
└── utils\
    ├── __init__.py (create empty file)
    └── logger.py
```

### 3. Windows Batch Script for Setup

Create `setup_project.bat`:

```batch
@echo off
echo Setting up Schwab AI Trading System...

REM Create directory structure
mkdir config schwab_api data models training trading backtesting web_app templates static\css static\js static\images\logos utils database monitoring logs data_storage\raw_data data_storage\processed_data data_storage\models data_storage\backups

REM Create __init__.py files
echo. > config\__init__.py
echo. > schwab_api\__init__.py
echo. > data\__init__.py
echo. > models\__init__.py
echo. > training\__init__.py
echo. > trading\__init__.py
echo. > backtesting\__init__.py
echo. > web_app\__init__.py
echo. > utils\__init__.py
echo. > database\__init__.py
echo. > monitoring\__init__.py

REM Create .env template
echo # Schwab AI Trading System Environment Variables > .env
echo SCHWAB_CLIENT_ID=your_client_id_here >> .env
echo SCHWAB_CLIENT_SECRET=your_client_secret_here >> .env
echo SCHWAB_REDIRECT_URI=https://127.0.0.1:8182 >> .env
echo MASTER_PASSWORD=your_secure_master_password >> .env
echo WEB_SECRET_KEY=your_secret_key_here >> .env
echo DB_PASSWORD=your_db_password >> .env
echo LOG_LEVEL=INFO >> .env
echo ENVIRONMENT=development >> .env

echo Project structure created successfully!
echo Please copy the generated files to their respective directories.
echo Then run: pip install -r requirements.txt
pause
```

### 4. Git Repository Setup

```bash
# Initialize Git repository (if not already done)
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
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
*.egg-info/
.installed.cfg
*.egg

# Environment variables
.env
.env.local
.env.production

# Credentials and keys
credentials.enc
.key
localhost.pem
schwab_token.json

# Database
*.db
*.sqlite3

# Logs
logs/
*.log

# Data storage
data_storage/raw_data/
data_storage/processed_data/
data_storage/backups/

# Models (large files)
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

# Jupyter
.ipynb_checkpoints/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Documentation
docs/_build/
EOF

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: Schwab AI Trading System

- Core configuration management
- Schwab API integration with OAuth2
- BiConNet AI model implementation
- Trading strategy engine
- Modern dark theme web interface
- Comprehensive logging system
- Complete project structure"

# Add remote repository (replace with your repo URL)
git remote add origin https://github.com/yourusername/schwab-ai-trading.git

# Push to repository
git push -u origin main
```

### 5. Python Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install TA-Lib (Windows)
pip install ta_lib-0.6.3-cp313-cp313-win_amd64.whl
```

### 6. Configuration Steps

1. **Update .env file** with your actual credentials:
   ```env
   SCHWAB_CLIENT_ID=your_actual_client_id
   SCHWAB_CLIENT_SECRET=your_actual_client_secret
   WEB_SECRET_KEY=generate_a_secure_random_key
   MASTER_PASSWORD=create_a_strong_password
   ```

2. **Set up database** (PostgreSQL recommended):
   ```bash
   # Install PostgreSQL and create database
   createdb schwab_ai
   ```

3. **Initialize the application**:
   ```bash
   # Authenticate with Schwab
   python main.py --authenticate

   # Start web interface
   python main.py --web
   ```

## Git Commands Cheat Sheet

```bash
# Check status
git status

# Add changes
git add .

# Commit changes
git commit -m "Description of changes"

# Push to remote
git push

# Pull latest changes
git pull

# Create new branch
git checkout -b feature/new-feature

# Switch branches
git checkout main

# Merge branch
git merge feature/new-feature
```

## Development Workflow

1. **Create feature branch**:
   ```bash
   git checkout -b feature/ai-improvements
   ```

2. **Make changes and test**:
   ```bash
   python main.py --train  # Train models
   python main.py --backtest 2023-01-01 2023-12-31  # Test strategy
   ```

3. **Commit and push**:
   ```bash
   git add .
   git commit -m "Improve BiConNet model accuracy"
   git push origin feature/ai-improvements
   ```

4. **Create pull request** on GitHub/GitLab

5. **Merge to main** after review

## Automated Deployment Script

Create `deploy.bat` for automated deployment:

```batch
@echo off
echo Deploying Schwab AI Trading System...

REM Pull latest changes
git pull origin main

REM Install/update dependencies
pip install -r requirements.txt

REM Run tests
python -m pytest tests/ -v

REM Start application
python main.py --web

echo Deployment completed!
pause
```

## Security Notes

- Never commit credentials to Git
- Use environment variables for sensitive data
- Keep `.env` file in `.gitignore`
- Regularly rotate API keys and passwords
- Use HTTPS for all external communications

## Next Steps

1. Copy all generated files to their proper locations
2. Run the setup script
3. Configure your credentials
4. Initialize Git repository
5. Push to your remote repository
6. Start developing and trading!

The system is designed to be modular and scalable. You can extend it by adding new AI models, trading strategies, or integrating additional data sources.
