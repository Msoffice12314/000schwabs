@echo off
echo 🔧 Fixing Git Issues for Schwab AI Trading System...
echo.

REM Step 1: Close Visual Studio if running
echo 📝 Step 1: Please close Visual Studio if it's running, then press any key...
pause >nul

REM Step 2: Remove .vs directory (Visual Studio cache)
echo 🗑️ Step 2: Removing Visual Studio cache directory...
if exist ".vs" (
    attrib -h -s -r ".vs" /s /d
    rmdir /s /q ".vs"
    echo    ✅ Removed .vs directory
) else (
    echo    ℹ️ .vs directory not found
)

REM Step 3: Update .gitignore to exclude Visual Studio files
echo 📄 Step 3: Updating .gitignore...
echo. >> .gitignore
echo # Visual Studio >> .gitignore
echo .vs/ >> .gitignore
echo *.user >> .gitignore
echo *.suo >> .gitignore
echo *.sln.docstates >> .gitignore
echo bin/ >> .gitignore
echo obj/ >> .gitignore
echo    ✅ Updated .gitignore

REM Step 4: Check current branch and rename if needed
echo 🌿 Step 4: Checking Git branch...
git branch
for /f "tokens=2" %%i in ('git branch 2^>nul ^| find "*"') do set current_branch=%%i
echo    Current branch: %current_branch%

if "%current_branch%"=="master" (
    echo    🔄 Renaming master to main...
    git branch -m master main
    echo    ✅ Renamed branch to main
)

REM Step 5: Remove existing remote and add correct one
echo 🌐 Step 5: Fixing Git remote...
git remote remove origin 2>nul
echo    Enter your GitHub repository URL (e.g., https://github.com/username/repo.git):
set /p repo_url="> "
git remote add origin %repo_url%
echo    ✅ Added remote origin: %repo_url%

REM Step 6: Add files to Git
echo 📦 Step 6: Adding files to Git...
git add .gitignore
git add .
echo    ✅ Files added to Git

REM Step 7: Create initial commit
echo 💾 Step 7: Creating initial commit...
git commit -m "Initial commit: Schwab AI Trading System

- Core configuration management with secure credentials
- Schwab API integration with OAuth2 authentication  
- BiConNet CNN-BiLSTM hybrid neural network for price prediction
- Advanced trading strategy engine with multiple modes
- Modern dark theme web interface with real-time updates
- Comprehensive logging and monitoring system
- Complete project structure with automated setup
- Risk management and portfolio optimization
- Backtesting engine for strategy validation
- WebSocket integration for live market data"

if %errorlevel% equ 0 (
    echo    ✅ Initial commit created successfully
) else (
    echo    ❌ Commit failed, checking status...
    git status
    pause
    exit /b 1
)

REM Step 8: Push to remote repository
echo 🚀 Step 8: Pushing to remote repository...
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ✅ SUCCESS! Repository uploaded successfully!
    echo.
    echo 🎉 Your Schwab AI Trading System is now on GitHub!
    echo 📍 Repository URL: %repo_url%
    echo.
) else (
    echo.
    echo ❌ Push failed. This might be because:
    echo    1. Repository doesn't exist on GitHub
    echo    2. Authentication issues
    echo    3. Network connectivity problems
    echo.
    echo 🔧 To fix:
    echo    1. Make sure the repository exists on GitHub
    echo    2. Configure Git credentials: git config --global user.name "Your Name"
    echo    3. Configure Git email: git config --global user.email "your.email@example.com"
    echo    4. Try: git push -u origin main
    echo.
)

echo.
echo 📋 Next Steps:
echo    1. Configure your .env file with Schwab API credentials
echo    2. Set up Python virtual environment: python -m venv venv
echo    3. Install dependencies: pip install -r requirements.txt
echo    4. Start the application: python main.py --web
echo.
pause
