@echo off
echo Starting Schwab AI Trading System...

REM Activate virtual environment
call venv\Scripts\activate

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
