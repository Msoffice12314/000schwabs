#!/bin/bash
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
