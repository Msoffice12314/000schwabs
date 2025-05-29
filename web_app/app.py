"""
Schwab AI Trading System - Main Web Application
FastAPI-based web interface with modern dark theme UI
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path
import websockets
from contextlib import asynccontextmanager

from config.settings import settings
from config.credentials import schwab_credentials
from schwab_api.auth_manager import auth_manager, is_authenticated
from schwab_api.market_data import market_data_client, get_quotes_async, is_market_open
from models.signal_detector import SignalDetector
from models.market_predictor import MarketPredictor
from trading.portfolio_manager import portfolio_manager
from trading.risk_manager import risk_manager
from utils.logger import get_logger
from web_app.websocket_handler import ConnectionManager

logger = get_logger(__name__)

# Global state management
app_state = {
    'signal_detector': None,
    'market_predictor': None,
    'active_connections': {},
    'market_data_task': None,
    'prediction_task': None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Schwab AI Trading System")
    
    # Initialize AI models
    try:
        app_state['signal_detector'] = SignalDetector()
        app_state['market_predictor'] = MarketPredictor()
        logger.info("AI models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI models: {e}")
    
    # Start background tasks
    app_state['market_data_task'] = asyncio.create_task(market_data_background_task())
    app_state['prediction_task'] = asyncio.create_task(prediction_background_task())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Schwab AI Trading System")
    
    # Cancel background tasks
    if app_state['market_data_task']:
        app_state['market_data_task'].cancel()
    if app_state['prediction_task']:
        app_state['prediction_task'].cancel()

# Initialize FastAPI app
app = FastAPI(
    title="Schwab AI Trading System",
    description="AI-powered trading system with modern dark theme interface",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.web_app.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Security
security = HTTPBearer(auto_error=False)
connection_manager = ConnectionManager()

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate authentication token"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # In a production environment, validate the token properly
    # For now, just check if Schwab authentication is valid
    if not is_authenticated():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Schwab authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {"authenticated": True}

# Background tasks
async def market_data_background_task():
    """Background task for real-time market data updates"""
    while True:
        try:
            if is_market_open() and connection_manager.active_connections:
                # Get watchlist symbols from active connections
                symbols = set()
                for connection in connection_manager.active_connections:
                    if hasattr(connection, 'watchlist'):
                        symbols.update(connection.watchlist)
                
                if symbols:
                    # Get real-time quotes
                    quotes = await get_quotes_async(list(symbols))
                    
                    # Broadcast to all connected clients
                    market_data = {
                        "type": "market_data",
                        "timestamp": datetime.now().isoformat(),
                        "data": {
                            symbol: {
                                "symbol": quote.symbol,
                                "last_price": quote.last_price,
                                "net_change": quote.net_change,
                                "net_percent_change": quote.net_percent_change,
                                "volume": quote.volume,
                                "bid": quote.bid_price,
                                "ask": quote.ask_price,
                                "high": quote.high_price,
                                "low": quote.low_price,
                                "quote_time": quote.quote_time.isoformat()
                            }
                            for symbol, quote in quotes.items()
                        }
                    }
                    
                    await connection_manager.broadcast(json.dumps(market_data))
            
            await asyncio.sleep(5)  # Update every 5 seconds during market hours
            
        except Exception as e:
            logger.error(f"Error in market data background task: {e}")
            await asyncio.sleep(30)  # Wait longer on error

async def prediction_background_task():
    """Background task for AI predictions"""
    while True:
        try:
            if app_state['market_predictor'] and connection_manager.active_connections:
                # Get predictions for watchlist symbols
                symbols = set()
                for connection in connection_manager.active_connections:
                    if hasattr(connection, 'watchlist'):
                        symbols.update(connection.watchlist)
                
                predictions = {}
                for symbol in list(symbols)[:10]:  # Limit to 10 symbols for performance
                    try:
                        prediction = await app_state['market_predictor'].predict_async(symbol)
                        predictions[symbol] = prediction
                    except Exception as e:
                        logger.error(f"Prediction failed for {symbol}: {e}")
                
                if predictions:
                    prediction_data = {
                        "type": "predictions",
                        "timestamp": datetime.now().isoformat(),
                        "data": predictions
                    }
                    
                    await connection_manager.broadcast(json.dumps(prediction_data))
            
            await asyncio.sleep(300)  # Update predictions every 5 minutes
            
        except Exception as e:
            logger.error(f"Error in prediction background task: {e}")
            await asyncio.sleep(600)  # Wait longer on error

# Web Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main trading dashboard"""
    context = {
        "request": request,
        "title": "Schwab AI Trading Dashboard",
        "is_authenticated": is_authenticated(),
        "market_open": is_market_open(),
        "current_time": datetime.now().isoformat()
    }
    return templates.TemplateResponse("dashboard.html", context)

@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio(request: Request, user: dict = Depends(get_current_user)):
    """Portfolio management interface"""
    context = {
        "request": request,
        "title": "Portfolio Management",
        "portfolio_data": await get_portfolio_data()
    }
    return templates.TemplateResponse("portfolio.html", context)

@app.get("/analysis", response_class=HTMLResponse)
async def analysis(request: Request, user: dict = Depends(get_current_user)):
    """Market analysis and predictions interface"""
    context = {
        "request": request,
        "title": "Market Analysis",
        "analysis_data": await get_analysis_data()
    }
    return templates.TemplateResponse("analysis.html", context)

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, user: dict = Depends(get_current_user)):
    """Settings and configuration interface"""
    context = {
        "request": request,
        "title": "Settings",
        "current_settings": get_current_settings()
    }
    return templates.TemplateResponse("settings.html", context)

# API Routes
@app.get("/api/auth/status")
async def auth_status():
    """Check authentication status"""
    return {
        "authenticated": is_authenticated(),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/auth/login")
async def login():
    """Initiate Schwab OAuth login"""
    try:
        auth_url = auth_manager.start_oauth_flow(auto_open_browser=False)
        return {"auth_url": auth_url, "status": "success"}
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auth/logout")
async def logout(user: dict = Depends(get_current_user)):
    """Logout and revoke tokens"""
    try:
        auth_manager.revoke_tokens()
        return {"status": "success", "message": "Logged out successfully"}
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/quote/{symbol}")
async def get_quote_api(symbol: str, user: dict = Depends(get_current_user)):
    """Get real-time quote for a symbol"""
    try:
        quote = market_data_client.get_quote(symbol)
        return {
            "symbol": quote.symbol,
            "last_price": quote.last_price,
            "net_change": quote.net_change,
            "net_percent_change": quote.net_percent_change,
            "volume": quote.volume,
            "bid": quote.bid_price,
            "ask": quote.ask_price,
            "high": quote.high_price,
            "low": quote.low_price,
            "timestamp": quote.quote_time.isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/history/{symbol}")
async def get_history_api(symbol: str, days: int = 30, user: dict = Depends(get_current_user)):
    """Get historical price data"""
    try:
        df = market_data_client.get_price_history(symbol, days=days)
        
        if df.empty:
            return {"data": [], "symbol": symbol}
        
        # Convert DataFrame to list of dicts
        data = []
        for index, row in df.iterrows():
            data.append({
                "datetime": index.isoformat(),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": int(row['volume'])
            })
        
        return {"data": data, "symbol": symbol}
    except Exception as e:
        logger.error(f"Failed to get history for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/{symbol}")
async def get_predictions_api(symbol: str, user: dict = Depends(get_current_user)):
    """Get AI predictions for a symbol"""
    try:
        if not app_state['market_predictor']:
            raise HTTPException(status_code=503, detail="Prediction service not available")
        
        predictions = await app_state['market_predictor'].predict_async(symbol)
        return predictions
    except Exception as e:
        logger.error(f"Failed to get predictions for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/signals/{symbol}")
async def get_signals_api(symbol: str, user: dict = Depends(get_current_user)):
    """Get trading signals for a symbol"""
    try:
        if not app_state['signal_detector']:
            raise HTTPException(status_code=503, detail="Signal detection service not available")
        
        signals = await app_state['signal_detector'].get_signals_async(symbol)
        return signals
    except Exception as e:
        logger.error(f"Failed to get signals for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio/summary")
async def get_portfolio_summary(user: dict = Depends(get_current_user)):
    """Get portfolio summary"""
    try:
        summary = await portfolio_manager.get_portfolio_summary()
        return summary
    except Exception as e:
        logger.error(f"Failed to get portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/risk/analysis")
async def get_risk_analysis(user: dict = Depends(get_current_user)):
    """Get risk analysis"""
    try:
        analysis = await risk_manager.get_risk_analysis()
        return analysis
    except Exception as e:
        logger.error(f"Failed to get risk analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/watchlist/add")
async def add_to_watchlist(symbol: str, user: dict = Depends(get_current_user)):
    """Add symbol to watchlist"""
    # Implementation depends on user management system
    return {"status": "success", "symbol": symbol}

@app.delete("/api/watchlist/remove")
async def remove_from_watchlist(symbol: str, user: dict = Depends(get_current_user)):
    """Remove symbol from watchlist"""
    # Implementation depends on user management system
    return {"status": "success", "symbol": symbol}

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time data streaming"""
    await connection_manager.connect(websocket)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe":
                # Subscribe to symbols for real-time updates
                symbols = message.get("symbols", [])
                websocket.watchlist = symbols
                await websocket.send_text(json.dumps({
                    "type": "subscription_confirmed",
                    "symbols": symbols
                }))
            
            elif message.get("type") == "ping":
                # Health check
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
    
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)

# Helper functions
async def get_portfolio_data():
    """Get portfolio data for template"""
    try:
        return await portfolio_manager.get_portfolio_summary()
    except Exception as e:
        logger.error(f"Failed to get portfolio data: {e}")
        return {}

async def get_analysis_data():
    """Get analysis data for template"""
    try:
        # Get market overview and predictions
        data = {
            "market_overview": {},
            "top_predictions": [],
            "sector_analysis": {}
        }
        return data
    except Exception as e:
        logger.error(f"Failed to get analysis data: {e}")
        return {}

def get_current_settings():
    """Get current settings for template"""
    try:
        return {
            "trading_enabled": True,
            "risk_level": "moderate",
            "max_position_size": settings.trading.max_position_size,
            "stop_loss_percent": settings.trading.stop_loss_pct * 100,
            "take_profit_percent": settings.trading.take_profit_pct * 100
        }
    except Exception as e:
        logger.error(f"Failed to get settings: {e}")
        return {}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Application health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "auth": is_authenticated(),
            "market_data": True,
            "predictions": app_state['market_predictor'] is not None,
            "signals": app_state['signal_detector'] is not None
        }
    }

# Main application runner
def run_app():
    """Run the FastAPI application"""
    logger.info(f"Starting Schwab AI Trading System on {settings.web_app.host}:{settings.web_app.port}")
    
    uvicorn.run(
        "web_app.app:app",
        host=settings.web_app.host,
        port=settings.web_app.port,
        reload=settings.web_app.debug,
        log_level="info" if not settings.web_app.debug else "debug",
        access_log=True,
        server_header=False,
        date_header=False
    )

if __name__ == "__main__":
    run_app()