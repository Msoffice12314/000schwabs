from flask import Flask, request, jsonify, session, g
from flask_restful import Api, Resource, reqparse
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Optional, Any
from functools import wraps
import jwt
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

# Import system components
from utils.database import get_database_manager
from utils.cache_manager import get_cache_manager
from trading.portfolio_manager import PortfolioManager
from trading.performance_tracker import PerformanceTracker
from ai.market_predictor import MarketPredictor
from schwab_api.client import SchwabClient
from monitoring.system_monitor import SystemMonitor
from monitoring.alert_manager import AlertManager
from web_app.auth import authenticate_user, require_auth, get_current_user

app = Flask(__name__)
api = Api(app)

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1000 per hour"]
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global components (initialized in main)
db_manager = None
cache_manager = None
portfolio_manager = None
performance_tracker = None
market_predictor = None
schwab_client = None
system_monitor = None
alert_manager = None

def init_components():
    """Initialize system components"""
    global db_manager, cache_manager, portfolio_manager, performance_tracker
    global market_predictor, schwab_client, system_monitor, alert_manager
    
    try:
        db_manager = get_database_manager()
        cache_manager = get_cache_manager()
        # Initialize other components as needed
        logger.info("API components initialized")
    except Exception as e:
        logger.error(f"Error initializing components: {e}")

def handle_api_error(func):
    """Decorator for handling API errors"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"API error in {func.__name__}: {e}")
            return {'error': 'Internal server error', 'message': str(e)}, 500
    return wrapper

class AuthResource(Resource):
    """Authentication endpoints"""
    
    @limiter.limit("10 per minute")
    def post(self):
        """User login"""
        parser = reqparse.RequestParser()
        parser.add_argument('username', required=True, help='Username is required')
        parser.add_argument('password', required=True, help='Password is required')
        args = parser.parse_args()
        
        try:
            user = authenticate_user(args['username'], args['password'])
            if user:
                # Generate JWT token
                token = jwt.encode({
                    'user_id': user.id,
                    'username': user.username,
                    'exp': datetime.utcnow() + timedelta(hours=24)
                }, app.config.get('SECRET_KEY', 'dev-secret'), algorithm='HS256')
                
                return {
                    'success': True,
                    'token': token,
                    'user': user.to_dict()
                }
            else:
                return {'error': 'Invalid credentials'}, 401
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            return {'error': 'Authentication failed'}, 500

class UserResource(Resource):
    """User management endpoints"""
    
    @require_auth
    @handle_api_error
    def get(self, user_id=None):
        """Get user information"""
        current_user = get_current_user()
        
        if user_id and user_id != current_user.id and not current_user.is_admin:
            return {'error': 'Access denied'}, 403
        
        target_user = current_user if not user_id else None
        if user_id and current_user.is_admin:
            with db_manager.get_session() as session:
                target_user = session.query(User).filter(User.id == user_id).first()
        
        if not target_user:
            return {'error': 'User not found'}, 404
        
        return {'user': target_user.to_dict()}
    
    @require_auth
    @handle_api_error
    def put(self, user_id=None):
        """Update user information"""
        current_user = get_current_user()
        
        if user_id and user_id != current_user.id and not current_user.is_admin:
            return {'error': 'Access denied'}, 403
        
        parser = reqparse.RequestParser()
        parser.add_argument('first_name', type=str)
        parser.add_argument('last_name', type=str)
        parser.add_argument('email', type=str)
        parser.add_argument('settings', type=dict)
        args = parser.parse_args()
        
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.id == user_id or current_user.id).first()
            
            if not user:
                return {'error': 'User not found'}, 404
            
            # Update fields
            for field, value in args.items():
                if value is not None:
                    setattr(user, field, value)
            
            session.commit()
            
            return {'success': True, 'user': user.to_dict()}

class PortfolioResource(Resource):
    """Portfolio management endpoints"""
    
    @require_auth
    @handle_api_error
    def get(self, portfolio_id=None):
        """Get portfolio(s)"""
        current_user = get_current_user()
        
        with db_manager.get_session() as session:
            if portfolio_id:
                portfolio = session.query(Portfolio).filter(
                    Portfolio.id == portfolio_id,
                    Portfolio.user_id == current_user.id
                ).first()
                
                if not portfolio:
                    return {'error': 'Portfolio not found'}, 404
                
                # Get positions
                positions = session.query(Position).filter(
                    Position.portfolio_id == portfolio_id,
                    Position.is_active == True
                ).all()
                
                return {
                    'portfolio': portfolio.to_dict(),
                    'positions': [pos.to_dict() for pos in positions]
                }
            else:
                portfolios = session.query(Portfolio).filter(
                    Portfolio.user_id == current_user.id,
                    Portfolio.is_active == True
                ).all()
                
                return {'portfolios': [p.to_dict() for p in portfolios]}
    
    @require_auth
    @handle_api_error
    def post(self):
        """Create new portfolio"""
        current_user = get_current_user()
        
        parser = reqparse.RequestParser()
        parser.add_argument('name', required=True, help='Portfolio name is required')
        parser.add_argument('description', type=str, default='')
        parser.add_argument('initial_cash', type=float, default=100000.0)
        parser.add_argument('is_paper_trading', type=bool, default=True)
        parser.add_argument('risk_tolerance', type=str, default='moderate')
        args = parser.parse_args()
        
        with db_manager.get_session() as session:
            portfolio = Portfolio(
                user_id=current_user.id,
                name=args['name'],
                description=args['description'],
                initial_cash=args['initial_cash'],
                current_cash=args['initial_cash'],
                total_value=args['initial_cash'],
                is_paper_trading=args['is_paper_trading'],
                risk_tolerance=args['risk_tolerance']
            )
            
            session.add(portfolio)
            session.commit()
            
            return {'success': True, 'portfolio': portfolio.to_dict()}, 201
    
    @require_auth
    @handle_api_error
    def put(self, portfolio_id):
        """Update portfolio"""
        current_user = get_current_user()
        
        parser = reqparse.RequestParser()
        parser.add_argument('name', type=str)
        parser.add_argument('description', type=str)
        parser.add_argument('risk_tolerance', type=str)
        parser.add_argument('max_position_size', type=float)
        parser.add_argument('stop_loss_percent', type=float)
        args = parser.parse_args()
        
        with db_manager.get_session() as session:
            portfolio = session.query(Portfolio).filter(
                Portfolio.id == portfolio_id,
                Portfolio.user_id == current_user.id
            ).first()
            
            if not portfolio:
                return {'error': 'Portfolio not found'}, 404
            
            # Update fields
            for field, value in args.items():
                if value is not None:
                    setattr(portfolio, field, value)
            
            session.commit()
            
            return {'success': True, 'portfolio': portfolio.to_dict()}

class TradeResource(Resource):
    """Trading endpoints"""
    
    @require_auth
    @handle_api_error
    def get(self, portfolio_id=None):
        """Get trade history"""
        current_user = get_current_user()
        
        parser = reqparse.RequestParser()
        parser.add_argument('limit', type=int, default=100)
        parser.add_argument('offset', type=int, default=0)
        parser.add_argument('symbol', type=str)
        parser.add_argument('start_date', type=str)
        parser.add_argument('end_date', type=str)
        args = parser.parse_args()
        
        with db_manager.get_session() as session:
            query = session.query(Trade).filter(Trade.user_id == current_user.id)
            
            if portfolio_id:
                query = query.filter(Trade.portfolio_id == portfolio_id)
            
            if args['symbol']:
                symbol = session.query(Symbol).filter(Symbol.symbol == args['symbol'].upper()).first()
                if symbol:
                    query = query.filter(Trade.symbol_id == symbol.id)
            
            if args['start_date']:
                start_date = datetime.fromisoformat(args['start_date'])
                query = query.filter(Trade.execution_time >= start_date)
            
            if args['end_date']:
                end_date = datetime.fromisoformat(args['end_date'])
                query = query.filter(Trade.execution_time <= end_date)
            
            trades = query.order_by(Trade.execution_time.desc()).offset(args['offset']).limit(args['limit']).all()
            
            return {'trades': [trade.to_dict() for trade in trades]}
    
    @require_auth
    @handle_api_error
    def post(self):
        """Execute trade"""
        current_user = get_current_user()
        
        parser = reqparse.RequestParser()
        parser.add_argument('portfolio_id', type=int, required=True)
        parser.add_argument('symbol', type=str, required=True)
        parser.add_argument('trade_type', type=str, required=True, choices=['BUY', 'SELL'])
        parser.add_argument('quantity', type=int, required=True)
        parser.add_argument('order_type', type=str, default='MARKET', choices=['MARKET', 'LIMIT'])
        parser.add_argument('limit_price', type=float)
        parser.add_argument('strategy_name', type=str)
        args = parser.parse_args()
        
        # Validate portfolio ownership
        with db_manager.get_session() as session:
            portfolio = session.query(Portfolio).filter(
                Portfolio.id == args['portfolio_id'],
                Portfolio.user_id == current_user.id
            ).first()
            
            if not portfolio:
                return {'error': 'Portfolio not found'}, 404
            
            # Get or create symbol
            symbol = session.query(Symbol).filter(Symbol.symbol == args['symbol'].upper()).first()
            if not symbol:
                symbol = Symbol(symbol=args['symbol'].upper(), is_active=True)
                session.add(symbol)
                session.commit()
        
        try:
            # Execute trade through portfolio manager
            if portfolio_manager:
                trade_result = portfolio_manager.execute_trade(
                    portfolio_id=args['portfolio_id'],
                    symbol=args['symbol'],
                    side=args['trade_type'],
                    quantity=args['quantity'],
                    order_type=args['order_type'],
                    limit_price=args.get('limit_price'),
                    strategy=args.get('strategy_name')
                )
                
                if trade_result['success']:
                    return {'success': True, 'trade': trade_result['trade']}, 201
                else:
                    return {'error': trade_result['error']}, 400
            else:
                return {'error': 'Trading system not available'}, 503
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {'error': 'Trade execution failed'}, 500

class MarketDataResource(Resource):
    """Market data endpoints"""
    
    @require_auth
    @handle_api_error
    @limiter.limit("100 per minute")
    def get(self, symbol=None):
        """Get market data"""
        parser = reqparse.RequestParser()
        parser.add_argument('timeframe', type=str, default='1d')
        parser.add_argument('limit', type=int, default=100)
        parser.add_argument('start_date', type=str)
        parser.add_argument('end_date', type=str)
        args = parser.parse_args()
        
        if not symbol:
            return {'error': 'Symbol is required'}, 400
        
        # Check cache first
        cache_key = f"market_data:{symbol}:{args['timeframe']}:{args['limit']}"
        cached_data = cache_manager.get(cache_key) if cache_manager else None
        
        if cached_data:
            return cached_data
        
        with db_manager.get_session() as session:
            symbol_obj = session.query(Symbol).filter(Symbol.symbol == symbol.upper()).first()
            
            if not symbol_obj:
                return {'error': 'Symbol not found'}, 404
            
            query = session.query(MarketData).filter(
                MarketData.symbol_id == symbol_obj.id,
                MarketData.timeframe == args['timeframe']
            )
            
            if args['start_date']:
                start_date = datetime.fromisoformat(args['start_date'])
                query = query.filter(MarketData.timestamp >= start_date)
            
            if args['end_date']:
                end_date = datetime.fromisoformat(args['end_date'])
                query = query.filter(MarketData.timestamp <= end_date)
            
            market_data = query.order_by(MarketData.timestamp.desc()).limit(args['limit']).all()
            
            result = {
                'symbol': symbol.upper(),
                'timeframe': args['timeframe'],
                'data': [
                    {
                        'timestamp': md.timestamp.isoformat(),
                        'open': md.open_price,
                        'high': md.high_price,
                        'low': md.low_price,
                        'close': md.close_price,
                        'volume': md.volume,
                        'vwap': md.vwap
                    }
                    for md in reversed(market_data)
                ]
            }
            
            # Cache for 1 minute
            if cache_manager:
                cache_manager.set(cache_key, result, ttl=60)
            
            return result

class PredictionResource(Resource):
    """AI prediction endpoints"""
    
    @require_auth
    @handle_api_error
    def get(self, symbol=None):
        """Get AI predictions"""
        parser = reqparse.RequestParser()
        parser.add_argument('model_name', type=str)
        parser.add_argument('prediction_type', type=str, default='price')
        parser.add_argument('horizon', type=str, default='1d')
        parser.add_argument('limit', type=int, default=10)
        args = parser.parse_args()
        
        if not symbol:
            return {'error': 'Symbol is required'}, 400
        
        with db_manager.get_session() as session:
            symbol_obj = session.query(Symbol).filter(Symbol.symbol == symbol.upper()).first()
            
            if not symbol_obj:
                return {'error': 'Symbol not found'}, 404
            
            query = session.query(AIPrediction).filter(
                AIPrediction.symbol_id == symbol_obj.id,
                AIPrediction.is_active == True
            )
            
            if args['model_name']:
                query = query.filter(AIPrediction.model_name == args['model_name'])
            
            if args['prediction_type']:
                query = query.filter(AIPrediction.prediction_type == args['prediction_type'])
            
            if args['horizon']:
                query = query.filter(AIPrediction.prediction_horizon == args['horizon'])
            
            predictions = query.order_by(AIPrediction.prediction_date.desc()).limit(args['limit']).all()
            
            return {
                'symbol': symbol.upper(),
                'predictions': [
                    {
                        'id': pred.id,
                        'model_name': pred.model_name,
                        'prediction_type': pred.prediction_type,
                        'horizon': pred.prediction_horizon,
                        'predicted_value': pred.predicted_value,
                        'confidence': pred.confidence_score,
                        'probability_up': pred.probability_up,
                        'probability_down': pred.probability_down,
                        'target_price': pred.target_price,
                        'support_level': pred.support_level,
                        'resistance_level': pred.resistance_level,
                        'prediction_date': pred.prediction_date.isoformat(),
                        'expiry_date': pred.expiry_date.isoformat() if pred.expiry_date else None
                    }
                    for pred in predictions
                ]
            }
    
    @require_auth
    @handle_api_error
    def post(self):
        """Generate new prediction"""
        parser = reqparse.RequestParser()
        parser.add_argument('symbol', type=str, required=True)
        parser.add_argument('model_name', type=str, default='default')
        parser.add_argument('prediction_type', type=str, default='price')
        parser.add_argument('horizon', type=str, default='1d')
        args = parser.parse_args()
        
        try:
            if market_predictor:
                prediction = market_predictor.predict(
                    symbol=args['symbol'],
                    model_name=args['model_name'],
                    prediction_type=args['prediction_type'],
                    horizon=args['horizon']
                )
                
                return {'success': True, 'prediction': prediction}, 201
            else:
                return {'error': 'Prediction system not available'}, 503
                
        except Exception as e:
            logger.error(f"Prediction generation error: {e}")
            return {'error': 'Prediction generation failed'}, 500

class PerformanceResource(Resource):
    """Performance analytics endpoints"""
    
    @require_auth
    @handle_api_error
    def get(self, portfolio_id):
        """Get portfolio performance metrics"""
        current_user = get_current_user()
        
        parser = reqparse.RequestParser()
        parser.add_argument('period_days', type=int, default=30)
        parser.add_argument('include_benchmark', type=bool, default=True)
        args = parser.parse_args()
        
        # Validate portfolio ownership
        with db_manager.get_session() as session:
            portfolio = session.query(Portfolio).filter(
                Portfolio.id == portfolio_id,
                Portfolio.user_id == current_user.id
            ).first()
            
            if not portfolio:
                return {'error': 'Portfolio not found'}, 404
        
        try:
            if performance_tracker:
                metrics = performance_tracker.get_performance_metrics(args['period_days'])
                
                return {
                    'portfolio_id': portfolio_id,
                    'period_days': args['period_days'],
                    'metrics': metrics
                }
            else:
                return {'error': 'Performance tracking not available'}, 503
                
        except Exception as e:
            logger.error(f"Performance calculation error: {e}")
            return {'error': 'Performance calculation failed'}, 500

class SystemStatusResource(Resource):
    """System status and monitoring endpoints"""
    
    @require_auth
    @handle_api_error
    def get(self):
        """Get system status"""
        current_user = get_current_user()
        
        if not current_user.is_admin:
            return {'error': 'Admin access required'}, 403
        
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'system_health': 'healthy',
                'components': {}
            }
            
            if system_monitor:
                system_status = system_monitor.get_performance_summary()
                status['system_metrics'] = system_status
                status['system_health'] = system_status.get('health_status', 'unknown')
            
            if alert_manager:
                active_alerts = alert_manager.get_active_alerts()
                status['active_alerts'] = len(active_alerts)
                status['alert_summary'] = alert_manager.get_statistics()
            
            # Component status
            status['components'] = {
                'database': 'healthy' if db_manager else 'unavailable',
                'cache': 'healthy' if cache_manager else 'unavailable',
                'portfolio_manager': 'healthy' if portfolio_manager else 'unavailable',
                'market_predictor': 'healthy' if market_predictor else 'unavailable',
                'schwab_client': 'healthy' if schwab_client else 'unavailable'
            }
            
            return status
            
        except Exception as e:
            logger.error(f"System status error: {e}")
            return {'error': 'System status unavailable'}, 500

class AlertResource(Resource):
    """Alert management endpoints"""
    
    @require_auth
    @handle_api_error
    def get(self):
        """Get alerts"""
        current_user = get_current_user()
        
        parser = reqparse.RequestParser()
        parser.add_argument('status', type=str, choices=['active', 'acknowledged', 'resolved'])
        parser.add_argument('severity', type=str, choices=['low', 'medium', 'high', 'critical'])
        parser.add_argument('limit', type=int, default=50)
        args = parser.parse_args()
        
        with db_manager.get_session() as session:
            query = session.query(Alert).filter(Alert.user_id == current_user.id)
            
            if args['status']:
                query = query.filter(Alert.status == args['status'])
            
            if args['severity']:
                query = query.filter(Alert.severity == args['severity'])
            
            alerts = query.order_by(Alert.triggered_at.desc()).limit(args['limit']).all()
            
            return {
                'alerts': [
                    {
                        'id': alert.id,
                        'title': alert.title,
                        'message': alert.message,
                        'severity': alert.severity,
                        'category': alert.alert_type,
                        'status': 'read' if alert.is_read else 'unread',
                        'triggered_at': alert.triggered_at.isoformat(),
                        'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None
                    }
                    for alert in alerts
                ]
            }
    
    @require_auth
    @handle_api_error
    def put(self, alert_id):
        """Update alert (acknowledge/resolve)"""
        current_user = get_current_user()
        
        parser = reqparse.RequestParser()
        parser.add_argument('action', type=str, required=True, choices=['acknowledge', 'resolve', 'mark_read'])
        args = parser.parse_args()
        
        with db_manager.get_session() as session:
            alert = session.query(Alert).filter(
                Alert.id == alert_id,
                Alert.user_id == current_user.id
            ).first()
            
            if not alert:
                return {'error': 'Alert not found'}, 404
            
            if args['action'] == 'acknowledge':
                alert.acknowledged_at = datetime.now()
            elif args['action'] == 'resolve':
                alert.is_active = False
            elif args['action'] == 'mark_read':
                alert.is_read = True
            
            session.commit()
            
            return {'success': True, 'alert_id': alert_id}

# Register API resources
api.add_resource(AuthResource, '/api/auth/login')
api.add_resource(UserResource, '/api/user', '/api/user/<int:user_id>')
api.add_resource(PortfolioResource, '/api/portfolio', '/api/portfolio/<int:portfolio_id>')
api.add_resource(TradeResource, '/api/trades', '/api/trades/<int:portfolio_id>')
api.add_resource(MarketDataResource, '/api/market-data', '/api/market-data/<string:symbol>')
api.add_resource(PredictionResource, '/api/predictions', '/api/predictions/<string:symbol>')
api.add_resource(PerformanceResource, '/api/performance/<int:portfolio_id>')
api.add_resource(SystemStatusResource, '/api/system/status')
api.add_resource(AlertResource, '/api/alerts', '/api/alerts/<int:alert_id>')

# Health check endpoint
@app.route('/api/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(429)
def rate_limit_handler(e):
    return jsonify({'error': 'Rate limit exceeded', 'message': str(e)}), 429

# Middleware
@app.before_request
def before_request():
    """Pre-request middleware"""
    g.start_time = datetime.now()

@app.after_request
def after_request(response):
    """Post-request middleware"""
    if hasattr(g, 'start_time'):
        duration = (datetime.now() - g.start_time).total_seconds()
        response.headers['X-Response-Time'] = f"{duration:.3f}s"
    
    response.headers['X-API-Version'] = '1.0.0'
    return response

if __name__ == '__main__':
    # Initialize components
    init_components()
    
    # Configure Flask app
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key')
    
    # Run development server
    app.run(debug=True, host='0.0.0.0', port=5000)
