"""
Schwab AI Trading System - Trading API Client
Official Schwab API integration for order execution, account management, and portfolio operations.
"""

import logging
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import time
import hashlib
import hmac
import base64
from urllib.parse import urlencode

from schwab_api.rate_limiter import RateLimiter
from utils.cache_manager import CacheManager
from utils.database import DatabaseManager
from config.settings import get_settings

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types supported by Schwab API"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP" 
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"

class OrderInstruction(Enum):
    """Order instructions"""
    BUY = "BUY"
    SELL = "SELL"
    BUY_TO_COVER = "BUY_TO_COVER"
    SELL_SHORT = "SELL_SHORT"

class OrderStatus(Enum):
    """Order status values"""
    PENDING = "PENDING_ACTIVATION"
    QUEUED = "QUEUED"
    WORKING = "WORKING"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class AssetType(Enum):
    """Asset types"""
    EQUITY = "EQUITY"
    OPTION = "OPTION"
    ETF = "ETF"
    MUTUAL_FUND = "MUTUAL_FUND"

@dataclass
class OrderRequest:
    """Order request structure"""
    symbol: str
    quantity: int
    instruction: OrderInstruction
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    duration: str = "DAY"
    session: str = "NORMAL"
    asset_type: AssetType = AssetType.EQUITY

@dataclass
class OrderResponse:
    """Order response from API"""
    order_id: str
    status: OrderStatus
    symbol: str
    quantity: int
    filled_quantity: int
    remaining_quantity: int
    price: Optional[float]
    filled_price: Optional[float]
    order_type: OrderType
    instruction: OrderInstruction
    created_time: datetime
    updated_time: datetime
    message: str

@dataclass
class Position:
    """Position information"""
    symbol: str
    quantity: int
    average_price: float
    market_value: float
    day_change: float
    day_change_percent: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    asset_type: AssetType

@dataclass
class AccountInfo:
    """Account information"""
    account_number: str
    account_type: str
    cash_balance: float
    buying_power: float
    market_value: float
    day_change: float
    day_change_percent: float
    total_value: float

class SchwabTradingClient:
    """
    Official Schwab API trading client with OAuth2 authentication,
    order management, and account operations.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.rate_limiter = RateLimiter()
        self.cache_manager = CacheManager()
        self.db_manager = DatabaseManager()
        
        # API configuration
        self.base_url = "https://api.schwabapi.com"
        self.client_id = self.settings.schwab_client_id
        self.client_secret = self.settings.schwab_client_secret
        self.redirect_uri = self.settings.schwab_redirect_uri
        
        # Authentication tokens
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        
        # Account info
        self.account_number = None
        self.account_hash = None
        
        # Session management
        self.session = None
        self.is_authenticated = False
        
        logger.info("SchwabTradingClient initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self) -> bool:
        """Initialize the trading client"""
        try:
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Load saved tokens
            await self._load_tokens()
            
            # Authenticate if needed
            if not self.is_authenticated:
                logger.info("Authentication required")
                return False
            
            # Get account information
            await self._get_account_info()
            
            logger.info("SchwabTradingClient initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SchwabTradingClient: {str(e)}")
            return False
    
    async def authenticate(self, auth_code: Optional[str] = None) -> bool:
        """
        Authenticate with Schwab API using OAuth2
        
        Args:
            auth_code: Authorization code from OAuth redirect
            
        Returns:
            True if authentication successful
        """
        try:
            if auth_code:
                # Exchange authorization code for tokens
                success = await self._exchange_code_for_tokens(auth_code)
                if success:
                    self.is_authenticated = True
                    await self._save_tokens()
                    await self._get_account_info()
                    return True
            else:
                # Try to refresh existing tokens
                if self.refresh_token:
                    success = await self._refresh_access_token()
                    if success:
                        self.is_authenticated = True
                        return True
                
                # Generate authorization URL
                auth_url = await self._get_authorization_url()
                logger.info(f"Please visit this URL to authorize: {auth_url}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False
    
    async def _get_authorization_url(self) -> str:
        """Generate OAuth2 authorization URL"""
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'response_type': 'code',
            'scope': 'readonly',
            'state': 'schwab_trading_system'
        }
        
        auth_url = f"https://api.schwabapi.com/oauth/authorize?{urlencode(params)}"
        return auth_url
    
    async def _exchange_code_for_tokens(self, auth_code: str) -> bool:
        """Exchange authorization code for access and refresh tokens"""
        try:
            url = f"{self.base_url}/oauth/token"
            
            data = {
                'grant_type': 'authorization_code',
                'code': auth_code,
                'redirect_uri': self.redirect_uri,
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json'
            }
            
            async with self.session.post(url, data=data, headers=headers) as response:
                if response.status == 200:
                    token_data = await response.json()
                    
                    self.access_token = token_data['access_token']
                    self.refresh_token = token_data['refresh_token']
                    expires_in = token_data.get('expires_in', 3600)
                    self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                    
                    logger.info("Successfully exchanged authorization code for tokens")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Token exchange failed: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error exchanging code for tokens: {str(e)}")
            return False
    
    async def _refresh_access_token(self) -> bool:
        """Refresh access token using refresh token"""
        try:
            if not self.refresh_token:
                return False
            
            url = f"{self.base_url}/oauth/token"
            
            data = {
                'grant_type': 'refresh_token',
                'refresh_token': self.refresh_token,
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json'
            }
            
            async with self.session.post(url, data=data, headers=headers) as response:
                if response.status == 200:
                    token_data = await response.json()
                    
                    self.access_token = token_data['access_token']
                    if 'refresh_token' in token_data:
                        self.refresh_token = token_data['refresh_token']
                    
                    expires_in = token_data.get('expires_in', 3600)
                    self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                    
                    await self._save_tokens()
                    logger.info("Successfully refreshed access token")
                    return True
                else:
                    logger.error(f"Token refresh failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error refreshing token: {str(e)}")
            return False
    
    async def _ensure_authenticated(self) -> bool:
        """Ensure we have a valid access token"""
        try:
            # Check if token is expired
            if (self.token_expires_at and 
                datetime.now() >= self.token_expires_at - timedelta(minutes=5)):
                
                if not await self._refresh_access_token():
                    logger.error("Failed to refresh expired token")
                    return False
            
            return self.access_token is not None
            
        except Exception as e:
            logger.error(f"Error ensuring authentication: {str(e)}")
            return False
    
    async def _save_tokens(self):
        """Save tokens to cache"""
        try:
            token_data = {
                'access_token': self.access_token,
                'refresh_token': self.refresh_token,
                'expires_at': self.token_expires_at.isoformat() if self.token_expires_at else None
            }
            
            await self.cache_manager.set('schwab_tokens', token_data, expire=86400)
            
        except Exception as e:
            logger.error(f"Error saving tokens: {str(e)}")
    
    async def _load_tokens(self):
        """Load tokens from cache"""
        try:
            token_data = await self.cache_manager.get('schwab_tokens')
            
            if token_data:
                self.access_token = token_data.get('access_token')
                self.refresh_token = token_data.get('refresh_token')
                
                expires_str = token_data.get('expires_at')
                if expires_str:
                    self.token_expires_at = datetime.fromisoformat(expires_str)
                
                # Check if tokens are still valid
                if (self.access_token and 
                    (not self.token_expires_at or datetime.now() < self.token_expires_at)):
                    self.is_authenticated = True
                    
        except Exception as e:
            logger.error(f"Error loading tokens: {str(e)}")
    
    async def _get_account_info(self):
        """Get account information"""
        try:
            accounts = await self.get_accounts()
            if accounts:
                # Use first account
                account = accounts[0]
                self.account_number = account['accountNumber']
                self.account_hash = account.get('hashValue', '')
                
                logger.info(f"Using account: {self.account_number}")
                
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
    
    async def _make_request(self, method: str, endpoint: str, 
                          params: Optional[Dict] = None,
                          data: Optional[Dict] = None,
                          headers: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated API request with rate limiting"""
        try:
            # Ensure we're authenticated
            if not await self._ensure_authenticated():
                logger.error("Not authenticated for API request")
                return None
            
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            # Build URL
            url = f"{self.base_url}{endpoint}"
            
            # Default headers
            request_headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
            
            if headers:
                request_headers.update(headers)
            
            # Make request
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=request_headers
            ) as response:
                
                # Handle different response types
                if response.status == 200 or response.status == 201:
                    if response.content_type == 'application/json':
                        return await response.json()
                    else:
                        return await response.text()
                
                elif response.status == 204:
                    return {}  # No content
                
                elif response.status == 401:
                    # Token expired, try to refresh
                    logger.warning("Received 401, attempting token refresh")
                    if await self._refresh_access_token():
                        # Retry the request once
                        request_headers['Authorization'] = f'Bearer {self.access_token}'
                        async with self.session.request(
                            method=method, url=url, params=params, 
                            json=data, headers=request_headers
                        ) as retry_response:
                            if retry_response.status == 200:
                                return await retry_response.json()
                    
                    logger.error("Authentication failed even after token refresh")
                    return None
                
                else:
                    error_text = await response.text()
                    logger.error(f"API request failed: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error making API request: {str(e)}")
            return None
    
    async def get_accounts(self) -> Optional[List[Dict]]:
        """Get account information"""
        try:
            response = await self._make_request('GET', '/trader/v1/accounts')
            return response if response else []
            
        except Exception as e:
            logger.error(f"Error getting accounts: {str(e)}")
            return None
    
    async def get_account_details(self, account_number: Optional[str] = None) -> Optional[AccountInfo]:
        """Get detailed account information"""
        try:
            acc_num = account_number or self.account_number
            if not acc_num:
                logger.error("No account number available")
                return None
            
            endpoint = f"/trader/v1/accounts/{acc_num}"
            response = await self._make_request('GET', endpoint, 
                                              params={'fields': 'positions,orders'})
            
            if not response:
                return None
            
            # Parse account info
            account_data = response.get('securitiesAccount', {})
            
            account_info = AccountInfo(
                account_number=account_data.get('accountNumber', ''),
                account_type=account_data.get('type', ''),
                cash_balance=account_data.get('currentBalances', {}).get('cashBalance', 0.0),
                buying_power=account_data.get('currentBalances', {}).get('buyingPower', 0.0),
                market_value=account_data.get('currentBalances', {}).get('longMarketValue', 0.0),
                day_change=account_data.get('currentBalances', {}).get('totalCash', 0.0),
                day_change_percent=0.0,  # Calculate if needed
                total_value=account_data.get('currentBalances', {}).get('liquidationValue', 0.0)
            )
            
            return account_info
            
        except Exception as e:
            logger.error(f"Error getting account details: {str(e)}")
            return None
    
    async def get_positions(self, account_number: Optional[str] = None) -> Optional[List[Position]]:
        """Get current positions"""
        try:
            acc_num = account_number or self.account_number
            if not acc_num:
                return None
            
            endpoint = f"/trader/v1/accounts/{acc_num}"
            response = await self._make_request('GET', endpoint, params={'fields': 'positions'})
            
            if not response:
                return None
            
            positions = []
            position_data = response.get('securitiesAccount', {}).get('positions', [])
            
            for pos in position_data:
                instrument = pos.get('instrument', {})
                
                position = Position(
                    symbol=instrument.get('symbol', ''),
                    quantity=int(pos.get('longQuantity', 0)),
                    average_price=float(pos.get('averagePrice', 0)),
                    market_value=float(pos.get('marketValue', 0)),
                    day_change=float(pos.get('currentDayProfitLoss', 0)),
                    day_change_percent=float(pos.get('currentDayProfitLossPercentage', 0)),
                    unrealized_pnl=float(pos.get('marketValue', 0)) - 
                                   (float(pos.get('longQuantity', 0)) * float(pos.get('averagePrice', 0))),
                    unrealized_pnl_percent=0.0,  # Calculate if needed
                    asset_type=AssetType.EQUITY  # Default, parse from instrument if needed
                )
                
                if position.quantity > 0:  # Only include actual positions
                    positions.append(position)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return None
    
    async def place_order(self, order_request: OrderRequest, 
                         account_number: Optional[str] = None) -> Optional[OrderResponse]:
        """
        Place a trading order
        
        Args:
            order_request: Order details
            account_number: Account to place order in
            
        Returns:
            OrderResponse with order details
        """
        try:
            acc_num = account_number or self.account_number
            if not acc_num:
                logger.error("No account number for order placement")
                return None
            
            # Build order payload
            order_payload = {
                "orderType": order_request.order_type.value,
                "session": order_request.session,
                "duration": order_request.duration,
                "orderStrategyType": "SINGLE",
                "orderLegCollection": [
                    {
                        "instruction": order_request.instruction.value,
                        "quantity": order_request.quantity,
                        "instrument": {
                            "symbol": order_request.symbol,
                            "assetType": order_request.asset_type.value
                        }
                    }
                ]
            }
            
            # Add price for limit orders
            if order_request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                if order_request.price is None:
                    logger.error("Price required for limit orders")
                    return None
                order_payload["price"] = order_request.price
            
            # Add stop price for stop orders
            if order_request.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                if order_request.stop_price is None:
                    logger.error("Stop price required for stop orders")
                    return None
                order_payload["stopPrice"] = order_request.stop_price
            
            # Place order
            endpoint = f"/trader/v1/accounts/{acc_num}/orders"
            response = await self._make_request('POST', endpoint, data=order_payload)
            
            if response is None:
                logger.error("Failed to place order")
                return None
            
            # Extract order ID from response headers (typically in Location header)
            # For this implementation, we'll generate a mock order ID
            order_id = f"ORDER_{int(time.time())}"
            
            # Create order response
            order_response = OrderResponse(
                order_id=order_id,
                status=OrderStatus.PENDING,
                symbol=order_request.symbol,
                quantity=order_request.quantity,
                filled_quantity=0,
                remaining_quantity=order_request.quantity,
                price=order_request.price,
                filled_price=None,
                order_type=order_request.order_type,
                instruction=order_request.instruction,
                created_time=datetime.now(),
                updated_time=datetime.now(),
                message="Order placed successfully"
            )
            
            # Log order placement
            await self._log_order(order_response, "PLACED")
            
            logger.info(f"Order placed: {order_id} - {order_request.instruction.value} "
                       f"{order_request.quantity} {order_request.symbol}")
            
            return order_response
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None
    
    async def cancel_order(self, order_id: str, 
                          account_number: Optional[str] = None) -> bool:
        """Cancel an existing order"""
        try:
            acc_num = account_number or self.account_number
            if not acc_num:
                return False
            
            endpoint = f"/trader/v1/accounts/{acc_num}/orders/{order_id}"
            response = await self._make_request('DELETE', endpoint)
            
            if response is not None:
                logger.info(f"Order cancelled: {order_id}")
                await self._log_order_action(order_id, "CANCELLED")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False
    
    async def get_order_status(self, order_id: str, 
                             account_number: Optional[str] = None) -> Optional[OrderResponse]:
        """Get status of a specific order"""
        try:
            acc_num = account_number or self.account_number
            if not acc_num:
                return None
            
            endpoint = f"/trader/v1/accounts/{acc_num}/orders/{order_id}"
            response = await self._make_request('GET', endpoint)
            
            if not response:
                return None
            
            # Parse order data
            order_data = response
            
            order_response = OrderResponse(
                order_id=order_data.get('orderId', order_id),
                status=OrderStatus(order_data.get('status', 'PENDING')),
                symbol=order_data.get('orderLegCollection', [{}])[0].get('instrument', {}).get('symbol', ''),
                quantity=int(order_data.get('quantity', 0)),
                filled_quantity=int(order_data.get('filledQuantity', 0)),
                remaining_quantity=int(order_data.get('remainingQuantity', 0)),
                price=order_data.get('price'),
                filled_price=order_data.get('filledPrice'),
                order_type=OrderType(order_data.get('orderType', 'MARKET')),
                instruction=OrderInstruction(order_data.get('orderLegCollection', [{}])[0].get('instruction', 'BUY')),
                created_time=datetime.fromisoformat(order_data.get('enteredTime', datetime.now().isoformat())),
                updated_time=datetime.fromisoformat(order_data.get('statusLastModified', datetime.now().isoformat())),
                message=order_data.get('statusDescription', '')
            )
            
            return order_response
            
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {str(e)}")
            return None
    
    async def get_orders(self, account_number: Optional[str] = None,
                        from_date: Optional[datetime] = None,
                        to_date: Optional[datetime] = None,
                        status: Optional[OrderStatus] = None) -> Optional[List[OrderResponse]]:
        """Get list of orders"""
        try:
            acc_num = account_number or self.account_number
            if not acc_num:
                return None
            
            endpoint = f"/trader/v1/accounts/{acc_num}/orders"
            
            params = {}
            if from_date:
                params['fromEnteredTime'] = from_date.isoformat()
            if to_date:
                params['toEnteredTime'] = to_date.isoformat()
            if status:
                params['status'] = status.value
            
            response = await self._make_request('GET', endpoint, params=params)
            
            if not response:
                return []
            
            orders = []
            for order_data in response:
                try:
                    order_response = OrderResponse(
                        order_id=order_data.get('orderId', ''),
                        status=OrderStatus(order_data.get('status', 'PENDING')),
                        symbol=order_data.get('orderLegCollection', [{}])[0].get('instrument', {}).get('symbol', ''),
                        quantity=int(order_data.get('quantity', 0)),
                        filled_quantity=int(order_data.get('filledQuantity', 0)),
                        remaining_quantity=int(order_data.get('remainingQuantity', 0)),
                        price=order_data.get('price'),
                        filled_price=order_data.get('filledPrice'),
                        order_type=OrderType(order_data.get('orderType', 'MARKET')),
                        instruction=OrderInstruction(order_data.get('orderLegCollection', [{}])[0].get('instruction', 'BUY')),
                        created_time=datetime.fromisoformat(order_data.get('enteredTime', datetime.now().isoformat())),
                        updated_time=datetime.fromisoformat(order_data.get('statusLastModified', datetime.now().isoformat())),
                        message=order_data.get('statusDescription', '')
                    )
                    orders.append(order_response)
                    
                except Exception as parse_error:
                    logger.error(f"Error parsing order data: {str(parse_error)}")
                    continue
            
            return orders
            
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            return None
    
    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote for a symbol"""
        try:
            endpoint = f"/marketdata/v1/quotes"
            params = {'symbols': symbol}
            
            response = await self._make_request('GET', endpoint, params=params)
            
            if response and symbol in response:
                return response[symbol]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {str(e)}")
            return None
    
    async def get_quotes(self, symbols: List[str]) -> Optional[Dict[str, Dict]]:
        """Get quotes for multiple symbols"""
        try:
            endpoint = f"/marketdata/v1/quotes"
            params = {'symbols': ','.join(symbols)}
            
            response = await self._make_request('GET', endpoint, params=params)
            return response if response else {}
            
        except Exception as e:
            logger.error(f"Error getting quotes: {str(e)}")
            return None
    
    async def _log_order(self, order: OrderResponse, action: str):
        """Log order to database"""
        try:
            insert_query = """
                INSERT INTO trade_log (order_id, symbol, action, quantity, price, 
                                     order_type, instruction, status, timestamp, message)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                order.order_id, order.symbol, action, order.quantity, order.price,
                order.order_type.value, order.instruction.value, order.status.value,
                order.created_time, order.message
            )
            
            await self.db_manager.execute_query(insert_query, values)
            
        except Exception as e:
            logger.error(f"Error logging order: {str(e)}")
    
    async def _log_order_action(self, order_id: str, action: str):
        """Log order action to database"""
        try:
            update_query = """
                UPDATE trade_log SET status = %s, updated_at = %s 
                WHERE order_id = %s
            """
            
            values = (action, datetime.now(), order_id)
            await self.db_manager.execute_query(update_query, values)
            
        except Exception as e:
            logger.error(f"Error logging order action: {str(e)}")
    
    async def validate_order(self, order_request: OrderRequest) -> Tuple[bool, List[str]]:
        """Validate order before placement"""
        try:
            errors = []
            
            # Check account authentication
            if not self.is_authenticated:
                errors.append("Not authenticated with Schwab API")
            
            # Validate symbol
            if not order_request.symbol or len(order_request.symbol) == 0:
                errors.append("Symbol is required")
            
            # Validate quantity
            if order_request.quantity <= 0:
                errors.append("Quantity must be positive")
            
            # Validate price for limit orders
            if order_request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                if order_request.price is None or order_request.price <= 0:
                    errors.append("Valid price required for limit orders")
            
            # Validate stop price for stop orders
            if order_request.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                if order_request.stop_price is None or order_request.stop_price <= 0:
                    errors.append("Valid stop price required for stop orders")
            
            # Check buying power (if available)
            account_info = await self.get_account_details()
            if account_info and order_request.instruction == OrderInstruction.BUY:
                estimated_cost = order_request.quantity * (order_request.price or 0)
                if estimated_cost > account_info.buying_power:
                    errors.append(f"Insufficient buying power: ${account_info.buying_power:.2f} available")
            
            # Check position for sell orders
            if order_request.instruction == OrderInstruction.SELL:
                positions = await self.get_positions()
                if positions:
                    position = next((p for p in positions if p.symbol == order_request.symbol), None)
                    if not position or position.quantity < order_request.quantity:
                        errors.append(f"Insufficient shares to sell: {position.quantity if position else 0} available")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Error validating order: {str(e)}")
            return False, ["Order validation error"]
    
    async def get_trading_hours(self) -> Optional[Dict]:
        """Get current trading hours"""
        try:
            endpoint = "/marketdata/v1/markets"
            params = {'markets': 'equity'}
            
            response = await self._make_request('GET', endpoint, params=params)
            return response if response else {}
            
        except Exception as e:
            logger.error(f"Error getting trading hours: {str(e)}")
            return None
    
    async def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            trading_hours = await self.get_trading_hours()
            
            if trading_hours and 'equity' in trading_hours:
                equity_hours = trading_hours['equity']
                # Parse trading hours and check current time
                # This is a simplified check
                now = datetime.now().time()
                return now.hour >= 9 and now.hour < 16  # 9 AM to 4 PM ET (simplified)
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking market hours: {str(e)}")
            return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            'authenticated': self.is_authenticated,
            'access_token_valid': self.access_token is not None,
            'token_expires_at': self.token_expires_at.isoformat() if self.token_expires_at else None,
            'account_number': self.account_number,
            'session_active': self.session is not None and not self.session.closed
        }
    
    async def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("SchwabTradingClient session closed")
