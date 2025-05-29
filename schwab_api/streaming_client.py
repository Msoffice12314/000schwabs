import asyncio
import websockets
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
import threading
import time
from dataclasses import dataclass, field
from collections import defaultdict, deque
import ssl
import base64
import hmac
import hashlib
from urllib.parse import urlencode
import aiohttp
import backoff
from enum import Enum

class StreamingService(Enum):
    """Schwab streaming services"""
    QUOTE = "QUOTE"
    OPTION = "OPTION"
    LEVELONE_FUTURES = "LEVELONE_FUTURES"
    LEVELONE_FOREX = "LEVELONE_FOREX"
    LEVELONE_FUTURES_OPTIONS = "LEVELONE_FUTURES_OPTIONS"
    NEWS_HEADLINE = "NEWS_HEADLINE"
    CHART_EQUITY = "CHART_EQUITY"
    CHART_FUTURES = "CHART_FUTURES"
    TIMESALE_EQUITY = "TIMESALE_EQUITY"
    TIMESALE_FUTURES = "TIMESALE_FUTURES"
    BOOK = "BOOK"
    ACTIVES_NYSE = "ACTIVES_NYSE"
    ACTIVES_NASDAQ = "ACTIVES_NASDAQ"
    ACTIVES_OPTIONS = "ACTIVES_OPTIONS"

@dataclass
class StreamingMessage:
    """Represents a streaming message from Schwab"""
    service: str
    command: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    symbols: List[str] = field(default_factory=list)

@dataclass
class SubscriptionRequest:
    """Subscription request configuration"""
    service: StreamingService
    symbols: List[str]
    fields: Optional[List[str]] = None
    command: str = "SUBS"

class SchwabStreamingClient:
    """Advanced Schwab API WebSocket streaming client"""
    
    def __init__(self, access_token: str, refresh_token: str, 
                 account_info: Dict[str, Any]):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.account_info = account_info
        self.logger = logging.getLogger(__name__)
        
        # Connection management
        self.websocket = None
        self.connection_active = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.heartbeat_interval = 30
        
        # Message handling
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.message_queue = deque(maxlen=10000)
        self.subscriptions: Dict[str, SubscriptionRequest] = {}
        
        # Performance tracking
        self.message_count = 0
        self.last_message_time = None
        self.connection_start_time = None
        self.latency_history = deque(maxlen=1000)
        
        # Data caching
        self.quote_cache: Dict[str, Dict] = {}
        self.level1_cache: Dict[str, Dict] = {}
        self.timesale_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Threading
        self.processing_thread = None
        self.heartbeat_thread = None
        self.stop_event = threading.Event()
        
        # Error handling
        self.error_count = 0
        self.last_error_time = None
        self.error_callbacks: List[Callable] = []
        
        # Quality of service
        self.qos_level = 0  # 0=500ms, 1=750ms, 2=1000ms, 3=1500ms, 4=3000ms, 5=5000ms
        
        # Streaming endpoint configuration
        self.streaming_url = "wss://streamer-api.schwab.com/ws"
        self.request_id = 0
        
    async def connect(self) -> bool:
        """Establish WebSocket connection to Schwab streaming API"""
        try:
            self.logger.info("Connecting to Schwab streaming API...")
            
            # Get streaming credentials
            streaming_info = await self._get_streaming_credentials()
            if not streaming_info:
                self.logger.error("Failed to get streaming credentials")
                return False
            
            # Build WebSocket URL with authentication
            ws_url = self._build_streaming_url(streaming_info)
            
            # SSL configuration
            ssl_context = ssl.create_default_context()
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(
                ws_url,
                ssl=ssl_context,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10
            )
            
            self.connection_active = True
            self.connection_start_time = datetime.now()
            self.reconnect_attempts = 0
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Send login request
            await self._send_login_request(streaming_info)
            
            self.logger.info("Successfully connected to Schwab streaming API")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to streaming API: {e}")
            self.connection_active = False
            return False
    
    async def _get_streaming_credentials(self) -> Optional[Dict[str, Any]]:
        """Get streaming credentials from Schwab API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                url = "https://api.schwabapi.com/trader/v1/userPreference"
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('streamerInfo', {})
                    else:
                        self.logger.error(f"Failed to get streaming credentials: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error getting streaming credentials: {e}")
            return None
    
    def _build_streaming_url(self, streaming_info: Dict[str, Any]) -> str:
        """Build streaming WebSocket URL with authentication"""
        try:
            # Extract streaming parameters
            app_id = streaming_info.get('schwabClientCustomerId')
            channel = streaming_info.get('schwabClientChannel')
            function_id = streaming_info.get('schwabClientFunctionId')
            
            # Build authentication parameters
            timestamp = int(time.time() * 1000)
            
            params = {
                'version': '1.0',
                'type': 'USER_PRINCIPAL_REQUEST',
                'token': self.access_token,
                'apikey': app_id,
                'timestamp': timestamp,
                'qoslevel': self.qos_level,
                'channel': channel,
                'function': function_id
            }
            
            return f"{self.streaming_url}?{urlencode(params)}"
            
        except Exception as e:
            self.logger.error(f"Error building streaming URL: {e}")
            return self.streaming_url
    
    async def _send_login_request(self, streaming_info: Dict[str, Any]):
        """Send login request to establish authenticated session"""
        try:
            login_request = {
                "requests": [{
                    "service": "ADMIN",
                    "command": "LOGIN",
                    "requestid": self._get_next_request_id(),
                    "account": self.account_info.get('accountNumber'),
                    "source": streaming_info.get('schwabClientCustomerId'),
                    "parameters": {
                        "credential": urlencode({
                            "userid": self.account_info.get('accountNumber'),
                            "token": self.access_token,
                            "company": streaming_info.get('schwabClientCustomerId'),
                            "segment": streaming_info.get('schwabClientChannel'),
                            "cddomain": streaming_info.get('schwabClientFunctionId'),
                            "usergroup": streaming_info.get('streamerBinaryUrl', ''),
                            "accesslevel": "SCHWAB",
                            "authorized": "Y",
                            "timestamp": int(time.time() * 1000),
                            "appid": "SCHWAB"
                        }),
                        "token": self.access_token,
                        "version": "1.0"
                    }
                }]
            }
            
            await self.websocket.send(json.dumps(login_request))
            self.logger.info("Login request sent")
            
        except Exception as e:
            self.logger.error(f"Error sending login request: {e}")
    
    async def _start_background_tasks(self):
        """Start background processing tasks"""
        try:
            # Start message processing thread
            if self.processing_thread is None or not self.processing_thread.is_alive():
                self.stop_event.clear()
                self.processing_thread = threading.Thread(
                    target=self._message_processing_loop,
                    daemon=True
                )
                self.processing_thread.start()
            
            # Start heartbeat thread
            if self.heartbeat_thread is None or not self.heartbeat_thread.is_alive():
                self.heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop,
                    daemon=True
                )
                self.heartbeat_thread.start()
            
        except Exception as e:
            self.logger.error(f"Error starting background tasks: {e}")
    
    def _message_processing_loop(self):
        """Background loop for processing incoming messages"""
        while not self.stop_event.is_set() and self.connection_active:
            try:
                # Get message from WebSocket
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    message = loop.run_until_complete(
                        asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    )
                    
                    # Process message
                    self._process_message(message)
                    
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    self.logger.warning("WebSocket connection closed")
                    self.connection_active = False
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {e}")
                time.sleep(1)
            finally:
                loop.close()
    
    def _heartbeat_loop(self):
        """Background loop for sending heartbeat messages"""
        while not self.stop_event.is_set() and self.connection_active:
            try:
                if self.websocket and not self.websocket.closed:
                    # Send heartbeat
                    heartbeat_msg = {
                        "requests": [{
                            "service": "ADMIN",
                            "command": "QOS",
                            "requestid": self._get_next_request_id(),
                            "parameters": {
                                "qoslevel": str(self.qos_level)
                            }
                        }]
                    }
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        loop.run_until_complete(
                            self.websocket.send(json.dumps(heartbeat_msg))
                        )
                    finally:
                        loop.close()
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(5)
    
    def _process_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            self.message_count += 1
            self.last_message_time = datetime.now()
            
            # Parse JSON message
            data = json.loads(message)
            
            # Handle different message types
            if 'response' in data:
                self._handle_response_message(data['response'])
            elif 'data' in data:
                self._handle_data_message(data['data'])
            elif 'notify' in data:
                self._handle_notification_message(data['notify'])
            
            # Add to message queue for debugging
            self.message_queue.append({
                'timestamp': self.last_message_time,
                'message': data
            })
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            self.error_count += 1
            self.last_error_time = datetime.now()
    
    def _handle_response_message(self, response_data: List[Dict]):
        """Handle response messages from the streaming API"""
        try:
            for response in response_data:
                service = response.get('service', '')
                command = response.get('command', '')
                
                if service == 'ADMIN' and command == 'LOGIN':
                    self._handle_login_response(response)
                elif service in ['QUOTE', 'OPTION', 'TIMESALE_EQUITY']:
                    self._handle_subscription_response(response)
                
        except Exception as e:
            self.logger.error(f"Error handling response message: {e}")
    
    def _handle_data_message(self, data_list: List[Dict]):
        """Handle streaming data messages"""
        try:
            for data_item in data_list:
                service = data_item.get('service', '')
                timestamp = datetime.now()
                
                # Create streaming message
                streaming_msg = StreamingMessage(
                    service=service,
                    command='DATA',
                    timestamp=timestamp,
                    data=data_item
                )
                
                # Update caches
                self._update_cache(streaming_msg)
                
                # Call registered handlers
                self._call_message_handlers(streaming_msg)
                
        except Exception as e:
            self.logger.error(f"Error handling data message: {e}")
    
    def _handle_notification_message(self, notify_data: List[Dict]):
        """Handle notification messages"""
        try:
            for notify in notify_data:
                self.logger.info(f"Notification: {notify}")
                
        except Exception as e:
            self.logger.error(f"Error handling notification: {e}")
    
    def _handle_login_response(self, response: Dict):
        """Handle login response"""
        try:
            if response.get('content', {}).get('code') == 0:
                self.logger.info("Successfully logged into streaming API")
            else:
                error_msg = response.get('content', {}).get('msg', 'Unknown error')
                self.logger.error(f"Login failed: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Error handling login response: {e}")
    
    def _handle_subscription_response(self, response: Dict):
        """Handle subscription response"""
        try:
            service = response.get('service', '')
            command = response.get('command', '')
            
            if response.get('content', {}).get('code') == 0:
                self.logger.info(f"Successfully subscribed to {service} - {command}")
            else:
                error_msg = response.get('content', {}).get('msg', 'Unknown error')
                self.logger.error(f"Subscription failed for {service}: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Error handling subscription response: {e}")
    
    def _update_cache(self, message: StreamingMessage):
        """Update internal data caches"""
        try:
            service = message.service
            data = message.data
            
            if service == 'QUOTE':
                # Update quote cache
                content = data.get('content', {})
                for symbol, quote_data in content.items():
                    self.quote_cache[symbol] = {
                        **quote_data,
                        'timestamp': message.timestamp
                    }
            
            elif service == 'TIMESALE_EQUITY':
                # Update timesale cache
                content = data.get('content', {})
                for symbol, timesale_data in content.items():
                    self.timesale_cache[symbol].append({
                        **timesale_data,
                        'timestamp': message.timestamp
                    })
            
            elif service in ['LEVELONE_FUTURES', 'LEVELONE_FOREX']:
                # Update level 1 cache
                content = data.get('content', {})
                for symbol, level1_data in content.items():
                    self.level1_cache[symbol] = {
                        **level1_data,
                        'timestamp': message.timestamp
                    }
            
        except Exception as e:
            self.logger.error(f"Error updating cache: {e}")
    
    def _call_message_handlers(self, message: StreamingMessage):
        """Call registered message handlers"""
        try:
            # Call handlers for specific service
            service_handlers = self.message_handlers.get(message.service, [])
            for handler in service_handlers:
                try:
                    handler(message)
                except Exception as e:
                    self.logger.error(f"Error in message handler: {e}")
            
            # Call global handlers
            global_handlers = self.message_handlers.get('*', [])
            for handler in global_handlers:
                try:
                    handler(message)
                except Exception as e:
                    self.logger.error(f"Error in global handler: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error calling message handlers: {e}")
    
    async def subscribe_quotes(self, symbols: List[str], 
                             fields: Optional[List[str]] = None) -> bool:
        """Subscribe to real-time quotes"""
        try:
            if not self.connection_active:
                self.logger.error("Connection not active")
                return False
            
            # Default quote fields
            if fields is None:
                fields = ["0", "1", "2", "3", "4", "8", "9", "10", "11", "12", "13"]
            
            subscription_request = {
                "requests": [{
                    "service": "QUOTE",
                    "command": "SUBS",
                    "requestid": self._get_next_request_id(),
                    "parameters": {
                        "keys": ",".join(symbols),
                        "fields": ",".join(fields)
                    }
                }]
            }
            
            await self.websocket.send(json.dumps(subscription_request))
            
            # Store subscription
            self.subscriptions[f"QUOTE_{','.join(symbols)}"] = SubscriptionRequest(
                service=StreamingService.QUOTE,
                symbols=symbols,
                fields=fields
            )
            
            self.logger.info(f"Subscribed to quotes for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            self.logger.error(f"Error subscribing to quotes: {e}")
            return False
    
    async def subscribe_timesales(self, symbols: List[str], 
                                fields: Optional[List[str]] = None) -> bool:
        """Subscribe to time and sales data"""
        try:
            if not self.connection_active:
                self.logger.error("Connection not active")
                return False
            
            # Default timesale fields
            if fields is None:
                fields = ["0", "1", "2", "3", "4"]
            
            subscription_request = {
                "requests": [{
                    "service": "TIMESALE_EQUITY",
                    "command": "SUBS",
                    "requestid": self._get_next_request_id(),
                    "parameters": {
                        "keys": ",".join(symbols),
                        "fields": ",".join(fields)
                    }
                }]
            }
            
            await self.websocket.send(json.dumps(subscription_request))
            
            # Store subscription
            self.subscriptions[f"TIMESALE_EQUITY_{','.join(symbols)}"] = SubscriptionRequest(
                service=StreamingService.TIMESALE_EQUITY,
                symbols=symbols,
                fields=fields
            )
            
            self.logger.info(f"Subscribed to timesales for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            self.logger.error(f"Error subscribing to timesales: {e}")
            return False
    
    async def subscribe_news(self, symbols: List[str]) -> bool:
        """Subscribe to news headlines"""
        try:
            if not self.connection_active:
                self.logger.error("Connection not active")
                return False
            
            subscription_request = {
                "requests": [{
                    "service": "NEWS_HEADLINE",
                    "command": "SUBS",
                    "requestid": self._get_next_request_id(),
                    "parameters": {
                        "keys": ",".join(symbols),
                        "fields": "0,1,2,3,4,5,6,7,8,9,10"
                    }
                }]
            }
            
            await self.websocket.send(json.dumps(subscription_request))
            
            self.logger.info(f"Subscribed to news for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            self.logger.error(f"Error subscribing to news: {e}")
            return False
    
    async def unsubscribe(self, service: StreamingService, symbols: List[str]) -> bool:
        """Unsubscribe from a service"""
        try:
            if not self.connection_active:
                return False
            
            unsubscribe_request = {
                "requests": [{
                    "service": service.value,
                    "command": "UNSUBS",
                    "requestid": self._get_next_request_id(),
                    "parameters": {
                        "keys": ",".join(symbols)
                    }
                }]
            }
            
            await self.websocket.send(json.dumps(unsubscribe_request))
            
            # Remove from subscriptions
            key = f"{service.value}_{','.join(symbols)}"
            if key in self.subscriptions:
                del self.subscriptions[key]
            
            self.logger.info(f"Unsubscribed from {service.value} for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unsubscribing: {e}")
            return False
    
    def add_message_handler(self, service: str, handler: Callable[[StreamingMessage], None]):
        """Add message handler for specific service or '*' for all"""
        self.message_handlers[service].append(handler)
    
    def remove_message_handler(self, service: str, handler: Callable):
        """Remove message handler"""
        if service in self.message_handlers:
            try:
                self.message_handlers[service].remove(handler)
            except ValueError:
                pass
    
    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest quote from cache"""
        return self.quote_cache.get(symbol)
    
    def get_timesales(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent timesales from cache"""
        if symbol in self.timesale_cache:
            return list(self.timesale_cache[symbol])[-limit:]
        return []
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        now = datetime.now()
        uptime = (now - self.connection_start_time).total_seconds() if self.connection_start_time else 0
        
        return {
            'connected': self.connection_active,
            'uptime_seconds': uptime,
            'message_count': self.message_count,
            'messages_per_second': self.message_count / max(uptime, 1),
            'last_message_time': self.last_message_time.isoformat() if self.last_message_time else None,
            'error_count': self.error_count,
            'reconnect_attempts': self.reconnect_attempts,
            'subscriptions': len(self.subscriptions),
            'cached_quotes': len(self.quote_cache),
            'qos_level': self.qos_level,
            'avg_latency_ms': np.mean(self.latency_history) if self.latency_history else 0
        }
    
    @backoff.on_exception(
        backoff.expo,
        (websockets.exceptions.ConnectionClosed, ConnectionError),
        max_tries=5,
        max_time=300
    )
    async def reconnect(self) -> bool:
        """Reconnect to streaming API with exponential backoff"""
        try:
            self.logger.info(f"Attempting to reconnect (attempt {self.reconnect_attempts + 1})")
            
            # Close existing connection
            if self.websocket:
                await self.websocket.close()
            
            self.connection_active = False
            self.reconnect_attempts += 1
            
            # Wait before reconnecting
            await asyncio.sleep(min(2 ** self.reconnect_attempts, 60))
            
            # Attempt connection
            success = await self.connect()
            
            if success:
                # Resubscribe to all services
                await self._resubscribe_all()
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error during reconnection: {e}")
            return False
    
    async def _resubscribe_all(self):
        """Resubscribe to all previously active subscriptions"""
        try:
            self.logger.info("Resubscribing to all services...")
            
            for key, subscription in self.subscriptions.items():
                if subscription.service == StreamingService.QUOTE:
                    await self.subscribe_quotes(subscription.symbols, subscription.fields)
                elif subscription.service == StreamingService.TIMESALE_EQUITY:
                    await self.subscribe_timesales(subscription.symbols, subscription.fields)
                # Add other services as needed
                
                # Small delay between subscriptions
                await asyncio.sleep(0.1)
            
            self.logger.info("Resubscription completed")
            
        except Exception as e:
            self.logger.error(f"Error resubscribing: {e}")
    
    def set_qos_level(self, level: int):
        """Set Quality of Service level (0-5)"""
        if 0 <= level <= 5:
            self.qos_level = level
            self.logger.info(f"QOS level set to {level}")
        else:
            self.logger.error("QOS level must be between 0 and 5")
    
    async def close(self):
        """Close WebSocket connection and cleanup"""
        try:
            self.logger.info("Closing streaming connection...")
            
            self.connection_active = False
            self.stop_event.set()
            
            # Close WebSocket
            if self.websocket:
                await self.websocket.close()
            
            # Wait for threads to finish
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
            
            if self.heartbeat_thread and self.heartbeat_thread.is_alive():
                self.heartbeat_thread.join(timeout=5.0)
            
            self.logger.info("Streaming connection closed")
            
        except Exception as e:
            self.logger.error(f"Error closing connection: {e}")
    
    def _get_next_request_id(self) -> int:
        """Get next request ID"""
        self.request_id += 1
        return self.request_id
    
    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Add callback for error handling"""
        self.error_callbacks.append(callback)
    
    def _handle_error(self, error: Exception):
        """Handle errors and call error callbacks"""
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(f"Error in error callback: {e}")
    
    def get_market_hours_status(self) -> Dict[str, Any]:
        """Get market hours status (simplified)"""
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        is_market_day = now.weekday() < 5  # Monday = 0, Friday = 4
        is_market_hours = market_open <= now <= market_close
        
        return {
            'is_market_open': is_market_day and is_market_hours,
            'is_market_day': is_market_day,
            'market_open_time': market_open.isoformat(),
            'market_close_time': market_close.isoformat(),
            'current_time': now.isoformat()
        }
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if self.connection_active:
                # Close connection in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.close())
                finally:
                    loop.close()
        except Exception as e:
            self.logger.error(f"Error in destructor: {e}")

# Utility functions
def create_streaming_client(access_token: str, refresh_token: str, 
                          account_info: Dict[str, Any]) -> SchwabStreamingClient:
    """Factory function to create streaming client"""
    return SchwabStreamingClient(access_token, refresh_token, account_info)

def format_schwab_symbol(symbol: str) -> str:
    """Format symbol for Schwab API compatibility"""
    return symbol.upper().strip()

def parse_quote_fields(quote_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse Schwab quote field numbers to readable names"""
    field_mapping = {
        "0": "Symbol",
        "1": "Bid Price",
        "2": "Ask Price", 
        "3": "Last Price",
        "4": "Bid Size",
        "5": "Ask Size",
        "6": "Bid ID",
        "7": "Ask ID",
        "8": "Volume",
        "9": "Last Size",
        "10": "Trade Time",
        "11": "Quote Time",
        "12": "High Price",
        "13": "Low Price",
        "14": "Close Price",
        "15": "Exchange ID",
        "16": "Marginable",
        "17": "Shortable",
        "18": "ISLAND Bid",
        "19": "ISLAND Ask",
        "20": "ISLAND Volume",
        "21": "Quote Day",
        "22": "Trade Day",
        "23": "Volatility",
        "24": "Description",
        "25": "Last ID",
        "26": "Digits",
        "27": "Open Price",
        "28": "Net Change",
        "29": "52 Week High",
        "30": "52 Week Low",
        "31": "PE Ratio",
        "32": "Dividend Amount",
        "33": "Dividend Yield",
        "34": "ISLAND Bid Size",
        "35": "ISLAND Ask Size",
        "36": "NAV",
        "37": "Fund Price",
        "38": "Exchange Name",
        "39": "Dividend Date",
        "40": "Regular Market Quote",
        "41": "Regular Market Trade",
        "42": "Regular Market Last Price",
        "43": "Regular Market Last Size",
        "44": "Regular Market Trade Time",
        "45": "Regular Market Trade Day",
        "46": "Regular Market Net Change",
        "47": "Security Status",
        "48": "Mark",
        "49": "Quote Time in Long",
        "50": "Trade Time in Long",
        "51": "Regular Market Trade Time in Long"
    }
    
    parsed = {}
    for field_num, value in quote_data.items():
        if field_num in field_mapping:
            parsed[field_mapping[field_num]] = value
        else:
            parsed[f"Field_{field_num}"] = value
    
    return parsed
