import asyncio
import websockets
import json
import logging
import jwt
from datetime import datetime, timedelta
from typing import Dict, Set, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import threading
import time
from enum import Enum
import uuid

class MessageType(Enum):
    """WebSocket message types"""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    DATA = "data"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    AUTHENTICATION = "auth"
    ALERT = "alert"
    TRADE_UPDATE = "trade_update"
    PORTFOLIO_UPDATE = "portfolio_update"
    MARKET_DATA = "market_data"
    SYSTEM_STATUS = "system_status"

@dataclass
class WebSocketClient:
    """WebSocket client connection"""
    id: str
    websocket: websockets.WebSocketServerProtocol
    user_id: Optional[int] = None
    username: Optional[str] = None
    subscriptions: Set[str] = field(default_factory=set)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    connected_at: datetime = field(default_factory=datetime.now)
    ip_address: str = ""
    user_agent: str = ""
    is_authenticated: bool = False

@dataclass
class Subscription:
    """Subscription configuration"""
    topic: str
    filters: Dict[str, Any] = field(default_factory=dict)
    client_ids: Set[str] = field(default_factory=set)

class WebSocketHandler:
    """Advanced WebSocket handler for real-time communication"""
    
    def __init__(self, host: str = "localhost", port: int = 8765, 
                 secret_key: str = "secret", heartbeat_interval: int = 30):
        self.host = host
        self.port = port
        self.secret_key = secret_key
        self.heartbeat_interval = heartbeat_interval
        
        self.logger = logging.getLogger(__name__)
        
        # Client management
        self.clients: Dict[str, WebSocketClient] = {}
        self.subscriptions: Dict[str, Subscription] = {}
        self.user_clients: Dict[int, Set[str]] = defaultdict(set)
        
        # Message routing
        self.message_handlers: Dict[MessageType, Callable] = {
            MessageType.SUBSCRIBE: self._handle_subscribe,
            MessageType.UNSUBSCRIBE: self._handle_unsubscribe,
            MessageType.AUTHENTICATION: self._handle_authentication,
            MessageType.HEARTBEAT: self._handle_heartbeat
        }
        
        # Broadcasting
        self.broadcast_queue = asyncio.Queue()
        self.broadcast_task = None
        
        # Statistics
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'authentication_attempts': 0,
            'failed_authentications': 0,
            'subscriptions_count': 0
        }
        
        # Background tasks
        self.cleanup_task = None
        self.heartbeat_task = None
        
        # Rate limiting
        self.rate_limits: Dict[str, Dict] = defaultdict(lambda: {'count': 0, 'reset_time': datetime.now()})
        self.rate_limit_threshold = 100  # messages per minute
        
        # Message history for debugging
        self.message_history = []
        self.max_history_size = 1000
    
    async def start_server(self):
        """Start WebSocket server"""
        try:
            self.logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
            
            # Start background tasks
            self.broadcast_task = asyncio.create_task(self._broadcast_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Start WebSocket server
            async with websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10
            ):
                self.logger.info("WebSocket server started successfully")
                await asyncio.Future()  # Keep server running
                
        except Exception as e:
            self.logger.error(f"Error starting WebSocket server: {e}")
            raise
    
    async def stop_server(self):
        """Stop WebSocket server"""
        try:
            # Cancel background tasks
            if self.broadcast_task:
                self.broadcast_task.cancel()
            if self.cleanup_task:
                self.cleanup_task.cancel()
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            
            # Close all client connections
            await self._close_all_clients()
            
            self.logger.info("WebSocket server stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping WebSocket server: {e}")
    
    async def _handle_client(self, websocket, path):
        """Handle new client connection"""
        client_id = str(uuid.uuid4())
        
        try:
            # Create client
            client = WebSocketClient(
                id=client_id,
                websocket=websocket,
                ip_address=websocket.remote_address[0] if websocket.remote_address else "unknown",
                user_agent=websocket.request_headers.get('User-Agent', 'unknown')
            )
            
            self.clients[client_id] = client
            self.stats['total_connections'] += 1
            self.stats['active_connections'] += 1
            
            self.logger.info(f"New WebSocket connection: {client_id} from {client.ip_address}")
            
            # Send welcome message
            await self._send_to_client(client_id, {
                'type': MessageType.DATA.value,
                'topic': 'connection',
                'data': {
                    'client_id': client_id,
                    'status': 'connected',
                    'server_time': datetime.now().isoformat(),
                    'requires_auth': True
                }
            })
            
            # Handle messages
            async for message in websocket:
                try:
                    await self._handle_message(client_id, message)
                except Exception as e:
                    self.logger.error(f"Error handling message from {client_id}: {e}")
                    await self._send_error(client_id, f"Message handling error: {str(e)}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            self.logger.error(f"Error handling client {client_id}: {e}")
        finally:
            await self._cleanup_client(client_id)
    
    async def _handle_message(self, client_id: str, message: str):
        """Handle incoming message from client"""
        try:
            # Check rate limiting
            if not self._check_rate_limit(client_id):
                await self._send_error(client_id, "Rate limit exceeded")
                return
            
            # Parse message
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await self._send_error(client_id, "Invalid JSON format")
                return
            
            # Validate message structure
            if not isinstance(data, dict) or 'type' not in data:
                await self._send_error(client_id, "Invalid message format")
                return
            
            message_type_str = data.get('type')
            try:
                message_type = MessageType(message_type_str)
            except ValueError:
                await self._send_error(client_id, f"Unknown message type: {message_type_str}")
                return
            
            # Add to message history
            self._add_to_history({
                'client_id': client_id,
                'timestamp': datetime.now().isoformat(),
                'type': message_type_str,
                'data': data
            })
            
            self.stats['messages_received'] += 1
            
            # Check authentication requirement
            client = self.clients.get(client_id)
            if not client:
                return
            
            if not client.is_authenticated and message_type != MessageType.AUTHENTICATION:
                await self._send_error(client_id, "Authentication required")
                return
            
            # Route message to handler
            handler = self.message_handlers.get(message_type)
            if handler:
                await handler(client_id, data)
            else:
                await self._send_error(client_id, f"No handler for message type: {message_type_str}")
                
        except Exception as e:
            self.logger.error(f"Error processing message from {client_id}: {e}")
            await self._send_error(client_id, "Message processing error")
    
    async def _handle_authentication(self, client_id: str, data: Dict[str, Any]):
        """Handle authentication message"""
        self.stats['authentication_attempts'] += 1
        
        try:
            token = data.get('token')
            if not token:
                await self._send_error(client_id, "Authentication token required")
                self.stats['failed_authentications'] += 1
                return
            
            # Verify JWT token
            try:
                payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
                user_id = payload.get('user_id')
                username = payload.get('username')
                
                if not user_id or not username:
                    raise jwt.InvalidTokenError("Missing user information")
                
            except jwt.ExpiredSignatureError:
                await self._send_error(client_id, "Token expired")
                self.stats['failed_authentications'] += 1
                return
            except jwt.InvalidTokenError as e:
                await self._send_error(client_id, f"Invalid token: {str(e)}")
                self.stats['failed_authentications'] += 1
                return
            
            # Update client
            client = self.clients.get(client_id)
            if client:
                client.user_id = user_id
                client.username = username
                client.is_authenticated = True
                
                # Track user connections
                self.user_clients[user_id].add(client_id)
                
                await self._send_to_client(client_id, {
                    'type': MessageType.DATA.value,
                    'topic': 'authentication',
                    'data': {
                        'status': 'authenticated',
                        'user_id': user_id,
                        'username': username
                    }
                })
                
                self.logger.info(f"Client {client_id} authenticated as user {username}")
            
        except Exception as e:
            self.logger.error(f"Authentication error for {client_id}: {e}")
            await self._send_error(client_id, "Authentication failed")
            self.stats['failed_authentications'] += 1
    
    async def _handle_subscribe(self, client_id: str, data: Dict[str, Any]):
        """Handle subscription message"""
        try:
            topic = data.get('topic')
            filters = data.get('filters', {})
            
            if not topic:
                await self._send_error(client_id, "Topic required for subscription")
                return
            
            # Validate topic
            valid_topics = [
                'market_data', 'portfolio_updates', 'trade_updates', 
                'alerts', 'system_status', 'ai_predictions'
            ]
            
            if topic not in valid_topics:
                await self._send_error(client_id, f"Invalid topic: {topic}")
                return
            
            # Add subscription
            if topic not in self.subscriptions:
                self.subscriptions[topic] = Subscription(topic=topic, filters=filters)
            
            self.subscriptions[topic].client_ids.add(client_id)
            
            # Update client subscriptions
            client = self.clients.get(client_id)
            if client:
                client.subscriptions.add(topic)
            
            self.stats['subscriptions_count'] += 1
            
            await self._send_to_client(client_id, {
                'type': MessageType.DATA.value,
                'topic': 'subscription',
                'data': {
                    'status': 'subscribed',
                    'topic': topic,
                    'filters': filters
                }
            })
            
            self.logger.info(f"Client {client_id} subscribed to {topic}")
            
        except Exception as e:
            self.logger.error(f"Subscription error for {client_id}: {e}")
            await self._send_error(client_id, "Subscription failed")
    
    async def _handle_unsubscribe(self, client_id: str, data: Dict[str, Any]):
        """Handle unsubscription message"""
        try:
            topic = data.get('topic')
            
            if not topic:
                await self._send_error(client_id, "Topic required for unsubscription")
                return
            
            # Remove subscription
            if topic in self.subscriptions:
                self.subscriptions[topic].client_ids.discard(client_id)
                
                # Remove empty subscriptions
                if not self.subscriptions[topic].client_ids:
                    del self.subscriptions[topic]
            
            # Update client subscriptions
            client = self.clients.get(client_id)
            if client:
                client.subscriptions.discard(topic)
            
            await self._send_to_client(client_id, {
                'type': MessageType.DATA.value,
                'topic': 'subscription',
                'data': {
                    'status': 'unsubscribed',
                    'topic': topic
                }
            })
            
            self.logger.info(f"Client {client_id} unsubscribed from {topic}")
            
        except Exception as e:
            self.logger.error(f"Unsubscription error for {client_id}: {e}")
            await self._send_error(client_id, "Unsubscription failed")
    
    async def _handle_heartbeat(self, client_id: str, data: Dict[str, Any]):
        """Handle heartbeat message"""
        client = self.clients.get(client_id)
        if client:
            client.last_heartbeat = datetime.now()
            
            await self._send_to_client(client_id, {
                'type': MessageType.HEARTBEAT.value,
                'data': {
                    'timestamp': datetime.now().isoformat(),
                    'status': 'alive'
                }
            })
    
    async def _send_to_client(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client"""
        client = self.clients.get(client_id)
        if not client:
            return False
        
        try:
            message_json = json.dumps(message, default=str)
            await client.websocket.send(message_json)
            self.stats['messages_sent'] += 1
            return True
            
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning(f"Cannot send message to closed connection: {client_id}")
            await self._cleanup_client(client_id)
            return False
        except Exception as e:
            self.logger.error(f"Error sending message to {client_id}: {e}")
            return False
    
    async def _send_error(self, client_id: str, error_message: str):
        """Send error message to client"""
        await self._send_to_client(client_id, {
            'type': MessageType.ERROR.value,
            'data': {
                'error': error_message,
                'timestamp': datetime.now().isoformat()
            }
        })
    
    async def broadcast_to_topic(self, topic: str, message: Dict[str, Any], 
                               filters: Optional[Dict[str, Any]] = None):
        """Broadcast message to all clients subscribed to a topic"""
        if topic not in self.subscriptions:
            return
        
        subscription = self.subscriptions[topic]
        
        # Apply filters if specified
        target_clients = set(subscription.client_ids)
        
        if filters:
            # Filter clients based on criteria (e.g., user_id, permissions)
            filtered_clients = set()
            for client_id in target_clients:
                client = self.clients.get(client_id)
                if client and self._matches_filters(client, filters):
                    filtered_clients.add(client_id)
            target_clients = filtered_clients
        
        # Queue broadcast message
        await self.broadcast_queue.put({
            'type': 'topic_broadcast',
            'topic': topic,
            'message': message,
            'client_ids': list(target_clients)
        })
    
    async def broadcast_to_user(self, user_id: int, message: Dict[str, Any]):
        """Broadcast message to all connections of a specific user"""
        client_ids = self.user_clients.get(user_id, set())
        
        if client_ids:
            await self.broadcast_queue.put({
                'type': 'user_broadcast',
                'user_id': user_id,
                'message': message,
                'client_ids': list(client_ids)
            })
    
    async def broadcast_to_all(self, message: Dict[str, Any], 
                             authenticated_only: bool = True):
        """Broadcast message to all connected clients"""
        target_clients = []
        
        for client_id, client in self.clients.items():
            if not authenticated_only or client.is_authenticated:
                target_clients.append(client_id)
        
        if target_clients:
            await self.broadcast_queue.put({
                'type': 'global_broadcast',
                'message': message,
                'client_ids': target_clients
            })
    
    async def _broadcast_loop(self):
        """Background loop for processing broadcast messages"""
        while True:
            try:
                broadcast_item = await self.broadcast_queue.get()
                
                message = broadcast_item['message']
                client_ids = broadcast_item['client_ids']
                
                # Send to all target clients
                for client_id in client_ids:
                    await self._send_to_client(client_id, message)
                
                self.broadcast_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in broadcast loop: {e}")
    
    async def _cleanup_loop(self):
        """Background loop for cleanup tasks"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_stale_connections()
                self._cleanup_message_history()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
    
    async def _heartbeat_loop(self):
        """Background loop for heartbeat checks"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                await self._check_heartbeats()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
    
    async def _check_heartbeats(self):
        """Check for stale connections based on heartbeat"""
        timeout_threshold = datetime.now() - timedelta(seconds=self.heartbeat_interval * 3)
        stale_clients = []
        
        for client_id, client in self.clients.items():
            if client.last_heartbeat < timeout_threshold:
                stale_clients.append(client_id)
        
        for client_id in stale_clients:
            self.logger.warning(f"Removing stale connection: {client_id}")
            await self._cleanup_client(client_id)
    
    async def _cleanup_stale_connections(self):
        """Clean up stale connections"""
        stale_clients = []
        
        for client_id, client in self.clients.items():
            try:
                # Try to ping the connection
                await client.websocket.ping()
            except:
                stale_clients.append(client_id)
        
        for client_id in stale_clients:
            await self._cleanup_client(client_id)
    
    async def _cleanup_client(self, client_id: str):
        """Clean up client connection and associated data"""
        try:
            client = self.clients.get(client_id)
            if not client:
                return
            
            # Remove from user connections
            if client.user_id:
                self.user_clients[client.user_id].discard(client_id)
                if not self.user_clients[client.user_id]:
                    del self.user_clients[client.user_id]
            
            # Remove from subscriptions
            for topic in client.subscriptions:
                if topic in self.subscriptions:
                    self.subscriptions[topic].client_ids.discard(client_id)
                    if not self.subscriptions[topic].client_ids:
                        del self.subscriptions[topic]
            
            # Close WebSocket connection
            if not client.websocket.closed:
                await client.websocket.close()
            
            # Remove from clients
            del self.clients[client_id]
            self.stats['active_connections'] = max(0, self.stats['active_connections'] - 1)
            
            self.logger.info(f"Cleaned up client: {client_id}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up client {client_id}: {e}")
    
    async def _close_all_clients(self):
        """Close all client connections"""
        for client_id in list(self.clients.keys()):
            await self._cleanup_client(client_id)
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits"""
        current_time = datetime.now()
        rate_limit = self.rate_limits[client_id]
        
        # Reset counter if minute has passed
        if current_time - rate_limit['reset_time'] > timedelta(minutes=1):
            rate_limit['count'] = 0
            rate_limit['reset_time'] = current_time
        
        # Check limit
        if rate_limit['count'] >= self.rate_limit_threshold:
            return False
        
        rate_limit['count'] += 1
        return True
    
    def _matches_filters(self, client: WebSocketClient, filters: Dict[str, Any]) -> bool:
        """Check if client matches broadcast filters"""
        for key, value in filters.items():
            if key == 'user_id' and client.user_id != value:
                return False
            elif key == 'username' and client.username != value:
                return False
            # Add more filter criteria as needed
        
        return True
    
    def _add_to_history(self, message: Dict[str, Any]):
        """Add message to history for debugging"""
        self.message_history.append(message)
        
        # Keep history size manageable
        if len(self.message_history) > self.max_history_size:
            self.message_history = self.message_history[-self.max_history_size:]
    
    def _cleanup_message_history(self):
        """Clean up old message history"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        self.message_history = [
            msg for msg in self.message_history
            if datetime.fromisoformat(msg['timestamp']) > cutoff_time
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket server statistics"""
        return {
            **self.stats,
            'active_subscriptions': len(self.subscriptions),
            'unique_users': len(self.user_clients),
            'subscription_topics': list(self.subscriptions.keys())
        }
    
    def get_client_info(self, client_id: str = None) -> Dict[str, Any]:
        """Get information about connected clients"""
        if client_id:
            client = self.clients.get(client_id)
            if client:
                return {
                    'id': client.id,
                    'user_id': client.user_id,
                    'username': client.username,
                    'ip_address': client.ip_address,
                    'user_agent': client.user_agent,
                    'is_authenticated': client.is_authenticated,
                    'subscriptions': list(client.subscriptions),
                    'connected_at': client.connected_at.isoformat(),
                    'last_heartbeat': client.last_heartbeat.isoformat()
                }
            else:
                return {}
        else:
            return {
                'total_clients': len(self.clients),
                'clients': [
                    {
                        'id': client.id,
                        'user_id': client.user_id,
                        'username': client.username,
                        'is_authenticated': client.is_authenticated,
                        'subscriptions': list(client.subscriptions),
                        'connected_at': client.connected_at.isoformat()
                    }
                    for client in self.clients.values()
                ]
            }

# Utility functions for external components
class WebSocketManager:
    """Manager class for WebSocket operations"""
    
    def __init__(self, handler: WebSocketHandler):
        self.handler = handler
    
    async def notify_portfolio_update(self, user_id: int, portfolio_data: Dict[str, Any]):
        """Notify user of portfolio updates"""
        message = {
            'type': MessageType.PORTFOLIO_UPDATE.value,
            'data': portfolio_data,
            'timestamp': datetime.now().isoformat()
        }
        
        await self.handler.broadcast_to_user(user_id, message)
    
    async def notify_trade_execution(self, user_id: int, trade_data: Dict[str, Any]):
        """Notify user of trade execution"""
        message = {
            'type': MessageType.TRADE_UPDATE.value,
            'data': trade_data,
            'timestamp': datetime.now().isoformat()
        }
        
        await self.handler.broadcast_to_user(user_id, message)
    
    async def broadcast_market_data(self, symbol: str, market_data: Dict[str, Any]):
        """Broadcast market data updates"""
        message = {
            'type': MessageType.MARKET_DATA.value,
            'data': {
                'symbol': symbol,
                **market_data
            },
            'timestamp': datetime.now().isoformat()
        }
        
        await self.handler.broadcast_to_topic('market_data', message, 
                                            filters={'symbol': symbol})
    
    async def notify_alert(self, user_id: int, alert_data: Dict[str, Any]):
        """Send alert notification to user"""
        message = {
            'type': MessageType.ALERT.value,
            'data': alert_data,
            'timestamp': datetime.now().isoformat()
        }
        
        await self.handler.broadcast_to_user(user_id, message)
    
    async def broadcast_system_status(self, status_data: Dict[str, Any]):
        """Broadcast system status updates"""
        message = {
            'type': MessageType.SYSTEM_STATUS.value,
            'data': status_data,
            'timestamp': datetime.now().isoformat()
        }
        
        await self.handler.broadcast_to_topic('system_status', message)

# Example usage
async def main():
    """Example of running the WebSocket server"""
    handler = WebSocketHandler(host="0.0.0.0", port=8765)
    
    try:
        await handler.start_server()
    except KeyboardInterrupt:
        print("Shutting down WebSocket server...")
        await handler.stop_server()

if __name__ == "__main__":
    asyncio.run(main())
