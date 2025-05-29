"""
Schwab AI Trading System - Order Management System
Advanced order execution with smart routing, partial fills, and order lifecycle management.
"""

import logging
import asyncio
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
from collections import defaultdict

from schwab_api.trading_client import SchwabTradingClient, OrderRequest, OrderResponse, OrderType, OrderInstruction, OrderStatus, AssetType
from trading.risk_manager import RiskManager, PositionSize
from models.signal_detector import TradingSignal, SignalType
from utils.cache_manager import CacheManager
from utils.database import DatabaseManager
from utils.notification import NotificationManager
from config.settings import get_settings

logger = logging.getLogger(__name__)

class OrderPriority(Enum):
    """Order execution priority"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

class ExecutionAlgorithm(Enum):
    """Order execution algorithms"""
    MARKET = "market"
    TWAP = "twap"  # Time Weighted Average Price
    VWAP = "vwap"  # Volume Weighted Average Price
    ICEBERG = "iceberg"
    SMART = "smart"

class OrderState(Enum):
    """Internal order states"""
    PENDING = "pending"
    QUEUED = "queued"
    ROUTING = "routing"
    WORKING = "working"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ERROR = "error"

@dataclass
class OrderExecution:
    """Order execution details"""
    execution_id: str
    order_id: str
    symbol: str
    quantity: int
    price: float
    timestamp: datetime
    commission: float = 0.0
    fees: float = 0.0
    execution_venue: str = "SCHWAB"
    liquidity_flag: str = "UNKNOWN"  # ADD, REMOVE, UNKNOWN

@dataclass
class OrderSlice:
    """Part of a larger order for algorithmic execution"""
    slice_id: str
    parent_order_id: str
    quantity: int
    target_price: Optional[float]
    min_quantity: int = 1
    max_quantity: Optional[int] = None
    time_constraint: Optional[datetime] = None
    executed_quantity: int = 0
    status: OrderState = OrderState.PENDING

@dataclass
class ManagedOrder:
    """Internally managed order with advanced features"""
    order_id: str
    original_request: OrderRequest
    signal: Optional[TradingSignal]
    position_size: Optional[PositionSize]
    
    # Order state
    status: OrderState
    created_time: datetime
    updated_time: datetime
    
    # Execution details
    schwab_order_id: Optional[str] = None
    filled_quantity: int = 0
    remaining_quantity: int = 0
    average_fill_price: float = 0.0
    executions: List[OrderExecution] = field(default_factory=list)
    
    # Advanced features
    priority: OrderPriority = OrderPriority.NORMAL
    execution_algorithm: ExecutionAlgorithm = ExecutionAlgorithm.SMART
    slices: List[OrderSlice] = field(default_factory=list)
    
    # Risk and monitoring
    risk_checks_passed: bool = False
    parent_order_id: Optional[str] = None
    child_order_ids: List[str] = field(default_factory=list)
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    notes: str = ""
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderState.PENDING, OrderState.QUEUED, OrderState.ROUTING, 
                              OrderState.WORKING, OrderState.PARTIALLY_FILLED]
    
    @property
    def fill_rate(self) -> float:
        if self.original_request.quantity == 0:
            return 0.0
        return self.filled_quantity / self.original_request.quantity

@dataclass
class ExecutionReport:
    """Order execution performance report"""
    order_id: str
    symbol: str
    total_quantity: int
    filled_quantity: int
    average_price: float
    benchmark_price: float  # VWAP, TWAP, etc.
    slippage: float
    implementation_shortfall: float
    execution_cost: float
    time_to_fill: float
    fill_rate: float
    execution_quality: str  # EXCELLENT, GOOD, FAIR, POOR

class OrderManager:
    """
    Advanced order management system with smart routing, algorithmic execution,
    and comprehensive order lifecycle management.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.trading_client = SchwabTradingClient()
        self.risk_manager = RiskManager()
        self.cache_manager = CacheManager()
        self.db_manager = DatabaseManager()
        self.notification_manager = NotificationManager()
        
        # Order management
        self.active_orders: Dict[str, ManagedOrder] = {}
        self.order_history: Dict[str, ManagedOrder] = {}
        self.execution_queue = asyncio.Queue(maxsize=1000)
        
        # Execution algorithms
        self.execution_algorithms: Dict[ExecutionAlgorithm, Callable] = {}
        
        # Order monitoring
        self.order_monitor_task = None
        self.is_monitoring = False
        
        # Execution statistics
        self.execution_stats = {
            'total_orders': 0,
            'successful_fills': 0,
            'partial_fills': 0,
            'cancellations': 0,
            'rejections': 0,
            'average_fill_time': 0.0,
            'average_slippage': 0.0
        }
        
        # Configuration
        self.config = {
            'max_order_size': 10000,  # Maximum single order size
            'slice_size': 100,  # Default slice size for large orders
            'max_slippage': 0.005,  # 0.5% maximum slippage
            'order_timeout': 3600,  # 1 hour order timeout
            'retry_attempts': 3,
            'monitoring_interval': 5  # seconds
        }
        
        logger.info("OrderManager initialized")
    
    async def initialize(self) -> bool:
        """Initialize the order manager"""
        try:
            # Initialize trading client
            await self.trading_client.initialize()
            
            # Initialize risk manager
            await self.risk_manager.initialize()
            
            # Setup execution algorithms
            self._setup_execution_algorithms()
            
            # Load active orders from database
            await self._load_active_orders()
            
            # Start order monitoring
            await self._start_order_monitoring()
            
            logger.info("OrderManager initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OrderManager: {str(e)}")
            return False
    
    def _setup_execution_algorithms(self):
        """Setup execution algorithm handlers"""
        self.execution_algorithms = {
            ExecutionAlgorithm.MARKET: self._execute_market_order,
            ExecutionAlgorithm.TWAP: self._execute_twap_order,
            ExecutionAlgorithm.VWAP: self._execute_vwap_order,
            ExecutionAlgorithm.ICEBERG: self._execute_iceberg_order,
            ExecutionAlgorithm.SMART: self._execute_smart_order
        }
    
    async def _load_active_orders(self):
        """Load active orders from database"""
        try:
            query = """
                SELECT order_id, order_data, status, created_time
                FROM managed_orders 
                WHERE status IN ('pending', 'queued', 'routing', 'working', 'partially_filled')
            """
            
            results = await self.db_manager.execute_query(query)
            
            for row in results:
                try:
                    # Deserialize order data
                    import json
                    order_data = json.loads(row['order_data'])
                    
                    # Reconstruct managed order (simplified)
                    # In production, use proper serialization
                    managed_order = ManagedOrder(
                        order_id=row['order_id'],
                        original_request=None,  # Would reconstruct from data
                        signal=None,
                        position_size=None,
                        status=OrderState(row['status']),
                        created_time=row['created_time'],
                        updated_time=datetime.now()
                    )
                    
                    self.active_orders[row['order_id']] = managed_order
                    
                except Exception as e:
                    logger.error(f"Error loading order {row['order_id']}: {str(e)}")
            
            logger.info(f"Loaded {len(self.active_orders)} active orders")
            
        except Exception as e:
            logger.error(f"Error loading active orders: {str(e)}")
    
    async def _start_order_monitoring(self):
        """Start order monitoring task"""
        try:
            if not self.is_monitoring:
                self.order_monitor_task = asyncio.create_task(self._order_monitoring_loop())
                self.is_monitoring = True
                logger.info("Order monitoring started")
                
        except Exception as e:
            logger.error(f"Error starting order monitoring: {str(e)}")
    
    async def _order_monitoring_loop(self):
        """Main order monitoring loop"""
        try:
            while self.is_monitoring:
                # Check active orders
                await self._check_active_orders()
                
                # Process execution queue
                await self._process_execution_queue()
                
                # Clean up old orders
                await self._cleanup_old_orders()
                
                # Update statistics
                await self._update_statistics()
                
                # Wait before next cycle
                await asyncio.sleep(self.config['monitoring_interval'])
                
        except Exception as e:
            logger.error(f"Error in order monitoring loop: {str(e)}")
        finally:
            self.is_monitoring = False
    
    async def place_order(self, signal: TradingSignal, position_size: PositionSize,
                         execution_algorithm: ExecutionAlgorithm = ExecutionAlgorithm.SMART,
                         priority: OrderPriority = OrderPriority.NORMAL,
                         tags: Optional[Dict[str, str]] = None) -> Optional[str]:
        """
        Place a trading order based on signal and position sizing
        
        Args:
            signal: Trading signal from AI model
            position_size: Position size from risk manager
            execution_algorithm: Execution algorithm to use
            priority: Order priority
            tags: Additional metadata tags
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Create order request
            order_request = self._create_order_request(signal, position_size)
            
            if not order_request:
                logger.error("Failed to create order request")
                return None
            
            # Validate order with risk manager
            is_valid, violations = await self.risk_manager.validate_trade(signal, position_size, {})
            
            if not is_valid:
                logger.warning(f"Order validation failed: {violations}")
                return None
            
            # Create managed order
            order_id = str(uuid.uuid4())
            managed_order = ManagedOrder(
                order_id=order_id,
                original_request=order_request,
                signal=signal,
                position_size=position_size,
                status=OrderState.PENDING,
                created_time=datetime.now(),
                updated_time=datetime.now(),
                priority=priority,
                execution_algorithm=execution_algorithm,
                remaining_quantity=order_request.quantity,
                risk_checks_passed=True,
                tags=tags or {},
                notes=f"Signal: {signal.signal_type.value}, Confidence: {signal.confidence:.3f}"
            )
            
            # Add to active orders
            self.active_orders[order_id] = managed_order
            
            # Store in database
            await self._store_managed_order(managed_order)
            
            # Queue for execution
            await self.execution_queue.put(order_id)
            
            logger.info(f"Order placed: {order_id} - {signal.signal_type.value} "
                       f"{position_size.recommended_shares} {signal.symbol}")
            
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None
    
    def _create_order_request(self, signal: TradingSignal, 
                            position_size: PositionSize) -> Optional[OrderRequest]:
        """Create Schwab API order request from signal and position size"""
        try:
            # Determine order instruction
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                instruction = OrderInstruction.BUY
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                instruction = OrderInstruction.SELL
            else:
                logger.warning(f"Cannot create order for signal type: {signal.signal_type}")
                return None
            
            # Determine order type and price
            if signal.confidence > 0.8:
                # High confidence - use market order
                order_type = OrderType.MARKET
                price = None
            else:
                # Lower confidence - use limit order
                order_type = OrderType.LIMIT
                # Set limit price slightly favorable to current price
                price_adjustment = 0.001 if instruction == OrderInstruction.BUY else -0.001
                price = signal.price * (1 + price_adjustment)
            
            order_request = OrderRequest(
                symbol=signal.symbol,
                quantity=position_size.recommended_shares,
                instruction=instruction,
                order_type=order_type,
                price=price,
                duration="DAY",
                session="NORMAL",
                asset_type=AssetType.EQUITY
            )
            
            return order_request
            
        except Exception as e:
            logger.error(f"Error creating order request: {str(e)}")
            return None
    
    async def _store_managed_order(self, managed_order: ManagedOrder):
        """Store managed order in database"""
        try:
            import json
            
            # Serialize order data
            order_data = {
                'original_request': {
                    'symbol': managed_order.original_request.symbol,
                    'quantity': managed_order.original_request.quantity,
                    'instruction': managed_order.original_request.instruction.value,
                    'order_type': managed_order.original_request.order_type.value,
                    'price': managed_order.original_request.price
                } if managed_order.original_request else {},
                'signal_data': {
                    'signal_type': managed_order.signal.signal_type.value,
                    'confidence': managed_order.signal.confidence,
                    'price': managed_order.signal.price
                } if managed_order.signal else {},
                'position_size_data': {
                    'recommended_shares': managed_order.position_size.recommended_shares,
                    'recommended_dollars': managed_order.position_size.recommended_dollars,
                    'risk_percentage': managed_order.position_size.risk_percentage
                } if managed_order.position_size else {},
                'execution_algorithm': managed_order.execution_algorithm.value,
                'priority': managed_order.priority.value,
                'tags': managed_order.tags,
                'notes': managed_order.notes
            }
            
            insert_query = """
                INSERT INTO managed_orders (order_id, symbol, quantity, instruction, order_type,
                                          status, created_time, updated_time, order_data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    status = VALUES(status),
                    updated_time = VALUES(updated_time),
                    order_data = VALUES(order_data)
            """
            
            values = (
                managed_order.order_id,
                managed_order.original_request.symbol if managed_order.original_request else '',
                managed_order.original_request.quantity if managed_order.original_request else 0,
                managed_order.original_request.instruction.value if managed_order.original_request else '',
                managed_order.original_request.order_type.value if managed_order.original_request else '',
                managed_order.status.value,
                managed_order.created_time,
                managed_order.updated_time,
                json.dumps(order_data)
            )
            
            await self.db_manager.execute_query(insert_query, values)
            
        except Exception as e:
            logger.error(f"Error storing managed order: {str(e)}")
    
    async def _check_active_orders(self):
        """Check status of active orders"""
        try:
            for order_id, managed_order in list(self.active_orders.items()):
                if managed_order.schwab_order_id:
                    # Check order status with Schwab
                    schwab_status = await self.trading_client.get_order_status(
                        managed_order.schwab_order_id
                    )
                    
                    if schwab_status:
                        await self._update_order_from_schwab_status(managed_order, schwab_status)
                
                # Check for timeouts
                if self._is_order_expired(managed_order):
                    await self._handle_expired_order(managed_order)
                
                # Update order in storage
                await self._store_managed_order(managed_order)
                
        except Exception as e:
            logger.error(f"Error checking active orders: {str(e)}")
    
    async def _update_order_from_schwab_status(self, managed_order: ManagedOrder, 
                                             schwab_status: OrderResponse):
        """Update managed order from Schwab API status"""
        try:
            old_status = managed_order.status
            
            # Map Schwab status to internal status
            status_mapping = {
                OrderStatus.PENDING: OrderState.PENDING,
                OrderStatus.QUEUED: OrderState.QUEUED,
                OrderStatus.WORKING: OrderState.WORKING,
                OrderStatus.FILLED: OrderState.FILLED,
                OrderStatus.CANCELED: OrderState.CANCELLED,
                OrderStatus.REJECTED: OrderState.REJECTED,
                OrderStatus.EXPIRED: OrderState.EXPIRED
            }
            
            new_status = status_mapping.get(schwab_status.status, OrderState.ERROR)
            
            # Update order details
            managed_order.status = new_status
            managed_order.filled_quantity = schwab_status.filled_quantity
            managed_order.remaining_quantity = schwab_status.remaining_quantity
            managed_order.updated_time = datetime.now()
            
            # Handle partial fills
            if schwab_status.filled_quantity > 0 and schwab_status.remaining_quantity > 0:
                managed_order.status = OrderState.PARTIALLY_FILLED
            
            # Update average fill price
            if schwab_status.filled_price and schwab_status.filled_quantity > 0:
                managed_order.average_fill_price = schwab_status.filled_price
            
            # Log status changes
            if old_status != new_status:
                logger.info(f"Order {managed_order.order_id} status changed: "
                           f"{old_status.value} -> {new_status.value}")
                
                # Send notifications for important status changes
                if new_status in [OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED]:
                    await self._send_order_notification(managed_order, new_status)
            
            # Move to history if no longer active
            if not managed_order.is_active:
                self.order_history[managed_order.order_id] = managed_order
                del self.active_orders[managed_order.order_id]
                
                # Generate execution report
                await self._generate_execution_report(managed_order)
            
        except Exception as e:
            logger.error(f"Error updating order from Schwab status: {str(e)}")
    
    def _is_order_expired(self, managed_order: ManagedOrder) -> bool:
        """Check if order has expired"""
        try:
            age = datetime.now() - managed_order.created_time
            return age.total_seconds() > self.config['order_timeout']
            
        except Exception as e:
            logger.error(f"Error checking order expiration: {str(e)}")
            return False
    
    async def _handle_expired_order(self, managed_order: ManagedOrder):
        """Handle expired order"""
        try:
            logger.warning(f"Order {managed_order.order_id} expired")
            
            # Cancel order if still active
            if managed_order.schwab_order_id and managed_order.is_active:
                await self.trading_client.cancel_order(managed_order.schwab_order_id)
            
            # Update status
            managed_order.status = OrderState.EXPIRED
            managed_order.updated_time = datetime.now()
            
            # Send notification
            await self._send_order_notification(managed_order, OrderState.EXPIRED)
            
        except Exception as e:
            logger.error(f"Error handling expired order: {str(e)}")
    
    async def _process_execution_queue(self):
        """Process orders in execution queue"""
        try:
            while not self.execution_queue.empty():
                try:
                    order_id = self.execution_queue.get_nowait()
                    
                    if order_id in self.active_orders:
                        managed_order = self.active_orders[order_id]
                        await self._execute_order(managed_order)
                    
                except asyncio.QueueEmpty:
                    break
                except Exception as e:
                    logger.error(f"Error processing order from queue: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error processing execution queue: {str(e)}")
    
    async def _execute_order(self, managed_order: ManagedOrder):
        """Execute a managed order using specified algorithm"""
        try:
            logger.info(f"Executing order {managed_order.order_id} using {managed_order.execution_algorithm.value}")
            
            # Update status
            managed_order.status = OrderState.ROUTING
            managed_order.updated_time = datetime.now()
            
            # Execute using specified algorithm
            execution_algorithm = self.execution_algorithms.get(managed_order.execution_algorithm)
            
            if execution_algorithm:
                await execution_algorithm(managed_order)
            else:
                logger.error(f"Unknown execution algorithm: {managed_order.execution_algorithm}")
                managed_order.status = OrderState.ERROR
            
        except Exception as e:
            logger.error(f"Error executing order {managed_order.order_id}: {str(e)}")
            managed_order.status = OrderState.ERROR
    
    async def _execute_market_order(self, managed_order: ManagedOrder):
        """Execute order as market order"""
        try:
            # Place market order with Schwab
            schwab_response = await self.trading_client.place_order(managed_order.original_request)
            
            if schwab_response:
                managed_order.schwab_order_id = schwab_response.order_id
                managed_order.status = OrderState.WORKING
                logger.info(f"Market order placed: {schwab_response.order_id}")
            else:
                managed_order.status = OrderState.REJECTED
                logger.error("Failed to place market order")
                
        except Exception as e:
            logger.error(f"Error executing market order: {str(e)}")
            managed_order.status = OrderState.ERROR
    
    async def _execute_smart_order(self, managed_order: ManagedOrder):
        """Execute order using smart algorithm"""
        try:
            # Smart algorithm logic:
            # 1. Check market conditions
            # 2. Determine best execution strategy
            # 3. Split large orders if needed
            # 4. Monitor and adjust
            
            quantity = managed_order.original_request.quantity
            
            # For large orders, use slicing
            if quantity > self.config['max_order_size']:
                await self._execute_with_slicing(managed_order)
            else:
                # Check current market conditions
                is_liquid = await self._check_market_liquidity(managed_order.original_request.symbol)
                
                if is_liquid:
                    # Use market order for liquid markets
                    await self._execute_market_order(managed_order)
                else:
                    # Use TWAP for illiquid markets
                    await self._execute_twap_order(managed_order)
                    
        except Exception as e:
            logger.error(f"Error executing smart order: {str(e)}")
            managed_order.status = OrderState.ERROR
    
    async def _execute_with_slicing(self, managed_order: ManagedOrder):
        """Execute large order with slicing"""
        try:
            total_quantity = managed_order.original_request.quantity
            slice_size = self.config['slice_size']
            
            # Create order slices
            slices = []
            remaining = total_quantity
            
            while remaining > 0:
                slice_quantity = min(slice_size, remaining)
                slice_id = str(uuid.uuid4())
                
                order_slice = OrderSlice(
                    slice_id=slice_id,
                    parent_order_id=managed_order.order_id,
                    quantity=slice_quantity,
                    target_price=managed_order.original_request.price
                )
                
                slices.append(order_slice)
                remaining -= slice_quantity
            
            managed_order.slices = slices
            
            # Execute slices sequentially with delays
            for slice_obj in slices:
                if managed_order.status != OrderState.ROUTING:
                    break
                
                await self._execute_slice(managed_order, slice_obj)
                
                # Wait between slices
                await asyncio.sleep(30)  # 30 second delay
            
        except Exception as e:
            logger.error(f"Error executing with slicing: {str(e)}")
            managed_order.status = OrderState.ERROR
    
    async def _execute_slice(self, managed_order: ManagedOrder, slice_obj: OrderSlice):
        """Execute individual slice"""
        try:
            # Create slice order request
            slice_request = OrderRequest(
                symbol=managed_order.original_request.symbol,
                quantity=slice_obj.quantity,
                instruction=managed_order.original_request.instruction,
                order_type=managed_order.original_request.order_type,
                price=slice_obj.target_price,
                duration="DAY",
                session="NORMAL"
            )
            
            # Place slice order
            schwab_response = await self.trading_client.place_order(slice_request)
            
            if schwab_response:
                slice_obj.status = OrderState.WORKING
                logger.info(f"Slice order placed: {schwab_response.order_id}")
                
                # Wait for fill (simplified)
                await asyncio.sleep(10)
                
                # Check status
                status = await self.trading_client.get_order_status(schwab_response.order_id)
                if status and status.status == OrderStatus.FILLED:
                    slice_obj.executed_quantity = slice_obj.quantity
                    managed_order.filled_quantity += slice_obj.quantity
                    
            if not schwab_response:
                slice_obj.status = OrderState.REJECTED
                
        except Exception as e:
            logger.error(f"Error executing slice: {str(e)}")
            slice_obj.status = OrderState.ERROR
    
    async def _execute_twap_order(self, managed_order: ManagedOrder):
        """Execute order using Time Weighted Average Price algorithm"""
        try:
            # TWAP implementation (simplified)
            total_quantity = managed_order.original_request.quantity
            duration_minutes = 30  # Execute over 30 minutes
            slice_count = 6  # 6 slices (5 minutes each)
            
            slice_size = total_quantity // slice_count
            
            for i in range(slice_count):
                if managed_order.status != OrderState.ROUTING:
                    break
                
                # Create slice
                slice_request = OrderRequest(
                    symbol=managed_order.original_request.symbol,
                    quantity=slice_size,
                    instruction=managed_order.original_request.instruction,
                    order_type=OrderType.MARKET,  # Use market for TWAP slices
                    duration="DAY",
                    session="NORMAL"
                )
                
                # Execute slice
                schwab_response = await self.trading_client.place_order(slice_request)
                
                if schwab_response:
                    logger.info(f"TWAP slice {i+1}/{slice_count} placed")
                    managed_order.filled_quantity += slice_size
                
                # Wait 5 minutes between slices
                if i < slice_count - 1:
                    await asyncio.sleep(300)
            
            managed_order.status = OrderState.FILLED if managed_order.fill_rate >= 0.95 else OrderState.PARTIALLY_FILLED
            
        except Exception as e:
            logger.error(f"Error executing TWAP order: {str(e)}")
            managed_order.status = OrderState.ERROR
    
    async def _execute_vwap_order(self, managed_order: ManagedOrder):
        """Execute order using Volume Weighted Average Price algorithm"""
        try:
            # VWAP implementation would analyze historical volume patterns
            # and execute slices based on volume distribution
            # For now, fall back to TWAP
            await self._execute_twap_order(managed_order)
            
        except Exception as e:
            logger.error(f"Error executing VWAP order: {str(e)}")
            managed_order.status = OrderState.ERROR
    
    async def _execute_iceberg_order(self, managed_order: ManagedOrder):
        """Execute iceberg order (show small portions)"""
        try:
            total_quantity = managed_order.original_request.quantity
            visible_size = min(100, total_quantity // 10)  # Show 10% or max 100 shares
            
            remaining = total_quantity
            
            while remaining > 0 and managed_order.status == OrderState.ROUTING:
                current_size = min(visible_size, remaining)
                
                # Place visible portion
                slice_request = OrderRequest(
                    symbol=managed_order.original_request.symbol,
                    quantity=current_size,
                    instruction=managed_order.original_request.instruction,
                    order_type=managed_order.original_request.order_type,
                    price=managed_order.original_request.price,
                    duration="DAY",
                    session="NORMAL"
                )
                
                schwab_response = await self.trading_client.place_order(slice_request)
                
                if schwab_response:
                    # Wait for fill
                    await asyncio.sleep(60)  # 1 minute
                    
                    # Check status
                    status = await self.trading_client.get_order_status(schwab_response.order_id)
                    if status and status.filled_quantity > 0:
                        managed_order.filled_quantity += status.filled_quantity
                        remaining -= status.filled_quantity
                        
                        if status.status != OrderStatus.FILLED:
                            # Cancel unfilled portion
                            await self.trading_client.cancel_order(schwab_response.order_id)
                    
                else:
                    break
            
            managed_order.status = OrderState.FILLED if remaining == 0 else OrderState.PARTIALLY_FILLED
            
        except Exception as e:
            logger.error(f"Error executing iceberg order: {str(e)}")
            managed_order.status = OrderState.ERROR
    
    async def _check_market_liquidity(self, symbol: str) -> bool:
        """Check if market is liquid for given symbol"""
        try:
            # Get recent volume data
            # This would analyze bid/ask spreads, volume, etc.
            # For now, assume liquid for major symbols
            major_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            return symbol in major_symbols
            
        except Exception as e:
            logger.error(f"Error checking market liquidity: {str(e)}")
            return False
    
    async def cancel_order(self, order_id: str, reason: str = "User request") -> bool:
        """Cancel an active order"""
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Order {order_id} not found in active orders")
                return False
            
            managed_order = self.active_orders[order_id]
            
            # Cancel with Schwab if placed
            if managed_order.schwab_order_id:
                success = await self.trading_client.cancel_order(managed_order.schwab_order_id)
                if not success:
                    logger.error(f"Failed to cancel order with Schwab: {managed_order.schwab_order_id}")
                    return False
            
            # Update order status
            managed_order.status = OrderState.CANCELLED
            managed_order.updated_time = datetime.now()
            managed_order.notes += f" | Cancelled: {reason}"
            
            # Move to history
            self.order_history[order_id] = managed_order
            del self.active_orders[order_id]
            
            # Store update
            await self._store_managed_order(managed_order)
            
            # Send notification
            await self._send_order_notification(managed_order, OrderState.CANCELLED)
            
            logger.info(f"Order cancelled: {order_id} - {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False
    
    async def _send_order_notification(self, managed_order: ManagedOrder, status: OrderState):
        """Send order status notification"""
        try:
            if managed_order.original_request:
                message = (
                    f"Order {managed_order.order_id[:8]}... {status.value.upper()}\n"
                    f"Symbol: {managed_order.original_request.symbol}\n"
                    f"Quantity: {managed_order.filled_quantity}/{managed_order.original_request.quantity}\n"
                    f"Average Price: ${managed_order.average_fill_price:.2f}"
                )
                
                await self.notification_manager.send_notification(
                    title=f"Order {status.value.title()}",
                    message=message,
                    priority="HIGH" if status in [OrderState.REJECTED, OrderState.ERROR] else "NORMAL"
                )
                
        except Exception as e:
            logger.error(f"Error sending order notification: {str(e)}")
    
    async def _generate_execution_report(self, managed_order: ManagedOrder):
        """Generate execution report for completed order"""
        try:
            if not managed_order.original_request:
                return
            
            # Calculate execution metrics
            total_quantity = managed_order.original_request.quantity
            filled_quantity = managed_order.filled_quantity
            avg_price = managed_order.average_fill_price
            
            # Get benchmark price (simplified)
            benchmark_price = managed_order.signal.price if managed_order.signal else avg_price
            
            # Calculate slippage
            if managed_order.original_request.instruction == OrderInstruction.BUY:
                slippage = (avg_price - benchmark_price) / benchmark_price if benchmark_price > 0 else 0
            else:
                slippage = (benchmark_price - avg_price) / benchmark_price if benchmark_price > 0 else 0
            
            # Calculate execution time
            execution_time = (managed_order.updated_time - managed_order.created_time).total_seconds()
            
            # Determine execution quality
            if abs(slippage) < 0.001:  # < 0.1%
                quality = "EXCELLENT"
            elif abs(slippage) < 0.003:  # < 0.3%
                quality = "GOOD"
            elif abs(slippage) < 0.005:  # < 0.5%
                quality = "FAIR"
            else:
                quality = "POOR"
            
            execution_report = ExecutionReport(
                order_id=managed_order.order_id,
                symbol=managed_order.original_request.symbol,
                total_quantity=total_quantity,
                filled_quantity=filled_quantity,
                average_price=avg_price,
                benchmark_price=benchmark_price,
                slippage=slippage,
                implementation_shortfall=abs(slippage) * filled_quantity * avg_price,
                execution_cost=0.0,  # Would calculate commissions/fees
                time_to_fill=execution_time,
                fill_rate=managed_order.fill_rate,
                execution_quality=quality
            )
            
            # Store execution report
            await self._store_execution_report(execution_report)
            
            logger.info(f"Execution report generated for {managed_order.order_id}: "
                       f"Fill rate {execution_report.fill_rate:.1%}, "
                       f"Slippage {execution_report.slippage:.3%}, "
                       f"Quality {execution_report.execution_quality}")
            
        except Exception as e:
            logger.error(f"Error generating execution report: {str(e)}")
    
    async def _store_execution_report(self, report: ExecutionReport):
        """Store execution report in database"""
        try:
            insert_query = """
                INSERT INTO execution_reports (order_id, symbol, total_quantity, filled_quantity,
                                             average_price, benchmark_price, slippage,
                                             implementation_shortfall, execution_cost, time_to_fill,
                                             fill_rate, execution_quality, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                report.order_id, report.symbol, report.total_quantity, report.filled_quantity,
                report.average_price, report.benchmark_price, report.slippage,
                report.implementation_shortfall, report.execution_cost, report.time_to_fill,
                report.fill_rate, report.execution_quality, datetime.now()
            )
            
            await self.db_manager.execute_query(insert_query, values)
            
        except Exception as e:
            logger.error(f"Error storing execution report: {str(e)}")
    
    async def _cleanup_old_orders(self):
        """Clean up old completed orders from memory"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            old_orders = [
                order_id for order_id, order in self.order_history.items()
                if order.updated_time < cutoff_time
            ]
            
            for order_id in old_orders:
                del self.order_history[order_id]
            
            if old_orders:
                logger.info(f"Cleaned up {len(old_orders)} old orders from memory")
                
        except Exception as e:
            logger.error(f"Error cleaning up old orders: {str(e)}")
    
    async def _update_statistics(self):
        """Update execution statistics"""
        try:
            # Query recent execution statistics
            query = """
                SELECT 
                    COUNT(*) as total_orders,
                    SUM(CASE WHEN fill_rate >= 1.0 THEN 1 ELSE 0 END) as successful_fills,
                    SUM(CASE WHEN fill_rate > 0 AND fill_rate < 1.0 THEN 1 ELSE 0 END) as partial_fills,
                    AVG(time_to_fill) as avg_fill_time,
                    AVG(ABS(slippage)) as avg_slippage
                FROM execution_reports 
                WHERE created_at >= %s
            """
            
            cutoff_time = datetime.now() - timedelta(days=1)
            results = await self.db_manager.execute_query(query, (cutoff_time,))
            
            if results and results[0]['total_orders']:
                stats = results[0]
                self.execution_stats.update({
                    'total_orders': stats['total_orders'],
                    'successful_fills': stats['successful_fills'],
                    'partial_fills': stats['partial_fills'],
                    'average_fill_time': stats['avg_fill_time'] or 0.0,
                    'average_slippage': stats['avg_slippage'] or 0.0
                })
                
        except Exception as e:
            logger.error(f"Error updating statistics: {str(e)}")
    
    def get_order_status(self, order_id: str) -> Optional[ManagedOrder]:
        """Get status of a managed order"""
        try:
            if order_id in self.active_orders:
                return self.active_orders[order_id]
            elif order_id in self.order_history:
                return self.order_history[order_id]
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting order status: {str(e)}")
            return None
    
    def get_active_orders(self) -> List[ManagedOrder]:
        """Get all active orders"""
        return list(self.active_orders.values())
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        stats = self.execution_stats.copy()
        stats.update({
            'active_orders': len(self.active_orders),
            'orders_in_history': len(self.order_history),
            'queue_size': self.execution_queue.qsize(),
            'monitoring_active': self.is_monitoring
        })
        return stats
    
    async def stop_monitoring(self):
        """Stop order monitoring"""
        try:
            self.is_monitoring = False
            
            if self.order_monitor_task:
                self.order_monitor_task.cancel()
                try:
                    await self.order_monitor_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Order monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")
    
    async def cleanup(self):
        """Cleanup order manager resources"""
        try:
            # Stop monitoring
            await self.stop_monitoring()
            
            # Cancel all active orders
            for order_id in list(self.active_orders.keys()):
                await self.cancel_order(order_id, "System shutdown")
            
            # Close trading client
            await self.trading_client.close()
            
            logger.info("OrderManager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
