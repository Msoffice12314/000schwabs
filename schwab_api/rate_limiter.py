"""
Schwab AI Trading System - API Rate Limiter
Advanced rate limiting system for Schwab API with adaptive throttling, priority queues, and burst handling.
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque, defaultdict
import heapq

from utils.cache_manager import CacheManager
from config.settings import get_settings

logger = logging.getLogger(__name__)

class RequestPriority(Enum):
    """Request priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class ApiEndpoint(Enum):
    """Schwab API endpoint categories"""
    MARKET_DATA = "market_data"
    TRADING = "trading" 
    ACCOUNT = "account"
    ORDERS = "orders"
    STREAMING = "streaming"
    AUTHENTICATION = "authentication"

@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests_per_second: float
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_size: int = 10
    window_size: int = 60  # seconds

@dataclass
class RequestToken:
    """Request token for rate limiting"""
    request_id: str
    endpoint: ApiEndpoint
    priority: RequestPriority
    timestamp: float
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        # Higher priority values come first (reverse order)
        return self.priority.value > other.priority.value

@dataclass
class RateMetrics:
    """Rate limiting metrics"""
    total_requests: int = 0
    requests_allowed: int = 0
    requests_throttled: int = 0
    requests_rejected: int = 0
    average_wait_time: float = 0.0
    current_rate: float = 0.0
    queue_size: int = 0
    burst_usage: int = 0

class TokenBucket:
    """Token bucket algorithm implementation"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens from bucket"""
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                return False
    
    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        self.last_refill = now
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
    
    def get_tokens(self) -> int:
        """Get current token count"""
        with self.lock:
            self._refill()
            return int(self.tokens)
    
    def time_until_token(self) -> float:
        """Time until next token is available"""
        with self.lock:
            self._refill()
            if self.tokens >= 1:
                return 0.0
            else:
                return (1.0 - self.tokens) / self.refill_rate

class SlidingWindowCounter:
    """Sliding window rate counter"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size  # seconds
        self.max_requests = max_requests
        self.requests = deque()
        self.lock = threading.Lock()
    
    def add_request(self, timestamp: float = None) -> bool:
        """Add request to window, return True if within limit"""
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            # Remove old requests outside window
            cutoff = timestamp - self.window_size
            while self.requests and self.requests[0] <= cutoff:
                self.requests.popleft()
            
            # Check if we can add new request
            if len(self.requests) < self.max_requests:
                self.requests.append(timestamp)
                return True
            else:
                return False
    
    def get_count(self) -> int:
        """Get current request count in window"""
        with self.lock:
            now = time.time()
            cutoff = now - self.window_size
            while self.requests and self.requests[0] <= cutoff:
                self.requests.popleft()
            return len(self.requests)
    
    def time_until_available(self) -> float:
        """Time until oldest request expires from window"""
        with self.lock:
            if len(self.requests) < self.max_requests:
                return 0.0
            else:
                oldest_request = self.requests[0]
                return max(0.0, oldest_request + self.window_size - time.time())

class RateLimiter:
    """
    Advanced rate limiter for Schwab API with multiple algorithms,
    priority queues, and adaptive throttling.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache_manager = CacheManager()
        
        # Rate limit configurations by endpoint
        self.rate_limits = {
            ApiEndpoint.MARKET_DATA: RateLimit(
                requests_per_second=5.0,
                requests_per_minute=120,
                requests_per_hour=1000,
                requests_per_day=10000,
                burst_size=10
            ),
            ApiEndpoint.TRADING: RateLimit(
                requests_per_second=2.0,
                requests_per_minute=60,
                requests_per_hour=500,
                requests_per_day=2000,
                burst_size=5
            ),
            ApiEndpoint.ACCOUNT: RateLimit(
                requests_per_second=1.0,
                requests_per_minute=30,
                requests_per_hour=200,
                requests_per_day=1000,
                burst_size=3
            ),
            ApiEndpoint.ORDERS: RateLimit(
                requests_per_second=1.0,
                requests_per_minute=30,
                requests_per_hour=200,
                requests_per_day=1000,
                burst_size=3
            ),
            ApiEndpoint.STREAMING: RateLimit(
                requests_per_second=0.1,
                requests_per_minute=5,
                requests_per_hour=50,
                requests_per_day=200,
                burst_size=1
            ),
            ApiEndpoint.AUTHENTICATION: RateLimit(
                requests_per_second=0.1,
                requests_per_minute=5,
                requests_per_hour=20,
                requests_per_day=50,
                burst_size=1
            )
        }
        
        # Token buckets for burst handling
        self.token_buckets = {}
        
        # Sliding window counters for different time periods
        self.sliding_windows = {}
        
        # Priority queue for pending requests
        self.request_queue = []
        self.queue_lock = asyncio.Lock()
        
        # Request processing
        self.processing_task = None
        self.is_processing = False
        
        # Metrics tracking
        self.metrics = defaultdict(lambda: RateMetrics())
        self.request_history = deque(maxlen=10000)
        
        # Adaptive throttling
        self.adaptive_throttling = True
        self.throttle_factor = 1.0
        self.error_count = 0
        self.success_count = 0
        
        # Initialize rate limiters
        self._initialize_limiters()
        
        logger.info("RateLimiter initialized")
    
    def _initialize_limiters(self):
        """Initialize token buckets and sliding windows"""
        for endpoint, limits in self.rate_limits.items():
            # Token bucket for burst handling
            self.token_buckets[endpoint] = TokenBucket(
                capacity=limits.burst_size,
                refill_rate=limits.requests_per_second
            )
            
            # Sliding windows for different time periods
            self.sliding_windows[endpoint] = {
                'minute': SlidingWindowCounter(60, limits.requests_per_minute),
                'hour': SlidingWindowCounter(3600, limits.requests_per_hour),
                'day': SlidingWindowCounter(86400, limits.requests_per_day)
            }
    
    async def acquire(self, endpoint: ApiEndpoint = ApiEndpoint.MARKET_DATA,
                     priority: RequestPriority = RequestPriority.NORMAL,
                     request_id: Optional[str] = None) -> bool:
        """
        Acquire permission to make API request
        
        Args:
            endpoint: API endpoint category
            priority: Request priority
            request_id: Optional request identifier
            
        Returns:
            True if request is allowed immediately, False if queued
        """
        try:
            if request_id is None:
                request_id = f"req_{int(time.time() * 1000000)}"
            
            # Check if request can be processed immediately
            if await self._can_process_immediately(endpoint, priority):
                await self._process_request(endpoint, request_id)
                return True
            else:
                # Queue the request
                await self._queue_request(endpoint, priority, request_id)
                return False
                
        except Exception as e:
            logger.error(f"Error acquiring rate limit: {str(e)}")
            return False
    
    async def _can_process_immediately(self, endpoint: ApiEndpoint, 
                                     priority: RequestPriority) -> bool:
        """Check if request can be processed immediately"""
        try:
            # Apply adaptive throttling
            if self.adaptive_throttling and self.throttle_factor < 1.0:
                # Reduce effective rate limits
                if not self._check_throttled_limits(endpoint):
                    return False
            
            # Check token bucket (burst handling)
            token_bucket = self.token_buckets[endpoint]
            if not token_bucket.consume(1):
                return False
            
            # Check sliding windows
            windows = self.sliding_windows[endpoint]
            now = time.time()
            
            for window_name, window in windows.items():
                if not window.add_request(now):
                    # Refund token since we couldn't proceed
                    token_bucket.tokens = min(token_bucket.capacity, token_bucket.tokens + 1)
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking immediate processing: {str(e)}")
            return False
    
    def _check_throttled_limits(self, endpoint: ApiEndpoint) -> bool:
        """Check limits with adaptive throttling applied"""
        try:
            limits = self.rate_limits[endpoint]
            windows = self.sliding_windows[endpoint]
            
            # Apply throttling factor to limits
            throttled_limits = {
                'minute': int(limits.requests_per_minute * self.throttle_factor),
                'hour': int(limits.requests_per_hour * self.throttle_factor),
                'day': int(limits.requests_per_day * self.throttle_factor)
            }
            
            # Check against throttled limits
            for window_name, window in windows.items():
                if window.get_count() >= throttled_limits[window_name]:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking throttled limits: {str(e)}")
            return True
    
    async def _queue_request(self, endpoint: ApiEndpoint, priority: RequestPriority,
                           request_id: str):
        """Queue request for later processing"""
        try:
            request_token = RequestToken(
                request_id=request_id,
                endpoint=endpoint,
                priority=priority,
                timestamp=time.time()
            )
            
            async with self.queue_lock:
                heapq.heappush(self.request_queue, request_token)
            
            # Start processing if not already running
            if not self.is_processing:
                await self._start_request_processing()
            
            # Update metrics
            self.metrics[endpoint].total_requests += 1
            self.metrics[endpoint].queue_size = len(self.request_queue)
            
        except Exception as e:
            logger.error(f"Error queuing request: {str(e)}")
    
    async def _start_request_processing(self):
        """Start background request processing"""
        try:
            if not self.is_processing:
                self.is_processing = True
                self.processing_task = asyncio.create_task(self._process_queue())
                
        except Exception as e:
            logger.error(f"Error starting request processing: {str(e)}")
    
    async def _process_queue(self):
        """Process queued requests"""
        try:
            while self.is_processing:
                async with self.queue_lock:
                    if not self.request_queue:
                        # No requests to process, stop processing
                        self.is_processing = False
                        break
                    
                    # Get highest priority request
                    request_token = heapq.heappop(self.request_queue)
                
                # Try to process the request
                if await self._can_process_immediately(request_token.endpoint, request_token.priority):
                    await self._process_request(request_token.endpoint, request_token.request_id)
                    
                    # Execute callback if provided
                    if request_token.callback:
                        try:
                            await request_token.callback()
                        except Exception as e:
                            logger.error(f"Error executing request callback: {str(e)}")
                else:
                    # Put request back in queue
                    async with self.queue_lock:
                        heapq.heappush(self.request_queue, request_token)
                    
                    # Wait before trying again
                    await asyncio.sleep(0.1)
                
                # Update queue size metric
                for endpoint in ApiEndpoint:
                    self.metrics[endpoint].queue_size = len(self.request_queue)
                    
        except Exception as e:
            logger.error(f"Error processing queue: {str(e)}")
        finally:
            self.is_processing = False
    
    async def _process_request(self, endpoint: ApiEndpoint, request_id: str):
        """Process an approved request"""
        try:
            now = time.time()
            
            # Record request
            self.request_history.append({
                'endpoint': endpoint.value,
                'request_id': request_id,
                'timestamp': now,
                'success': True
            })
            
            # Update metrics
            metrics = self.metrics[endpoint]
            metrics.requests_allowed += 1
            metrics.current_rate = self._calculate_current_rate(endpoint)
            metrics.burst_usage = self.rate_limits[endpoint].burst_size - self.token_buckets[endpoint].get_tokens()
            
            # Update adaptive throttling
            self.success_count += 1
            await self._update_adaptive_throttling()
            
            logger.debug(f"Request processed: {request_id} for {endpoint.value}")
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
    
    def _calculate_current_rate(self, endpoint: ApiEndpoint) -> float:
        """Calculate current request rate for endpoint"""
        try:
            # Get requests in last minute
            now = time.time()
            cutoff = now - 60
            
            recent_requests = [
                req for req in self.request_history
                if req['endpoint'] == endpoint.value and req['timestamp'] > cutoff
            ]
            
            return len(recent_requests) / 60.0  # requests per second
            
        except Exception as e:
            logger.error(f"Error calculating current rate: {str(e)}")
            return 0.0
    
    async def _update_adaptive_throttling(self):
        """Update adaptive throttling based on success/error rates"""
        try:
            if not self.adaptive_throttling:
                return
            
            total_requests = self.success_count + self.error_count
            
            if total_requests >= 100:  # Update every 100 requests
                error_rate = self.error_count / total_requests
                
                if error_rate > 0.1:  # > 10% error rate
                    # Increase throttling
                    self.throttle_factor = max(0.5, self.throttle_factor * 0.9)
                    logger.warning(f"Increased throttling due to high error rate: {error_rate:.2%}")
                    
                elif error_rate < 0.02:  # < 2% error rate
                    # Decrease throttling
                    self.throttle_factor = min(1.0, self.throttle_factor * 1.1)
                    if self.throttle_factor > 0.95:
                        logger.info("Throttling reduced due to low error rate")
                
                # Reset counters
                self.success_count = 0
                self.error_count = 0
                
        except Exception as e:
            logger.error(f"Error updating adaptive throttling: {str(e)}")
    
    async def report_error(self, endpoint: ApiEndpoint, error_code: Optional[int] = None):
        """Report API error for adaptive throttling"""
        try:
            self.error_count += 1
            
            # Update metrics
            self.metrics[endpoint].requests_rejected += 1
            
            # Handle specific error codes
            if error_code == 429:  # Too Many Requests
                # Immediately increase throttling
                self.throttle_factor = max(0.3, self.throttle_factor * 0.7)
                logger.warning(f"Rate limit exceeded (429), throttling increased: {self.throttle_factor:.2f}")
                
                # Clear some request history to reduce apparent rate
                while len(self.request_history) > 5000:
                    self.request_history.popleft()
            
            elif error_code in [500, 502, 503, 504]:  # Server errors
                # Moderate throttling increase
                self.throttle_factor = max(0.6, self.throttle_factor * 0.85)
                logger.warning(f"Server error ({error_code}), throttling increased: {self.throttle_factor:.2f}")
            
            await self._update_adaptive_throttling()
            
        except Exception as e:
            logger.error(f"Error reporting API error: {str(e)}")
    
    def get_wait_time(self, endpoint: ApiEndpoint) -> float:
        """Get estimated wait time for endpoint"""
        try:
            # Check token bucket wait time
            token_bucket = self.token_buckets[endpoint]
            token_wait = token_bucket.time_until_token()
            
            # Check sliding window wait times
            windows = self.sliding_windows[endpoint]
            window_waits = [window.time_until_available() for window in windows.values()]
            
            # Return maximum wait time
            return max(token_wait, max(window_waits) if window_waits else 0.0)
            
        except Exception as e:
            logger.error(f"Error getting wait time: {str(e)}")
            return 1.0  # Default 1 second wait
    
    def get_remaining_requests(self, endpoint: ApiEndpoint) -> Dict[str, int]:
        """Get remaining requests for different time windows"""
        try:
            limits = self.rate_limits[endpoint]
            windows = self.sliding_windows[endpoint]
            
            return {
                'per_minute': limits.requests_per_minute - windows['minute'].get_count(),
                'per_hour': limits.requests_per_hour - windows['hour'].get_count(),
                'per_day': limits.requests_per_day - windows['day'].get_count(),
                'burst_tokens': self.token_buckets[endpoint].get_tokens()
            }
            
        except Exception as e:
            logger.error(f"Error getting remaining requests: {str(e)}")
            return {'per_minute': 0, 'per_hour': 0, 'per_day': 0, 'burst_tokens': 0}
    
    def get_metrics(self, endpoint: Optional[ApiEndpoint] = None) -> Dict[str, Any]:
        """Get rate limiting metrics"""
        try:
            if endpoint:
                # Return metrics for specific endpoint
                metrics = self.metrics[endpoint]
                remaining = self.get_remaining_requests(endpoint)
                
                return {
                    'endpoint': endpoint.value,
                    'total_requests': metrics.total_requests,
                    'requests_allowed': metrics.requests_allowed,
                    'requests_throttled': metrics.requests_throttled,
                    'requests_rejected': metrics.requests_rejected,
                    'current_rate': metrics.current_rate,
                    'queue_size': metrics.queue_size,
                    'burst_usage': metrics.burst_usage,
                    'wait_time': self.get_wait_time(endpoint),
                    'remaining_requests': remaining,
                    'throttle_factor': self.throttle_factor
                }
            else:
                # Return aggregate metrics
                total_metrics = {
                    'total_requests': sum(m.total_requests for m in self.metrics.values()),
                    'requests_allowed': sum(m.requests_allowed for m in self.metrics.values()),
                    'requests_throttled': sum(m.requests_throttled for m in self.metrics.values()),
                    'requests_rejected': sum(m.requests_rejected for m in self.metrics.values()),
                    'queue_size': len(self.request_queue),
                    'throttle_factor': self.throttle_factor,
                    'adaptive_throttling': self.adaptive_throttling,
                    'is_processing': self.is_processing,
                    'by_endpoint': {}
                }
                
                for ep in ApiEndpoint:
                    total_metrics['by_endpoint'][ep.value] = self.get_metrics(ep)
                
                return total_metrics
                
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return {}
    
    def reset_throttling(self):
        """Reset adaptive throttling to normal levels"""
        try:
            self.throttle_factor = 1.0
            self.error_count = 0
            self.success_count = 0
            
            logger.info("Rate limiter throttling reset to normal levels")
            
        except Exception as e:
            logger.error(f"Error resetting throttling: {str(e)}")
    
    def set_throttling(self, factor: float):
        """Manually set throttling factor"""
        try:
            self.throttle_factor = max(0.1, min(1.0, factor))
            logger.info(f"Throttling factor manually set to: {self.throttle_factor:.2f}")
            
        except Exception as e:
            logger.error(f"Error setting throttling: {str(e)}")
    
    async def wait_for_capacity(self, endpoint: ApiEndpoint, 
                              timeout: float = 60.0) -> bool:
        """Wait until capacity is available for endpoint"""
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if await self._can_process_immediately(endpoint, RequestPriority.NORMAL):
                    return True
                
                wait_time = min(self.get_wait_time(endpoint), 1.0)
                await asyncio.sleep(wait_time)
            
            logger.warning(f"Timeout waiting for capacity on {endpoint.value}")
            return False
            
        except Exception as e:
            logger.error(f"Error waiting for capacity: {str(e)}")
            return False
    
    def clear_queue(self, endpoint: Optional[ApiEndpoint] = None):
        """Clear request queue for endpoint or all endpoints"""
        try:
            if endpoint:
                # Remove requests for specific endpoint
                new_queue = [req for req in self.request_queue if req.endpoint != endpoint]
                self.request_queue = new_queue
                heapq.heapify(self.request_queue)
                logger.info(f"Cleared queue for {endpoint.value}")
            else:
                # Clear entire queue
                self.request_queue.clear()
                logger.info("Cleared entire request queue")
                
        except Exception as e:
            logger.error(f"Error clearing queue: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on rate limiter"""
        try:
            health_status = {
                'healthy': True,
                'issues': [],
                'queue_size': len(self.request_queue),
                'processing_active': self.is_processing,
                'throttle_factor': self.throttle_factor,
                'endpoints': {}
            }
            
            # Check each endpoint
            for endpoint in ApiEndpoint:
                endpoint_health = {
                    'healthy': True,
                    'current_rate': self._calculate_current_rate(endpoint),
                    'wait_time': self.get_wait_time(endpoint),
                    'remaining_requests': self.get_remaining_requests(endpoint)
                }
                
                # Check for issues
                if endpoint_health['wait_time'] > 30:
                    endpoint_health['healthy'] = False
                    health_status['issues'].append(f"High wait time for {endpoint.value}: {endpoint_health['wait_time']:.1f}s")
                
                if endpoint_health['remaining_requests']['per_minute'] < 5:
                    endpoint_health['healthy'] = False
                    health_status['issues'].append(f"Low remaining requests for {endpoint.value}")
                
                health_status['endpoints'][endpoint.value] = endpoint_health
                
                if not endpoint_health['healthy']:
                    health_status['healthy'] = False
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error in health check: {str(e)}")
            return {'healthy': False, 'error': str(e)}
    
    async def stop(self):
        """Stop rate limiter processing"""
        try:
            self.is_processing = False
            
            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Rate limiter stopped")
            
        except Exception as e:
            logger.error(f"Error stopping rate limiter: {str(e)}")
