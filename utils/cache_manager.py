import redis
import json
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Set, Callable
import asyncio
import threading
import time
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
import gzip
import base64
from enum import Enum
import warnings

class CacheLevel(Enum):
    """Cache storage levels"""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"
    MULTI_LEVEL = "multi_level"

class SerializationMethod(Enum):
    """Serialization methods for cached data"""
    JSON = "json"
    PICKLE = "pickle"
    COMPRESSED_PICKLE = "compressed_pickle"
    STRING = "string"

@dataclass
class CacheEntry:
    """Individual cache entry with metadata"""
    key: str
    value: Any
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    tags: Set[str] = field(default_factory=set)
    size_bytes: int = 0

@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    memory_usage: int = 0
    redis_usage: int = 0

class MemoryCache:
    """In-memory LRU cache implementation"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_memory = 0
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check expiration
                if entry.expires_at and datetime.now() > entry.expires_at:
                    self._remove_entry(key)
                    return None
                
                # Update access info
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                
                return entry.value
            
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: Set[str] = None) -> bool:
        """Set value in memory cache"""
        with self.lock:
            # Calculate size
            size = self._calculate_size(value)
            
            # Check if we need to make room
            self._make_room(size)
            
            # Create entry
            expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None
            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
                tags=tags or set(),
                size_bytes=size
            )
            
            # Remove old entry if exists
            if key in self.cache:
                self._remove_entry(key)
            
            # Add new entry
            self.cache[key] = entry
            self.current_memory += size
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from memory cache"""
        with self.lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Clear all entries from memory cache"""
        with self.lock:
            self.cache.clear()
            self.current_memory = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory cache statistics"""
        with self.lock:
            return {
                'entries': len(self.cache),
                'memory_usage': self.current_memory,
                'max_memory': self.max_memory_bytes,
                'memory_utilization': self.current_memory / self.max_memory_bytes,
                'max_size': self.max_size
            }
    
    def _remove_entry(self, key: str):
        """Remove entry and update memory usage"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_memory -= entry.size_bytes
    
    def _make_room(self, needed_size: int):
        """Make room for new entry by evicting old ones"""
        while (len(self.cache) >= self.max_size or 
               self.current_memory + needed_size > self.max_memory_bytes):
            if not self.cache:
                break
            
            # Remove least recently used entry
            oldest_key = next(iter(self.cache))
            self._remove_entry(oldest_key)
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate memory size of value"""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float, bool)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) 
                          for k, v in value.items())
            else:
                # Fallback to pickle size
                return len(pickle.dumps(value))
        except:
            return 1024  # Default estimate

class CacheManager:
    """Advanced multi-level cache manager with Redis support"""
    
    def __init__(self, 
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 redis_password: Optional[str] = None,
                 default_ttl: int = 3600,
                 memory_cache_size: int = 1000,
                 memory_cache_mb: int = 100,
                 enable_compression: bool = True):
        
        self.logger = logging.getLogger(__name__)
        self.default_ttl = default_ttl
        self.enable_compression = enable_compression
        
        # Initialize memory cache
        self.memory_cache = MemoryCache(memory_cache_size, memory_cache_mb)
        
        # Initialize Redis connection
        self.redis_client = None
        self.redis_available = False
        
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=False,  # We handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            self.redis_available = True
            self.logger.info("Redis cache connected successfully")
            
        except Exception as e:
            self.logger.warning(f"Redis not available, using memory cache only: {e}")
            self.redis_available = False
        
        # Statistics
        self.stats = CacheStats()
        self.stats_lock = threading.Lock()
        
        # Background cleanup
        self.cleanup_thread = None
        self.stop_cleanup = threading.Event()
        self.start_background_cleanup()
        
        # Cache patterns and prefixes
        self.key_prefixes = {
            'quotes': 'quote:',
            'market_data': 'market:',
            'analysis': 'analysis:',
            'user_data': 'user:',
            'ai_predictions': 'ai:',
            'portfolio': 'portfolio:',
            'temporary': 'temp:'
        }
    
    def start_background_cleanup(self):
        """Start background thread for cache cleanup"""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.stop_cleanup.clear()
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True
            )
            self.cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Background loop for cache maintenance"""
        while not self.stop_cleanup.is_set():
            try:
                self._cleanup_expired_entries()
                time.sleep(300)  # Run every 5 minutes
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                time.sleep(60)
    
    def _cleanup_expired_entries(self):
        """Clean up expired entries from all cache levels"""
        try:
            # Clean memory cache
            with self.memory_cache.lock:
                expired_keys = []
                for key, entry in self.memory_cache.cache.items():
                    if entry.expires_at and datetime.now() > entry.expires_at:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    self.memory_cache._remove_entry(key)
                    with self.stats_lock:
                        self.stats.evictions += 1
            
            # Clean Redis cache if available
            if self.redis_available:
                # Redis handles TTL automatically, but we can clean up manually tagged entries
                pass
                
        except Exception as e:
            self.logger.error(f"Error cleaning expired entries: {e}")
    
    def get(self, key: str, level: CacheLevel = CacheLevel.MULTI_LEVEL) -> Optional[Any]:
        """Get value from cache with multi-level support"""
        try:
            full_key = self._build_key(key)
            
            # Try memory cache first (fastest)
            if level in [CacheLevel.MEMORY, CacheLevel.MULTI_LEVEL]:
                value = self.memory_cache.get(full_key)
                if value is not None:
                    with self.stats_lock:
                        self.stats.hits += 1
                    return value
            
            # Try Redis cache
            if (level in [CacheLevel.REDIS, CacheLevel.MULTI_LEVEL] and 
                self.redis_available):
                value = self._get_from_redis(full_key)
                if value is not None:
                    # Populate memory cache for faster future access
                    if level == CacheLevel.MULTI_LEVEL:
                        self.memory_cache.set(full_key, value, ttl=self.default_ttl)
                    
                    with self.stats_lock:
                        self.stats.hits += 1
                    return value
            
            # Cache miss
            with self.stats_lock:
                self.stats.misses += 1
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting from cache: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            level: CacheLevel = CacheLevel.MULTI_LEVEL,
            tags: Optional[Set[str]] = None,
            serialization: SerializationMethod = SerializationMethod.PICKLE) -> bool:
        """Set value in cache with multi-level support"""
        try:
            full_key = self._build_key(key)
            ttl = ttl or self.default_ttl
            
            success = True
            
            # Set in memory cache
            if level in [CacheLevel.MEMORY, CacheLevel.MULTI_LEVEL]:
                success &= self.memory_cache.set(full_key, value, ttl, tags)
            
            # Set in Redis cache
            if (level in [CacheLevel.REDIS, CacheLevel.MULTI_LEVEL] and 
                self.redis_available):
                success &= self._set_in_redis(full_key, value, ttl, tags, serialization)
            
            if success:
                with self.stats_lock:
                    self.stats.sets += 1
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error setting in cache: {e}")
            return False
    
    def delete(self, key: str, level: CacheLevel = CacheLevel.MULTI_LEVEL) -> bool:
        """Delete value from cache"""
        try:
            full_key = self._build_key(key)
            success = True
            
            # Delete from memory cache
            if level in [CacheLevel.MEMORY, CacheLevel.MULTI_LEVEL]:
                success &= self.memory_cache.delete(full_key)
            
            # Delete from Redis cache
            if (level in [CacheLevel.REDIS, CacheLevel.MULTI_LEVEL] and 
                self.redis_available):
                success &= bool(self.redis_client.delete(full_key))
            
            if success:
                with self.stats_lock:
                    self.stats.deletes += 1
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error deleting from cache: {e}")
            return False
    
    def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis with deserialization"""
        try:
            raw_data = self.redis_client.get(key)
            if raw_data is None:
                return None
            
            # Deserialize based on metadata
            metadata_key = f"{key}:meta"
            metadata = self.redis_client.get(metadata_key)
            
            if metadata:
                meta_info = json.loads(metadata.decode('utf-8'))
                serialization = SerializationMethod(meta_info.get('serialization', 'pickle'))
            else:
                serialization = SerializationMethod.PICKLE
            
            return self._deserialize(raw_data, serialization)
            
        except Exception as e:
            self.logger.error(f"Error getting from Redis: {e}")
            return None
    
    def _set_in_redis(self, key: str, value: Any, ttl: int, 
                     tags: Optional[Set[str]], 
                     serialization: SerializationMethod) -> bool:
        """Set value in Redis with serialization"""
        try:
            # Serialize value
            serialized_data = self._serialize(value, serialization)
            
            # Set value with TTL
            success = self.redis_client.setex(key, ttl, serialized_data)
            
            # Store metadata
            metadata = {
                'serialization': serialization.value,
                'created_at': datetime.now().isoformat(),
                'tags': list(tags or []),
                'size': len(serialized_data)
            }
            
            metadata_key = f"{key}:meta"
            self.redis_client.setex(metadata_key, ttl, json.dumps(metadata))
            
            # Add to tag indexes if tags provided
            if tags:
                for tag in tags:
                    tag_key = f"tag:{tag}"
                    self.redis_client.sadd(tag_key, key)
                    self.redis_client.expire(tag_key, ttl)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error setting in Redis: {e}")
            return False
    
    def _serialize(self, value: Any, method: SerializationMethod) -> bytes:
        """Serialize value based on method"""
        try:
            if method == SerializationMethod.JSON:
                return json.dumps(value, default=str).encode('utf-8')
            
            elif method == SerializationMethod.STRING:
                return str(value).encode('utf-8')
            
            elif method == SerializationMethod.PICKLE:
                return pickle.dumps(value)
            
            elif method == SerializationMethod.COMPRESSED_PICKLE:
                pickled = pickle.dumps(value)
                if self.enable_compression:
                    return gzip.compress(pickled)
                return pickled
            
            else:
                return pickle.dumps(value)
                
        except Exception as e:
            self.logger.error(f"Error serializing data: {e}")
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes, method: SerializationMethod) -> Any:
        """Deserialize value based on method"""
        try:
            if method == SerializationMethod.JSON:
                return json.loads(data.decode('utf-8'))
            
            elif method == SerializationMethod.STRING:
                return data.decode('utf-8')
            
            elif method == SerializationMethod.PICKLE:
                return pickle.loads(data)
            
            elif method == SerializationMethod.COMPRESSED_PICKLE:
                if self.enable_compression:
                    decompressed = gzip.decompress(data)
                    return pickle.loads(decompressed)
                return pickle.loads(data)
            
            else:
                return pickle.loads(data)
                
        except Exception as e:
            self.logger.error(f"Error deserializing data: {e}")
            # Try pickle as fallback
            return pickle.loads(data)
    
    def _build_key(self, key: str) -> str:
        """Build full cache key with prefix"""
        # Auto-detect key type and add prefix
        for key_type, prefix in self.key_prefixes.items():
            if key_type in key.lower():
                return f"{prefix}{key}"
        
        return key
    
    def clear_by_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern"""
        try:
            count = 0
            
            # Clear from memory cache
            with self.memory_cache.lock:
                keys_to_remove = []
                for key in self.memory_cache.cache.keys():
                    if self._match_pattern(key, pattern):
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    self.memory_cache._remove_entry(key)
                    count += 1
            
            # Clear from Redis cache
            if self.redis_available:
                redis_keys = self.redis_client.keys(pattern)
                if redis_keys:
                    deleted = self.redis_client.delete(*redis_keys)
                    count += deleted
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error clearing by pattern: {e}")
            return 0
    
    def clear_by_tags(self, tags: Union[str, List[str]]) -> int:
        """Clear cache entries by tags"""
        try:
            if isinstance(tags, str):
                tags = [tags]
            
            count = 0
            
            for tag in tags:
                if self.redis_available:
                    tag_key = f"tag:{tag}"
                    keys = self.redis_client.smembers(tag_key)
                    
                    if keys:
                        # Delete from Redis
                        deleted = self.redis_client.delete(*keys)
                        count += deleted
                        
                        # Delete from memory cache
                        for key in keys:
                            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                            if self.memory_cache.delete(key_str):
                                count += 1
                        
                        # Clean up tag index
                        self.redis_client.delete(tag_key)
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error clearing by tags: {e}")
            return 0
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """Simple pattern matching for cache keys"""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            memory_stats = self.memory_cache.get_stats()
            
            redis_stats = {}
            if self.redis_available:
                redis_info = self.redis_client.info()
                redis_stats = {
                    'connected': True,
                    'used_memory': redis_info.get('used_memory', 0),
                    'connected_clients': redis_info.get('connected_clients', 0),
                    'keyspace_hits': redis_info.get('keyspace_hits', 0),
                    'keyspace_misses': redis_info.get('keyspace_misses', 0),
                    'expired_keys': redis_info.get('expired_keys', 0)
                }
            else:
                redis_stats = {'connected': False}
            
            with self.stats_lock:
                return {
                    'overall': {
                        'hits': self.stats.hits,
                        'misses': self.stats.misses,
                        'hit_ratio': self.stats.hits / max(self.stats.hits + self.stats.misses, 1),
                        'sets': self.stats.sets,
                        'deletes': self.stats.deletes,
                        'evictions': self.stats.evictions
                    },
                    'memory_cache': memory_stats,
                    'redis_cache': redis_stats
                }
                
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache systems"""
        health = {
            'memory_cache': True,
            'redis_cache': False,
            'overall_health': 'degraded'
        }
        
        try:
            # Test memory cache
            test_key = "health_check_test"
            test_value = {"test": True, "timestamp": datetime.now().isoformat()}
            
            # Test memory cache operations
            self.memory_cache.set(test_key, test_value, ttl=10)
            retrieved = self.memory_cache.get(test_key)
            self.memory_cache.delete(test_key)
            
            health['memory_cache'] = retrieved is not None
            
            # Test Redis cache
            if self.redis_available:
                self.redis_client.ping()
                self.redis_client.setex(test_key, 10, pickle.dumps(test_value))
                redis_value = self.redis_client.get(test_key)
                self.redis_client.delete(test_key)
                
                health['redis_cache'] = redis_value is not None
            
            # Overall health
            if health['memory_cache'] and health['redis_cache']:
                health['overall_health'] = 'healthy'
            elif health['memory_cache']:
                health['overall_health'] = 'degraded'
            else:
                health['overall_health'] = 'unhealthy'
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            health['overall_health'] = 'unhealthy'
            health['error'] = str(e)
        
        return health
    
    def optimize_memory_usage(self):
        """Optimize memory usage by cleaning up least used entries"""
        try:
            with self.memory_cache.lock:
                # Sort by access count and last access time
                entries = list(self.memory_cache.cache.items())
                entries.sort(key=lambda x: (x[1].access_count, x[1].last_accessed))
                
                # Remove bottom 20% of entries if cache is getting full
                memory_usage = self.memory_cache.current_memory / self.memory_cache.max_memory_bytes
                
                if memory_usage > 0.8:  # 80% full
                    entries_to_remove = int(len(entries) * 0.2)
                    for key, _ in entries[:entries_to_remove]:
                        self.memory_cache._remove_entry(key)
                        with self.stats_lock:
                            self.stats.evictions += 1
                    
                    self.logger.info(f"Optimized memory cache, removed {entries_to_remove} entries")
                    
        except Exception as e:
            self.logger.error(f"Error optimizing memory usage: {e}")
    
    def backup_to_disk(self, file_path: str) -> bool:
        """Backup cache contents to disk"""
        try:
            backup_data = {
                'timestamp': datetime.now().isoformat(),
                'memory_cache': {},
                'redis_keys': []
            }
            
            # Backup memory cache
            with self.memory_cache.lock:
                for key, entry in self.memory_cache.cache.items():
                    backup_data['memory_cache'][key] = {
                        'value': entry.value,
                        'expires_at': entry.expires_at.isoformat() if entry.expires_at else None,
                        'tags': list(entry.tags)
                    }
            
            # Backup Redis keys (metadata only)
            if self.redis_available:
                all_keys = self.redis_client.keys('*')
                backup_data['redis_keys'] = [k.decode('utf-8') if isinstance(k, bytes) else k 
                                           for k in all_keys]
            
            # Save to disk
            with open(file_path, 'wb') as f:
                pickle.dump(backup_data, f)
            
            self.logger.info(f"Cache backed up to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error backing up cache: {e}")
            return False
    
    def restore_from_disk(self, file_path: str) -> bool:
        """Restore cache contents from disk backup"""
        try:
            with open(file_path, 'rb') as f:
                backup_data = pickle.load(f)
            
            # Restore memory cache
            memory_data = backup_data.get('memory_cache', {})
            for key, entry_data in memory_data.items():
                expires_at = None
                if entry_data.get('expires_at'):
                    expires_at = datetime.fromisoformat(entry_data['expires_at'])
                    
                    # Skip if already expired
                    if expires_at < datetime.now():
                        continue
                
                self.memory_cache.set(
                    key, 
                    entry_data['value'],
                    ttl=(expires_at - datetime.now()).total_seconds() if expires_at else None,
                    tags=set(entry_data.get('tags', []))
                )
            
            self.logger.info(f"Cache restored from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error restoring cache: {e}")
            return False
    
    def close(self):
        """Close cache connections and cleanup"""
        try:
            self.stop_cleanup.set()
            
            if self.cleanup_thread and self.cleanup_thread.is_alive():
                self.cleanup_thread.join(timeout=5.0)
            
            if self.redis_client:
                self.redis_client.close()
            
            self.logger.info("Cache manager closed")
            
        except Exception as e:
            self.logger.error(f"Error closing cache manager: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.close()
        except:
            pass

# Global cache instance
_cache_manager = None

def get_cache_manager(**kwargs) -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(**kwargs)
    return _cache_manager

def cache_result(ttl: int = 3600, 
                tags: Optional[Set[str]] = None,
                key_func: Optional[Callable] = None):
    """Decorator for caching function results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            cache = get_cache_manager()
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl, tags=tags)
            
            return result
        
        return wrapper
    return decorator

# Example usage functions
def cache_market_data(symbol: str, data: Dict[str, Any], ttl: int = 300):
    """Cache market data with specific TTL"""
    cache = get_cache_manager()
    key = f"market_data:{symbol}"
    return cache.set(key, data, ttl=ttl, tags={'market_data', symbol})

def get_cached_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Get cached market data"""
    cache = get_cache_manager()
    key = f"market_data:{symbol}"
    return cache.get(key)

def cache_ai_prediction(model_name: str, symbol: str, prediction: Dict[str, Any], ttl: int = 1800):
    """Cache AI prediction results"""
    cache = get_cache_manager()
    key = f"ai_prediction:{model_name}:{symbol}"
    return cache.set(key, prediction, ttl=ttl, tags={'ai_predictions', model_name, symbol})

def clear_symbol_cache(symbol: str):
    """Clear all cached data for a specific symbol"""
    cache = get_cache_manager()
    return cache.clear_by_tags([symbol])
