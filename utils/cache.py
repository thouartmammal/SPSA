import logging
import json
import hashlib
import pickle
from typing import Any, Optional, List, Dict, Union, Callable
from datetime import datetime, timedelta
import redis
from redis.exceptions import ConnectionError, TimeoutError
import threading
import time

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Advanced Redis-based cache manager for the Sales Sentiment RAG system.
    Provides intelligent caching for embeddings, LLM responses, search results,
    and other computationally expensive operations.
    """
    
    def __init__(
        self,
        redis_url: str = None,
        default_ttl: int = None,
        enable_compression: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize Cache Manager
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default time-to-live in seconds
            enable_compression: Whether to compress cached data
            max_retries: Maximum connection retry attempts
        """
        self.redis_url = redis_url or settings.REDIS_URL
        self.default_ttl = default_ttl or settings.CACHE_TTL
        self.enable_compression = enable_compression
        self.max_retries = max_retries
        
        # Connection pool for better performance
        self.connection_pool = None
        self.redis_client = None

        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        
        # Thread lock for statistics
        self.stats_lock = threading.Lock()
        
        # Cache key prefixes for different data types
        self.key_prefixes = {
            'embedding': 'emb:',
            'llm_response': 'llm:',
            'vector_search': 'vec:',
            'deal_pattern': 'deal:',
            'rag_context': 'rag:',
            'sentiment_analysis': 'sent:',
            'knowledge_base': 'kb:',
            'user_session': 'sess:'
        }
        
        self._initialize_connection()
        logger.info("Cache Manager initialized")
    
    def _initialize_connection(self):
        """Initialize Redis connection with retry logic"""
        
        for attempt in range(self.max_retries):
            try:
                # Parse Redis URL
                if self.redis_url.startswith('redis://'):
                    # Create connection pool
                    self.connection_pool = redis.ConnectionPool.from_url(
                        self.redis_url,
                        max_connections=20,
                        socket_timeout=5,
                        socket_connect_timeout=5,
                        retry_on_timeout=True
                    )
                    
                    # Create Redis client
                    self.redis_client = redis.Redis(
                        connection_pool=self.connection_pool,
                        decode_responses=False  # We handle encoding ourselves
                    )
                    
                    # Test connection
                    self.redis_client.ping()
                    logger.info(f"Connected to Redis: {self.redis_url}")
                    return
                
                else:
                    raise ValueError(f"Invalid Redis URL format: {self.redis_url}")
                    
            except Exception as e:
                logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    logger.error("Failed to connect to Redis after all retries")
                    self.redis_client = None
                else:
                    time.sleep(2 ** attempt)  # Exponential backoff
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: int = None,
        cache_type: str = None,
        compress: bool = None
    ) -> bool:
        """
        Set a value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            cache_type: Type of cached data for key prefixing
            compress: Whether to compress the data
            
        Returns:
            True if successful, False otherwise
        """
        
        if not self.redis_client:
            logger.debug("Redis not available, skipping cache set")
            return False
        
        try:
            # Prepare cache key
            cache_key = self._prepare_cache_key(key, cache_type)
            
            # Serialize value
            serialized_value = self._serialize_value(value, compress)
            
            # Set TTL
            ttl = ttl or self.default_ttl
            
            # Store in Redis
            result = self.redis_client.setex(cache_key, ttl, serialized_value)
            
            if result:
                with self.stats_lock:
                    self.stats['sets'] += 1
                logger.debug(f"Cached: {cache_key} (TTL: {ttl}s)")
                return True
            else:
                logger.warning(f"Failed to cache: {cache_key}")
                return False
                
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            with self.stats_lock:
                self.stats['errors'] += 1
            return False
    
    def get(
        self,
        key: str,
        cache_type: str = None,
        default: Any = None
    ) -> Any:
        """
        Get a value from cache
        
        Args:
            key: Cache key
            cache_type: Type of cached data for key prefixing
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        
        if not self.redis_client:
            logger.debug("Redis not available, returning default")
            return default
        
        try:
            # Prepare cache key
            cache_key = self._prepare_cache_key(key, cache_type)
            
            # Get from Redis
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data is not None:
                # Deserialize value
                value = self._deserialize_value(cached_data)
                
                with self.stats_lock:
                    self.stats['hits'] += 1
                
                logger.debug(f"Cache hit: {cache_key}")
                return value
            
            else:
                with self.stats_lock:
                    self.stats['misses'] += 1
                
                logger.debug(f"Cache miss: {cache_key}")
                return default
                
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            with self.stats_lock:
                self.stats['errors'] += 1
            return default
    
    def delete(self, key: str, cache_type: str = None) -> bool:
        """
        Delete a key from cache
        
        Args:
            key: Cache key
            cache_type: Type of cached data
            
        Returns:
            True if successful, False otherwise
        """
        
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._prepare_cache_key(key, cache_type)
            result = self.redis_client.delete(cache_key)
            
            if result:
                with self.stats_lock:
                    self.stats['deletes'] += 1
                logger.debug(f"Cache delete: {cache_key}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            with self.stats_lock:
                self.stats['errors'] += 1
            return False
    
    def exists(self, key: str, cache_type: str = None) -> bool:
        """
        Check if a key exists in cache
        
        Args:
            key: Cache key
            cache_type: Type of cached data
            
        Returns:
            True if key exists, False otherwise
        """
        
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._prepare_cache_key(key, cache_type)
            return bool(self.redis_client.exists(cache_key))
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    def expire(self, key: str, ttl: int, cache_type: str = None) -> bool:
        """
        Set expiration time for a key
        
        Args:
            key: Cache key
            ttl: Time-to-live in seconds
            cache_type: Type of cached data
            
        Returns:
            True if successful, False otherwise
        """
        
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._prepare_cache_key(key, cache_type)
            return bool(self.redis_client.expire(cache_key, ttl))
        except Exception as e:
            logger.error(f"Cache expire error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern
        
        Args:
            pattern: Pattern to match (supports wildcards)
            
        Returns:
            Number of keys deleted
        """
        
        if not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                with self.stats_lock:
                    self.stats['deletes'] += deleted
                logger.info(f"Cleared {deleted} keys matching pattern: {pattern}")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cache clear pattern error: {e}")
            return 0
    
    def clear_all(self) -> bool:
        """
        Clear all cached data (use with caution)
        
        Returns:
            True if successful, False otherwise
        """
        
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.flushdb()
            logger.warning("Cleared all cache data")
            return True
        except Exception as e:
            logger.error(f"Cache clear all error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary of cache statistics
        """
        
        with self.stats_lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests) if total_requests > 0 else 0
            
            stats = {
                **self.stats.copy(),
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }
        
        # Add Redis info if available
        if self.redis_client:
            try:
                redis_info = self.redis_client.info()
                stats['redis_info'] = {
                    'used_memory': redis_info.get('used_memory', 0),
                    'used_memory_human': redis_info.get('used_memory_human', '0B'),
                    'connected_clients': redis_info.get('connected_clients', 0),
                    'total_commands_processed': redis_info.get('total_commands_processed', 0)
                }
            except Exception as e:
                logger.debug(f"Could not get Redis info: {e}")
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform cache health check
        
        Returns:
            Health check results
        """
        
        health_info = {
            'status': 'unhealthy',
            'redis_connected': False,
            'latency_ms': None,
            'error': None,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if not self.redis_client:
            health_info['error'] = 'Redis client not initialized'
            return health_info
        
        try:
            # Test connection with latency measurement
            start_time = time.time()
            self.redis_client.ping()
            latency = (time.time() - start_time) * 1000
            
            health_info.update({
                'status': 'healthy',
                'redis_connected': True,
                'latency_ms': round(latency, 2)
            })
            
        except Exception as e:
            health_info['error'] = str(e)
        
        return health_info
    
    # Specialized caching methods for different data types
    
    def cache_embedding(
        self,
        text: str,
        embedding: List[float],
        model_name: str = "default",
        ttl: int = None
    ) -> bool:
        """
        Cache text embedding
        
        Args:
            text: Original text
            embedding: Generated embedding
            model_name: Embedding model name
            ttl: Time-to-live
            
        Returns:
            True if successful
        """
        
        cache_key = self._generate_embedding_key(text, model_name)
        return self.set(cache_key, embedding, ttl, cache_type='embedding')
    
    def get_cached_embedding(
        self,
        text: str,
        model_name: str = "default"
    ) -> Optional[List[float]]:
        """
        Get cached embedding
        
        Args:
            text: Original text
            model_name: Embedding model name
            
        Returns:
            Cached embedding or None
        """
        
        cache_key = self._generate_embedding_key(text, model_name)
        return self.get(cache_key, cache_type='embedding')
    
    def cache_llm_response(
        self,
        prompt: str,
        response: str,
        model_name: str = "default",
        ttl: int = None
    ) -> bool:
        """
        Cache LLM response
        
        Args:
            prompt: LLM prompt
            response: LLM response
            model_name: LLM model name
            ttl: Time-to-live
            
        Returns:
            True if successful
        """
        
        cache_key = self._generate_llm_key(prompt, model_name)
        cache_data = {
            'response': response,
            'timestamp': datetime.utcnow().isoformat(),
            'model': model_name
        }
        
        ttl = ttl or settings.LLM_CACHE_TTL
        return self.set(cache_key, cache_data, ttl, cache_type='llm_response')
    
    def get_cached_llm_response(
        self,
        prompt: str,
        model_name: str = "default"
    ) -> Optional[str]:
        """
        Get cached LLM response
        
        Args:
            prompt: LLM prompt
            model_name: LLM model name
            
        Returns:
            Cached response or None
        """
        
        cache_key = self._generate_llm_key(prompt, model_name)
        cache_data = self.get(cache_key, cache_type='llm_response')
        
        if cache_data and isinstance(cache_data, dict):
            return cache_data.get('response')
        
        return cache_data  # Backward compatibility
    
    def cache_vector_search_results(
        self,
        query_embedding: List[float],
        results: List[Any],
        search_params: Dict[str, Any] = None,
        ttl: int = None
    ) -> bool:
        """
        Cache vector search results
        
        Args:
            query_embedding: Query embedding
            results: Search results
            search_params: Search parameters
            ttl: Time-to-live
            
        Returns:
            True if successful
        """
        
        cache_key = self._generate_vector_search_key(query_embedding, search_params)
        cache_data = {
            'results': results,
            'timestamp': datetime.utcnow().isoformat(),
            'search_params': search_params or {}
        }
        
        ttl = ttl or settings.VECTOR_SEARCH_CACHE_TTL
        return self.set(cache_key, cache_data, ttl, cache_type='vector_search')
    
    def get_cached_vector_search_results(
        self,
        query_embedding: List[float],
        search_params: Dict[str, Any] = None
    ) -> Optional[List[Any]]:
        """
        Get cached vector search results
        
        Args:
            query_embedding: Query embedding
            search_params: Search parameters
            
        Returns:
            Cached results or None
        """
        
        cache_key = self._generate_vector_search_key(query_embedding, search_params)
        cache_data = self.get(cache_key, cache_type='vector_search')
        
        if cache_data and isinstance(cache_data, dict):
            return cache_data.get('results')
        
        return None
    
    def cache_sentiment_analysis(
        self,
        deal_id: str,
        analysis_result: Dict[str, Any],
        ttl: int = None
    ) -> bool:
        """
        Cache sentiment analysis result
        
        Args:
            deal_id: Deal identifier
            analysis_result: Analysis result
            ttl: Time-to-live
            
        Returns:
            True if successful
        """
        
        cache_key = f"sentiment_{deal_id}"
        analysis_result['cached_at'] = datetime.utcnow().isoformat()
        
        ttl = ttl or 7200  # 2 hours default for sentiment analysis
        return self.set(cache_key, analysis_result, ttl, cache_type='sentiment_analysis')
    
    def get_cached_sentiment_analysis(self, deal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached sentiment analysis
        
        Args:
            deal_id: Deal identifier
            
        Returns:
            Cached analysis or None
        """
        
        cache_key = f"sentiment_{deal_id}"
        return self.get(cache_key, cache_type='sentiment_analysis')
    
    # Private helper methods
    
    def _prepare_cache_key(self, key: str, cache_type: str = None) -> str:
        """Prepare cache key with optional prefix"""
        
        if cache_type and cache_type in self.key_prefixes:
            return f"{self.key_prefixes[cache_type]}{key}"
        
        return key
    
    def _serialize_value(self, value: Any, compress: bool = None) -> bytes:
        """Serialize value for caching"""
        
        compress = compress if compress is not None else self.enable_compression
        
        try:
            # Use pickle for complex objects, JSON for simple ones
            if isinstance(value, (dict, list, str, int, float, bool)):
                serialized = json.dumps(value, default=str).encode('utf-8')
            else:
                serialized = pickle.dumps(value)
            
            # Compress if enabled and beneficial
            if compress and len(serialized) > 1024:  # Only compress if > 1KB
                import gzip
                compressed = gzip.compress(serialized)
                # Only use compression if it actually reduces size
                if len(compressed) < len(serialized):
                    return b'GZIP:' + compressed
            
            return serialized
            
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise
    
    def _deserialize_value(self, cached_data: bytes) -> Any:
        """Deserialize cached value"""
        
        try:
            # Check if data is compressed
            if cached_data.startswith(b'GZIP:'):
                import gzip
                cached_data = gzip.decompress(cached_data[5:])
            
            # Try JSON first (most common)
            try:
                return json.loads(cached_data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fall back to pickle
                return pickle.loads(cached_data)
                
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise
    
    def _generate_embedding_key(self, text: str, model_name: str) -> str:
        """Generate cache key for embeddings"""
        
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"{model_name}_{text_hash}"
    
    def _generate_llm_key(self, prompt: str, model_name: str) -> str:
        """Generate cache key for LLM responses"""
        
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        return f"{model_name}_{prompt_hash}"
    
    def _generate_vector_search_key(
        self,
        query_embedding: List[float],
        search_params: Dict[str, Any] = None
    ) -> str:
        """Generate cache key for vector search results"""
        
        # Create hash from embedding and parameters
        embedding_str = ','.join(map(str, query_embedding[:10]))  # Use first 10 dims
        params_str = json.dumps(search_params or {}, sort_keys=True)
        
        combined = f"{embedding_str}_{params_str}"
        search_hash = hashlib.md5(combined.encode('utf-8')).hexdigest()
        
        return f"search_{search_hash}"


# Factory function
def create_cache_manager(
    redis_url: str = None,
    default_ttl: int = None,
    enable_compression: bool = True
) -> CacheManager:
    """Create cache manager instance"""
    
    return CacheManager(
        redis_url=redis_url,
        default_ttl=default_ttl,
        enable_compression=enable_compression
    )


# Context manager for cache operations
class CacheContext:
    """Context manager for cache operations with automatic cleanup"""
    
    def __init__(self, cache_manager: CacheManager, prefix: str = "temp"):
        self.cache_manager = cache_manager
        self.prefix = prefix
        self.keys_created = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up any keys created during this context
        for key in self.keys_created:
            self.cache_manager.delete(key)
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value and track key for cleanup"""
        full_key = f"{self.prefix}:{key}"
        success = self.cache_manager.set(full_key, value, ttl)
        if success:
            self.keys_created.append(full_key)
        return success
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value"""
        full_key = f"{self.prefix}:{key}"
        return self.cache_manager.get(full_key, default=default)


# Decorators for caching
def cache_result(
    cache_manager: CacheManager,
    ttl: int = None,
    cache_type: str = None,
    key_func: Callable = None
):
    """Decorator to cache function results"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                cache_key = hashlib.md5('_'.join(key_parts).encode()).hexdigest()
            
            # Try to get cached result
            cached_result = cache_manager.get(cache_key, cache_type=cache_type)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl=ttl, cache_type=cache_type)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        return wrapper
    return decorator


# Example usage and testing
def test_cache_manager():
    """Test cache manager functionality"""
    
    try:
        # Create cache manager
        cache = create_cache_manager()
        
        # Test basic operations
        print("Testing basic cache operations...")
        
        # Set and get
        success = cache.set("test_key", {"data": "test_value"}, ttl=300)
        print(f"Set operation: {'Success' if success else 'Failed'}")
        
        value = cache.get("test_key")
        print(f"Get operation: {value}")
        
        # Test specialized caching
        print("\nTesting specialized caching...")
        
        # Cache embedding
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        cache.cache_embedding("test text", embedding, "test_model")
        
        cached_embedding = cache.get_cached_embedding("test text", "test_model")
        print(f"Cached embedding: {cached_embedding}")
        
        # Test health check
        health = cache.health_check()
        print(f"\nHealth check: {health}")
        
        # Test statistics
        stats = cache.get_stats()
        print(f"\nCache statistics: {stats}")
        
        print("\n✅ Cache Manager test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_cache_manager()