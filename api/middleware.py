import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Optional
from collections import defaultdict, deque

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from config.settings import settings

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent", "unknown")
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            # Log response
            logger.info(
                f"Request completed: {response.status_code}",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time": process_time
                }
            )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            logger.error(
                f"Request failed: {str(e)}",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "process_time": process_time
                }
            )
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to client host
        return request.client.host if request.client else "unknown"

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.enabled = settings.RATE_LIMIT_ENABLED
        self.calls_per_minute = settings.RATE_LIMIT_CALLS_PER_MINUTE
        self.calls_per_hour = settings.RATE_LIMIT_CALLS_PER_HOUR
        
        # Storage for rate limiting data
        self.minute_requests: Dict[str, deque] = defaultdict(deque)
        self.hour_requests: Dict[str, deque] = defaultdict(deque)
        
        # Last cleanup time
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
        
        if self.enabled:
            logger.info(f"Rate limiting enabled: {self.calls_per_minute}/min, {self.calls_per_hour}/hour")
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting if disabled
        if not self.enabled:
            return await call_next(request)
        
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Cleanup old requests periodically
        self._cleanup_old_requests()
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Check rate limits
        is_limited, message, retry_after = self._is_rate_limited(client_ip)
        
        if is_limited:
            logger.warning(f"Rate limit exceeded for {client_ip}: {message}")
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "code": 429,
                        "message": message,
                        "retry_after_seconds": retry_after,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                },
                headers={"Retry-After": str(retry_after)}
            )
        
        # Record this request
        self._record_request(client_ip)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        minute_requests_count = len(self.minute_requests[client_ip])
        hour_requests_count = len(self.hour_requests[client_ip])
        
        response.headers["X-RateLimit-Minute-Limit"] = str(self.calls_per_minute)
        response.headers["X-RateLimit-Minute-Remaining"] = str(max(0, self.calls_per_minute - minute_requests_count))
        response.headers["X-RateLimit-Hour-Limit"] = str(self.calls_per_hour)
        response.headers["X-RateLimit-Hour-Remaining"] = str(max(0, self.calls_per_hour - hour_requests_count))
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to client host
        return request.client.host if request.client else "unknown"
    
    def _cleanup_old_requests(self):
        """Clean up old request records"""
        current_time = time.time()
        
        # Only cleanup every cleanup_interval seconds
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff_minute = current_time - 60
        cutoff_hour = current_time - 3600
        
        # Cleanup minute requests
        for client_ip in list(self.minute_requests.keys()):
            requests = self.minute_requests[client_ip]
            while requests and requests[0] < cutoff_minute:
                requests.popleft()
            
            if not requests:
                del self.minute_requests[client_ip]
        
        # Cleanup hour requests
        for client_ip in list(self.hour_requests.keys()):
            requests = self.hour_requests[client_ip]
            while requests and requests[0] < cutoff_hour:
                requests.popleft()
            
            if not requests:
                del self.hour_requests[client_ip]
        
        self.last_cleanup = current_time
    
    def _is_rate_limited(self, client_ip: str) -> tuple[bool, str, int]:
        """Check if client is rate limited"""
        current_time = time.time()
        
        # Check minute limit
        minute_requests = self.minute_requests[client_ip]
        minute_cutoff = current_time - 60
        
        # Remove old requests
        while minute_requests and minute_requests[0] < minute_cutoff:
            minute_requests.popleft()
        
        if len(minute_requests) >= self.calls_per_minute:
            retry_after = int(60 - (current_time - minute_requests[0]))
            return True, "Rate limit exceeded: too many requests per minute", retry_after
        
        # Check hour limit
        hour_requests = self.hour_requests[client_ip]
        hour_cutoff = current_time - 3600
        
        # Remove old requests
        while hour_requests and hour_requests[0] < hour_cutoff:
            hour_requests.popleft()
        
        if len(hour_requests) >= self.calls_per_hour:
            retry_after = int(3600 - (current_time - hour_requests[0]))
            return True, "Rate limit exceeded: too many requests per hour", retry_after
        
        return False, "", 0
    
    def _record_request(self, client_ip: str):
        """Record a request for rate limiting"""
        current_time = time.time()
        
        self.minute_requests[client_ip].append(current_time)
        self.hour_requests[client_ip].append(current_time)

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for API key validation"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.enabled = settings.REQUIRE_AUTH
        self.api_key = settings.API_KEY
        
        # Paths that don't require authentication
        self.public_paths = {
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json"
        }
        
        if self.enabled:
            logger.info("Authentication middleware enabled")
    
    async def dispatch(self, request: Request, call_next):
        # Skip authentication if disabled
        if not self.enabled:
            return await call_next(request)
        
        # Skip authentication for public paths
        if request.url.path in self.public_paths:
            return await call_next(request)
        
        # Extract API key
        api_key = self._extract_api_key(request)
        
        if not api_key:
            logger.warning(f"Authentication required for {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "code": 401,
                        "message": "API key required",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                },
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Validate API key
        if not self._validate_api_key(api_key):
            logger.warning(f"Invalid API key for {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "code": 401,
                        "message": "Invalid API key",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                },
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Add authentication info to request state
        request.state.authenticated = True
        request.state.api_key = api_key
        
        return await call_next(request)
    
    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request"""
        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        
        # Check X-API-Key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key
        
        # Check query parameter
        api_key = request.query_params.get("api_key")
        if api_key:
            return api_key
        
        return None
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        if not self.api_key:
            return True  # No API key configured, allow access
        
        return api_key == self.api_key

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for standardized error handling"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
            
        except HTTPException:
            # Let FastAPI handle HTTP exceptions
            raise
            
        except Exception as e:
            # Handle unexpected errors
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            logger.error(
                f"Unhandled error in request {request_id}: {str(e)}",
                exc_info=True
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "code": 500,
                        "message": "Internal server error",
                        "request_id": request_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
            )

class CORSMiddleware(BaseHTTPMiddleware):
    """CORS middleware for external API access"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
        response.headers["Access-Control-Max-Age"] = "86400"
        
        return response

# Utility functions
def get_request_id(request: Request) -> str:
    """Get request ID from request state"""
    return getattr(request.state, 'request_id', 'unknown')

def is_authenticated(request: Request) -> bool:
    """Check if request is authenticated"""
    return getattr(request.state, 'authenticated', False)

def get_api_key(request: Request) -> Optional[str]:
    """Get API key from request state"""
    return getattr(request.state, 'api_key', None)