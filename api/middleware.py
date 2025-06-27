import time
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Set, Optional
from collections import defaultdict, deque

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

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
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else "unknown",
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
                f"Request completed",
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
                f"Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "process_time": process_time
                }
            )
            raise


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


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(
        self,
        app: ASGIApp,
        calls_per_minute: int = 60,
        calls_per_hour: int = 1000
    ):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour
        
        # Storage for rate limiting data
        self.minute_requests: Dict[str, deque] = defaultdict(deque)
        self.hour_requests: Dict[str, deque] = defaultdict(deque)
        
        # Cleanup interval (in seconds)
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
    
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
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
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


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication middleware (basic implementation)"""
    
    def __init__(self, app: ASGIApp, require_auth: bool = False):
        super().__init__(app)
        self.require_auth = require_auth
        
        # Paths that don't require authentication
        self.public_paths = {
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json"
        }
    
    def _extract_token(self, request: Request) -> Optional[str]:
        """Extract authentication token from request"""
        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        
        # Check API key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key
        
        return None
    
    def _validate_token(self, token: str) -> bool:
        """Validate authentication token"""
        # In a real implementation, this would:
        # 1. Validate JWT tokens
        # 2. Check API keys against database
        # 3. Verify token expiration
        # 4. Check user permissions
        
        # For now, just check if it's not empty
        # TODO: Implement proper token validation
        return len(token) > 0
    
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for public paths
        if request.url.path in self.public_paths:
            return await call_next(request)
        
        # Skip if authentication is not required
        if not self.require_auth:
            return await call_next(request)
        
        # Extract token
        token = self._extract_token(request)
        
        if not token:
            logger.warning(f"Authentication required for {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "code": 401,
                        "message": "Authentication required",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                },
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Validate token
        if not self._validate_token(token):
            logger.warning(f"Invalid authentication token for {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "code": 401,
                        "message": "Invalid authentication token",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                },
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Add user info to request state for use in endpoints
        request.state.authenticated = True
        request.state.token = token
        
        return await call_next(request)


# Utility functions for middleware
def get_request_id(request: Request) -> str:
    """Get request ID from request state"""
    return getattr(request.state, 'request_id', 'unknown')


def is_authenticated(request: Request) -> bool:
    """Check if request is authenticated"""
    return getattr(request.state, 'authenticated', False)


def get_auth_token(request: Request) -> Optional[str]:
    """Get authentication token from request state"""
    return getattr(request.state, 'token', None)