import logging
import sys
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes import router
from api.middleware import (
    RequestLoggingMiddleware,
    ErrorHandlingMiddleware,
    RateLimitMiddleware,
    AuthenticationMiddleware
)
from config.settings import settings
from utils.logging_config import setup_logging

# Initialize logging
logger = setup_logging()

# Global service instances
service_registry = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager
    Handles startup and shutdown events
    """
    # Startup
    logger.info("Starting Sales Sentiment RAG API")
    
    try:
        # Initialize core services
        await initialize_services()
        logger.info("All services initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down Sales Sentiment RAG API")
        await cleanup_services()

async def initialize_services():
    """Initialize all core services"""
    global service_registry
    
    try:
        # Initialize embedding service
        from core.embedding_service import get_embedding_service
        service_registry['embedding_service'] = get_embedding_service()
        logger.info("Embedding service initialized")
        
        # Initialize vector store
        from core.vector_store import get_vector_store
        service_registry['vector_store'] = get_vector_store()
        logger.info("Vector store initialized")
        
        # Initialize cache manager
        from utils.cache import create_cache_manager
        service_registry['cache_manager'] = create_cache_manager()
        logger.info("Cache manager initialized")
        
        # Initialize sentiment analyzer
        from llm.sentiment_analyzer import create_sentiment_analyzer
        #service_registry['sentiment_analyzer'] = create_sentiment_analyzer(provider_name=os.getenv('LLM_PROVIDER'), provider_config={'api_key': os.getenv('GROQ_API_KEY'), 'model': os.getenv('GROQ_MODEL')})
        service_registry['sentiment_analyzer'] = create_sentiment_analyzer(provider_name=os.getenv('LLM_PROVIDER'), provider_config={'api_key': os.getenv('AZURE_OPENAI_API_KEY'), 'endpoint': os.getenv('AZURE_OPENAI_ENDPOINT'), 'deployment_name': os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'), 'api_version':os.getenv('AZURE_OPENAI_API_VERSION')})
        logger.info("Sentiment analyzer initialized")
        
        # Initialize RAG retriever
        from rag.retriever import create_rag_retriever
        service_registry['rag_retriever'] = create_rag_retriever()
        logger.info("RAG retriever initialized")
        
        # Initialize knowledge base manager
        from rag.knowledge_base import create_knowledge_base_manager
        service_registry['knowledge_base_manager'] = create_knowledge_base_manager()
        logger.info("Knowledge base manager initialized")
        
        # Health check for all services
        await perform_health_checks()
        
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        logger.error(traceback.format_exc())
        raise

async def cleanup_services():
    """Cleanup services on shutdown"""
    global service_registry
    
    try:
        # Close cache connections
        cache_manager = service_registry.get('cache_manager')
        if cache_manager and hasattr(cache_manager, 'redis_client'):
            if cache_manager.redis_client:
                cache_manager.redis_client.close()
        
        # Clear service registry
        service_registry.clear()
        logger.info("Services cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during service cleanup: {e}")

async def perform_health_checks():
    """Perform health checks on all services"""
    health_results = {}
    
    # Check cache
    cache_manager = service_registry.get('cache_manager')
    if cache_manager:
        health_results['cache'] = cache_manager.health_check()
    
    # Check vector store
    vector_store = service_registry.get('vector_store')
    if vector_store:
        try:
            stats = vector_store.get_stats()
            health_results['vector_store'] = {'status': 'healthy', 'stats': stats}
        except Exception as e:
            health_results['vector_store'] = {'status': 'unhealthy', 'error': str(e)}
    
    # Log health check results
    for service, health in health_results.items():
        status = health.get('status', 'unknown')
        logger.info(f"Health check - {service}: {status}")
        if status != 'healthy':
            logger.warning(f"Service {service} health issue: {health}")

def get_service(service_name: str):
    """Get service instance from registry"""
    service = service_registry.get(service_name)
    if not service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service {service_name} not available"
        )
    return service

# Create FastAPI application
app = FastAPI(
    title="Sales Sentiment RAG API",
    description="""
    Advanced Sales Sentiment Analysis API with Retrieval-Augmented Generation (RAG) capabilities.
    
    This API analyzes salesperson performance and sentiment from CRM activities using:
    - Historical deal pattern matching
    - LLM-powered sentiment analysis
    - Performance benchmarking
    - Risk assessment and coaching recommendations
    
    ## Features
    - **Sentiment Analysis**: Analyze salesperson sentiment from deal activities
    - **Performance Benchmarking**: Compare against successful deal patterns
    - **Risk Assessment**: Identify behavioral risks that could jeopardize deals
    - **Coaching Recommendations**: Get actionable improvement suggestions
    - **Deal Insights**: Comprehensive analysis with historical context
    - **Knowledge Base Management**: Manage and optimize the pattern database
    """,
    version="1.0.0",
    contact={
        "name": "Sales Sentiment RAG Team",
        "email": "aman.jaiswar@glynac.ai",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthenticationMiddleware)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url)
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url)
            }
        }
    )

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Sales Sentiment RAG API",
        "version": "1.0.0",
        "description": "Advanced Sales Sentiment Analysis with RAG capabilities",
        "docs_url": "/docs",
        "health_url": "/health",
        "timestamp": datetime.utcnow().isoformat()
    }

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """System health check endpoint"""
    try:
        health_results = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {}
        }
        
        # Check cache
        cache_manager = service_registry.get('cache_manager')
        if cache_manager:
            health_results["services"]["cache"] = cache_manager.health_check()
        
        # Check vector store
        vector_store = service_registry.get('vector_store')
        if vector_store:
            try:
                stats = vector_store.get_stats()
                health_results["services"]["vector_store"] = {
                    "status": "healthy",
                    "stats": stats
                }
            except Exception as e:
                health_results["services"]["vector_store"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        # Check overall health
        unhealthy_services = [
            name for name, health in health_results["services"].items()
            if health.get("status") != "healthy"
        ]
        
        if unhealthy_services:
            health_results["status"] = "degraded"
            health_results["unhealthy_services"] = unhealthy_services
        
        return health_results
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

# Make service registry available to routes
app.state.services = service_registry

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )