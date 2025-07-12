import logging
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from routes import router
from middleware import (
    RequestLoggingMiddleware,
    ErrorHandlingMiddleware,
    RateLimitMiddleware,
    AuthenticationMiddleware,
    CORSMiddleware
)
from config.settings import settings
from utils.logging_config import setup_logging

# Initialize logging
logger = setup_logging()

# Global service registry
service_registry = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    logger.info("Starting Sales Sentiment RAG API")
    
    try:
        # Initialize services
        await initialize_services()
        logger.info("All services initialized successfully")
        
        # Build knowledge base if configured
        if settings.KB_REBUILD_ON_STARTUP:
            await build_knowledge_base_on_startup()
        
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
        # Initialize cache manager
        logger.info("Initializing cache manager...")
        from utils.cache import get_cache_manager
        service_registry['cache_manager'] = get_cache_manager()
        
        # Initialize embedding service
        logger.info("Initializing embedding service...")
        from core.embedding_service import get_embedding_service
        service_registry['embedding_service'] = get_embedding_service()
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        from core.vector_store import get_vector_store
        service_registry['vector_store'] = get_vector_store()
        
        # Initialize data processor
        logger.info("Initializing data processor...")
        from core.data_processor import DealDataProcessor
        service_registry['data_processor'] = DealDataProcessor(
            service_registry['embedding_service']
        )
        
        # Initialize RAG retriever
        logger.info("Initializing RAG retriever...")
        from rag.retriever import create_rag_retriever
        service_registry['rag_retriever'] = create_rag_retriever(
            embedding_service=service_registry['embedding_service'],
            vector_store=service_registry['vector_store']
        )
        
        # Initialize knowledge base builder
        logger.info("Initializing knowledge base builder...")
        from rag.knowledge_base import create_knowledge_base_builder
        service_registry['knowledge_base_builder'] = create_knowledge_base_builder(
            embedding_service=service_registry['embedding_service'],
            vector_store=service_registry['vector_store'],
            data_processor=service_registry['data_processor']
        )

        # Initialize sales sentiment analyzer
        logger.info("Initializing sales sentiment analyzer...")
        from llm.sales_sentiment_analyzer import create_sales_sentiment_analyzer
        service_registry['sales_sentiment_analyzer'] = create_sales_sentiment_analyzer(
            llm_provider=settings.LLM_PROVIDER,
            llm_config=settings.get_llm_config()
        )

        # Initialize client sentiment analyzer
        logger.info("Initializing client sentiment analyzer...")
        from llm.client_sentiment_analyzer import create_client_sentiment_analyzer
        service_registry['client_sentiment_analyzer'] = create_client_sentiment_analyzer(
            llm_provider=settings.LLM_PROVIDER,
            llm_config=settings.get_llm_config()
        )
        
        # Perform health checks
        await perform_health_checks()
        
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        raise
    
async def build_knowledge_base_on_startup():
    """Build knowledge base on startup if configured"""
    try:
        logger.info("Building knowledge base on startup...")
        
        kb_builder = service_registry.get('knowledge_base_builder')
        if kb_builder:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                kb_builder.build_knowledge_base,
                settings.DATA_PATH,
                False  # Don't force rebuild
            )
            
            if result.get('status') == 'completed':
                logger.info(f"Knowledge base built successfully: {result.get('total_deals_processed', 0)} deals processed")
            else:
                logger.warning(f"Knowledge base build result: {result.get('status', 'unknown')}")
        
    except Exception as e:
        logger.error(f"Error building knowledge base on startup: {e}")

async def cleanup_services():
    """Cleanup services on shutdown"""
    global service_registry
    
    try:
        # Close cache connections
        cache_manager = service_registry.get('cache_manager')
        if cache_manager:
            cache_manager.close()
        
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
    
    # Check sales sentiment analyzer
    sales_sentiment_analyzer = service_registry.get('sales_sentiment_analyzer')
    if sales_sentiment_analyzer:
        try:
            stats = sales_sentiment_analyzer.get_analyzer_stats()
            health_results['sales_sentiment_analyzer'] = {'status': 'healthy', 'stats': stats}
        except Exception as e:
            health_results['sales_sentiment_analyzer'] = {'status': 'unhealthy', 'error': str(e)}
    
    # Check client sentiment analyzer
    client_sentiment_analyzer = service_registry.get('client_sentiment_analyzer')
    if client_sentiment_analyzer:
        try:
            stats = client_sentiment_analyzer.get_analyzer_stats()
            health_results['client_sentiment_analyzer'] = {'status': 'healthy', 'stats': stats}
        except Exception as e:
            health_results['client_sentiment_analyzer'] = {'status': 'unhealthy', 'error': str(e)}
    
    # Log health check results
    for service, health in health_results.items():
        status = health.get('status', 'unknown')
        logger.info(f"Health check - {service}: {status}")
        if status != 'healthy':
            logger.warning(f"Service {service} health issue: {health}")

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    Advanced Sales Sentiment Analysis API with Retrieval-Augmented Generation (RAG) capabilities.
    
    This API analyzes both salesperson performance and client engagement from CRM activities using:
    - Historical deal pattern matching through RAG
    - LLM-powered sentiment analysis for sales and client perspectives
    - Modular context engineering
    - Multiple LLM provider support
    
    ## Features
    - **Sales Sentiment Analysis**: Analyze salesperson sentiment from deal activities
    - **Client Sentiment Analysis**: Analyze client engagement and buying intent
    - **RAG Context**: Retrieve relevant examples from past deals for better analysis
    - **Knowledge Base Management**: Build and manage historical deal patterns
    - **Multi-Provider Support**: Azure OpenAI, OpenAI, Anthropic, Groq
    - **Production Ready**: Rate limiting, caching, authentication, monitoring
    """,
    contact={
        "name": "Sales Sentiment RAG API",
        "email": "support@company.com",
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

# Add middleware (order matters!)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthenticationMiddleware)

# Include API routes
app.include_router(router, prefix=settings.API_PREFIX)

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
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Advanced Sales Sentiment Analysis with RAG capabilities",
        "docs_url": "/docs",
        "health_url": "/health",
        "api_prefix": settings.API_PREFIX,
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
        
        # Check sales sentiment analyzer
        sales_sentiment_analyzer = service_registry.get('sales_sentiment_analyzer')
        if sales_sentiment_analyzer:
            try:
                stats = sales_sentiment_analyzer.get_analyzer_stats()
                health_results["services"]["sales_sentiment_analyzer"] = {
                    "status": "healthy",
                    "stats": stats
                }
            except Exception as e:
                health_results["services"]["sales_sentiment_analyzer"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        # Check client sentiment analyzer
        client_sentiment_analyzer = service_registry.get('client_sentiment_analyzer')
        if client_sentiment_analyzer:
            try:
                stats = client_sentiment_analyzer.get_analyzer_stats()
                health_results["services"]["client_sentiment_analyzer"] = {
                    "status": "healthy",
                    "stats": stats
                }
            except Exception as e:
                health_results["services"]["client_sentiment_analyzer"] = {
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
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )