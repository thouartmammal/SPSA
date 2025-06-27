import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, status, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Pydantic models for API
class ActivityData(BaseModel):
    """Single CRM activity data"""
    activity_type: str = Field(..., description="Type of activity (email, call, meeting, note, task)")
    timestamp: Optional[str] = None
    content: Optional[str] = None
    direction: Optional[str] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    sent_at: Optional[str] = None
    createdate: Optional[str] = None
    call_title: Optional[str] = None
    call_body: Optional[str] = None
    call_duration: Optional[int] = None
    meeting_title: Optional[str] = None
    meeting_start_time: Optional[str] = None
    internal_meeting_notes: Optional[str] = None
    note_body: Optional[str] = None
    lastmodifieddate: Optional[str] = None
    task_subject: Optional[str] = None
    task_body: Optional[str] = None
    task_status: Optional[str] = None
    task_priority: Optional[str] = None
    to: Optional[List[str]] = None

class DealData(BaseModel):
    """Deal data for analysis"""
    deal_id: str = Field(..., description="Unique deal identifier")
    activities: List[ActivityData] = Field(..., description="List of deal activities")
    amount: Optional[float] = Field(None, description="Deal amount")
    dealstage: Optional[str] = Field(None, description="Current deal stage")
    dealtype: Optional[str] = Field(None, description="Type of deal")
    deal_stage_probability: Optional[float] = Field(None, description="Deal probability percentage")
    createdate: Optional[str] = Field(None, description="Deal creation date")
    closedate: Optional[str] = Field(None, description="Deal close date")

class AnalysisRequest(BaseModel):
    """Request for sentiment analysis"""
    deal_data: DealData = Field(..., description="Deal data to analyze")
    analysis_type: str = Field("sentiment", description="Type of analysis")
    include_rag_context: bool = Field(True, description="Whether to include RAG context")
    analysis_options: Optional[Dict[str, Any]] = Field(None, description="Additional analysis options")

class BatchAnalysisRequest(BaseModel):
    """Request for batch analysis"""
    deals: List[DealData] = Field(..., description="List of deals to analyze")
    analysis_type: str = Field("sentiment", description="Type of analysis")
    include_rag_context: bool = Field(True, description="Whether to include RAG context")
    batch_size: int = Field(10, description="Batch processing size")

class SearchRequest(BaseModel):
    """Request for knowledge base search"""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(10, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    include_metadata: bool = Field(True, description="Whether to include metadata")

class KnowledgeBaseBuildRequest(BaseModel):
    """Request to build knowledge base"""
    data_sources: List[str] = Field(..., description="List of data file paths")
    rebuild: bool = Field(False, description="Whether to rebuild existing knowledge base")
    batch_size: int = Field(50, description="Batch processing size")

# Response models
class SentimentAnalysisResponse(BaseModel):
    """Sentiment analysis response"""
    deal_id: str
    overall_sentiment: str
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    activity_breakdown: Dict[str, Any]
    deal_momentum_indicators: Dict[str, Any]
    performance_analysis: Dict[str, Any]
    reasoning: str
    professional_gaps: List[str]
    excellence_indicators: List[str]
    risk_indicators: List[str]
    opportunity_indicators: List[str]
    temporal_trend: str
    recommended_actions: List[str]
    context_analysis_notes: List[str]
    benchmark_comparison: str
    analysis_metadata: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class SearchResponse(BaseModel):
    """Search response"""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time_ms: float
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class BatchAnalysisResponse(BaseModel):
    """Batch analysis response"""
    total_deals: int
    successful_analyses: int
    failed_analyses: int
    results: List[Dict[str, Any]]
    processing_time_seconds: float
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class MessageResponse(BaseModel):
    """Simple message response"""
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class KnowledgeBaseStatus(BaseModel):
    """Knowledge base status"""
    status: str
    stats: Dict[str, Any]
    health_metrics: Dict[str, Any]
    last_checked: str

class KnowledgeBaseBuildResponse(BaseModel):
    """Knowledge base build response"""
    success: bool
    total_deals_processed: int
    build_duration_seconds: float
    knowledge_base_statistics: Dict[str, Any]
    processing_statistics: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

# Dependency to get services from app state
def get_services(request: Request):
    """Get services from application state"""
    return request.app.state.services

def get_sentiment_analyzer(services: dict = Depends(get_services)):
    """Get sentiment analyzer service"""
    analyzer = services.get('sentiment_analyzer')
    if not analyzer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment analyzer service not available"
        )
    return analyzer

def get_rag_retriever(services: dict = Depends(get_services)):
    """Get RAG retriever service"""
    retriever = services.get('rag_retriever')
    if not retriever:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG retriever service not available"
        )
    return retriever

def get_knowledge_base_manager(services: dict = Depends(get_services)):
    """Get knowledge base manager service"""
    kb_manager = services.get('knowledge_base_manager')
    if not kb_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base manager service not available"
        )
    return kb_manager

def get_cache_manager(services: dict = Depends(get_services)):
    """Get cache manager service"""
    cache = services.get('cache_manager')
    if not cache:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache manager service not available"
        )
    return cache

def get_request_id(request: Request) -> str:
    """Get request ID from request state"""
    return getattr(request.state, 'request_id', 'unknown')

# Analysis endpoints
@router.post(
    "/analyze/sentiment",
    response_model=SentimentAnalysisResponse,
    summary="Analyze salesperson sentiment",
    description="Analyze salesperson sentiment and performance from deal activities",
    tags=["Analysis"]
)
async def analyze_sentiment(
    request: AnalysisRequest,
    req: Request,
    analyzer=Depends(get_sentiment_analyzer)
):
    """Analyze salesperson sentiment from deal activities"""
    
    request_id = get_request_id(req)
    logger.info(f"[{request_id}] Starting sentiment analysis for deal {request.deal_data.deal_id}")
    
    try:
        start_time = time.time()
        
        # Convert request to dict format expected by analyzer
        deal_data_dict = {
            "deal_id": request.deal_data.deal_id,
            "activities": [activity.dict() for activity in request.deal_data.activities],
            **request.deal_data.dict(exclude={"deal_id", "activities"})
        }
        
        # Perform analysis
        result = analyzer.analyze_deal_sentiment(
            deal_data=deal_data_dict,
            include_rag_context=request.include_rag_context,
            analysis_options=request.analysis_options or {}
        )
        
        processing_time = time.time() - start_time
        
        # Add processing metadata
        result["analysis_metadata"]["processing_time_seconds"] = processing_time
        result["analysis_metadata"]["request_id"] = request_id
        result["analysis_metadata"]["analysis_type"] = request.analysis_type
        
        logger.info(f"[{request_id}] Sentiment analysis completed in {processing_time:.2f}s")
        
        return SentimentAnalysisResponse(**result)
        
    except Exception as e:
        logger.error(f"[{request_id}] Sentiment analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@router.post(
    "/analyze/batch",
    response_model=BatchAnalysisResponse,
    summary="Batch analyze multiple deals",
    description="Analyze sentiment for multiple deals in a single request",
    tags=["Analysis"]
)
async def analyze_batch(
    request: BatchAnalysisRequest,
    req: Request,
    background_tasks: BackgroundTasks,
    analyzer=Depends(get_sentiment_analyzer)
):
    """Analyze sentiment for multiple deals"""
    
    request_id = get_request_id(req)
    logger.info(f"[{request_id}] Starting batch analysis for {len(request.deals)} deals")
    
    try:
        start_time = time.time()
        
        # Convert deals to expected format
        deals_data = []
        for deal in request.deals:
            deal_dict = {
                "deal_id": deal.deal_id,
                "activities": [activity.dict() for activity in deal.activities],
                **deal.dict(exclude={"deal_id", "activities"})
            }
            deals_data.append(deal_dict)
        
        # Perform batch analysis
        results = analyzer.batch_analyze_sentiment(
            deals_data=deals_data,
            include_rag_context=request.include_rag_context,
            batch_size=request.batch_size
        )
        
        processing_time = time.time() - start_time
        
        # Process results
        batch_results = []
        successful_count = 0
        failed_count = 0
        
        for result in results:
            if result.get("error"):
                batch_results.append({
                    "deal_id": result.get("deal_id", "unknown"),
                    "success": False,
                    "result": None,
                    "error": result["error"]
                })
                failed_count += 1
            else:
                batch_results.append({
                    "deal_id": result["deal_context"]["deal_id"],
                    "success": True,
                    "result": SentimentAnalysisResponse(**result),
                    "error": None
                })
                successful_count += 1
        
        logger.info(f"[{request_id}] Batch analysis completed: {successful_count} successful, {failed_count} failed")
        
        return BatchAnalysisResponse(
            total_deals=len(request.deals),
            successful_analyses=successful_count,
            failed_analyses=failed_count,
            results=batch_results,
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Batch analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch analysis failed: {str(e)}"
        )

# Search endpoints
@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search knowledge base",
    description="Search for similar deal patterns in the knowledge base",
    tags=["Search"]
)
async def search_knowledge_base(
    request: SearchRequest,
    req: Request,
    retriever=Depends(get_rag_retriever)
):
    """Search for similar patterns in knowledge base"""
    
    request_id = get_request_id(req)
    logger.info(f"[{request_id}] Searching knowledge base for: {request.query[:100]}...")
    
    try:
        start_time = time.time()
        
        # Perform search
        results = retriever.retrieve_similar_patterns(
            query_text=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
        
        search_time = time.time() - start_time
        
        # Format results
        search_results = []
        for result in results:
            search_results.append({
                "deal_id": result.deal_id,
                "similarity_score": result.similarity_score,
                "content_snippet": result.combined_text[:200] + "..." if len(result.combined_text) > 200 else result.combined_text,
                "metadata": result.metadata if request.include_metadata else {}
            })
        
        logger.info(f"[{request_id}] Search completed in {search_time:.3f}s, found {len(search_results)} results")
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=search_time * 1000
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@router.get(
    "/search/success-patterns/{query}",
    response_model=SearchResponse,
    summary="Search successful deal patterns",
    description="Search for patterns from successful (won) deals only",
    tags=["Search"]
)
async def search_success_patterns(
    query: str,
    top_k: int = 5,
    req: Request = None,
    retriever=Depends(get_rag_retriever)
):
    """Search for successful deal patterns"""
    
    request_id = get_request_id(req)
    logger.info(f"[{request_id}] Searching success patterns for: {query[:100]}...")
    
    try:
        start_time = time.time()
        
        results = retriever.get_success_patterns(query, top_k=top_k)
        search_time = time.time() - start_time
        
        # Format results
        search_results = []
        for result in results:
            search_results.append({
                "deal_id": result.deal_id,
                "similarity_score": result.similarity_score,
                "content_snippet": result.combined_text[:200] + "..." if len(result.combined_text) > 200 else result.combined_text,
                "metadata": result.metadata
            })
        
        return SearchResponse(
            query=f"Success patterns: {query}",
            results=search_results,
            total_results=len(search_results),
            search_time_ms=search_time * 1000
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Success patterns search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Success patterns search failed: {str(e)}"
        )

# Knowledge Base Management endpoints
@router.get(
    "/knowledge-base/status",
    response_model=KnowledgeBaseStatus,
    summary="Get knowledge base status",
    description="Get comprehensive status and health metrics for the knowledge base",
    tags=["Knowledge Base"]
)
async def get_knowledge_base_status(
    req: Request,
    kb_manager=Depends(get_knowledge_base_manager)
):
    """Get knowledge base status and health metrics"""
    
    request_id = get_request_id(req)
    logger.info(f"[{request_id}] Getting knowledge base status")
    
    try:
        status_data = kb_manager.get_knowledge_base_status()
        
        # Format for response model
        formatted_status = {
            "status": status_data.get("status", "unknown"),
            "stats": {
                "total_deals": status_data.get("metadata", {}).get("total_deals", 0),
                "deal_outcomes": {},
                "deal_sizes": {},
                "last_updated": status_data.get("metadata", {}).get("last_updated", datetime.utcnow().isoformat()),
                "embedding_dimension": status_data.get("metadata", {}).get("vector_dimension", 0)
            },
            "health_metrics": status_data.get("health_metrics", {}),
            "last_checked": status_data.get("last_checked", datetime.utcnow().isoformat())
        }
        
        return KnowledgeBaseStatus(**formatted_status)
        
    except Exception as e:
        logger.error(f"[{request_id}] Failed to get knowledge base status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get knowledge base status: {str(e)}"
        )

@router.post(
    "/knowledge-base/build",
    response_model=KnowledgeBaseBuildResponse,
    summary="Build knowledge base",
    description="Build or rebuild the knowledge base from data sources",
    tags=["Knowledge Base"]
)
async def build_knowledge_base(
    request: KnowledgeBaseBuildRequest,
    req: Request,
    background_tasks: BackgroundTasks,
    kb_manager=Depends(get_knowledge_base_manager)
):
    """Build knowledge base from data sources"""
    
    request_id = get_request_id(req)
    logger.info(f"[{request_id}] Starting knowledge base build with {len(request.data_sources)} sources")
    
    try:
        # Validate data sources exist
        from pathlib import Path
        for source in request.data_sources:
            if not Path(source).exists():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Data source not found: {source}"
                )
        
        # Start build process
        result = kb_manager.build_knowledge_base(
            data_sources=request.data_sources,
            rebuild=request.rebuild,
            batch_size=request.batch_size
        )
        
        if result["success"]:
            logger.info(f"[{request_id}] Knowledge base build completed successfully")
            return KnowledgeBaseBuildResponse(**result)
        else:
            logger.error(f"[{request_id}] Knowledge base build failed: {result.get('error')}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Build failed")
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Knowledge base build failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge base build failed: {str(e)}"
        )

# Utility endpoints
@router.get(
    "/deals/{deal_id}/insights",
    summary="Get deal insights",
    description="Get comprehensive insights for a specific deal",
    tags=["Insights"]
)
async def get_deal_insights(
    deal_id: str,
    req: Request,
    retriever=Depends(get_rag_retriever),
    cache=Depends(get_cache_manager)
):
    """Get comprehensive insights for a specific deal"""
    
    request_id = get_request_id(req)
    logger.info(f"[{request_id}] Getting insights for deal {deal_id}")
    
    try:
        # Check cache first
        cache_key = f"deal_insights_{deal_id}"
        cached_insights = cache.get(cache_key)
        
        if cached_insights:
            logger.info(f"[{request_id}] Returning cached insights for deal {deal_id}")
            return cached_insights
        
        # Generate insights (placeholder)
        insights = {
            "deal_id": deal_id,
            "summary": "Deal insights would be generated here",
            "similar_deals_count": 0,
            "success_probability": 0.0,
            "risk_factors": [],
            "opportunities": [],
            "recommendations": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Cache insights for 1 hour
        cache.set(cache_key, insights, ttl=3600)
        
        return insights
        
    except Exception as e:
        logger.error(f"[{request_id}] Failed to get deal insights: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get deal insights: {str(e)}"
        )

@router.post(
    "/cache/clear",
    response_model=MessageResponse,
    summary="Clear cache",
    description="Clear application cache",
    tags=["Utilities"]
)
async def clear_cache(
    pattern: Optional[str] = None,
    req: Request = None,
    cache=Depends(get_cache_manager)
):
    """Clear application cache"""
    
    request_id = get_request_id(req)
    logger.info(f"[{request_id}] Clearing cache with pattern: {pattern}")
    
    try:
        if pattern:
            cleared_count = cache.clear_pattern(pattern)
            message = f"Cleared {cleared_count} cache entries matching pattern: {pattern}"
        else:
            cache.clear_all()
            message = "Cleared all cache entries"
        
        logger.info(f"[{request_id}] {message}")
        
        return MessageResponse(message=message)
        
    except Exception as e:
        logger.error(f"[{request_id}] Failed to clear cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )

@router.get(
    "/cache/stats",
    summary="Get cache statistics",
    description="Get cache performance statistics",
    tags=["Utilities"]
)
async def get_cache_stats(
    req: Request,
    cache=Depends(get_cache_manager)
):
    """Get cache statistics"""
    
    request_id = get_request_id(req)
    logger.info(f"[{request_id}] Getting cache statistics")
    
    try:
        stats = cache.get_stats()
        
        return {
            "cache_stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Failed to get cache stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache stats: {str(e)}"
        )