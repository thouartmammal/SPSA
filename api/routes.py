import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Literal

from fastapi import APIRouter, HTTPException, status, Depends, Request, BackgroundTasks, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from config.settings import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# =================== REQUEST/RESPONSE MODELS ===================

class BaseActivity(BaseModel):
    """Base activity model with common fields"""
    activity_type: str = Field(..., description="Type of activity")
    
    class Config:
        extra = "allow"  # Allow extra fields for flexibility

class EmailActivity(BaseActivity):
    """Email activity data"""
    activity_type: Literal["email"] = "email"
    sent_at: Optional[str] = None
    from_email: Optional[str] = Field(None, alias="from")
    to: Optional[List[str]] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    direction: Optional[str] = None

class CallActivity(BaseActivity):
    """Call activity data"""
    activity_type: Literal["call"] = "call"
    id: Optional[str] = None
    createdate: Optional[str] = None
    call_title: Optional[str] = None
    call_body: Optional[str] = None
    call_direction: Optional[str] = None
    call_duration: Optional[int] = None
    call_status: Optional[str] = None

class MeetingActivity(BaseActivity):
    """Meeting activity data"""
    activity_type: Literal["meeting"] = "meeting"
    id: Optional[str] = None
    internal_meeting_notes: Optional[str] = None
    meeting_title: Optional[str] = None
    meeting_location: Optional[str] = None
    meeting_location_type: Optional[str] = None
    meeting_outcome: Optional[str] = None
    meeting_start_time: Optional[str] = None
    meeting_end_time: Optional[str] = None

class NoteActivity(BaseActivity):
    """Note activity data"""
    activity_type: Literal["note"] = "note"
    id: Optional[str] = None
    createdate: Optional[str] = None
    lastmodifieddate: Optional[str] = None
    note_body: Optional[str] = None

class TaskActivity(BaseActivity):
    """Task activity data"""
    activity_type: Literal["task"] = "task"
    id: Optional[str] = None
    createdate: Optional[str] = None
    task_priority: Optional[str] = None
    task_status: Optional[str] = None
    task_type: Optional[str] = None
    task_subject: Optional[str] = None
    task_body: Optional[str] = None

# Union type for activities
ActivityData = Union[EmailActivity, CallActivity, MeetingActivity, NoteActivity, TaskActivity]

class DealData(BaseModel):
    """Deal data structure"""
    deal_id: str = Field(..., description="Deal identifier")
    activities: List[Dict[str, Any]] = Field(..., description="List of activities")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Deal metadata")
    
    @validator('deal_id')
    def validate_deal_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Deal ID cannot be empty")
        return str(v).strip()
    
    @validator('activities')
    def validate_activities(cls, v):
        if not v or not isinstance(v, list):
            raise ValueError("Activities must be a non-empty list")
        return v

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
class ActivityBreakdownItem(BaseModel):
    """Individual activity breakdown"""
    sentiment: str
    sentiment_score: float
    key_indicators: List[str]
    count: int

class DealMomentumIndicators(BaseModel):
    """Deal momentum indicators"""
    stage_progression: str
    client_engagement_trend: str
    competitive_position: str

class SentimentAnalysisResponse(BaseModel):
    """Sentiment analysis response"""
    overall_sentiment: str
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    activity_breakdown: Dict[str, ActivityBreakdownItem]
    deal_momentum_indicators: DealMomentumIndicators
    reasoning: str
    professional_gaps: List[str]
    excellence_indicators: List[str]
    risk_indicators: List[str]
    opportunity_indicators: List[str]
    temporal_trend: str
    recommended_actions: List[str]
    context_analysis_notes: List[str]
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

# =================== DEPENDENCY FUNCTIONS ===================

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

def get_knowledge_base_builder(services: dict = Depends(get_services)):
    """Get knowledge base builder service"""
    kb_builder = services.get('knowledge_base_builder')
    if not kb_builder:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base builder service not available"
        )
    return kb_builder

def get_vector_store(services: dict = Depends(get_services)):
    """Get vector store service"""
    vector_store = services.get('vector_store')
    if not vector_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store service not available"
        )
    return vector_store

def get_cache_manager(services: dict = Depends(get_services)):
    """Get cache manager service"""
    cache_manager = services.get('cache_manager')
    if not cache_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache manager service not available"
        )
    return cache_manager

def get_request_id(request: Request) -> str:
    """Get request ID from request state"""
    return getattr(request.state, 'request_id', 'unknown')

# =================== API ENDPOINTS ===================

@router.post("/analyze/sentiment", response_model=SentimentAnalysisResponse, tags=["Analysis"])
async def analyze_sentiment(
    request: Request,
    sentiment_analyzer = Depends(get_sentiment_analyzer),
    analysis_request: AnalysisRequest = Body(
        ...,
        example={
            "deal_data": {
                "deal_id": "deal_12345678",
                "activities": [
                    {
                        "sent_at": "2024-02-13T18:40:10.028Z",
                        "from": None,
                        "to": ["contact1@client-company.com", "contact2@client-company.com"],
                        "subject": "Proposal Document for Review",
                        "body": "Hi [Client Name], Great to speak with you yesterday. I've prepared the proposal document which I'm attaching here for your review. As discussed, I've included the optional extended support package and outlined the implementation timeline. Let me know if you have any questions about this document!",
                        "state": "email",
                        "direction": "outgoing",
                        "activity_type": "email"
                    },
                    {
                        "sent_at": "2024-02-15T14:30:42.690Z",
                        "from": "decision-maker@client-company.com",
                        "to": ["salesperson@your-company.com"],
                        "subject": "RE: Proposal Document for Review",
                        "body": "Hi [Salesperson], Thanks for sending this over so quickly. I have a few questions about the pricing structure and would like to discuss the implementation timeline. Can we schedule a call this week?",
                        "state": "thread",
                        "direction": "incoming",
                        "activity_type": "email"
                    },
                    {
                        "id": "call_001",
                        "createdate": "2024-02-16T15:30:00Z",
                        "call_title": "Proposal Discussion Call",
                        "call_body": "Discussed proposal terms, pricing structure, and implementation timeline. Client had questions about ROI projections and integration requirements. Positive reception overall, they're moving forward with internal review process.",
                        "call_direction": "outgoing",
                        "call_duration": 45,
                        "call_status": "completed",
                        "activity_type": "call"
                    },
                    {
                        "id": "meeting_001", 
                        "meeting_title": "Account Strategy Planning Session",
                        "internal_meeting_notes": "Internal team meeting to discuss account strategy and next steps. Reviewed client's current pain points, budget parameters, and decision timeline. Need to emphasize value proposition and provide additional case studies.",
                        "meeting_location": "Conference Room A",
                        "meeting_location_type": "office",
                        "meeting_outcome": "action_items_defined",
                        "meeting_start_time": "2024-02-14T14:00:00Z",
                        "meeting_end_time": "2024-02-14T15:00:00Z",
                        "activity_type": "meeting"
                    },
                    {
                        "id": "note_001",
                        "createdate": "2024-02-17T16:20:00Z",
                        "lastmodifieddate": "2024-02-17T16:25:00Z",
                        "note_body": "Client mentioned their procurement process requires additional approvals. Important to clarify timeline expectations and provide references from similar implementations. Decision maker prefers morning calls due to time zone differences.",
                        "activity_type": "note"
                    },
                    {
                        "id": "task_001",
                        "createdate": "2024-02-16T20:47:18.511Z",
                        "task_priority": "HIGH",
                        "task_status": "COMPLETED",
                        "task_type": "EMAIL",
                        "task_subject": "Send Additional Case Studies",
                        "task_body": "Client requested case studies from similar implementations. Need to send 2-3 relevant examples showing ROI and implementation success. Also need to clarify if pilot program option is available.",
                        "activity_type": "task"
                    },
                    {
                        "id": "task_002",
                        "createdate": "2024-02-18T10:15:04.145Z",
                        "task_priority": "MEDIUM",
                        "task_status": "IN_PROGRESS",
                        "task_type": "FOLLOW_UP",
                        "task_subject": "Schedule Technical Demo",
                        "task_body": "Client expressed interest in technical demonstration. Need to coordinate with product team and schedule demo for next week. Ensure technical requirements are documented beforehand.",
                        "activity_type": "task"
                    }
                ],
                "metadata": {
                    "deal_amount": 25000,
                    "deal_stage": "Proposal",
                    "deal_type": "newbusiness",
                    "deal_probability": 0.75,
                    "createdate": "2024-01-15T09:00:00Z"
                }
            },
            "include_rag_context": True,
            "analysis_options": {
                "focus_area": "communication_effectiveness",
                "include_benchmarking": True
            }
        }
    )
):
    """
    Analyze sentiment for a single deal
    
    This endpoint analyzes salesperson sentiment from deal activities using:
    - RAG retrieval of relevant historical examples
    - LLM-powered sentiment analysis
    - Modular context engineering
    
    Returns detailed sentiment analysis including:
    - Overall sentiment score and confidence
    - Activity breakdown by type
    - Deal momentum indicators
    - Professional gaps and excellence indicators
    - Risk and opportunity indicators
    - Actionable recommendations
    """
    
    try:
        request_id = get_request_id(request)
        logger.info(f"[{request_id}] Analyzing sentiment for deal {analysis_request.deal_data.deal_id}")
        
        # Convert deal data to dict format expected by analyzer
        deal_dict = {
            'deal_id': analysis_request.deal_data.deal_id,
            'activities': analysis_request.deal_data.activities,
            'metadata': analysis_request.deal_data.metadata or {}
        }
        
        # Perform sentiment analysis
        result = sentiment_analyzer.analyze_deal_sentiment(
            deal_data=deal_dict,
            include_rag_context=analysis_request.include_rag_context
        )
        
        # Check for errors
        if 'error' in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Analysis failed: {result['error']}"
            )
        
        logger.info(f"[{request_id}] Sentiment analysis completed for deal {analysis_request.deal_data.deal_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@router.post("/analyze/batch", response_model=BatchAnalysisResponse, tags=["Analysis"])
async def analyze_batch_sentiment(
    request: Request,
    background_tasks: BackgroundTasks,
    sentiment_analyzer = Depends(get_sentiment_analyzer),
    batch_request: BatchAnalysisRequest = Body(
        ...,
        example={
            "deals": [
                {
                    "deal_id": "deal_001",
                    "activities": [
                        {
                            "activity_type": "email",
                            "subject": "Follow-up on proposal",
                            "body": "Thanks for the meeting. I'll send the revised proposal.",
                            "direction": "outgoing"
                        }
                    ],
                    "metadata": {
                        "deal_amount": 50000,
                        "deal_stage": "proposal"
                    }
                },
                {
                    "deal_id": "deal_002",
                    "activities": [
                        {
                            "activity_type": "call",
                            "call_title": "Discovery call",
                            "call_body": "Discussed requirements and next steps"
                        }
                    ],
                    "metadata": {
                        "deal_amount": 75000,
                        "deal_stage": "discovery"
                    }
                }
            ],
            "include_rag_context": True,
            "batch_size": 10
        }
    )
):
    """
    Analyze sentiment for multiple deals in batch
    
    Processes multiple deals efficiently with:
    - Batch processing for performance
    - Individual error handling
    - Progress tracking
    - Detailed batch statistics
    """
    
    try:
        request_id = get_request_id(request)
        logger.info(f"[{request_id}] Starting batch analysis for {len(batch_request.deals)} deals")
        
        # Convert deals to dict format
        deals_dict = [
            {
                'deal_id': deal.deal_id,
                'activities': deal.activities,
                'metadata': deal.metadata or {}
            }
            for deal in batch_request.deals
        ]
        
        # Perform batch analysis
        result = sentiment_analyzer.analyze_batch_sentiment(
            deals_data=deals_dict,
            include_rag_context=batch_request.include_rag_context
        )
        
        logger.info(f"[{request_id}] Batch analysis completed: {result.get('successful_analyses', 0)} successful")
        return result
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch analysis failed: {str(e)}"
        )

@router.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_knowledge_base(
    request: Request,
    rag_retriever = Depends(get_rag_retriever),
    vector_store = Depends(get_vector_store),
    search_request: SearchRequest = Body(
        ...,
        example={
            "query": "proposal discussion client questions pricing",
            "top_k": 5,
            "filters": {
                "deal_stage": "proposal",
                "deal_outcome": "won"
            },
            "include_metadata": True
        }
    )
):
    """
    Search knowledge base for relevant deals
    
    Searches historical deals using:
    - Vector similarity search
    - Metadata filtering
    - Relevance ranking
    - Configurable result limits
    """
    
    try:
        request_id = get_request_id(request)
        logger.info(f"[{request_id}] Searching knowledge base with query: {search_request.query[:50]}...")
        
        start_time = datetime.utcnow()
        
        # Perform vector search
        from core.embedding_service import get_embedding_service
        embedding_service = get_embedding_service()
        
        # Generate query embedding
        query_embedding = embedding_service.encode(search_request.query)
        
        # Search vector store
        search_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=search_request.top_k,
            filters=search_request.filters
        )
        
        # Format results
        results = []
        for result in search_results:
            formatted_result = {
                'deal_id': result.deal_id,
                'similarity_score': result.similarity_score,
                'content': result.combined_text[:500] + "..." if len(result.combined_text) > 500 else result.combined_text
            }
            
            if search_request.include_metadata:
                formatted_result['metadata'] = result.metadata
            
            results.append(formatted_result)
        
        end_time = datetime.utcnow()
        search_time_ms = (end_time - start_time).total_seconds() * 1000
        
        response = SearchResponse(
            query=search_request.query,
            results=results,
            total_results=len(results),
            search_time_ms=search_time_ms
        )
        
        logger.info(f"[{request_id}] Search completed: {len(results)} results in {search_time_ms:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Error in knowledge base search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@router.post("/knowledge-base/build", response_model=KnowledgeBaseBuildResponse, tags=["Knowledge Base"])
async def build_knowledge_base(
    request: Request,
    background_tasks: BackgroundTasks,
    kb_builder = Depends(get_knowledge_base_builder),
    build_request: KnowledgeBaseBuildRequest = Body(
        ...,
        example={
            "data_sources": ["data/final_deal_details.json"],
            "rebuild": True,
            "batch_size": 50
        }
    )
):
    """
    Build knowledge base from deal data
    
    Processes deal data to create searchable knowledge base:
    - Batch processing for efficiency
    - Embedding generation
    - Vector storage
    - Incremental updates support
    """
    
    try:
        request_id = get_request_id(request)
        logger.info(f"[{request_id}] Building knowledge base from {len(build_request.data_sources)} data sources")
        
        # Use primary data source (first one or default)
        data_source = build_request.data_sources[0] if build_request.data_sources else settings.DATA_PATH
        
        # Build knowledge base
        result = kb_builder.build_knowledge_base(
            data_file_path=data_source,
            rebuild=build_request.rebuild
        )
        
        # Format response
        response = KnowledgeBaseBuildResponse(
            success=result.get('status') == 'completed',
            total_deals_processed=result.get('total_deals_processed', 0),
            build_duration_seconds=result.get('build_duration_seconds', 0),
            knowledge_base_statistics=result.get('knowledge_base_statistics', {}),
            processing_statistics={
                'successful_embeddings': result.get('successful_embeddings', 0),
                'failed_embeddings': result.get('failed_embeddings', 0),
                'status': result.get('status', 'unknown')
            }
        )
        
        logger.info(f"[{request_id}] Knowledge base build completed: {response.total_deals_processed} deals processed")
        return response
        
    except Exception as e:
        logger.error(f"Error building knowledge base: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge base build failed: {str(e)}"
        )

@router.get("/knowledge-base/status", response_model=KnowledgeBaseStatus, tags=["Knowledge Base"])
async def get_knowledge_base_status(
    request: Request,
    kb_builder = Depends(get_knowledge_base_builder),
    vector_store = Depends(get_vector_store)
):
    """
    Get knowledge base status and statistics
    
    Returns:
    - Current status
    - Statistics (total deals, vectors, etc.)
    - Health metrics
    - Last update information
    """
    
    try:
        request_id = get_request_id(request)
        logger.info(f"[{request_id}] Getting knowledge base status")
        
        # Get knowledge base stats
        kb_stats = kb_builder.get_knowledge_base_stats()
        
        # Get vector store stats
        vector_stats = vector_store.get_stats()
        
        # Determine status
        status_value = "healthy" if vector_stats.get('total_vectors', 0) > 0 else "empty"
        
        response = KnowledgeBaseStatus(
            status=status_value,
            stats=kb_stats,
            health_metrics=vector_stats,
            last_checked=datetime.utcnow().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting knowledge base status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get knowledge base status: {str(e)}"
        )

@router.delete("/knowledge-base/clear", response_model=MessageResponse, tags=["Knowledge Base"])
async def clear_knowledge_base(
    request: Request,
    kb_builder = Depends(get_knowledge_base_builder)
):
    """
    Clear knowledge base
    
    Removes all stored deal patterns and embeddings.
    This action cannot be undone.
    """
    
    try:
        request_id = get_request_id(request)
        logger.warning(f"[{request_id}] Clearing knowledge base")
        
        result = kb_builder.clear_knowledge_base()
        
        if result.get('status') == 'cleared':
            return MessageResponse(message="Knowledge base cleared successfully")
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to clear knowledge base: {result.get('error', 'Unknown error')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear knowledge base: {str(e)}"
        )

@router.get("/system/stats", tags=["System"])
async def get_system_stats(
    request: Request,
    services: dict = Depends(get_services)
):
    """
    Get system statistics and performance metrics
    
    Returns comprehensive system information including:
    - Service health status
    - Performance metrics
    - Configuration settings
    - Resource usage
    """
    
    try:
        request_id = get_request_id(request)
        logger.info(f"[{request_id}] Getting system statistics")
        
        stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'configuration': {
                'llm_provider': settings.LLM_PROVIDER,
                'embedding_service': settings.EMBEDDING_SERVICE,
                'vector_db': settings.VECTOR_DB,
                'cache_enabled': settings.CACHE_ENABLED,
                'rate_limit_enabled': settings.RATE_LIMIT_ENABLED,
                'auth_required': settings.REQUIRE_AUTH
            },
            'services': {}
        }
        
        # Get sentiment analyzer stats
        sentiment_analyzer = services.get('sentiment_analyzer')
        if sentiment_analyzer:
            stats['services']['sentiment_analyzer'] = sentiment_analyzer.get_analyzer_stats()
        
        # Get RAG retriever stats
        rag_retriever = services.get('rag_retriever')
        if rag_retriever:
            stats['services']['rag_retriever'] = rag_retriever.get_retrieval_stats()
        
        # Get knowledge base stats
        kb_builder = services.get('knowledge_base_builder')
        if kb_builder:
            stats['services']['knowledge_base'] = kb_builder.get_knowledge_base_stats()
        
        # Get cache stats
        cache_manager = services.get('cache_manager')
        if cache_manager:
            stats['services']['cache'] = cache_manager.get_stats()
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system stats: {str(e)}"
        )

@router.post("/cache/clear", response_model=MessageResponse, tags=["System"])
async def clear_cache(
    request: Request,
    cache_manager = Depends(get_cache_manager),
    cache_type: Optional[str] = Body(
        None,
        example="embedding",
        description="Type of cache to clear (general, embedding, llm) or null for all"
    )
):
    """
    Clear cache by type or all cache
    
    Args:
        cache_type: Type of cache to clear (general, embedding, llm) or None for all
    """
    
    try:
        request_id = get_request_id(request)
        logger.info(f"[{request_id}] Clearing cache: {cache_type or 'all'}")
        
        success = cache_manager.clear_cache(cache_type)
        
        if success:
            message = f"Cache cleared successfully: {cache_type or 'all'}"
            return MessageResponse(message=message)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear cache"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )