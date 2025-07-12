from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

class ActivityType(str, Enum):
    EMAIL = "email"
    CALL = "call"
    MEETING = "meeting"
    NOTE = "note"
    TASK = "task"

class ProcessedActivity(BaseModel):
    """Structured representation of a deal activity"""
    deal_id: str
    activity_type: ActivityType
    timestamp: Optional[datetime] = None
    content: str
    direction: Optional[str] = None
    metadata: Dict[str, Any] = {}

class EmailActivityRaw(BaseModel):
    """Raw email activity from CRM data"""
    activity_type: str = Field(default="email")
    sent_at: Optional[str] = None
    from_email: Optional[str] = Field(None, alias="from")
    to: Optional[List[str]] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    state: Optional[str] = None
    direction: Optional[str] = None

class CallActivityRaw(BaseModel):
    """Raw call activity from CRM data"""
    activity_type: str = Field(default="call")
    id: Optional[str] = None
    createdate: Optional[str] = None
    call_title: Optional[str] = None
    call_body: Optional[str] = None
    call_direction: Optional[str] = None
    call_duration: Optional[int] = None
    call_status: Optional[str] = None

class MeetingActivityRaw(BaseModel):
    """Raw meeting activity from CRM data"""
    activity_type: str = Field(default="meeting")
    id: Optional[str] = None
    internal_meeting_notes: Optional[str] = None
    meeting_title: Optional[str] = None
    meeting_location: Optional[str] = None
    meeting_location_type: Optional[str] = None
    meeting_outcome: Optional[str] = None
    meeting_start_time: Optional[str] = None
    meeting_end_time: Optional[str] = None

class NoteActivityRaw(BaseModel):
    """Raw note activity from CRM data"""
    activity_type: str = Field(default="note")
    id: Optional[str] = None
    createdate: Optional[str] = None
    lastmodifieddate: Optional[str] = None
    note_body: Optional[str] = None

class TaskActivityRaw(BaseModel):
    """Raw task activity from CRM data"""
    activity_type: str = Field(default="task")
    id: Optional[str] = None
    createdate: Optional[str] = None
    task_priority: Optional[str] = None
    task_status: Optional[str] = None
    task_type: Optional[str] = None
    task_subject: Optional[str] = None
    task_body: Optional[str] = None

# Union type for raw activities
RawActivityData = Union[EmailActivityRaw, CallActivityRaw, MeetingActivityRaw, NoteActivityRaw, TaskActivityRaw]

class DealPattern(BaseModel):
    """Complete deal pattern for vector storage"""
    deal_id: str
    combined_text: str
    activities_count: int
    activity_types: List[str]
    time_span_days: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class VectorSearchResult(BaseModel):
    """Result from vector similarity search"""
    deal_id: str
    similarity_score: float
    metadata: Dict[str, Any]
    combined_text: str

class DealMetrics(BaseModel):
    """Calculated metrics for a deal"""
    deal_id: str
    total_activities: int
    activity_type_counts: Dict[str, int]
    time_span_days: int
    avg_time_between_activities_hours: float
    response_time_metrics: Dict[str, float]
    communication_gaps_count: int
    business_hours_ratio: float
    weekend_activity_ratio: float
    activity_frequency_trend: str
    email_ratio: float
    
class DealCharacteristics(BaseModel):
    """Deal characteristics and metadata"""
    deal_id: str
    deal_amount: float
    deal_stage: str
    deal_type: str
    deal_probability: float
    deal_outcome: str
    deal_size_category: str
    probability_category: str
    create_date: str
    close_date: Optional[str] = None
    deal_age_days: int
    is_closed: bool
    is_won: bool
    is_lost: bool
    is_open: bool
    is_new_business: bool

class ProcessedDealData(BaseModel):
    """Complete processed deal data"""
    deal_id: str
    raw_deal_data: Dict[str, Any]
    processed_activities: List[ProcessedActivity]
    deal_metrics: DealMetrics
    deal_characteristics: DealCharacteristics
    combined_text: str
    embedding: Optional[List[float]] = None
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)

class ActivityAnalysis(BaseModel):
    """Analysis result for individual activity"""
    activity_type: str
    sentiment: str
    sentiment_score: float
    key_indicators: List[str]
    performance_rating: str
    count: int

class DealAnalysisResult(BaseModel):
    """Complete deal analysis result"""
    deal_id: str
    overall_sentiment: str
    sentiment_score: float
    confidence: float
    activity_breakdown: Dict[str, ActivityAnalysis]
    deal_momentum_indicators: Dict[str, str]
    performance_analysis: Dict[str, str]
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
    timestamp: datetime = Field(default_factory=datetime.utcnow)