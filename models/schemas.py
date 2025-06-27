from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
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
