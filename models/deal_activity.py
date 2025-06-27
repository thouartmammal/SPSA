from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, validator

class RawDealActivity(BaseModel):
    """Raw deal activity as loaded from JSON"""
    deal_id: str
    activities: List[Dict[str, Any]]
    
    @validator('deal_id')
    def deal_id_must_be_string(cls, v):
        return str(v)

class DealActivityMetrics(BaseModel):
    """Calculated metrics for a deal"""
    deal_id: str
    total_activities: int
    activity_types: Dict[str, int]
    time_span_days: int
    avg_time_between_activities_hours: float
    response_time_metrics: Dict[str, float]
    communication_gaps_count: int
    business_hours_ratio: float
    activity_frequency_trend: str