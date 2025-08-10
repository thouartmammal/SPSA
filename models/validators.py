from pydantic import BaseModel, Field, ValidationError
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class LLM_response_RV(BaseModel):
    
    """ 
        Base response schema returned by the LLM analysis. 
    """
    
    overall_sentiment: str
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    activity_breakdown: str
    reasoning: str

class sales_specific_RV(LLM_response_RV):
    
    """
    Response schema for sales-specific sentiment analysis.

    Extends:
        LLMResponseRV
    """
    
    deal_momentum_indicators: str

class client_specific_RV(LLM_response_RV):

    """
    Response schema for client-specific sentiment analysis.

    Extends:
        LLMResponseRV
    """
    
    client_engagement_indicators: str

def parse_activity_safely(result: dict[str, Any]) -> Optional[LLM_response_RV]:
    
    """
    Attempts to parse and validate an LLM response dict using the appropriate schema
    based on the 'analysis_type' field.

    Args:
        result (Dict[str, Any]): The dictionary containing the raw response from the LLM.

    Returns:
        Optional[LLMResponseRV]: A validated model object if parsing succeeds, otherwise None.
    """
    
    analysis_type = result.get("analysis_type", "").lower()

    model_map = {
        'sales': sales_specific_RV,
        'client': client_specific_RV
    }
    
    model = model_map.get(analysis_type)

    if not model:
        logger.error(f"Unknown analysis type: '{analysis_type}'")
        return None

    try:
        validated = model.parse_obj(result)
        return validated
    except ValidationError as e:
        logger.error(f"Validation error for analysis type '{analysis_type}': {e}")
        logger.debug(f"Invalid response data: {result}")
        return None