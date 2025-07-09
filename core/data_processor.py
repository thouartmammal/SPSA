import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from models.schemas import ProcessedActivity, DealPattern, ProcessedDealData, DealMetrics, DealCharacteristics
from core.embedding_service import EmbeddingService
from config.settings import settings

logger = logging.getLogger(__name__)

class DealDataProcessor:
    """Simplified deal data processor focused on core functionality"""
    
    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize processor with embedding service
        
        Args:
            embedding_service: Initialized embedding service instance
        """
        self.embedding_service = embedding_service
        logger.info("Deal Data Processor initialized")
    
    def load_deal_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load deal data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} deals from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def process_deal(self, deal_data: Dict[str, Any]) -> ProcessedDealData:
        """
        Process a single deal into structured format
        
        Args:
            deal_data: Raw deal data
            
        Returns:
            Processed deal data
        """
        
        deal_id = str(deal_data.get('deal_id', 'unknown'))
        
        try:
            # Process activities
            activities = deal_data.get('activities', [])
            processed_activities = [
                self._parse_activity(activity, deal_id) 
                for activity in activities
            ]
            
            # Filter out None activities
            processed_activities = [a for a in processed_activities if a is not None]
            
            # Calculate metrics
            deal_metrics = self._calculate_deal_metrics(processed_activities, deal_data)
            
            # Extract characteristics
            deal_characteristics = self._extract_deal_characteristics(deal_data, deal_metrics)
            
            # Create combined text
            combined_text = self._create_combined_text(processed_activities)
            
            # Generate embedding
            embedding = None
            if combined_text.strip():
                try:
                    embedding = self.embedding_service.encode(combined_text)
                except Exception as e:
                    logger.error(f"Error generating embedding for deal {deal_id}: {e}")
            
            return ProcessedDealData(
                deal_id=deal_id,
                raw_deal_data=deal_data,
                processed_activities=processed_activities,
                deal_metrics=deal_metrics,
                deal_characteristics=deal_characteristics,
                combined_text=combined_text,
                embedding=embedding,
                processing_timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error processing deal {deal_id}: {e}")
            raise
    
    def _parse_activity(self, activity: Dict[str, Any], deal_id: str) -> Optional[ProcessedActivity]:
        """Parse a single activity into structured format"""
        
        try:
            activity_type = activity.get('activity_type', 'unknown')
            
            # Extract content based on activity type
            content = self._extract_activity_content(activity)
            
            if not content.strip():
                return None
            
            # Extract timestamp
            timestamp = self._extract_activity_timestamp(activity)
            
            # Extract direction
            direction = activity.get('direction') or activity.get('call_direction') or 'unknown'
            
            # Create metadata
            metadata = {
                'id': activity.get('id'),
                'createdate': activity.get('createdate'),
                'lastmodifieddate': activity.get('lastmodifieddate'),
                'original_activity_type': activity_type
            }
            
            # Add type-specific metadata
            if activity_type == 'email':
                metadata.update({
                    'subject': activity.get('subject'),
                    'from_email': activity.get('from'),
                    'to': activity.get('to'),
                    'state': activity.get('state')
                })
            elif activity_type == 'call':
                metadata.update({
                    'duration': activity.get('call_duration'),
                    'status': activity.get('call_status')
                })
            elif activity_type == 'meeting':
                metadata.update({
                    'location': activity.get('meeting_location'),
                    'location_type': activity.get('meeting_location_type'),
                    'outcome': activity.get('meeting_outcome'),
                    'start_time': activity.get('meeting_start_time'),
                    'end_time': activity.get('meeting_end_time')
                })
            elif activity_type == 'task':
                metadata.update({
                    'priority': activity.get('task_priority'),
                    'status': activity.get('task_status'),
                    'task_type': activity.get('task_type')
                })
            
            return ProcessedActivity(
                deal_id=deal_id,
                activity_type=activity_type,
                timestamp=timestamp,
                content=content,
                direction=direction,
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(f"Error parsing activity for deal {deal_id}: {e}")
            return None
    
    def _extract_activity_content(self, activity: Dict[str, Any]) -> str:
        """Extract content from activity based on type"""
        
        activity_type = activity.get('activity_type', 'unknown')
        content_parts = []
        
        if activity_type == 'email':
            subject = activity.get('subject', '').strip()
            body = activity.get('body', '').strip()
            
            if subject:
                content_parts.append(f"Subject: {subject}")
            if body:
                content_parts.append(f"Body: {body}")
        
        elif activity_type == 'call':
            title = activity.get('call_title', '').strip()
            body = activity.get('call_body', '').strip()
            
            if title:
                content_parts.append(f"Call: {title}")
            if body:
                content_parts.append(f"Notes: {body}")
        
        elif activity_type == 'meeting':
            title = activity.get('meeting_title', '').strip()
            notes = activity.get('internal_meeting_notes', '').strip()
            
            if title:
                content_parts.append(f"Meeting: {title}")
            if notes:
                content_parts.append(f"Notes: {notes}")
        
        elif activity_type == 'note':
            body = activity.get('note_body', '').strip()
            if body:
                content_parts.append(f"Note: {body}")
        
        elif activity_type == 'task':
            subject = activity.get('task_subject', '').strip()
            body = activity.get('task_body', '').strip()
            
            if subject:
                content_parts.append(f"Task: {subject}")
            if body:
                content_parts.append(f"Details: {body}")
        
        return '\n'.join(content_parts)
    
    def _extract_activity_timestamp(self, activity: Dict[str, Any]) -> Optional[datetime]:
        """Extract timestamp from activity"""
        
        activity_type = activity.get('activity_type', 'unknown')
        
        # Define timestamp fields for each activity type
        timestamp_fields = {
            'email': 'sent_at',
            'call': 'createdate',
            'meeting': 'meeting_start_time',
            'note': 'lastmodifieddate',
            'task': 'createdate'
        }
        
        timestamp_field = timestamp_fields.get(activity_type, 'createdate')
        timestamp_str = activity.get(timestamp_field)
        
        if not timestamp_str:
            return None
        
        # Try to parse various timestamp formats
        timestamp_formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d'
        ]
        
        for fmt in timestamp_formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse timestamp: {timestamp_str}")
        return None
    
    def _calculate_deal_metrics(self, activities: List[ProcessedActivity], deal_data: Dict[str, Any]) -> DealMetrics:
        """Calculate metrics for a deal"""
        
        deal_id = deal_data.get('deal_id', 'unknown')
        
        # Basic counts
        total_activities = len(activities)
        activity_type_counts = {}
        
        for activity in activities:
            activity_type = activity.activity_type
            activity_type_counts[activity_type] = activity_type_counts.get(activity_type, 0) + 1
        
        # Time span calculation
        timestamps = [a.timestamp for a in activities if a.timestamp]
        if timestamps:
            timestamps.sort()
            time_span = (timestamps[-1] - timestamps[0]).days
        else:
            time_span = 0
        
        # Simple metrics (complex calculations removed for focus)
        return DealMetrics(
            deal_id=deal_id,
            total_activities=total_activities,
            activity_type_counts=activity_type_counts,
            time_span_days=time_span,
            avg_time_between_activities_hours=0.0,  # Simplified
            response_time_metrics={},
            communication_gaps_count=0,
            business_hours_ratio=0.0,
            weekend_activity_ratio=0.0,
            activity_frequency_trend="stable",
            email_ratio=activity_type_counts.get('email', 0) / max(total_activities, 1)
        )
    
    def _extract_deal_characteristics(self, deal_data: Dict[str, Any], deal_metrics: DealMetrics) -> DealCharacteristics:
        """Extract deal characteristics"""
        
        # Extract from metadata or deal data
        metadata = deal_data.get('metadata', {})
        
        deal_amount = self._safe_float(metadata.get('deal_amount') or metadata.get('amount', 0))
        deal_stage = str(metadata.get('deal_stage') or metadata.get('stage', 'unknown'))
        deal_type = str(metadata.get('deal_type') or metadata.get('type', 'unknown'))
        deal_probability = self._safe_float(metadata.get('deal_probability') or metadata.get('probability', 0))
        deal_outcome = str(metadata.get('deal_outcome') or metadata.get('outcome', 'unknown'))
        
        # Create date strings
        create_date = metadata.get('create_date', '')
        close_date = metadata.get('close_date', '')
        
        # Calculate deal age
        deal_age_days = deal_metrics.time_span_days
        
        # Determine deal status
        is_closed = deal_outcome in ['won', 'lost', 'closed']
        is_won = deal_outcome == 'won'
        is_lost = deal_outcome == 'lost'
        is_open = not is_closed
        is_new_business = deal_type == 'newbusiness'
        
        # Categorize deal size
        if deal_amount >= 100000:
            deal_size_category = 'large'
        elif deal_amount >= 10000:
            deal_size_category = 'medium'
        else:
            deal_size_category = 'small'
        
        # Categorize probability
        if deal_probability >= 0.8:
            probability_category = 'high'
        elif deal_probability >= 0.5:
            probability_category = 'medium'
        else:
            probability_category = 'low'
        
        return DealCharacteristics(
            deal_id=deal_metrics.deal_id,
            deal_amount=deal_amount,
            deal_stage=deal_stage,
            deal_type=deal_type,
            deal_probability=deal_probability,
            deal_outcome=deal_outcome,
            deal_size_category=deal_size_category,
            probability_category=probability_category,
            create_date=create_date,
            close_date=close_date,
            deal_age_days=deal_age_days,
            is_closed=is_closed,
            is_won=is_won,
            is_lost=is_lost,
            is_open=is_open,
            is_new_business=is_new_business
        )
    
    def _create_combined_text(self, activities: List[ProcessedActivity]) -> str:
        """Create combined text from activities"""
        
        text_parts = []
        
        for activity in activities:
            if activity.content.strip():
                text_parts.append(f"[{activity.activity_type.upper()}] {activity.content}")
        
        return '\n'.join(text_parts)
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float"""
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def process_batch(self, deals_data: List[Dict[str, Any]]) -> List[ProcessedDealData]:
        """Process multiple deals in batch"""
        
        processed_deals = []
        
        for deal_data in deals_data:
            try:
                processed_deal = self.process_deal(deal_data)
                processed_deals.append(processed_deal)
            except Exception as e:
                logger.error(f"Error processing deal {deal_data.get('deal_id', 'unknown')}: {e}")
                continue
        
        return processed_deals