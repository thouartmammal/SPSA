import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pathlib import Path

from models.schemas import ProcessedActivity, DealPattern
from core.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class DealDataProcessor:
    """Process deal activities and convert to vector embeddings"""
    
    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize processor with embedding service
        
        Args:
            embedding_service: Initialized embedding service instance
        """
        self.embedding_service = embedding_service
        logger.info("DealDataProcessor initialized")
    
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
    
    def parse_activity(self, activity: Dict[str, Any], deal_id: str) -> ProcessedActivity:
        """Parse a single activity into structured format"""
        
        # Define correct date fields for each activity type
        DATE_FIELDS = {
            'email': 'sent_at',
            'call': 'createdate',
            'note': 'lastmodifieddate',
            'meeting': 'meeting_start_time',
            'task': 'createdate'
        }
        
        # Extract content based on activity type
        content_parts = []
        
        if activity['activity_type'] == 'email':
            if activity.get('subject'):
                content_parts.append(f"Subject: {activity['subject']}")
            if activity.get('body'):
                # Clean up the body text
                body = activity['body'].strip()
                if body:
                    content_parts.append(f"Body: {body}")
                
        elif activity['activity_type'] == 'note':
            if activity.get('note_body'):
                note_body = activity['note_body'].strip()
                if note_body:
                    content_parts.append(f"Note: {note_body}")
                
        elif activity['activity_type'] == 'task':
            if activity.get('task_subject'):
                task_subject = activity['task_subject'].strip()
                if task_subject:
                    content_parts.append(f"Task: {task_subject}")
            if activity.get('task_body'):
                task_body = activity['task_body'].strip()
                if task_body:
                    content_parts.append(f"Details: {task_body}")
                
        elif activity['activity_type'] == 'call':
            if activity.get('call_title'):
                call_title = activity['call_title'].strip()
                if call_title:
                    content_parts.append(f"Call: {call_title}")
            if activity.get('call_body'):
                call_body = activity['call_body'].strip()
                if call_body:
                    content_parts.append(f"Notes: {call_body}")
                
        elif activity['activity_type'] == 'meeting':
            if activity.get('meeting_title'):
                meeting_title = activity['meeting_title'].strip()
                if meeting_title:
                    content_parts.append(f"Meeting: {meeting_title}")
            if activity.get('internal_meeting_notes'):
                meeting_notes = activity['internal_meeting_notes'].strip()
                if meeting_notes:
                    content_parts.append(f"Notes: {meeting_notes}")
        
        # Create combined content, return empty if no meaningful content
        combined_content = " | ".join(content_parts) if content_parts else ""
        
        # Parse timestamp using correct field for activity type
        timestamp = None
        activity_type = activity['activity_type']
        date_field = DATE_FIELDS.get(activity_type)
        
        if date_field and activity.get(date_field):
            timestamp_value = activity[date_field]
            # Handle null timestamps
            if timestamp_value is not None:
                try:
                    # Handle different timestamp formats
                    if isinstance(timestamp_value, str):
                        if timestamp_value.endswith('Z'):
                            timestamp = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                        else:
                            timestamp = datetime.fromisoformat(timestamp_value)
                except Exception as e:
                    logger.warning(f"Could not parse timestamp '{timestamp_value}' for activity type '{activity_type}' in deal {deal_id}: {e}")
        
        # Extract metadata
        metadata = {
            'direction': activity.get('direction') or '',
            'status': activity.get('task_status') or activity.get('call_status') or '',
            'priority': activity.get('task_priority') or '',
            'duration': activity.get('call_duration') or 0,
            'participants': activity.get('to', []),
            'raw_activity': activity  # Keep original for reference
        }
        
        return ProcessedActivity(
            deal_id=deal_id,
            activity_type=activity['activity_type'],
            timestamp=timestamp,
            content=combined_content,
            direction=activity.get('direction'),
            metadata=metadata
        )
    
    def calculate_time_metrics(self, activities: List[ProcessedActivity]) -> Dict[str, Any]:
        """Calculate detailed time-based metrics from activities"""
        
        # Sort activities by timestamp - only include those with valid timestamps
        timestamped_activities = [a for a in activities if a.timestamp is not None]
        timestamped_activities.sort(key=lambda x: x.timestamp)
        
        # Default return for insufficient data
        default_response = {
            'avg_time_between_activities_hours': 0,
            'min_time_gap_hours': 0,
            'max_time_gap_hours': 0,
            'response_time_metrics': {
                'avg_response_time_hours': 0,
                'fastest_response_hours': 0,
                'slowest_response_hours': 0,
                'response_count': 0
            },
            'activity_frequency_trend': 'insufficient_data',
            'communication_gaps': [],
            'communication_gaps_count': 0,
            'business_hours_ratio': 0,
            'weekend_activity_ratio': 0,
            'total_time_gaps': 0
        }
        
        if len(timestamped_activities) < 2:
            return default_response
        
        # Calculate time gaps between consecutive activities
        time_gaps_hours = []
        for i in range(1, len(timestamped_activities)):
            gap = (timestamped_activities[i].timestamp - timestamped_activities[i-1].timestamp).total_seconds() / 3600
            time_gaps_hours.append(gap)
        
        # Calculate response time patterns (outgoing email responses to incoming)
        response_times = []
        email_activities = [a for a in timestamped_activities if a.activity_type == 'email']
        
        for i, email in enumerate(email_activities):
            if email.direction == 'outgoing' and i > 0:
                # Find the previous incoming email
                prev_incoming = None
                for j in range(i-1, -1, -1):
                    if email_activities[j].direction == 'incoming':
                        prev_incoming = email_activities[j]
                        break
                
                if prev_incoming:
                    response_time = (email.timestamp - prev_incoming.timestamp).total_seconds() / 3600
                    response_times.append(response_time)
        
        # Identify communication gaps (> 7 days between activities)
        communication_gaps = []
        for i, gap_hours in enumerate(time_gaps_hours):
            if gap_hours > 168:  # 7 days
                gap_info = {
                    'start_date': timestamped_activities[i].timestamp.isoformat(),
                    'end_date': timestamped_activities[i+1].timestamp.isoformat(),
                    'gap_days': gap_hours / 24
                }
                communication_gaps.append(gap_info)
        
        # Calculate business hours activity ratio
        business_hours_count = 0
        weekend_count = 0
        
        for activity in timestamped_activities:
            # Business hours: 9 AM to 6 PM, Monday to Friday
            if activity.timestamp.weekday() < 5:  # Monday = 0, Friday = 4
                if 9 <= activity.timestamp.hour <= 18:
                    business_hours_count += 1
            else:  # Weekend
                weekend_count += 1
        
        total_activities = len(timestamped_activities)
        business_hours_ratio = business_hours_count / total_activities if total_activities > 0 else 0
        weekend_activity_ratio = weekend_count / total_activities if total_activities > 0 else 0
        
        # Activity frequency trend (first half vs second half)
        mid_point = len(timestamped_activities) // 2
        if mid_point > 0:
            first_half = timestamped_activities[:mid_point]
            second_half = timestamped_activities[mid_point:]
            
            if len(first_half) > 1 and len(second_half) > 1:
                first_half_days = (first_half[-1].timestamp - first_half[0].timestamp).days or 1
                second_half_days = (second_half[-1].timestamp - second_half[0].timestamp).days or 1
                
                first_half_frequency = len(first_half) / first_half_days
                second_half_frequency = len(second_half) / second_half_days
                
                if second_half_frequency > first_half_frequency * 1.2:
                    frequency_trend = 'accelerating'
                elif second_half_frequency < first_half_frequency * 0.8:
                    frequency_trend = 'declining'
                else:
                    frequency_trend = 'stable'
            else:
                frequency_trend = 'insufficient_data'
        else:
            frequency_trend = 'insufficient_data'
        
        return {
            'avg_time_between_activities_hours': sum(time_gaps_hours) / len(time_gaps_hours) if time_gaps_hours else 0,
            'min_time_gap_hours': min(time_gaps_hours) if time_gaps_hours else 0,
            'max_time_gap_hours': max(time_gaps_hours) if time_gaps_hours else 0,
            'response_time_metrics': {
                'avg_response_time_hours': sum(response_times) / len(response_times) if response_times else 0,
                'fastest_response_hours': min(response_times) if response_times else 0,
                'slowest_response_hours': max(response_times) if response_times else 0,
                'response_count': len(response_times)
            },
            'activity_frequency_trend': frequency_trend,
            'communication_gaps': communication_gaps,
            'communication_gaps_count': len(communication_gaps),
            'business_hours_ratio': business_hours_ratio,
            'weekend_activity_ratio': weekend_activity_ratio,
            'total_time_gaps': len(time_gaps_hours)
        }
    
    def categorize_deal_size(self, amount: float, all_amounts: List[float]) -> str:
        """Categorize deal size based on percentiles of all deals"""
        if amount <= 0:
            return 'unknown'
        
        if not all_amounts:
            return 'unknown'
        
        # Filter out zero amounts for percentile calculation
        valid_amounts = [a for a in all_amounts if a > 0]
        if not valid_amounts:
            return 'unknown'
        
        valid_amounts.sort()
        n = len(valid_amounts)
        
        # Calculate percentiles safely
        p25_idx = max(0, int(n * 0.25) - 1)
        p50_idx = max(0, int(n * 0.50) - 1)
        p75_idx = max(0, int(n * 0.75) - 1)
        
        p25 = valid_amounts[p25_idx]
        p50 = valid_amounts[p50_idx]
        p75 = valid_amounts[p75_idx]
        
        if amount <= p25:
            return 'small'
        elif amount <= p50:
            return 'medium'
        elif amount <= p75:
            return 'large'
        else:
            return 'enterprise'
    
    def categorize_probability(self, probability: float) -> str:
        """Categorize probability - these can be universal ranges"""
        if probability >= 80:
            return 'high'
        elif probability >= 60:
            return 'medium_high'
        elif probability >= 40:
            return 'medium'
        elif probability >= 20:
            return 'low'
        else:
            return 'very_low'
    
    def extract_deal_characteristics(self, deal_data: Dict[str, Any], all_deals_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract and process deal characteristics from HubSpot data"""
        
        # Parse deal amount
        amount = 0.0
        if deal_data.get('amount'):
            try:
                amount = float(deal_data['amount'])
            except (ValueError, TypeError):
                amount = 0.0
        
        # Parse deal stage probability
        probability = 0.0
        if deal_data.get('deal_stage_probability'):
            try:
                probability = float(deal_data['deal_stage_probability'])
            except (ValueError, TypeError):
                probability = 0.0
        
        # Parse dates
        create_date = None
        close_date = None
        
        if deal_data.get('createdate'):
            try:
                create_date = datetime.fromisoformat(deal_data['createdate'].replace('Z', '+00:00'))
            except Exception as e:
                logger.warning(f"Could not parse createdate: {e}")
        
        if deal_data.get('closedate'):
            try:
                close_date = datetime.fromisoformat(deal_data['closedate'].replace('Z', '+00:00'))
            except Exception as e:
                logger.warning(f"Could not parse closedate: {e}")
        
        # Calculate deal lifecycle metrics
        deal_age_days = 0
        if create_date:
            current_time = datetime.now(create_date.tzinfo)
            deal_age_days = (current_time - create_date).days
        
        # Determine deal outcome category
        dealstage = deal_data.get('dealstage', '').lower()
        if 'closed won' in dealstage or 'won' in dealstage:
            outcome = 'won'
        elif 'closed lost' in dealstage or 'lost' in dealstage:
            outcome = 'lost'
        else:
            outcome = 'open'
        
        # Get all amounts for relative categorization
        all_amounts = []
        if all_deals_data:
            for deal in all_deals_data:
                try:
                    deal_amount = float(deal.get('amount', 0))
                    if deal_amount > 0:
                        all_amounts.append(deal_amount)
                except (ValueError, TypeError):
                    continue
        
        # Categorize deal size based on data distribution
        deal_size_category = self.categorize_deal_size(amount, all_amounts)
        
        # Categorize probability
        prob_category = self.categorize_probability(probability)
        
        return {
            # Raw deal data
            'deal_amount': amount,
            'deal_stage': deal_data.get('dealstage') or '',
            'deal_probability': probability,
            'deal_type': deal_data.get('dealtype') or '',
            'deal_outcome': outcome,
            
            # Processed categories
            'deal_size_category': deal_size_category,
            'probability_category': prob_category,
            
            # Date information
            'create_date': create_date.isoformat() if create_date else '',
            'close_date': close_date.isoformat() if close_date else '',
            'deal_age_days': deal_age_days,
            
            # Lifecycle metrics
            'is_closed': outcome in ['won', 'lost'],
            'is_won': outcome == 'won',
            'is_lost': outcome == 'lost',
            'is_open': outcome == 'open',
            
            # Business logic flags
            'is_new_business': deal_data.get('dealtype', '').lower() == 'newbusiness',
            'has_amount': amount > 0,
            'high_probability': probability >= 70,
            'low_probability': probability <= 30
        }
    
    def process_deal(self, deal_data: Dict[str, Any], all_deals_data: List[Dict[str, Any]] = None) -> DealPattern:
        """Process a complete deal into a pattern for vector storage"""
        
        deal_id = str(deal_data['deal_id'])
        activities = []
        
        # Process each activity
        for activity in deal_data['activities']:
            try:
                processed_activity = self.parse_activity(activity, deal_id)
                # Only add activities with valid content (not empty or "No content")
                if (processed_activity.content and 
                    processed_activity.content.strip() and 
                    processed_activity.content != "No content"):
                    activities.append(processed_activity)
                else:
                    logger.debug(f"Skipping activity with no content in deal {deal_id}: {activity.get('activity_type', 'unknown')}")
            except Exception as e:
                logger.warning(f"Error processing activity in deal {deal_id}: {e}")
                continue
        
        if not activities:
            raise ValueError(f"No valid activities found for deal {deal_id}")
        
        # Extract deal characteristics
        deal_characteristics = self.extract_deal_characteristics(deal_data, all_deals_data)
        
        # Combine all activities into searchable text - handle activities without timestamps
        combined_text_parts = []
        for activity in activities:
            if activity.timestamp:
                timestamp_str = activity.timestamp.strftime("%Y-%m-%d")
            else:
                timestamp_str = "Unknown date"
            activity_text = f"[{timestamp_str}] {activity.activity_type.upper()}: {activity.content}"
            combined_text_parts.append(activity_text)
        
        combined_text = "\n".join(combined_text_parts)
        
        # Calculate time span - only for activities with timestamps
        timestamps = [a.timestamp for a in activities if a.timestamp is not None]
        time_span_days = 0
        if len(timestamps) > 1:
            time_span_days = (max(timestamps) - min(timestamps)).days
        
        # Calculate detailed time metrics
        time_metrics = self.calculate_time_metrics(activities)
        
        # Extract activity patterns
        activity_types = [a.activity_type for a in activities]
        activity_type_counts = {
            'email': activity_types.count('email'),
            'call': activity_types.count('call'),
            'meeting': activity_types.count('meeting'),
            'note': activity_types.count('note'),
            'task': activity_types.count('task')
        }
        
        # Calculate engagement metrics
        outgoing_emails = len([a for a in activities if a.activity_type == 'email' and a.direction == 'outgoing'])
        incoming_emails = len([a for a in activities if a.activity_type == 'email' and a.direction == 'incoming'])
        
        # Combine all metadata
        metadata = {
            # Activity metrics
            'activities_count': len(activities),
            'time_span_days': time_span_days,
            'activity_types': activity_type_counts,
            'outgoing_emails': outgoing_emails,
            'incoming_emails': incoming_emails,
            'email_ratio': outgoing_emails / max(incoming_emails, 1),  # Avoid division by zero
            'last_activity_date': max(timestamps).isoformat() if timestamps else '',
            'first_activity_date': min(timestamps).isoformat() if timestamps else '',
            
            # Deal characteristics (new)
            **deal_characteristics,
            
            # Time metrics
            **time_metrics
        }
        
        return DealPattern(
            deal_id=deal_id,
            combined_text=combined_text,
            activities_count=len(activities),
            activity_types=list(set(activity_types)),
            time_span_days=time_span_days,
            metadata=metadata
        )
    
    def process_all_deals(self, file_path: str) -> List[DealPattern]:
        """Process all deals from file and create embeddings"""
        
        # Load data
        deals_data = self.load_deal_data(file_path)
        processed_deals = []
        
        logger.info(f"Processing {len(deals_data)} deals...")
        
        for deal_data in deals_data:
            try:
                # Process deal with all deals data for relative categorization
                deal_pattern = self.process_deal(deal_data, deals_data)
                
                # Create embedding using the injected service
                logger.info(f"Creating embedding for deal {deal_pattern.deal_id}")
                deal_pattern.embedding = self.embedding_service.encode(deal_pattern.combined_text)
                
                processed_deals.append(deal_pattern)
                logger.info(f"Successfully processed deal {deal_pattern.deal_id}")
                
            except Exception as e:
                logger.error(f"Error processing deal {deal_data.get('deal_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_deals)} deals")
        return processed_deals


# Example usage and testing
def main():
    """Example usage of the DealDataProcessor"""
    from core.embedding_service import get_embedding_service
    from core.vector_store import get_vector_store
    from config.settings import settings
    from utils.logging_config import setup_logging
    
    # Setup logging
    logger = setup_logging()
    
    # Initialize services
    embedding_service = get_embedding_service()
    vector_store = get_vector_store()
    
    # Initialize processor
    processor = DealDataProcessor(embedding_service=embedding_service)
    
    # Process your sample data
    file_path = settings.DATA_PATH
    
    try:
        # Process all deals
        processed_deals = processor.process_all_deals(file_path)
        
        # Print summary
        print(f"\n{'='*50}")
        print("PROCESSING SUMMARY")
        print(f"{'='*50}")
        print(f"Total deals processed: {len(processed_deals)}")
        
        for deal in processed_deals:
            print(f"\nDeal ID: {deal.deal_id}")
            print(f"Outcome: {deal.metadata['deal_outcome']}")
            print(f"Amount: ${deal.metadata['deal_amount']:,.2f} ({deal.metadata['deal_size_category']})")
            print(f"Stage: {deal.metadata['deal_stage']}")
            print(f"Probability: {deal.metadata['deal_probability']:.1f}% ({deal.metadata['probability_category']})")
            print(f"Type: {deal.metadata['deal_type']}")
            print(f"Age: {deal.metadata['deal_age_days']} days")
            print(f"Activities: {deal.activities_count}")
            print(f"Time span: {deal.time_span_days} days")
            print(f"Activity types: {', '.join(deal.activity_types)}")
            print(f"Embedding dimension: {len(deal.embedding)}")
            print(f"Email ratio (out/in): {deal.metadata['email_ratio']:.2f}")
            print(f"Avg time between activities: {deal.metadata['avg_time_between_activities_hours']:.1f} hours")
            print(f"Avg response time: {deal.metadata['response_time_metrics']['avg_response_time_hours']:.1f} hours")
            print(f"Communication gaps: {deal.metadata['communication_gaps_count']}")
            print(f"Activity trend: {deal.metadata['activity_frequency_trend']}")
            print(f"Business hours ratio: {deal.metadata['business_hours_ratio']:.2f}")
            
            # Show concerning patterns
            if deal.metadata['communication_gaps_count'] > 0:
                print(f"⚠️  Warning: {deal.metadata['communication_gaps_count']} communication gaps detected")
            if deal.metadata['response_time_metrics']['avg_response_time_hours'] > 24:
                print(f"🐌 Slow response time: {deal.metadata['response_time_metrics']['avg_response_time_hours']:.1f} hours avg")
            if deal.metadata['activity_frequency_trend'] == 'declining':
                print(f"📉 Declining activity frequency trend")
            
            # Deal-specific insights
            if deal.metadata['is_won']:
                print(f"✅ Won deal - good patterns to learn from")
            elif deal.metadata['is_lost']:
                print(f"❌ Lost deal - warning patterns identified")
            
            if deal.metadata['deal_size_category'] == 'enterprise' and deal.metadata['communication_gaps_count'] > 0:
                print(f"🚨 Enterprise deal with communication gaps - high risk!")
                
            if deal.metadata['is_new_business'] and deal.metadata['response_time_metrics']['avg_response_time_hours'] > 12:
                print(f"⏰ New business deal with slow response times - potential issue")
        
        # Save to vector database
        print(f"\n{'='*50}")
        print("SAVING TO VECTOR DATABASE")
        print(f"{'='*50}")
        
        vector_store.store_patterns(processed_deals)
        
        print("✅ Successfully processed and saved all deals!")
        
        # Test similarity search
        print(f"\n{'='*50}")
        print("TESTING SIMILARITY SEARCH")
        print(f"{'='*50}")
        
        # Create a test query
        test_query = "Client is interested in proposal and wants to schedule a call"
        test_embedding = embedding_service.encode(test_query)
        
        # Search for similar patterns
        results = vector_store.search_similar(test_embedding, top_k=3)
        
        print(f"Query: {test_query}")
        print(f"Most similar deals:")
        for i, result in enumerate(results):
            print(f"{i+1}. Deal {result.deal_id} (similarity: {result.similarity_score:.3f})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()