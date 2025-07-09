import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from llm.llm_clients import LLMClient
from rag.retriever import RAGRetriever
from core.data_processor import DealDataProcessor
from models.schemas import DealAnalysisResult
from config.settings import settings

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Simplified sentiment analyzer focused on RAG-based sentiment analysis
    Uses RAG retriever to get relevant examples for context
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        rag_retriever: RAGRetriever,
        data_processor: DealDataProcessor = None
    ):
        """
        Initialize sentiment analyzer
        
        Args:
            llm_client: LLM client for sentiment analysis
            rag_retriever: RAG retriever for historical context
            data_processor: Data processor for parsing deal activities
        """
        self.llm_client = llm_client
        self.rag_retriever = rag_retriever
        self.data_processor = data_processor
        
        logger.info("Sentiment Analyzer initialized")
    
    def analyze_deal_sentiment(
        self,
        deal_data: Dict[str, Any],
        include_rag_context: bool = True
    ) -> Dict[str, Any]:
        """
        Perform sentiment analysis on a deal
        
        Args:
            deal_data: Deal data including activities and metadata
            include_rag_context: Whether to include RAG context
            
        Returns:
            Sentiment analysis result
        """
        
        try:
            deal_id = deal_data.get('deal_id', 'unknown')
            logger.info(f"Analyzing sentiment for deal {deal_id}")
            
            # Process deal data if data processor is available
            if self.data_processor:
                processed_deal = self.data_processor.process_deal(deal_data)
                activities = [
                    {
                        'activity_type': activity.activity_type,
                        'content': activity.content,
                        'timestamp': activity.timestamp.isoformat() if activity.timestamp else None,
                        'direction': activity.direction,
                        'metadata': activity.metadata
                    }
                    for activity in processed_deal.processed_activities
                ]
                metadata = {
                    'deal_amount': processed_deal.deal_characteristics.deal_amount,
                    'deal_stage': processed_deal.deal_characteristics.deal_stage,
                    'deal_type': processed_deal.deal_characteristics.deal_type,
                    'deal_probability': processed_deal.deal_characteristics.deal_probability,
                    'outcome': processed_deal.deal_characteristics.deal_outcome,
                    'total_activities': processed_deal.deal_metrics.total_activities,
                    'time_span_days': processed_deal.deal_metrics.time_span_days,
                    'communication_gaps_count': processed_deal.deal_metrics.communication_gaps_count
                }
                activities_text = processed_deal.combined_text
            else:
                # Use raw data if no processor
                activities = deal_data.get('activities', [])
                metadata = deal_data.get('metadata', {})
                activities_text = self._create_activities_text(activities)
            
            # Get RAG context if requested
            rag_context = ""
            if include_rag_context:
                try:
                    rag_context = self.rag_retriever.retrieve_relevant_examples(
                        deal_id=deal_id,
                        activities=activities,
                        metadata=metadata
                    )
                except Exception as e:
                    logger.error(f"Error retrieving RAG context: {e}")
                    rag_context = "Error retrieving historical context."
            
            # Analyze sentiment using LLM
            sentiment_result = self.llm_client.analyze_sentiment(
                deal_id=deal_id,
                activities_text=activities_text,
                rag_context=rag_context,
                activity_frequency=len(activities),
                total_activities=len(activities)
            )
            
            # Add analysis metadata
            sentiment_result.update({
                'analysis_metadata': {
                    'deal_id': deal_id,
                    'analysis_timestamp': datetime.utcnow().isoformat(),
                    'llm_provider': self.llm_client.provider.get_provider_name(),
                    'included_rag_context': include_rag_context,
                    'total_activities_analyzed': len(activities),
                    'rag_context_length': len(rag_context) if rag_context else 0
                }
            })
            
            logger.info(f"Sentiment analysis completed for deal {deal_id}")
            return sentiment_result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for deal {deal_data.get('deal_id', 'unknown')}: {e}")
            return {
                'error': str(e),
                'deal_id': deal_data.get('deal_id', 'unknown'),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _create_activities_text(self, activities: List[Dict[str, Any]]) -> str:
        """Create combined activities text from raw activities"""
        
        text_parts = []
        
        for activity in activities:
            activity_type = activity.get('activity_type', 'unknown')
            
            # Extract content based on activity type
            if activity_type == 'email':
                subject = activity.get('subject', '')
                body = activity.get('body', '')
                if subject:
                    text_parts.append(f"EMAIL: Subject: {subject}")
                if body:
                    text_parts.append(f"Body: {body}")
            
            elif activity_type == 'call':
                title = activity.get('call_title', '')
                body = activity.get('call_body', '')
                if title:
                    text_parts.append(f"CALL: {title}")
                if body:
                    text_parts.append(f"Notes: {body}")
            
            elif activity_type == 'meeting':
                title = activity.get('meeting_title', '')
                notes = activity.get('internal_meeting_notes', '')
                if title:
                    text_parts.append(f"MEETING: {title}")
                if notes:
                    text_parts.append(f"Notes: {notes}")
            
            elif activity_type == 'note':
                body = activity.get('note_body', '')
                if body:
                    text_parts.append(f"NOTE: {body}")
            
            elif activity_type == 'task':
                subject = activity.get('task_subject', '')
                body = activity.get('task_body', '')
                if subject:
                    text_parts.append(f"TASK: {subject}")
                if body:
                    text_parts.append(f"Details: {body}")
        
        return "\n".join(text_parts)
    
    def analyze_batch_sentiment(
        self,
        deals_data: List[Dict[str, Any]],
        include_rag_context: bool = True
    ) -> Dict[str, Any]:
        """
        Perform batch sentiment analysis on multiple deals
        
        Args:
            deals_data: List of deal data
            include_rag_context: Whether to include RAG context
            
        Returns:
            Batch analysis results
        """
        
        start_time = datetime.utcnow()
        
        results = []
        successful_analyses = 0
        failed_analyses = 0
        
        logger.info(f"Starting batch sentiment analysis for {len(deals_data)} deals")
        
        for i, deal_data in enumerate(deals_data):
            try:
                result = self.analyze_deal_sentiment(deal_data, include_rag_context)
                
                if 'error' not in result:
                    successful_analyses += 1
                else:
                    failed_analyses += 1
                
                results.append(result)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(deals_data)} deals")
                
            except Exception as e:
                logger.error(f"Error in batch analysis for deal {deal_data.get('deal_id', 'unknown')}: {e}")
                failed_analyses += 1
                results.append({
                    'error': str(e),
                    'deal_id': deal_data.get('deal_id', 'unknown'),
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        batch_summary = {
            'total_deals': len(deals_data),
            'successful_analyses': successful_analyses,
            'failed_analyses': failed_analyses,
            'results': results,
            'processing_time_seconds': processing_time,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Batch sentiment analysis completed: {successful_analyses} successful, {failed_analyses} failed")
        return batch_summary
    
    def get_analyzer_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        
        try:
            return {
                'llm_provider': self.llm_client.provider.get_provider_name(),
                'rag_retriever_stats': self.rag_retriever.get_retrieval_stats(),
                'configuration': {
                    'llm_max_tokens': settings.LLM_MAX_TOKENS,
                    'llm_temperature': settings.LLM_TEMPERATURE,
                    'rag_top_k': settings.RAG_TOP_K,
                    'rag_similarity_threshold': settings.RAG_SIMILARITY_THRESHOLD
                },
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting analyzer stats: {e}")
            return {'error': str(e)}

# Factory function
def create_sentiment_analyzer(
    llm_provider: str = None,
    llm_config: Dict[str, Any] = None
) -> SentimentAnalyzer:
    """Create sentiment analyzer instance"""
    
    # Get LLM provider and config from settings if not provided
    if not llm_provider:
        llm_provider = settings.LLM_PROVIDER
    
    if not llm_config:
        llm_config = settings.get_llm_config()
    
    # Create LLM client
    from llm.llm_clients import create_llm_client
    llm_client = create_llm_client(llm_provider, **llm_config)
    
    # Create RAG retriever
    from rag.retriever import create_rag_retriever
    rag_retriever = create_rag_retriever()
    
    # Create data processor
    from core.embedding_service import get_embedding_service
    from core.data_processor import DealDataProcessor
    embedding_service = get_embedding_service()
    data_processor = DealDataProcessor(embedding_service)
    
    return SentimentAnalyzer(
        llm_client=llm_client,
        rag_retriever=rag_retriever,
        data_processor=data_processor
    )