import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from core.embedding_service import EmbeddingService
from core.vector_store import VectorStore
from rag.context_builder import RAGContextBuilder
from models.schemas import VectorSearchResult, DealPattern
from config.settings import settings

logger = logging.getLogger(__name__)

class RAGRetriever:
    """
    Focused RAG retriever that finds relevant examples from past deals
    based on client behavior, sentiment patterns, language tone, and deal progression
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        context_builder: RAGContextBuilder = None
    ):
        """
        Initialize RAG retriever
        
        Args:
            embedding_service: Service for generating embeddings
            vector_store: Vector database for similarity search
            context_builder: Context builder for formatting results
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.context_builder = context_builder or RAGContextBuilder()
        
        # Configuration from settings
        self.top_k = settings.RAG_TOP_K
        self.similarity_threshold = settings.RAG_SIMILARITY_THRESHOLD
        
        logger.info("RAG Retriever initialized")
    
    def retrieve_relevant_examples(
        self,
        deal_id: str,
        activities: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> str:
        """
        Retrieve relevant examples from past deals and build context
        
        Args:
            deal_id: Current deal identifier
            activities: Deal activities
            metadata: Deal metadata
            
        Returns:
            Formatted context string with relevant examples
        """
        
        try:
            # Step 1: Create search query from current deal
            search_query = self._create_search_query(activities, metadata)
            
            if not search_query:
                logger.warning(f"No search query generated for deal {deal_id}")
                return "No relevant historical context available."
            
            # Step 2: Find similar deals using vector search
            similar_deals = self._find_similar_deals(search_query)
            
            if not similar_deals:
                logger.info(f"No similar deals found for deal {deal_id}")
                return "No similar deals found for historical context."
            
            # Step 3: Filter and rank by relevance
            relevant_deals = self._filter_and_rank_deals(similar_deals, metadata)
            
            # Step 4: Build context using context builder
            context = self.context_builder.build_context(
                deal_id=deal_id,
                activities=activities,
                metadata=metadata,
                similar_deals=relevant_deals
            )
            
            logger.info(f"Retrieved {len(relevant_deals)} relevant examples for deal {deal_id}")
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving examples for deal {deal_id}: {e}")
            return "Error retrieving historical context."
    
    def _create_search_query(
        self, 
        activities: List[Dict[str, Any]], 
        metadata: Dict[str, Any]
    ) -> str:
        """
        Create search query focusing on client behavior and sentiment patterns
        
        Args:
            activities: Deal activities
            metadata: Deal metadata
            
        Returns:
            Search query string
        """
        
        from datetime import datetime
        
        def get_activity_timestamp(activity):
            """Extract timestamp for sorting with priority: sent_at -> meeting_start_time -> createdate"""
            
            # Priority 1: sent_at
            if 'sent_at' in activity and activity['sent_at']:
                timestamp_str = activity['sent_at']
            # Priority 2: meeting_start_time  
            elif 'meeting_start_time' in activity and activity['meeting_start_time']:
                timestamp_str = activity['meeting_start_time']
            # Priority 3: createdate
            elif 'createdate' in activity and activity['createdate']:
                timestamp_str = activity['createdate']
            else:
                # No timestamp found, return max datetime to make it latest
                return datetime.max
            
            try:
                # Handle timezone - if no timezone, assume UTC
                if timestamp_str.endswith('Z'):
                    return datetime.fromisoformat(timestamp_str[:-1] + '+00:00')
                elif '+' in timestamp_str or timestamp_str.count('-') > 2:
                    return datetime.fromisoformat(timestamp_str)
                else:
                    # No timezone info, assume UTC
                    return datetime.fromisoformat(timestamp_str + '+00:00')
            except:
                # If parsing fails, return max datetime to make it latest
                return datetime.max
        
        # Sort activities by timestamp (most recent first) and take latest 7
        sorted_activities = sorted(
            activities, 
            key=get_activity_timestamp, 
            reverse=True
        )[:7]
        
        query_parts = []
        
        # Extract key content from latest activities
        for activity in sorted_activities:
            activity_type = activity.get('activity_type', 'unknown')
            content_parts = []
            
            if activity_type == 'email':
                subject = activity.get('subject', '').strip()
                body = activity.get('body', '').strip()
                
                if subject:
                    content_parts.append(subject)
                if body:
                    content_parts.append(body[:150])
            
            elif activity_type == 'call':
                title = activity.get('call_title', '').strip()
                body = activity.get('call_body', '').strip()
                
                if title:
                    content_parts.append(title)
                if body:
                    content_parts.append(body[:150])
            
            elif activity_type == 'meeting':
                title = activity.get('meeting_title', '').strip()
                notes = activity.get('internal_meeting_notes', '').strip()
                
                if title:
                    content_parts.append(title)
                if notes:
                    content_parts.append(notes[:150])
            
            elif activity_type == 'note':
                body = activity.get('note_body', '').strip()
                if body:
                    content_parts.append(body[:150])
            
            elif activity_type == 'task':
                subject = activity.get('task_subject', '').strip()
                body = activity.get('task_body', '').strip()
                
                if subject:
                    content_parts.append(subject)
                if body:
                    content_parts.append(body[:150])
            
            # Join content parts for this activity
            if content_parts:
                query_parts.append(" ".join(content_parts))
        
        # Add deal characteristics for context
        deal_stage = metadata.get('deal_stage', '')
        if deal_stage:
            query_parts.append(f"deal stage {deal_stage}")
        
        deal_type = metadata.get('deal_type', '')
        if deal_type:
            query_parts.append(f"deal type {deal_type}")
        
        # Combine into search query
        search_query = " ".join(query_parts)
        
        # Limit query length for better search performance
        max_query_length = 1000
        if len(search_query) > max_query_length:
            search_query = search_query[:max_query_length]
        
        return search_query
    
    def _find_similar_deals(self, search_query: str) -> List[Dict[str, Any]]:
        """
        Find similar deals using vector search
        
        Args:
            search_query: Query string for similarity search
            
        Returns:
            List of similar deals
        """
        
        try:
            # Generate embedding for search query
            query_embedding = self.embedding_service.encode(search_query)
            
            # Search vector database
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=self.top_k,
                min_similarity=self.similarity_threshold
            )

            # Convert search results to deal format
            similar_deals = []
            for result in search_results:
                # logger.info(f"Processing deal: {result.deal_id} with similarity score: {result.similarity_score} \n Combined text : {result.combined_text}")
                deal_data = {
                    'deal_id': result.deal_id,
                    'similarity_score': result.similarity_score,
                    'activities': self._parse_activities_from_content(result.combined_text),
                    'metadata': result.metadata
                }
                similar_deals.append(deal_data)

            logger.info(f"Got {len(similar_deals)} similar deals")
            
            return similar_deals
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def _parse_activities_from_content(self, combined_text: str) -> List[Dict[str, Any]]:
        """
        Parse activities from combined text content
        
        Args:
            combined_text: Combined activities text
            
        Returns:
            List of parsed activities
        """
        
        activities = []
        
        # Split by activity markers while keeping the markers
        import re
        
        # Pattern to split on activity markers while keeping them
        pattern = r'(\[EMAIL\]|\[CALL\]|\[MEETING\]|\[NOTE\]|\[TASK\])'
        parts = re.split(pattern, combined_text)
        
        # Process parts in pairs (marker + content)
        for i in range(1, len(parts), 2):  # Start from 1 to skip first empty part
            if i + 1 < len(parts):
                marker = parts[i]
                content = parts[i + 1].strip()
                
                if not content:
                    continue
                
                # Determine activity type from marker
                activity_type = 'unknown'
                if marker == '[EMAIL]':
                    activity_type = 'email'
                elif marker == '[CALL]':
                    activity_type = 'call'
                elif marker == '[MEETING]':
                    activity_type = 'meeting'
                elif marker == '[NOTE]':
                    activity_type = 'note'
                elif marker == '[TASK]':
                    activity_type = 'task'
                
                activities.append({
                    'activity_type': activity_type,
                    'content': content
                })
        
        return activities
    
    def _filter_and_rank_deals(
        self, 
        similar_deals: List[Dict[str, Any]], 
        current_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter and rank deals by relevance to current deal
        
        Args:
            similar_deals: List of similar deals from vector search
            current_metadata: Current deal metadata
            
        Returns:
            Filtered and ranked deals
        """
        
        scored_deals = []
        
        for deal in similar_deals:
            relevance_score = self._calculate_relevance_score(deal, current_metadata)
            deal['relevance_score'] = relevance_score
            scored_deals.append(deal)
        
        # Sort by relevance score (descending) and keep top deals
        scored_deals.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Filter by minimum relevance threshold
        min_relevance = 0.3
        filtered_deals = [d for d in scored_deals if d['relevance_score'] >= min_relevance]
        
        # Return top deals
        return filtered_deals[:settings.RAG_TOP_K]
    
    def _calculate_relevance_score(
        self, 
        similar_deal: Dict[str, Any], 
        current_metadata: Dict[str, Any]
    ) -> float:
        """
        Calculate relevance score based on multiple factors
        
        Args:
            similar_deal: Similar deal data
            current_metadata: Current deal metadata
            
        Returns:
            Relevance score (0-1)
        """
        
        # logger.info(f"Similar deal: {similar_deal.get('deal_id')} similarity score {similar_deal.get('similarity_score')}")

        similar_metadata = similar_deal.get('metadata', {})
        score = similar_deal.get('similarity_score', 0.0)

        # Deal size similarity (much tighter range)
        current_amount = current_metadata.get('deal_amount', 0)
        similar_amount = similar_metadata.get('deal_amount', 0)

        if current_amount > 0 and similar_amount > 0:
            # Calculate percentage difference
            larger = max(current_amount, similar_amount)
            smaller = min(current_amount, similar_amount)
            percentage_diff = (larger - smaller) / smaller
            
            if percentage_diff <= 0.1:  # Within 10% difference
                score *= 1.3  # 30% boost
            elif percentage_diff <= 0.3:  # Within 30% difference
                score *= 1.15  # 15% boost
            elif percentage_diff <= 0.4:  # Within 40% difference
                score *= 1.1  # 10% boost
            # else no boost
        
        # Deal type match
        current_type = current_metadata.get('deal_type', '')
        similar_type = similar_metadata.get('deal_type', '')
        if current_type and similar_type and current_type == similar_type:
            score *= 1.05  # tiny 5% boost

        logger.info(f"Score: {score}")
        
        return min(score, 1.0)  # Cap at 1.0
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        try:
            vector_stats = self.vector_store.get_stats()
            return {
                'vector_store_stats': vector_stats,
                'top_k': self.top_k,
                'similarity_threshold': self.similarity_threshold,
                'enabled_context_components': self.context_builder.get_enabled_components()
            }
        except Exception as e:
            logger.error(f"Error getting retrieval stats: {e}")
            return {'error': str(e)}

# Factory function
def create_rag_retriever(
    embedding_service: EmbeddingService = None,
    vector_store: VectorStore = None,
    context_builder: RAGContextBuilder = None
) -> RAGRetriever:
    """Create RAG retriever instance"""
    
    if not embedding_service:
        from core.embedding_service import get_embedding_service
        embedding_service = get_embedding_service()
    
    if not vector_store:
        from core.vector_store import get_vector_store
        vector_store = get_vector_store()
    
    if not context_builder:
        # Import LLM client creation
        from llm.llm_clients import create_llm_client
        from config.settings import settings
        
        # Create LLM client for context generation
        llm_client = create_llm_client(
            provider_name=settings.LLM_PROVIDER,
            **settings.get_llm_config()
        )
        
        # Create context builder with LLM client
        from rag.context_builder import create_context_builder
        context_builder = create_context_builder(llm_client=llm_client)
    
    return RAGRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        context_builder=context_builder
    )