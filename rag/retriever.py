import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

from core.embedding_service import EmbeddingService
from core.vector_store import VectorStore
from core.data_processor import DealDataProcessor
from rag.context_builder import RAGContextBuilder
from models.schemas import VectorSearchResult, DealPattern, ProcessedActivity
from config.settings import settings

logger = logging.getLogger(__name__)

class RAGRetriever:
    """
    Main RAG retrieval engine that finds similar historical patterns
    and builds contextual insights for sentiment analysis
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
        
        # Configuration
        self.default_top_k = settings.MAX_RETRIEVED_DOCS
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        
        logger.info("RAG Retriever initialized")
    
    def retrieve_similar_patterns(
        self,
        query_text: str,
        top_k: int = None,
        min_similarity: float = None,
        filters: Dict[str, Any] = None
    ) -> List[VectorSearchResult]:
        """
        Retrieve similar deal patterns based on query text
        
        Args:
            query_text: Text to find similar patterns for
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            filters: Additional filters for search
            
        Returns:
            List of similar deal patterns
        """
        
        if not query_text or not query_text.strip():
            logger.warning("Empty query text provided")
            return []
        
        # Use defaults if not provided
        top_k = top_k or self.default_top_k
        min_similarity = min_similarity or self.similarity_threshold
        
        try:
            # Generate embedding for query
            logger.debug(f"Generating embedding for query: {query_text[:100]}...")
            query_embedding = self.embedding_service.encode(query_text)
            
            # Search vector store
            logger.debug(f"Searching vector store with top_k={top_k}")
            results = self.vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=top_k * 2  # Get more results to filter
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result.similarity_score >= min_similarity
            ]
            
            # Apply additional filters if provided
            if filters:
                filtered_results = self._apply_filters(filtered_results, filters)
            
            # Limit to requested number
            final_results = filtered_results[:top_k]
            
            logger.info(f"Retrieved {len(final_results)} similar patterns (filtered from {len(results)} total)")
            return final_results
            
        except Exception as e:
            logger.error(f"Error retrieving similar patterns: {e}")
            return []
    
    def retrieve_for_deal_analysis(
        self,
        current_deal_text: str,
        current_deal_metadata: Dict[str, Any],
        analysis_type: str = "sentiment"
    ) -> Tuple[List[VectorSearchResult], str]:
        """
        Retrieve patterns specifically for deal sentiment analysis
        
        Args:
            current_deal_text: Combined text of current deal activities
            current_deal_metadata: Metadata about current deal
            analysis_type: Type of analysis being performed
            
        Returns:
            Tuple of (similar_deals, formatted_context)
        """
        
        # Enhance query with deal characteristics for better matching
        enhanced_query = self._enhance_query_for_analysis(
            current_deal_text, 
            current_deal_metadata,
            analysis_type
        )
        
        # Retrieve similar patterns
        similar_deals = self.retrieve_similar_patterns(
            query_text=enhanced_query,
            top_k=self.default_top_k,
            filters=self._get_analysis_filters(current_deal_metadata, analysis_type)
        )
        
        # Build context using context builder
        context = self.context_builder.build_context(
            similar_deals=similar_deals,
            current_deal_metadata=current_deal_metadata,
            max_context_length=settings.CONTEXT_WINDOW_SIZE
        )
        
        logger.info(f"Built analysis context from {len(similar_deals)} similar deals")
        return similar_deals, context
    
    def retrieve_by_deal_characteristics(
        self,
        deal_amount: float = None,
        deal_stage: str = None,
        deal_type: str = None,
        deal_outcome: str = None,
        activity_types: List[str] = None,
        top_k: int = None
    ) -> List[VectorSearchResult]:
        """
        Retrieve deals by specific characteristics
        
        Args:
            deal_amount: Deal amount range
            deal_stage: Deal stage
            deal_type: Type of deal
            deal_outcome: Deal outcome (won/lost/open)
            activity_types: Types of activities
            top_k: Number of results
            
        Returns:
            List of matching deals
        """
        
        # Build query based on characteristics
        query_parts = []
        
        if deal_stage:
            query_parts.append(f"deal stage {deal_stage}")
        
        if deal_type:
            query_parts.append(f"deal type {deal_type}")
        
        if activity_types:
            query_parts.append(f"activities {' '.join(activity_types)}")
        
        if deal_outcome:
            query_parts.append(f"outcome {deal_outcome}")
        
        # Default query if no characteristics provided
        if not query_parts:
            query_parts.append("sales deal activities")
        
        query_text = " ".join(query_parts)
        
        # Build filters
        filters = {}
        if deal_outcome:
            filters['deal_outcome'] = deal_outcome
        if deal_amount:
            filters['deal_amount_range'] = deal_amount
        
        return self.retrieve_similar_patterns(
            query_text=query_text,
            top_k=top_k,
            filters=filters
        )
    
    def get_success_patterns(self, query_text: str, top_k: int = 5) -> List[VectorSearchResult]:
        """Get patterns from successful deals only"""
        return self.retrieve_similar_patterns(
            query_text=query_text,
            top_k=top_k,
            filters={'deal_outcome': 'won'}
        )
    
    def get_failure_patterns(self, query_text: str, top_k: int = 5) -> List[VectorSearchResult]:
        """Get patterns from failed deals only"""
        return self.retrieve_similar_patterns(
            query_text=query_text,
            top_k=top_k,
            filters={'deal_outcome': 'lost'}
        )
    
    def get_contextual_insights(
        self,
        deal_text: str,
        deal_metadata: Dict[str, Any],
        insight_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Get comprehensive contextual insights for a deal
        
        Args:
            deal_text: Deal activity text
            deal_metadata: Deal metadata
            insight_type: Type of insights to generate
            
        Returns:
            Dictionary of insights and recommendations
        """
        
        try:
            # Get similar deals for analysis
            similar_deals, context = self.retrieve_for_deal_analysis(
                current_deal_text=deal_text,
                current_deal_metadata=deal_metadata,
                analysis_type=insight_type
            )
            
            # Analyze patterns
            insights = self._analyze_patterns(similar_deals, deal_metadata)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(insights, deal_metadata)
            
            return {
                'similar_deals_count': len(similar_deals),
                'context': context,
                'insights': insights,
                'recommendations': recommendations,
                'analysis_metadata': {
                    'timestamp': datetime.utcnow().isoformat(),
                    'insight_type': insight_type,
                    'similarity_threshold': self.similarity_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating contextual insights: {e}")
            return {
                'error': str(e),
                'similar_deals_count': 0,
                'context': '',
                'insights': {},
                'recommendations': []
            }
    
    def _enhance_query_for_analysis(
        self,
        deal_text: str,
        deal_metadata: Dict[str, Any],
        analysis_type: str
    ) -> str:
        """Enhance query with deal characteristics for better matching"""
        
        query_parts = [deal_text]
        
        # Add deal characteristics to improve matching
        if deal_metadata.get('deal_stage'):
            query_parts.append(f"stage {deal_metadata['deal_stage']}")
        
        if deal_metadata.get('deal_type'):
            query_parts.append(f"type {deal_metadata['deal_type']}")
        
        if deal_metadata.get('deal_size_category'):
            query_parts.append(f"size {deal_metadata['deal_size_category']}")
        
        # Add analysis-specific terms
        if analysis_type == "sentiment":
            query_parts.append("salesperson behavior communication patterns")
        elif analysis_type == "risk":
            query_parts.append("deal risk factors warning signs")
        elif analysis_type == "opportunity":
            query_parts.append("deal opportunities success factors")
        
        return " ".join(query_parts)
    
    def _get_analysis_filters(
        self,
        deal_metadata: Dict[str, Any],
        analysis_type: str
    ) -> Dict[str, Any]:
        """Get filters for analysis-specific retrieval"""
        
        filters = {}
        
        # Filter by deal size category for better matching
        if deal_metadata.get('deal_size_category'):
            filters['deal_size_category'] = deal_metadata['deal_size_category']
        
        # Filter by deal type for better matching
        if deal_metadata.get('deal_type'):
            filters['deal_type'] = deal_metadata['deal_type']
        
        return filters
    
    def _apply_filters(
        self,
        results: List[VectorSearchResult],
        filters: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Apply additional filters to search results"""
        
        filtered_results = []
        
        for result in results:
            include = True
            
            for filter_key, filter_value in filters.items():
                if filter_key == 'deal_outcome':
                    if result.metadata.get('deal_outcome') != filter_value:
                        include = False
                        break
                        
                elif filter_key == 'deal_size_category':
                    if result.metadata.get('deal_size_category') != filter_value:
                        include = False
                        break
                        
                elif filter_key == 'deal_type':
                    if result.metadata.get('deal_type') != filter_value:
                        include = False
                        break
                        
                elif filter_key == 'deal_amount_range':
                    result_amount = result.metadata.get('deal_amount', 0)
                    if isinstance(filter_value, tuple):
                        min_amount, max_amount = filter_value
                        if not (min_amount <= result_amount <= max_amount):
                            include = False
                            break
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    def _analyze_patterns(
        self,
        similar_deals: List[VectorSearchResult],
        current_deal_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze patterns from similar deals"""
        
        if not similar_deals:
            return {'pattern_analysis': 'insufficient_data'}
        
        # Group by outcomes
        won_deals = [d for d in similar_deals if d.metadata.get('deal_outcome') == 'won']
        lost_deals = [d for d in similar_deals if d.metadata.get('deal_outcome') == 'lost']
        open_deals = [d for d in similar_deals if d.metadata.get('deal_outcome') == 'open']
        
        # Analyze current deal position
        current_metrics = self._extract_current_metrics(current_deal_metadata)
        
        # Compare against patterns
        insights = {
            'total_similar_deals': len(similar_deals),
            'won_deals_count': len(won_deals),
            'lost_deals_count': len(lost_deals),
            'open_deals_count': len(open_deals),
            'success_rate': len(won_deals) / len(similar_deals) if similar_deals else 0,
            'current_deal_position': current_metrics,
            'risk_factors': self._identify_risk_factors(current_metrics, lost_deals),
            'success_indicators': self._identify_success_indicators(current_metrics, won_deals)
        }
        
        return insights
    
    def _extract_current_metrics(self, deal_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from current deal"""
        return {
            'activities_count': deal_metadata.get('activities_count', 0),
            'response_time': deal_metadata.get('response_time_metrics', {}).get('avg_response_time_hours', 0),
            'communication_gaps': deal_metadata.get('communication_gaps_count', 0),
            'business_hours_ratio': deal_metadata.get('business_hours_ratio', 0),
            'activity_trend': deal_metadata.get('activity_frequency_trend', 'unknown'),
            'deal_age': deal_metadata.get('deal_age_days', 0)
        }
    
    def _identify_risk_factors(
        self,
        current_metrics: Dict[str, Any],
        lost_deals: List[VectorSearchResult]
    ) -> List[str]:
        """Identify risk factors based on lost deal patterns"""
        
        risk_factors = []
        
        if not lost_deals:
            return risk_factors
        
        # Analyze lost deal patterns
        avg_lost_response_time = sum(
            d.metadata.get('response_time_metrics', {}).get('avg_response_time_hours', 0)
            for d in lost_deals
        ) / len(lost_deals)
        
        avg_lost_gaps = sum(
            d.metadata.get('communication_gaps_count', 0)
            for d in lost_deals
        ) / len(lost_deals)
        
        # Compare current deal
        if current_metrics['response_time'] > avg_lost_response_time:
            risk_factors.append("Response time exceeds typical lost deal pattern")
        
        if current_metrics['communication_gaps'] >= avg_lost_gaps:
            risk_factors.append("Communication gaps match lost deal pattern")
        
        if current_metrics['activity_trend'] == 'declining':
            declining_lost = sum(
                1 for d in lost_deals 
                if d.metadata.get('activity_frequency_trend') == 'declining'
            )
            if declining_lost / len(lost_deals) > 0.5:
                risk_factors.append("Declining activity trend common in lost deals")
        
        return risk_factors
    
    def _identify_success_indicators(
        self,
        current_metrics: Dict[str, Any],
        won_deals: List[VectorSearchResult]
    ) -> List[str]:
        """Identify success indicators based on won deal patterns"""
        
        success_indicators = []
        
        if not won_deals:
            return success_indicators
        
        # Analyze won deal patterns
        avg_won_response_time = sum(
            d.metadata.get('response_time_metrics', {}).get('avg_response_time_hours', 0)
            for d in won_deals
        ) / len(won_deals)
        
        avg_won_activities = sum(
            d.metadata.get('activities_count', 0)
            for d in won_deals
        ) / len(won_deals)
        
        # Compare current deal
        if current_metrics['response_time'] <= avg_won_response_time:
            success_indicators.append("Response time matches successful deal pattern")
        
        if current_metrics['activities_count'] >= avg_won_activities:
            success_indicators.append("Activity level matches successful deal pattern")
        
        if current_metrics['communication_gaps'] == 0:
            no_gap_won = sum(
                1 for d in won_deals 
                if d.metadata.get('communication_gaps_count', 0) == 0
            )
            if no_gap_won / len(won_deals) > 0.7:
                success_indicators.append("No communication gaps aligns with won deals")
        
        return success_indicators
    
    def _generate_recommendations(
        self,
        insights: Dict[str, Any],
        deal_metadata: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on insights"""
        
        recommendations = []
        
        # Risk-based recommendations
        risk_factors = insights.get('risk_factors', [])
        for risk in risk_factors:
            if "response time" in risk.lower():
                recommendations.append({
                    'type': 'urgent',
                    'action': 'Improve response time',
                    'details': 'Reduce response time to emails and calls to match successful deal patterns'
                })
            
            elif "communication gaps" in risk.lower():
                recommendations.append({
                    'type': 'critical',
                    'action': 'Maintain consistent communication',
                    'details': 'Avoid communication gaps longer than 2-3 days'
                })
            
            elif "declining activity" in risk.lower():
                recommendations.append({
                    'type': 'important',
                    'action': 'Increase engagement',
                    'details': 'Boost activity frequency to prevent deal momentum loss'
                })
        
        # Success-based recommendations
        success_indicators = insights.get('success_indicators', [])
        if success_indicators:
            recommendations.append({
                'type': 'positive',
                'action': 'Continue current approach',
                'details': 'Your current patterns match successful deals - maintain momentum'
            })
        
        # General recommendations based on success rate
        success_rate = insights.get('success_rate', 0)
        if success_rate < 0.3:
            recommendations.append({
                'type': 'warning',
                'action': 'Review deal strategy',
                'details': 'Similar deals have low success rate - consider different approach'
            })
        elif success_rate > 0.7:
            recommendations.append({
                'type': 'positive',
                'action': 'High success probability',
                'details': 'Similar deals have high success rate - maintain current trajectory'
            })
        
        return recommendations


# Factory function
def create_rag_retriever(
    embedding_service: EmbeddingService = None,
    vector_store: VectorStore = None
) -> RAGRetriever:
    """Create RAG retriever with default services if not provided"""
    
    if not embedding_service:
        from core.embedding_service import get_embedding_service
        embedding_service = get_embedding_service()
    
    if not vector_store:
        from core.vector_store import get_vector_store
        vector_store = get_vector_store()
    
    return RAGRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store
    )


# Example usage and testing
def test_rag_retriever():
    """Test the RAG retriever with sample data"""
    
    try:
        # Create retriever
        retriever = create_rag_retriever()
        
        # Test query
        test_query = "Client interested in proposal, scheduled follow-up meeting"
        
        print("Testing RAG Retriever...")
        print(f"Query: {test_query}")
        
        # Retrieve similar patterns
        similar_deals = retriever.retrieve_similar_patterns(test_query, top_k=3)
        
        print(f"\nFound {len(similar_deals)} similar deals:")
        for i, deal in enumerate(similar_deals):
            print(f"{i+1}. Deal {deal.deal_id} (similarity: {deal.similarity_score:.3f})")
            print(f"   Outcome: {deal.metadata.get('deal_outcome', 'unknown')}")
            print(f"   Activities: {deal.metadata.get('activities_count', 0)}")
        
        # Test contextual insights
        sample_metadata = {
            'deal_amount': 50000,
            'deal_stage': 'proposal',
            'deal_type': 'newbusiness',
            'activities_count': 12,
            'response_time_metrics': {'avg_response_time_hours': 6.2},
            'communication_gaps_count': 1,
            'business_hours_ratio': 0.8,
            'activity_frequency_trend': 'stable'
        }
        
        insights = retriever.get_contextual_insights(test_query, sample_metadata)
        
        print(f"\nContextual Insights:")
        print(f"Similar deals: {insights['similar_deals_count']}")
        print(f"Risk factors: {insights['insights'].get('risk_factors', [])}")
        print(f"Success indicators: {insights['insights'].get('success_indicators', [])}")
        print(f"Recommendations: {len(insights['recommendations'])}")
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_rag_retriever()