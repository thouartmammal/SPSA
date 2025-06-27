import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import math

from models.schemas import VectorSearchResult, DealPattern
from utils.cache import CacheManager
from utils.helpers import calculate_similarity_score, normalize_scores, validate_search_parameters
from config.settings import settings

logger = logging.getLogger(__name__)

class EnhancedSimilaritySearch:
    """
    Enhanced similarity search with intelligent ranking, filtering, and optimization
    for the Sales Sentiment RAG system. Provides advanced search capabilities
    beyond basic vector similarity matching.
    """
    
    def __init__(self, cache_manager: CacheManager = None):
        """
        Initialize Enhanced Similarity Search
        
        Args:
            cache_manager: Cache manager for search optimization
        """
        self.cache_manager = cache_manager or CacheManager()
        
        # Search configuration
        self.similarity_weights = {
            'vector_similarity': 0.6,      # Core vector similarity
            'metadata_relevance': 0.2,     # Metadata matching
            'temporal_relevance': 0.1,     # Time-based relevance
            'performance_boost': 0.1       # Success pattern boost
        }
        
        # Filtering thresholds
        self.quality_thresholds = {
            'min_activities': 3,
            'min_time_span': 1,
            'min_similarity': 0.3
        }
        
        logger.info("Enhanced Similarity Search initialized")
    
    def enhanced_search(
        self,
        query_embedding: List[float],
        raw_results: List[VectorSearchResult],
        search_context: Dict[str, Any] = None,
        ranking_strategy: str = "comprehensive",
        enable_caching: bool = True
    ) -> List[VectorSearchResult]:
        """
        Perform enhanced similarity search with intelligent ranking
        
        Args:
            query_embedding: Query vector embedding
            raw_results: Raw search results from vector store
            search_context: Context for search enhancement
            ranking_strategy: Strategy for result ranking
            enable_caching: Whether to use caching
            
        Returns:
            Enhanced and ranked search results
        """
        
        search_context = search_context or {}
        
        if not raw_results:
            logger.debug("No raw results to enhance")
            return []
        
        try:
            # Validate search parameters
            if not validate_search_parameters(query_embedding, raw_results):
                logger.warning("Invalid search parameters")
                return raw_results
            
            # Apply quality filtering
            quality_results = self._apply_quality_filters(raw_results)
            if not quality_results:
                logger.debug("No results passed quality filters")
                return []
            
            # Calculate enhanced similarity scores
            enhanced_results = self._calculate_enhanced_similarity(
                query_embedding, quality_results, search_context
            )
            
            # Apply ranking strategy
            ranked_results = self._apply_ranking_strategy(
                enhanced_results, ranking_strategy, search_context
            )
            
            # Apply diversification if needed
            diversified_results = self._apply_diversification(
                ranked_results, search_context
            )
            
            # Apply final filtering and normalization
            final_results = self._finalize_results(diversified_results, search_context)
            
            logger.debug(f"Enhanced search: {len(raw_results)} → {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            return raw_results  # Fall back to raw results
    
    def contextual_search(
        self,
        query_embedding: List[float],
        raw_results: List[VectorSearchResult],
        current_deal_metadata: Dict[str, Any],
        analysis_type: str = "sentiment"
    ) -> List[VectorSearchResult]:
        """
        Perform contextual search optimized for specific analysis types
        
        Args:
            query_embedding: Query vector embedding
            raw_results: Raw search results
            current_deal_metadata: Current deal context
            analysis_type: Type of analysis (sentiment, risk, coaching, etc.)
            
        Returns:
            Contextually relevant search results
        """
        
        search_context = {
            'current_deal': current_deal_metadata,
            'analysis_type': analysis_type,
            'contextual_boost': True
        }
        
        # Apply analysis-type specific enhancements
        if analysis_type == "sentiment":
            return self._sentiment_optimized_search(query_embedding, raw_results, search_context)
        elif analysis_type == "risk":
            return self._risk_optimized_search(query_embedding, raw_results, search_context)
        elif analysis_type == "coaching":
            return self._coaching_optimized_search(query_embedding, raw_results, search_context)
        elif analysis_type == "benchmarking":
            return self._benchmarking_optimized_search(query_embedding, raw_results, search_context)
        else:
            return self.enhanced_search(query_embedding, raw_results, search_context)
    
    def multi_stage_search(
        self,
        query_embedding: List[float],
        raw_results: List[VectorSearchResult],
        stages: List[Dict[str, Any]]
    ) -> List[VectorSearchResult]:
        """
        Perform multi-stage search with different criteria at each stage
        
        Args:
            query_embedding: Query vector embedding
            raw_results: Raw search results
            stages: List of stage configurations
            
        Returns:
            Multi-stage filtered and ranked results
        """
        
        current_results = raw_results
        
        for i, stage_config in enumerate(stages):
            logger.debug(f"Applying search stage {i+1}: {stage_config.get('name', 'unnamed')}")
            
            stage_context = {
                'stage_number': i + 1,
                'total_stages': len(stages),
                **stage_config
            }
            
            current_results = self.enhanced_search(
                query_embedding, 
                current_results, 
                stage_context,
                ranking_strategy=stage_config.get('ranking_strategy', 'comprehensive')
            )
            
            # Apply stage-specific result limit
            stage_limit = stage_config.get('max_results')
            if stage_limit and len(current_results) > stage_limit:
                current_results = current_results[:stage_limit]
        
        return current_results
    
    def similarity_explanation(
        self,
        query_embedding: List[float],
        result: VectorSearchResult,
        search_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Provide detailed explanation of similarity calculation
        
        Args:
            query_embedding: Query vector embedding
            result: Search result to explain
            search_context: Search context
            
        Returns:
            Detailed similarity explanation
        """
        
        search_context = search_context or {}
        
        try:
            # Calculate component scores
            vector_similarity = calculate_similarity_score(query_embedding, result.embedding) if hasattr(result, 'embedding') else result.similarity_score
            
            metadata_score = self._calculate_metadata_relevance(result, search_context)
            temporal_score = self._calculate_temporal_relevance(result, search_context)
            performance_score = self._calculate_performance_boost(result, search_context)
            
            # Calculate weighted final score
            final_score = (
                vector_similarity * self.similarity_weights['vector_similarity'] +
                metadata_score * self.similarity_weights['metadata_relevance'] +
                temporal_score * self.similarity_weights['temporal_relevance'] +
                performance_score * self.similarity_weights['performance_boost']
            )
            
            return {
                'final_similarity_score': final_score,
                'component_scores': {
                    'vector_similarity': vector_similarity,
                    'metadata_relevance': metadata_score,
                    'temporal_relevance': temporal_score,
                    'performance_boost': performance_score
                },
                'weights': self.similarity_weights,
                'explanation': self._generate_similarity_explanation(
                    vector_similarity, metadata_score, temporal_score, performance_score
                )
            }
            
        except Exception as e:
            logger.error(f"Error generating similarity explanation: {e}")
            return {'error': str(e)}
    
    # Private methods for search enhancement
    
    def _apply_quality_filters(self, results: List[VectorSearchResult]) -> List[VectorSearchResult]:
        """Apply quality filters to search results"""
        
        quality_results = []
        
        for result in results:
            # Check minimum activity count
            if result.metadata.get('activities_count', 0) < self.quality_thresholds['min_activities']:
                continue
            
            # Check minimum time span
            if result.metadata.get('time_span_days', 0) < self.quality_thresholds['min_time_span']:
                continue
            
            # Check minimum similarity
            if result.similarity_score < self.quality_thresholds['min_similarity']:
                continue
            
            # Check for valid deal outcome
            if not result.metadata.get('deal_outcome'):
                continue
            
            quality_results.append(result)
        
        logger.debug(f"Quality filters: {len(results)} → {len(quality_results)} results")
        return quality_results
    
    def _calculate_enhanced_similarity(
        self,
        query_embedding: List[float],
        results: List[VectorSearchResult],
        search_context: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Calculate enhanced similarity scores"""
        
        enhanced_results = []
        
        for result in results:
            # Calculate component scores
            metadata_score = self._calculate_metadata_relevance(result, search_context)
            temporal_score = self._calculate_temporal_relevance(result, search_context)
            performance_score = self._calculate_performance_boost(result, search_context)
            
            # Calculate enhanced similarity
            enhanced_similarity = (
                result.similarity_score * self.similarity_weights['vector_similarity'] +
                metadata_score * self.similarity_weights['metadata_relevance'] +
                temporal_score * self.similarity_weights['temporal_relevance'] +
                performance_score * self.similarity_weights['performance_boost']
            )
            
            # Create enhanced result
            enhanced_result = VectorSearchResult(
                deal_id=result.deal_id,
                similarity_score=enhanced_similarity,
                metadata={
                    **result.metadata,
                    'original_similarity': result.similarity_score,
                    'metadata_relevance': metadata_score,
                    'temporal_relevance': temporal_score,
                    'performance_boost': performance_score
                },
                combined_text=result.combined_text
            )
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _calculate_metadata_relevance(
        self,
        result: VectorSearchResult,
        search_context: Dict[str, Any]
    ) -> float:
        """Calculate metadata relevance score"""
        
        relevance_score = 0.0
        current_deal = search_context.get('current_deal', {})
        
        if not current_deal:
            return 0.5  # Neutral score if no context
        
        # Deal size category matching
        if (result.metadata.get('deal_size_category') == 
            current_deal.get('deal_size_category')):
            relevance_score += 0.3
        
        # Deal type matching
        if (result.metadata.get('deal_type') == 
            current_deal.get('deal_type')):
            relevance_score += 0.3
        
        # Deal stage proximity
        current_stage = current_deal.get('deal_stage', '')
        result_stage = result.metadata.get('deal_stage', '')
        if current_stage and result_stage:
            if current_stage == result_stage:
                relevance_score += 0.2
            elif self._are_adjacent_stages(current_stage, result_stage):
                relevance_score += 0.1
        
        # Activity count similarity
        current_activities = current_deal.get('activities_count', 0)
        result_activities = result.metadata.get('activities_count', 0)
        if current_activities > 0 and result_activities > 0:
            activity_similarity = 1.0 - abs(current_activities - result_activities) / max(current_activities, result_activities)
            relevance_score += activity_similarity * 0.2
        
        return min(1.0, relevance_score)
    
    def _calculate_temporal_relevance(
        self,
        result: VectorSearchResult,
        search_context: Dict[str, Any]
    ) -> float:
        """Calculate temporal relevance score"""
        
        # More recent deals get higher scores
        last_activity = result.metadata.get('last_activity_date')
        if not last_activity:
            return 0.5  # Neutral score for unknown dates
        
        try:
            last_activity_date = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
            days_ago = (datetime.now(last_activity_date.tzinfo) - last_activity_date).days
            
            # Exponential decay: more recent = higher score
            temporal_score = math.exp(-days_ago / 365.0)  # 1-year half-life
            
            return temporal_score
            
        except Exception as e:
            logger.debug(f"Error calculating temporal relevance: {e}")
            return 0.5
    
    def _calculate_performance_boost(
        self,
        result: VectorSearchResult,
        search_context: Dict[str, Any]
    ) -> float:
        """Calculate performance-based boost"""
        
        analysis_type = search_context.get('analysis_type', 'sentiment')
        deal_outcome = result.metadata.get('deal_outcome', 'unknown')
        
        # Boost successful deals for sentiment and coaching analysis
        if analysis_type in ['sentiment', 'coaching', 'benchmarking']:
            if deal_outcome == 'won':
                return 1.0
            elif deal_outcome == 'open':
                # Boost high-probability open deals
                probability = result.metadata.get('deal_probability', 0)
                return 0.5 + (probability / 100.0) * 0.5
            else:  # lost deals
                return 0.3
        
        # For risk analysis, boost failed deals
        elif analysis_type == 'risk':
            if deal_outcome == 'lost':
                return 1.0
            elif deal_outcome == 'won':
                return 0.3
            else:  # open deals
                return 0.5
        
        return 0.5  # Neutral boost
    
    def _apply_ranking_strategy(
        self,
        results: List[VectorSearchResult],
        strategy: str,
        search_context: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Apply ranking strategy to results"""
        
        if strategy == "similarity_only":
            return sorted(results, key=lambda x: x.similarity_score, reverse=True)
        
        elif strategy == "outcome_weighted":
            return self._outcome_weighted_ranking(results, search_context)
        
        elif strategy == "temporal_weighted":
            return self._temporal_weighted_ranking(results, search_context)
        
        elif strategy == "comprehensive":
            return self._comprehensive_ranking(results, search_context)
        
        else:
            logger.warning(f"Unknown ranking strategy: {strategy}")
            return sorted(results, key=lambda x: x.similarity_score, reverse=True)
    
    def _outcome_weighted_ranking(
        self,
        results: List[VectorSearchResult],
        search_context: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Rank results with outcome weighting"""
        
        analysis_type = search_context.get('analysis_type', 'sentiment')
        
        def outcome_weight(result):
            outcome = result.metadata.get('deal_outcome', 'unknown')
            base_score = result.similarity_score
            
            if analysis_type in ['sentiment', 'coaching']:
                if outcome == 'won':
                    return base_score * 1.2
                elif outcome == 'lost':
                    return base_score * 0.8
            elif analysis_type == 'risk':
                if outcome == 'lost':
                    return base_score * 1.2
                elif outcome == 'won':
                    return base_score * 0.8
            
            return base_score
        
        return sorted(results, key=outcome_weight, reverse=True)
    
    def _temporal_weighted_ranking(
        self,
        results: List[VectorSearchResult],
        search_context: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Rank results with temporal weighting"""
        
        def temporal_weight(result):
            base_score = result.similarity_score
            temporal_score = result.metadata.get('temporal_relevance', 0.5)
            return base_score * (0.7 + 0.3 * temporal_score)
        
        return sorted(results, key=temporal_weight, reverse=True)
    
    def _comprehensive_ranking(
        self,
        results: List[VectorSearchResult],
        search_context: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Comprehensive ranking using all factors"""
        
        # Results are already enhanced with all factors in similarity_score
        return sorted(results, key=lambda x: x.similarity_score, reverse=True)
    
    def _apply_diversification(
        self,
        results: List[VectorSearchResult],
        search_context: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Apply diversification to avoid too similar results"""
        
        diversify = search_context.get('diversify', True)
        if not diversify or len(results) <= 3:
            return results
        
        diversified = []
        used_characteristics = set()
        
        for result in results:
            # Create characteristic signature
            signature = (
                result.metadata.get('deal_outcome', 'unknown'),
                result.metadata.get('deal_size_category', 'unknown'),
                result.metadata.get('deal_type', 'unknown')
            )
            
            # If we haven't seen this combination, add it
            if signature not in used_characteristics or len(diversified) < 3:
                diversified.append(result)
                used_characteristics.add(signature)
            
            # Stop when we have enough diverse results
            if len(diversified) >= min(len(results), search_context.get('max_results', 10)):
                break
        
        # Fill remaining slots with best remaining results
        remaining_slots = search_context.get('max_results', 10) - len(diversified)
        if remaining_slots > 0:
            remaining_results = [r for r in results if r not in diversified]
            diversified.extend(remaining_results[:remaining_slots])
        
        return diversified
    
    def _finalize_results(
        self,
        results: List[VectorSearchResult],
        search_context: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Apply final processing to results"""
        
        if not results:
            return results
        
        # Normalize similarity scores to 0-1 range
        max_score = max(r.similarity_score for r in results)
        min_score = min(r.similarity_score for r in results)
        
        if max_score > min_score:
            for result in results:
                normalized_score = (result.similarity_score - min_score) / (max_score - min_score)
                result.similarity_score = normalized_score
        
        # Apply final result limit
        max_results = search_context.get('max_results', len(results))
        return results[:max_results]
    
    # Analysis-type specific search methods
    
    def _sentiment_optimized_search(
        self,
        query_embedding: List[float],
        raw_results: List[VectorSearchResult],
        search_context: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Optimize search for sentiment analysis"""
        
        # Boost deals with similar communication patterns
        search_context['boost_communication_patterns'] = True
        search_context['ranking_strategy'] = 'outcome_weighted'
        
        return self.enhanced_search(query_embedding, raw_results, search_context)
    
    def _risk_optimized_search(
        self,
        query_embedding: List[float],
        raw_results: List[VectorSearchResult],
        search_context: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Optimize search for risk assessment"""
        
        # Boost failed deals and deals with warning signs
        search_context['boost_failed_deals'] = True
        search_context['ranking_strategy'] = 'outcome_weighted'
        
        return self.enhanced_search(query_embedding, raw_results, search_context)
    
    def _coaching_optimized_search(
        self,
        query_embedding: List[float],
        raw_results: List[VectorSearchResult],
        search_context: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Optimize search for coaching recommendations"""
        
        # Boost successful deals for learning patterns
        search_context['boost_successful_deals'] = True
        search_context['ranking_strategy'] = 'outcome_weighted'
        search_context['diversify'] = True  # Want diverse examples
        
        return self.enhanced_search(query_embedding, raw_results, search_context)
    
    def _benchmarking_optimized_search(
        self,
        query_embedding: List[float],
        raw_results: List[VectorSearchResult],
        search_context: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Optimize search for performance benchmarking"""
        
        # Focus on deals with similar characteristics
        search_context['boost_similar_characteristics'] = True
        search_context['ranking_strategy'] = 'comprehensive'
        search_context['diversify'] = False  # Want similar deals for benchmarking
        
        return self.enhanced_search(query_embedding, raw_results, search_context)
    
    # Helper methods
    
    def _are_adjacent_stages(self, stage1: str, stage2: str) -> bool:
        """Check if two deal stages are adjacent"""
        
        stage_order = [
            'prospecting', 'qualification', 'proposal', 
            'negotiation', 'closed_won', 'closed_lost'
        ]
        
        try:
            idx1 = stage_order.index(stage1.lower())
            idx2 = stage_order.index(stage2.lower())
            return abs(idx1 - idx2) == 1
        except ValueError:
            return False
    
    def _generate_similarity_explanation(
        self,
        vector_sim: float,
        metadata_sim: float,
        temporal_sim: float,
        performance_sim: float
    ) -> str:
        """Generate human-readable similarity explanation"""
        
        explanations = []
        
        if vector_sim > 0.8:
            explanations.append("Very high content similarity")
        elif vector_sim > 0.6:
            explanations.append("High content similarity")
        elif vector_sim > 0.4:
            explanations.append("Moderate content similarity")
        else:
            explanations.append("Low content similarity")
        
        if metadata_sim > 0.7:
            explanations.append("Strong metadata match")
        elif metadata_sim > 0.4:
            explanations.append("Moderate metadata match")
        
        if temporal_sim > 0.7:
            explanations.append("Recent deal")
        elif temporal_sim < 0.3:
            explanations.append("Older deal")
        
        if performance_sim > 0.7:
            explanations.append("High-performing pattern")
        elif performance_sim < 0.3:
            explanations.append("Lower-performing pattern")
        
        return "; ".join(explanations)


# Factory function
def create_enhanced_similarity_search(cache_manager: CacheManager = None) -> EnhancedSimilaritySearch:
    """Create enhanced similarity search instance"""
    return EnhancedSimilaritySearch(cache_manager=cache_manager)


# Example usage and testing
def test_enhanced_similarity_search():
    """Test enhanced similarity search functionality"""
    
    try:
        # Create search engine
        search_engine = create_enhanced_similarity_search()
        
        # Mock query embedding
        query_embedding = [0.1] * 384  # Mock embedding
        
        # Mock search results
        mock_results = [
            VectorSearchResult(
                deal_id="deal_001",
                similarity_score=0.85,
                metadata={
                    'deal_outcome': 'won',
                    'deal_size_category': 'medium',
                    'activities_count': 15,
                    'time_span_days': 30
                },
                combined_text="Sample deal activities"
            )
        ]
        
        # Mock search context
        search_context = {
            'current_deal': {
                'deal_size_category': 'medium',
                'deal_type': 'newbusiness'
            },
            'analysis_type': 'sentiment'
        }
        
        # Test enhanced search
        enhanced_results = search_engine.enhanced_search(
            query_embedding, mock_results, search_context
        )
        
        print(f"Enhanced Search Results: {len(enhanced_results)}")
        
        if enhanced_results:
            # Test similarity explanation
            explanation = search_engine.similarity_explanation(
                query_embedding, enhanced_results[0], search_context
            )
            print("Similarity Explanation:")
            print(json.dumps(explanation, indent=2))
        
        print("✅ Enhanced Similarity Search test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    import json
    test_enhanced_similarity_search()