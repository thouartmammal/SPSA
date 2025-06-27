import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import statistics
from collections import defaultdict

from models.schemas import VectorSearchResult

logger = logging.getLogger(__name__)

class RAGContextBuilder:
    """Build intelligent context from similar historical deals for LLM analysis"""
    
    def __init__(self):
        self.min_sample_size = 2  # Minimum deals needed for reliable patterns
        logger.info("RAG Context Builder initialized")
    
    def build_context(
        self, 
        similar_deals: List[VectorSearchResult], 
        current_deal_metadata: Dict[str, Any],
        max_context_length: int = 2000
    ) -> str:
        """
        Build comprehensive context from similar deals for LLM analysis
        
        Args:
            similar_deals: List of similar deals from vector search
            current_deal_metadata: Metadata of current deal being analyzed
            max_context_length: Maximum length of context string
            
        Returns:
            Structured context string for LLM prompt
        """
        
        if not similar_deals:
            return self._create_no_context_message()
        
        # Filter and prepare deals for analysis
        filtered_deals = self._filter_quality_deals(similar_deals)
        
        if len(filtered_deals) < self.min_sample_size:
            return self._create_insufficient_context_message(len(filtered_deals))
        
        # Group deals by outcome
        deal_groups = self._group_by_outcome(filtered_deals)
        
        # Extract patterns from each group
        success_patterns = self._extract_success_patterns(deal_groups.get('won', []))
        failure_patterns = self._extract_failure_patterns(deal_groups.get('lost', []))
        open_patterns = self._extract_open_patterns(deal_groups.get('open', []))
        
        # Generate performance benchmarks
        benchmarks = self._generate_benchmarks(filtered_deals)
        
        # Create deal characteristics context
        characteristics_context = self._create_characteristics_context(
            current_deal_metadata, filtered_deals
        )
        
        # Format final context
        context = self._format_context(
            characteristics_context=characteristics_context,
            success_patterns=success_patterns,
            failure_patterns=failure_patterns,
            open_patterns=open_patterns,
            benchmarks=benchmarks,
            total_similar_deals=len(filtered_deals),
            similarity_scores=[deal.similarity_score for deal in filtered_deals]
        )
        
        # Truncate if too long
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n[Context truncated for length]"
        
        logger.info(f"Built context from {len(filtered_deals)} similar deals")
        return context
    
    def _filter_quality_deals(self, deals: List[VectorSearchResult]) -> List[VectorSearchResult]:
        """Filter deals to ensure quality data for pattern analysis"""
        quality_deals = []
        
        for deal in deals:
            # Check for minimum data quality
            metadata = deal.metadata
            
            # Must have basic activity data
            if metadata.get('activities_count', 0) < 2:
                continue
                
            # Must have clear outcome or be open
            if not metadata.get('deal_outcome'):
                continue
                
            # Must have reasonable similarity score (> 0.3)
            if deal.similarity_score < 0.3:
                continue
                
            quality_deals.append(deal)
        
        logger.debug(f"Filtered {len(deals)} deals to {len(quality_deals)} quality deals")
        return quality_deals
    
    def _group_by_outcome(self, deals: List[VectorSearchResult]) -> Dict[str, List[VectorSearchResult]]:
        """Group deals by their outcomes (won/lost/open)"""
        groups = defaultdict(list)
        
        for deal in deals:
            outcome = deal.metadata.get('deal_outcome', 'unknown')
            groups[outcome].append(deal)
        
        return dict(groups)
    
    def _extract_success_patterns(self, won_deals: List[VectorSearchResult]) -> Dict[str, Any]:
        """Extract patterns from successful deals"""
        if len(won_deals) < self.min_sample_size:
            return {'insufficient_data': True, 'count': len(won_deals)}
        
        # Extract key metrics
        response_times = []
        activity_counts = []
        communication_gaps = []
        business_hours_ratios = []
        deal_ages = []
        
        for deal in won_deals:
            metadata = deal.metadata
            
            # Response time metrics
            response_time = metadata.get('response_time_metrics', {}).get('avg_response_time_hours', 0)
            if response_time > 0:
                response_times.append(response_time)
            
            # Activity patterns
            activity_count = metadata.get('activities_count', 0)
            if activity_count > 0:
                activity_counts.append(activity_count)
            
            # Communication gaps
            gaps = metadata.get('communication_gaps_count', 0)
            communication_gaps.append(gaps)
            
            # Business hours activity
            bh_ratio = metadata.get('business_hours_ratio', 0)
            if bh_ratio > 0:
                business_hours_ratios.append(bh_ratio)
            
            # Deal lifecycle
            age = metadata.get('deal_age_days', 0)
            if age > 0:
                deal_ages.append(age)
        
        return {
            'count': len(won_deals),
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'avg_activity_count': statistics.mean(activity_counts) if activity_counts else 0,
            'avg_communication_gaps': statistics.mean(communication_gaps),
            'avg_business_hours_ratio': statistics.mean(business_hours_ratios) if business_hours_ratios else 0,
            'avg_deal_age': statistics.mean(deal_ages) if deal_ages else 0,
            'response_time_range': (min(response_times), max(response_times)) if len(response_times) > 1 else None,
            'common_characteristics': self._extract_common_characteristics(won_deals)
        }
    
    def _extract_failure_patterns(self, lost_deals: List[VectorSearchResult]) -> Dict[str, Any]:
        """Extract patterns from failed deals"""
        if len(lost_deals) < self.min_sample_size:
            return {'insufficient_data': True, 'count': len(lost_deals)}
        
        # Extract key failure indicators
        response_times = []
        activity_counts = []
        communication_gaps = []
        business_hours_ratios = []
        deal_ages = []
        
        for deal in lost_deals:
            metadata = deal.metadata
            
            # Response time metrics  
            response_time = metadata.get('response_time_metrics', {}).get('avg_response_time_hours', 0)
            if response_time > 0:
                response_times.append(response_time)
            
            # Activity patterns
            activity_count = metadata.get('activities_count', 0)
            if activity_count > 0:
                activity_counts.append(activity_count)
            
            # Communication gaps (key failure indicator)
            gaps = metadata.get('communication_gaps_count', 0)
            communication_gaps.append(gaps)
            
            # Business hours activity
            bh_ratio = metadata.get('business_hours_ratio', 0)
            if bh_ratio > 0:
                business_hours_ratios.append(bh_ratio)
            
            # Deal lifecycle
            age = metadata.get('deal_age_days', 0)
            if age > 0:
                deal_ages.append(age)
        
        return {
            'count': len(lost_deals),
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'avg_activity_count': statistics.mean(activity_counts) if activity_counts else 0,
            'avg_communication_gaps': statistics.mean(communication_gaps),
            'avg_business_hours_ratio': statistics.mean(business_hours_ratios) if business_hours_ratios else 0,
            'avg_deal_age': statistics.mean(deal_ages) if deal_ages else 0,
            'response_time_range': (min(response_times), max(response_times)) if len(response_times) > 1 else None,
            'common_warning_signs': self._extract_warning_signs(lost_deals)
        }
    
    def _extract_open_patterns(self, open_deals: List[VectorSearchResult]) -> Dict[str, Any]:
        """Extract patterns from currently open deals"""
        if not open_deals:
            return {'count': 0}
        
        activity_counts = []
        communication_gaps = []
        probabilities = []
        
        for deal in open_deals:
            metadata = deal.metadata
            
            activity_count = metadata.get('activities_count', 0)
            if activity_count > 0:
                activity_counts.append(activity_count)
            
            gaps = metadata.get('communication_gaps_count', 0)
            communication_gaps.append(gaps)
            
            prob = metadata.get('deal_probability', 0)
            if prob > 0:
                probabilities.append(prob)
        
        return {
            'count': len(open_deals),
            'avg_activity_count': statistics.mean(activity_counts) if activity_counts else 0,
            'avg_communication_gaps': statistics.mean(communication_gaps),
            'avg_probability': statistics.mean(probabilities) if probabilities else 0
        }
    
    def _extract_common_characteristics(self, deals: List[VectorSearchResult]) -> Dict[str, Any]:
        """Extract common characteristics from a group of deals"""
        deal_types = []
        deal_sizes = []
        industries = []
        
        for deal in deals:
            metadata = deal.metadata
            
            deal_type = metadata.get('deal_type', '')
            if deal_type:
                deal_types.append(deal_type)
            
            deal_size = metadata.get('deal_size_category', '')
            if deal_size:
                deal_sizes.append(deal_size)
        
        return {
            'most_common_deal_type': max(set(deal_types), key=deal_types.count) if deal_types else None,
            'most_common_deal_size': max(set(deal_sizes), key=deal_sizes.count) if deal_sizes else None
        }
    
    def _extract_warning_signs(self, lost_deals: List[VectorSearchResult]) -> List[str]:
        """Extract specific warning signs from lost deals"""
        warning_signs = []
        
        high_gap_count = sum(1 for deal in lost_deals 
                           if deal.metadata.get('communication_gaps_count', 0) > 2)
        
        if high_gap_count / len(lost_deals) > 0.5:
            warning_signs.append("Multiple communication gaps (>2)")
        
        slow_response_count = sum(1 for deal in lost_deals 
                                if deal.metadata.get('response_time_metrics', {}).get('avg_response_time_hours', 0) > 24)
        
        if slow_response_count / len(lost_deals) > 0.5:
            warning_signs.append("Slow response times (>24 hours)")
        
        declining_trend_count = sum(1 for deal in lost_deals 
                                  if deal.metadata.get('activity_frequency_trend') == 'declining')
        
        if declining_trend_count / len(lost_deals) > 0.3:
            warning_signs.append("Declining activity frequency")
        
        return warning_signs
    
    def _generate_benchmarks(self, all_deals: List[VectorSearchResult]) -> Dict[str, Any]:
        """Generate performance benchmarks from all similar deals"""
        response_times = []
        activity_counts = []
        success_rate_by_response_time = defaultdict(list)
        
        for deal in all_deals:
            metadata = deal.metadata
            outcome = metadata.get('deal_outcome')
            
            response_time = metadata.get('response_time_metrics', {}).get('avg_response_time_hours', 0)
            if response_time > 0:
                response_times.append(response_time)
                
                # Track success rate by response time buckets
                if response_time <= 4:
                    bucket = 'fast'
                elif response_time <= 12:
                    bucket = 'medium'
                else:
                    bucket = 'slow'
                
                success_rate_by_response_time[bucket].append(outcome == 'won')
            
            activity_count = metadata.get('activities_count', 0)
            if activity_count > 0:
                activity_counts.append(activity_count)
        
        # Calculate success rates by response time
        success_rates = {}
        for bucket, outcomes in success_rate_by_response_time.items():
            if outcomes:
                success_rates[f'{bucket}_response_success_rate'] = sum(outcomes) / len(outcomes)
        
        return {
            'median_response_time': statistics.median(response_times) if response_times else 0,
            'median_activity_count': statistics.median(activity_counts) if activity_counts else 0,
            **success_rates,
            'total_deals_analyzed': len(all_deals)
        }
    
    def _create_characteristics_context(
        self, 
        current_deal_metadata: Dict[str, Any], 
        similar_deals: List[VectorSearchResult]
    ) -> str:
        """Create context about deal characteristics and similarity"""
        
        current_amount = current_deal_metadata.get('deal_amount', 0)
        current_size = current_deal_metadata.get('deal_size_category', 'unknown')
        current_type = current_deal_metadata.get('deal_type', 'unknown')
        current_stage = current_deal_metadata.get('deal_stage', 'unknown')
        
        # Analyze similar deals characteristics
        similar_amounts = [deal.metadata.get('deal_amount', 0) for deal in similar_deals]
        avg_similar_amount = statistics.mean([a for a in similar_amounts if a > 0]) if similar_amounts else 0
        
        similarity_scores = [deal.similarity_score for deal in similar_deals]
        avg_similarity = statistics.mean(similarity_scores)
        
        context = f"""DEAL CHARACTERISTICS ANALYSIS:
Current Deal: ${current_amount:,.0f} ({current_size}) - {current_type} - {current_stage}
Similar Deals Found: {len(similar_deals)} (avg similarity: {avg_similarity:.2f})
Average Similar Deal Size: ${avg_similar_amount:,.0f}"""
        
        return context
    
    def _format_context(
        self,
        characteristics_context: str,
        success_patterns: Dict[str, Any],
        failure_patterns: Dict[str, Any], 
        open_patterns: Dict[str, Any],
        benchmarks: Dict[str, Any],
        total_similar_deals: int,
        similarity_scores: List[float]
    ) -> str:
        """Format all patterns into structured context for LLM"""
        
        context_parts = [
            "=" * 60,
            "HISTORICAL DEAL ANALYSIS FOR SENTIMENT PREDICTION",
            "=" * 60,
            "",
            characteristics_context,
            ""
        ]
        
        # Success patterns section
        if not success_patterns.get('insufficient_data', False):
            context_parts.extend([
                f"✅ SUCCESSFUL DEAL PATTERNS ({success_patterns['count']} won deals):",
                f"• Average Response Time: {success_patterns['avg_response_time']:.1f} hours",
                f"• Average Activities: {success_patterns['avg_activity_count']:.0f}",
                f"• Communication Gaps: {success_patterns['avg_communication_gaps']:.1f} avg",
                f"• Business Hours Activity: {success_patterns['avg_business_hours_ratio']:.0%}",
                f"• Average Deal Duration: {success_patterns['avg_deal_age']:.0f} days"
            ])
            
            if success_patterns.get('response_time_range'):
                min_rt, max_rt = success_patterns['response_time_range']
                context_parts.append(f"• Response Time Range: {min_rt:.1f}-{max_rt:.1f} hours")
            
            context_parts.append("")
        
        # Failure patterns section
        if not failure_patterns.get('insufficient_data', False):
            context_parts.extend([
                f"❌ FAILED DEAL PATTERNS ({failure_patterns['count']} lost deals):",
                f"• Average Response Time: {failure_patterns['avg_response_time']:.1f} hours",
                f"• Average Activities: {failure_patterns['avg_activity_count']:.0f}",
                f"• Communication Gaps: {failure_patterns['avg_communication_gaps']:.1f} avg",
                f"• Business Hours Activity: {failure_patterns['avg_business_hours_ratio']:.0%}",
                f"• Average Deal Duration: {failure_patterns['avg_deal_age']:.0f} days"
            ])
            
            warning_signs = failure_patterns.get('common_warning_signs', [])
            if warning_signs:
                context_parts.append(f"• Common Warning Signs: {', '.join(warning_signs)}")
            
            context_parts.append("")
        
        # Open deals context
        if open_patterns['count'] > 0:
            context_parts.extend([
                f"📊 OPEN DEAL PATTERNS ({open_patterns['count']} open deals):",
                f"• Average Activities: {open_patterns['avg_activity_count']:.0f}",
                f"• Average Probability: {open_patterns['avg_probability']:.0f}%",
                f"• Communication Gaps: {open_patterns['avg_communication_gaps']:.1f} avg",
                ""
            ])
        
        # Benchmarks section
        context_parts.extend([
            "📈 PERFORMANCE BENCHMARKS:",
            f"• Median Response Time: {benchmarks['median_response_time']:.1f} hours",
            f"• Median Activity Count: {benchmarks['median_activity_count']:.0f}"
        ])
        
        # Success rate benchmarks
        if 'fast_response_success_rate' in benchmarks:
            context_parts.extend([
                f"• Fast Response (<4h) Success Rate: {benchmarks['fast_response_success_rate']:.0%}",
                f"• Medium Response (4-12h) Success Rate: {benchmarks.get('medium_response_success_rate', 0):.0%}",
                f"• Slow Response (>12h) Success Rate: {benchmarks.get('slow_response_success_rate', 0):.0%}"
            ])
        
        context_parts.extend([
            "",
            "🎯 KEY INSIGHTS FOR ANALYSIS:",
            "Use these historical patterns to evaluate the current deal's sentiment.",
            "Compare current deal metrics against successful vs. failed patterns.",
            "Consider response times, communication gaps, and activity trends as key indicators.",
            ""
        ])
        
        return "\n".join(context_parts)
    
    def _create_no_context_message(self) -> str:
        """Create message when no similar deals found"""
        return """
HISTORICAL CONTEXT: No similar deals found in database.
This analysis will be based on general salesperson performance standards
without historical deal pattern comparison.
"""
    
    def _create_insufficient_context_message(self, deal_count: int) -> str:
        """Create message when insufficient similar deals found"""
        return f"""
HISTORICAL CONTEXT: Only {deal_count} similar deal(s) found.
Insufficient data for reliable pattern analysis.
Analysis will use general standards with limited historical context.
"""


# Example usage and testing
def test_context_builder():
    """Test the RAG Context Builder with sample data"""
    
    # Sample similar deals (would come from vector search)
    from models.schemas import VectorSearchResult
    
    sample_deals = [
        VectorSearchResult(
            deal_id="deal_001",
            similarity_score=0.85,
            metadata={
                'deal_outcome': 'won',
                'deal_amount': 50000,
                'deal_size_category': 'medium',
                'activities_count': 15,
                'response_time_metrics': {'avg_response_time_hours': 4.2},
                'communication_gaps_count': 0,
                'business_hours_ratio': 0.8,
                'deal_age_days': 45
            },
            combined_text="Sample won deal activities..."
        ),
        VectorSearchResult(
            deal_id="deal_002", 
            similarity_score=0.78,
            metadata={
                'deal_outcome': 'lost',
                'deal_amount': 45000,
                'deal_size_category': 'medium',
                'activities_count': 8,
                'response_time_metrics': {'avg_response_time_hours': 18.5},
                'communication_gaps_count': 3,
                'business_hours_ratio': 0.4,
                'deal_age_days': 67
            },
            combined_text="Sample lost deal activities..."
        )
    ]
    
    # Current deal metadata
    current_deal = {
        'deal_amount': 48000,
        'deal_size_category': 'medium', 
        'deal_type': 'newbusiness',
        'deal_stage': 'proposal'
    }
    
    # Build context
    builder = RAGContextBuilder()
    context = builder.build_context(sample_deals, current_deal)
    
    print("Generated RAG Context:")
    print("=" * 80)
    print(context)
    print("=" * 80)


if __name__ == "__main__":
    test_context_builder()