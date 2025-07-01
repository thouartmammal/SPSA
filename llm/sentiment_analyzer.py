import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from llm.llm_clients import LLMClient, create_llm_client
from rag.retriever import RAGRetriever, create_rag_retriever
from core.data_processor import DealDataProcessor
from core.embedding_service import get_embedding_service
from models.schemas import ProcessedActivity, DealPattern
from config.settings import settings

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Main sentiment analysis engine that orchestrates RAG retrieval,
    LLM analysis, and provides comprehensive sentiment insights
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
        self.data_processor = data_processor or DealDataProcessor(get_embedding_service())
        
        logger.info("Sentiment Analyzer initialized")
    
    def analyze_deal_sentiment(
        self,
        deal_data: Dict[str, Any],
        include_rag_context: bool = True,
        analysis_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis on a deal
        
        Args:
            deal_data: Deal data including activities and metadata
            include_rag_context: Whether to include RAG context
            analysis_options: Additional analysis options
            
        Returns:
            Comprehensive sentiment analysis result
        """
        
        analysis_options = analysis_options or {}
        
        try:
            # Process deal data
            logger.info(f"Analyzing sentiment for deal {deal_data.get('deal_id', 'unknown')}")
            
            # Parse and process activities
            processed_deal = self._prepare_deal_for_analysis(deal_data)
            
            # Get RAG context if requested
            rag_context = ""
            similar_deals = []
            contextual_insights = {}
            
            if include_rag_context:
                logger.debug("Retrieving RAG context...")
                similar_deals, rag_context = self.rag_retriever.retrieve_for_deal_analysis(
                    current_deal_text=processed_deal.combined_text,
                    current_deal_metadata=processed_deal.metadata,
                    analysis_type="sentiment"
                )
                
                # Get additional contextual insights
                contextual_insights = self.rag_retriever.get_contextual_insights(
                    deal_text=processed_deal.combined_text,
                    deal_metadata=processed_deal.metadata,
                    insight_type="comprehensive"
                )

            logger.info(f"RAG context retrieved: {len(rag_context)} characters")
            logger.debug(f"RAG context content: {rag_context}")  # First 500 chars
            
            # Perform LLM sentiment analysis
            logger.debug("Performing LLM sentiment analysis...")
            sentiment_result = self.llm_client.analyze_sentiment(
                deal_id=processed_deal.deal_id,
                activities_text=processed_deal.combined_text,
                rag_context=rag_context,
                activity_frequency=processed_deal.metadata.get('activities_count', 0),
                total_activities=processed_deal.activities_count,
                **self._prepare_llm_context(processed_deal, analysis_options)
            )
            
            # Enhance results with additional analysis
            enhanced_result = self._enhance_sentiment_result(
                sentiment_result=sentiment_result,
                processed_deal=processed_deal,
                similar_deals=similar_deals,
                contextual_insights=contextual_insights,
                analysis_options=analysis_options
            )
            
            logger.info(f"Completed sentiment analysis for deal {processed_deal.deal_id}")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._create_error_result(str(e), deal_data.get('deal_id', 'unknown'))
    
    def analyze_activities_sentiment(
        self,
        activities: List[Dict[str, Any]],
        deal_id: str,
        deal_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze sentiment for a list of activities
        
        Args:
            activities: List of activity dictionaries
            deal_id: Deal identifier
            deal_metadata: Additional deal metadata
            
        Returns:
            Sentiment analysis result
        """
        
        # Create deal data structure
        deal_data = {
            'deal_id': deal_id,
            'activities': activities,
            **(deal_metadata or {})
        }
        
        return self.analyze_deal_sentiment(deal_data)
    
    def batch_analyze_sentiment(
        self,
        deals_data: List[Dict[str, Any]],
        include_rag_context: bool = True,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for multiple deals in batches
        
        Args:
            deals_data: List of deal data dictionaries
            include_rag_context: Whether to include RAG context
            batch_size: Number of deals to process in each batch
            
        Returns:
            List of sentiment analysis results
        """
        
        results = []
        total_deals = len(deals_data)
        
        logger.info(f"Starting batch sentiment analysis for {total_deals} deals")
        
        for i in range(0, total_deals, batch_size):
            batch = deals_data[i:i + batch_size]
            batch_results = []
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_deals + batch_size - 1)//batch_size}")
            
            for deal_data in batch:
                try:
                    result = self.analyze_deal_sentiment(
                        deal_data=deal_data,
                        include_rag_context=include_rag_context
                    )
                    batch_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing deal {deal_data.get('deal_id', 'unknown')}: {e}")
                    batch_results.append(
                        self._create_error_result(str(e), deal_data.get('deal_id', 'unknown'))
                    )
            
            results.extend(batch_results)
        
        logger.info(f"Completed batch sentiment analysis for {len(results)} deals")
        return results
    
    def get_sentiment_summary(
        self,
        deal_data: Dict[str, Any],
        summary_type: str = "executive"
    ) -> Dict[str, Any]:
        """
        Get executive summary of sentiment analysis
        
        Args:
            deal_data: Deal data for analysis
            summary_type: Type of summary (executive, detailed, alerts)
            
        Returns:
            Sentiment summary
        """
        
        # Perform full analysis
        full_analysis = self.analyze_deal_sentiment(deal_data)
        
        # Extract summary based on type
        if summary_type == "executive":
            return self._create_executive_summary(full_analysis)
        elif summary_type == "detailed":
            return self._create_detailed_summary(full_analysis)
        elif summary_type == "alerts":
            return self._create_alerts_summary(full_analysis)
        else:
            return full_analysis
    
    def _prepare_deal_for_analysis(self, deal_data: Dict[str, Any]) -> DealPattern:
        """Prepare deal data for sentiment analysis"""
        
        # Use data processor to create structured deal pattern
        processed_deal = self.data_processor.process_deal(deal_data)
        
        return processed_deal
    
    def _prepare_llm_context(
        self,
        processed_deal: DealPattern,
        analysis_options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare additional context for LLM analysis"""
        
        context = {
            'deal_stage': processed_deal.metadata.get('deal_stage', ''),
            'deal_amount': processed_deal.metadata.get('deal_amount', 0),
            'deal_age_days': processed_deal.metadata.get('deal_age_days', 0),
            'communication_pattern': self._analyze_communication_pattern(processed_deal),
            'activity_distribution': self._analyze_activity_distribution(processed_deal),
            'timing_analysis': self._analyze_timing_patterns(processed_deal)
        }
        
        # Add analysis-specific context
        if analysis_options.get('focus_area'):
            context['focus_area'] = analysis_options['focus_area']
        
        if analysis_options.get('risk_assessment'):
            context['risk_factors'] = self._identify_immediate_risk_factors(processed_deal)
        
        return context
    
    def _analyze_communication_pattern(self, deal: DealPattern) -> Dict[str, Any]:
        """Analyze communication patterns in the deal"""
        
        metadata = deal.metadata
        
        return {
            'email_ratio': metadata.get('email_ratio', 0),
            'response_time_avg': metadata.get('response_time_metrics', {}).get('avg_response_time_hours', 0),
            'communication_gaps': metadata.get('communication_gaps_count', 0),
            'business_hours_ratio': metadata.get('business_hours_ratio', 0)
        }
    
    def _analyze_activity_distribution(self, deal: DealPattern) -> Dict[str, Any]:
        """Analyze distribution of activity types"""
        
        activity_types = deal.metadata.get('activity_types', {})
        total_activities = deal.activities_count
        
        if total_activities == 0:
            return {}
        
        return {
            'email_percentage': activity_types.get('email', 0) / total_activities,
            'call_percentage': activity_types.get('call', 0) / total_activities,
            'meeting_percentage': activity_types.get('meeting', 0) / total_activities,
            'note_percentage': activity_types.get('note', 0) / total_activities,
            'task_percentage': activity_types.get('task', 0) / total_activities
        }
    
    def _analyze_timing_patterns(self, deal: DealPattern) -> Dict[str, Any]:
        """Analyze timing patterns in activities"""
        
        metadata = deal.metadata
        
        return {
            'avg_time_between_activities': metadata.get('avg_time_between_activities_hours', 0),
            'activity_frequency_trend': metadata.get('activity_frequency_trend', 'unknown'),
            'weekend_activity_ratio': metadata.get('weekend_activity_ratio', 0),
            'time_span_days': deal.time_span_days
        }
    
    def _identify_immediate_risk_factors(self, deal: DealPattern) -> List[str]:
        """Identify immediate risk factors in the deal"""
        
        risk_factors = []
        metadata = deal.metadata
        
        # Communication gaps
        if metadata.get('communication_gaps_count', 0) > 1:
            risk_factors.append("Multiple communication gaps detected")
        
        # Slow response times
        avg_response = metadata.get('response_time_metrics', {}).get('avg_response_time_hours', 0)
        if avg_response > 24:
            risk_factors.append("Slow response times (>24 hours)")
        
        # Declining activity
        if metadata.get('activity_frequency_trend') == 'declining':
            risk_factors.append("Declining activity frequency")
        
        # Low engagement
        if metadata.get('email_ratio', 0) < 0.5:
            risk_factors.append("Low email engagement ratio")
        
        # Long deal duration without progress
        if metadata.get('deal_age_days', 0) > 90 and metadata.get('activities_count', 0) < 10:
            risk_factors.append("Long deal duration with low activity")
        
        return risk_factors
    
    def _enhance_sentiment_result(
        self,
        sentiment_result: Dict[str, Any],
        processed_deal: DealPattern,
        similar_deals: List,
        contextual_insights: Dict[str, Any],
        analysis_options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance sentiment result with additional analysis"""
        
        enhanced_result = sentiment_result.copy()
        
        # Add deal context
        enhanced_result['deal_context'] = {
            'deal_id': processed_deal.deal_id,
            'activities_count': processed_deal.activities_count,
            'time_span_days': processed_deal.time_span_days,
            'activity_types': processed_deal.activity_types,
            'deal_characteristics': {
                'amount': processed_deal.metadata.get('deal_amount', 0),
                'stage': processed_deal.metadata.get('deal_stage', ''),
                'type': processed_deal.metadata.get('deal_type', ''),
                'outcome': processed_deal.metadata.get('deal_outcome', 'open')
            }
        }
        
        # Add RAG insights
        enhanced_result['rag_insights'] = {
            'similar_deals_count': len(similar_deals),
            'contextual_analysis': contextual_insights.get('insights', {}),
            'recommendations': contextual_insights.get('recommendations', []),
            'historical_success_rate': self._calculate_historical_success_rate(similar_deals)
        }
        
        # Add risk assessment
        enhanced_result['risk_assessment'] = {
            'risk_level': self._calculate_risk_level(sentiment_result, contextual_insights),
            'risk_factors': contextual_insights.get('insights', {}).get('risk_factors', []),
            'mitigation_suggestions': self._generate_mitigation_suggestions(contextual_insights)
        }
        
        # Add performance metrics
        enhanced_result['performance_metrics'] = {
            'response_time_analysis': self._analyze_response_performance(processed_deal),
            'activity_frequency_analysis': self._analyze_activity_frequency(processed_deal),
            'communication_effectiveness': self._analyze_communication_effectiveness(processed_deal)
        }
        
        # Add actionable insights
        enhanced_result['actionable_insights'] = self._generate_actionable_insights(
            sentiment_result, processed_deal, contextual_insights
        )
        
        return enhanced_result
    
    def _calculate_historical_success_rate(self, similar_deals: List) -> float:
        """Calculate success rate from similar historical deals"""
        
        if not similar_deals:
            return 0.0
        
        won_deals = sum(1 for deal in similar_deals 
                       if deal.metadata.get('deal_outcome') == 'won')
        
        return won_deals / len(similar_deals)
    
    def _calculate_risk_level(
        self,
        sentiment_result: Dict[str, Any],
        contextual_insights: Dict[str, Any]
    ) -> str:
        """Calculate overall risk level"""
        
        sentiment_score = sentiment_result.get('sentiment_score', 0)
        risk_factors = contextual_insights.get('insights', {}).get('risk_factors', [])
        
        # Risk calculation based on sentiment and risk factors
        if sentiment_score < -0.5 or len(risk_factors) >= 3:
            return 'high'
        elif sentiment_score < 0 or len(risk_factors) >= 2:
            return 'medium'
        elif len(risk_factors) >= 1:
            return 'low'
        else:
            return 'minimal'
    
    def _generate_mitigation_suggestions(self, contextual_insights: Dict[str, Any]) -> List[str]:
        """Generate suggestions to mitigate identified risks"""
        
        suggestions = []
        risk_factors = contextual_insights.get('insights', {}).get('risk_factors', [])
        
        for risk in risk_factors:
            if 'response time' in risk.lower():
                suggestions.append("Set up automated response systems and response time targets")
            elif 'communication gap' in risk.lower():
                suggestions.append("Implement regular check-in schedule and communication cadence")
            elif 'declining activity' in risk.lower():
                suggestions.append("Schedule immediate client touchpoint and increase engagement")
        
        return suggestions
    
    def _analyze_response_performance(self, deal: DealPattern) -> Dict[str, Any]:
        """Analyze response time performance"""
        
        metrics = deal.metadata.get('response_time_metrics', {})
        avg_response = metrics.get('avg_response_time_hours', 0)
        
        # Performance categories
        if avg_response <= 4:
            performance = 'excellent'
        elif avg_response <= 12:
            performance = 'good'
        elif avg_response <= 24:
            performance = 'fair'
        else:
            performance = 'poor'
        
        return {
            'average_response_time': avg_response,
            'performance_rating': performance,
            'response_count': metrics.get('response_count', 0)
        }
    
    def _analyze_activity_frequency(self, deal: DealPattern) -> Dict[str, Any]:
        """Analyze activity frequency patterns"""
        
        activities_per_day = deal.activities_count / max(deal.time_span_days, 1)
        trend = deal.metadata.get('activity_frequency_trend', 'unknown')
        
        return {
            'activities_per_day': activities_per_day,
            'frequency_trend': trend,
            'total_activities': deal.activities_count,
            'time_span_days': deal.time_span_days
        }
    
    def _analyze_communication_effectiveness(self, deal: DealPattern) -> Dict[str, Any]:
        """Analyze effectiveness of communication"""
        
        metadata = deal.metadata
        
        return {
            'email_ratio': metadata.get('email_ratio', 0),
            'business_hours_ratio': metadata.get('business_hours_ratio', 0),
            'communication_gaps': metadata.get('communication_gaps_count', 0),
            'effectiveness_score': self._calculate_communication_score(metadata)
        }
    
    def _calculate_communication_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate communication effectiveness score (0-1)"""
        
        score = 1.0
        
        # Penalize for communication gaps
        gaps = metadata.get('communication_gaps_count', 0)
        score -= (gaps * 0.2)
        
        # Penalize for slow response times
        avg_response = metadata.get('response_time_metrics', {}).get('avg_response_time_hours', 0)
        if avg_response > 24:
            score -= 0.3
        elif avg_response > 12:
            score -= 0.2
        
        # Penalize for low business hours activity
        bh_ratio = metadata.get('business_hours_ratio', 0)
        if bh_ratio < 0.5:
            score -= 0.2
        
        return max(0.0, score)
    
    def _generate_actionable_insights(
        self,
        sentiment_result: Dict[str, Any],
        processed_deal: DealPattern,
        contextual_insights: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate actionable insights based on analysis"""
        
        insights = []
        
        # Sentiment-based insights
        sentiment_score = sentiment_result.get('sentiment_score', 0)
        if sentiment_score < -0.3:
            insights.append({
                'type': 'urgent_action',
                'insight': 'Negative sentiment detected',
                'action': 'Schedule immediate client check-in call'
            })
        elif sentiment_score > 0.5:
            insights.append({
                'type': 'opportunity',
                'insight': 'Positive sentiment momentum',
                'action': 'Consider accelerating to close or next stage'
            })
        
        # Performance-based insights
        avg_response = processed_deal.metadata.get('response_time_metrics', {}).get('avg_response_time_hours', 0)
        if avg_response > 12:
            insights.append({
                'type': 'improvement',
                'insight': 'Response times need improvement',
                'action': 'Set up response time alerts and targets'
            })
        
        # Historical comparison insights
        success_rate = contextual_insights.get('insights', {}).get('success_rate', 0)
        if success_rate > 0.8:
            insights.append({
                'type': 'confidence',
                'insight': 'Deal matches high-success patterns',
                'action': 'Maintain current approach and prepare for close'
            })
        elif success_rate < 0.3:
            insights.append({
                'type': 'concern',
                'insight': 'Deal matches low-success patterns',
                'action': 'Review strategy and consider different approach'
            })
        
        return insights
    
    def _create_executive_summary(self, full_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary of sentiment analysis"""
        
        return {
            'deal_id': full_analysis.get('deal_context', {}).get('deal_id'),
            'overall_sentiment': full_analysis.get('overall_sentiment'),
            'sentiment_score': full_analysis.get('sentiment_score'),
            'confidence': full_analysis.get('confidence'),
            'risk_level': full_analysis.get('risk_assessment', {}).get('risk_level'),
            'key_insights': full_analysis.get('actionable_insights', [])[:3],
            'recommendations': full_analysis.get('rag_insights', {}).get('recommendations', [])[:3],
            'historical_success_rate': full_analysis.get('rag_insights', {}).get('historical_success_rate', 0)
        }
    
    def _create_detailed_summary(self, full_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed summary of sentiment analysis"""
        
        return {
            'executive_summary': self._create_executive_summary(full_analysis),
            'sentiment_breakdown': full_analysis.get('activity_breakdown', {}),
            'performance_metrics': full_analysis.get('performance_metrics', {}),
            'risk_assessment': full_analysis.get('risk_assessment', {}),
            'historical_context': full_analysis.get('rag_insights', {}),
            'full_reasoning': full_analysis.get('reasoning', '')
        }
    
    def _create_alerts_summary(self, full_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create alerts-focused summary"""
        
        alerts = []
        
        # Risk-based alerts
        risk_level = full_analysis.get('risk_assessment', {}).get('risk_level', 'minimal')
        if risk_level in ['high', 'medium']:
            alerts.append({
                'type': 'risk',
                'level': risk_level,
                'message': f"Deal has {risk_level} risk level",
                'factors': full_analysis.get('risk_assessment', {}).get('risk_factors', [])
            })
        
        # Sentiment-based alerts
        sentiment_score = full_analysis.get('sentiment_score', 0)
        if sentiment_score < -0.5:
            alerts.append({
                'type': 'sentiment',
                'level': 'critical',
                'message': 'Negative sentiment detected',
                'score': sentiment_score
            })
        
        return {
            'deal_id': full_analysis.get('deal_context', {}).get('deal_id'),
            'alerts': alerts,
            'urgent_actions': [
                insight for insight in full_analysis.get('actionable_insights', [])
                if insight.get('type') in ['urgent_action', 'concern']
            ]
        }
    
    def _create_error_result(self, error_message: str, deal_id: str) -> Dict[str, Any]:
        """Create error result structure"""
        
        return {
            'deal_id': deal_id,
            'error': True,
            'error_message': error_message,
            'timestamp': datetime.utcnow().isoformat(),
            'overall_sentiment': 'unknown',
            'sentiment_score': 0.0,
            'confidence': 0.0
        }


# Factory functions
def create_sentiment_analyzer(
    provider_name: str = "openai",
    provider_config: Dict[str, Any] = None
) -> SentimentAnalyzer:
    """Create sentiment analyzer with specified LLM provider"""
    
    provider_config = provider_config or {}
    
    # Create LLM client
    llm_client = create_llm_client(provider_name, **provider_config)
    
    # Create RAG retriever
    rag_retriever = create_rag_retriever()
    
    return SentimentAnalyzer(
        llm_client=llm_client,
        rag_retriever=rag_retriever
    )


# Example usage and testing
def test_sentiment_analyzer():
    """Test the sentiment analyzer with sample data"""
    
    # Sample deal data
    sample_deal = {
        'deal_id': 'test_deal_001',
        'amount': 75000,
        'dealstage': 'proposal',
        'dealtype': 'newbusiness',
        'deal_stage_probability': 60,
        'createdate': '2024-01-01T00:00:00Z',
        'activities': [
            {
                'activity_type': 'email',
                'sent_at': '2024-01-02T09:00:00Z',
                'subject': 'Follow up on our proposal discussion',
                'body': 'Hi John, wanted to follow up on our conversation about the proposal.',
                'direction': 'outgoing'
            },
            {
                'activity_type': 'call',
                'createdate': '2024-01-03T14:00:00Z',
                'call_title': 'Proposal review call',
                'call_body': 'Discussed proposal details and timeline with client.',
                'call_duration': 30
            },
            {
                'activity_type': 'email',
                'sent_at': '2024-01-04T16:00:00Z',
                'subject': 'Re: Proposal timeline',
                'body': 'Thanks for the call. Looking forward to moving forward.',
                'direction': 'incoming'
            }
        ]
    }
    
    try:
        print("Testing Sentiment Analyzer...")
        
        # Create analyzer
        analyzer = create_sentiment_analyzer(
            provider_name="groq",
            provider_config={
                'api_key': '',  # Replace with actual key
                'model': 'llama-3.3-70b-versatile'
            }
        )
        
        # Analyze sentiment
        result = analyzer.analyze_deal_sentiment(sample_deal)
        
        print(f"\nSentiment Analysis Result:")
        print(f"Deal ID: {result.get('deal_context', {}).get('deal_id')}")
        print(f"Overall Sentiment: {result.get('overall_sentiment')}")
        print(f"Sentiment Score: {result.get('sentiment_score')}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"Risk Level: {result.get('risk_assessment', {}).get('risk_level')}")
        
        # Get executive summary
        summary = analyzer.get_sentiment_summary(sample_deal, summary_type="executive")
        print(f"\nExecutive Summary:")
        print(f"Key Insights: {len(summary.get('key_insights', []))}")
        print(f"Recommendations: {len(summary.get('recommendations', []))}")
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":

    test_sentiment_analyzer()