import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings

logger = logging.getLogger(__name__)

class PromptBuilder:
    """
    Advanced prompt builder for sales sentiment analysis focused on salesperson behavior patterns.
    Builds contextual prompts that analyze salesperson performance against professional standards
    using historical deal patterns and adaptive context prioritization.
    """
    
    def __init__(self, template_dir: str = "prompts/"):
        """
        Initialize prompt builder for salesperson sentiment analysis
        
        Args:
            template_dir: Directory containing prompt templates
        """
        self.template_dir = template_dir
        self.templates = {}
        self._load_templates()
        
        logger.info("Sales Sentiment Prompt Builder initialized")
    
    def _load_templates(self):
        """Load prompt templates from files"""
        
        template_files = {
            'salesperson_sentiment': 'salesperson_sentiment_prompt.txt',
            'deal_risk_assessment': 'deal_risk_assessment_prompt.txt',
            'coaching_recommendations': 'coaching_recommendations_prompt.txt',
            'performance_benchmarking': 'performance_benchmarking_prompt.txt',
            'deal_momentum_analysis': 'deal_momentum_analysis_prompt.txt'
        }
        
        for template_name, filename in template_files.items():
            template_path = Path(self.template_dir) / filename
            
            if template_path.exists():
                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        self.templates[template_name] = f.read()
                    logger.debug(f"Loaded template: {template_name}")
                except Exception as e:
                    logger.warning(f"Error loading template {template_name}: {e}")
                    self.templates[template_name] = self._get_default_template(template_name)
            else:
                logger.warning(f"Template file not found: {template_path}")
                self.templates[template_name] = self._get_default_template(template_name)
    
    def build_salesperson_sentiment_prompt(
        self,
        deal_id: str,
        activities_text: str,
        rag_context: str = "",
        deal_metadata: Dict[str, Any] = None,
        activity_frequency: int = 0,
        total_activities: int = 0,
        analysis_focus: str = "comprehensive",
        **kwargs
    ) -> str:
        """
        Build prompt for salesperson sentiment analysis with adaptive context prioritization
        
        Args:
            deal_id: Deal identifier
            activities_text: Combined activities text (already prioritized by importance)
            rag_context: Historical context from similar deals
            deal_metadata: Deal metadata with calculated metrics
            activity_frequency: Recent activity frequency
            total_activities: Total number of activities
            analysis_focus: Focus area for analysis
            **kwargs: Additional context variables
            
        Returns:
            Formatted prompt for salesperson sentiment analysis
        """
        
        deal_metadata = deal_metadata or {}
        
        # Build comprehensive context for salesperson analysis
        context_vars = {
            'deal_id': deal_id,
            'activities_text': activities_text,
            'rag_context': rag_context,
            'activity_frequency': activity_frequency,
            'total_activities': total_activities,
            'current_date': datetime.now().strftime("%Y-%m-%d"),
            'analysis_focus': analysis_focus,
            
            # Deal characteristics
            **self._prepare_deal_context(deal_metadata),
            
            # Performance metrics for salesperson evaluation
            **self._prepare_performance_context(deal_metadata),
            
            # Historical benchmarking context
            **self._prepare_benchmarking_context(rag_context, deal_metadata),
            
            # Analysis instructions based on focus
            'focus_instructions': self._get_salesperson_focus_instructions(analysis_focus),
            'evaluation_standards': self._get_salesperson_evaluation_standards(),
            'response_format': self._get_salesperson_response_format(),
            
            **kwargs
        }
        
        # Use the main salesperson sentiment template
        template = self.templates.get('salesperson_sentiment', self._get_default_template('salesperson_sentiment'))
        
        try:
            prompt = template.format(**context_vars)
            logger.debug(f"Built salesperson sentiment prompt for deal {deal_id}")
            return prompt
            
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            # Fall back to basic template
            return self._build_basic_salesperson_prompt(deal_id, activities_text, rag_context, deal_metadata)
    
    def build_deal_risk_assessment_prompt(
        self,
        deal_id: str,
        activities_text: str,
        rag_context: str = "",
        deal_metadata: Dict[str, Any] = None,
        identified_risks: List[str] = None,
        **kwargs
    ) -> str:
        """
        Build prompt for deal risk assessment focusing on salesperson behavior risks
        
        Args:
            deal_id: Deal identifier
            activities_text: Combined activities text
            rag_context: Historical context from similar deals
            deal_metadata: Deal metadata
            identified_risks: Pre-identified risk factors
            **kwargs: Additional context
            
        Returns:
            Risk assessment prompt
        """
        
        deal_metadata = deal_metadata or {}
        identified_risks = identified_risks or []
        
        context_vars = {
            'deal_id': deal_id,
            'activities_text': activities_text,
            'rag_context': rag_context,
            'current_date': datetime.now().strftime("%Y-%m-%d"),
            
            # Deal and performance context
            **self._prepare_deal_context(deal_metadata),
            **self._prepare_performance_context(deal_metadata),
            
            # Risk-specific context
            'identified_risks': self._format_risk_factors(identified_risks),
            'risk_indicators': self._identify_behavioral_risk_indicators(deal_metadata),
            'historical_risk_patterns': self._extract_risk_patterns_from_rag(rag_context),
            
            **kwargs
        }
        
        template = self.templates.get('deal_risk_assessment', self._get_default_template('deal_risk_assessment'))
        
        try:
            prompt = template.format(**context_vars)
            logger.debug(f"Built risk assessment prompt for deal {deal_id}")
            return prompt
            
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            return self._build_basic_risk_prompt(deal_id, activities_text, rag_context, deal_metadata)
    
    def build_coaching_prompt(
        self,
        deal_id: str,
        activities_text: str,
        rag_context: str = "",
        deal_metadata: Dict[str, Any] = None,
        performance_gaps: List[str] = None,
        coaching_focus: str = "comprehensive",
        **kwargs
    ) -> str:
        """
        Build prompt for salesperson coaching recommendations
        
        Args:
            deal_id: Deal identifier
            activities_text: Combined activities text
            rag_context: Historical context with success patterns
            deal_metadata: Deal metadata
            performance_gaps: Identified performance issues
            coaching_focus: Focus area for coaching
            **kwargs: Additional context
            
        Returns:
            Coaching recommendations prompt
        """
        
        deal_metadata = deal_metadata or {}
        performance_gaps = performance_gaps or []
        
        context_vars = {
            'deal_id': deal_id,
            'activities_text': activities_text,
            'rag_context': rag_context,
            'current_date': datetime.now().strftime("%Y-%m-%d"),
            'coaching_focus': coaching_focus,
            
            # Performance analysis context
            **self._prepare_deal_context(deal_metadata),
            **self._prepare_performance_context(deal_metadata),
            
            # Coaching-specific context
            'performance_gaps': self._format_performance_gaps(performance_gaps),
            'success_patterns': self._extract_success_patterns_from_rag(rag_context),
            'improvement_opportunities': self._identify_improvement_opportunities(deal_metadata),
            'coaching_recommendations_format': self._get_coaching_response_format(),
            
            **kwargs
        }
        
        template = self.templates.get('coaching_recommendations', self._get_default_template('coaching_recommendations'))
        
        try:
            prompt = template.format(**context_vars)
            logger.debug(f"Built coaching prompt for deal {deal_id}")
            return prompt
            
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            return self._build_basic_coaching_prompt(deal_id, activities_text, rag_context, deal_metadata)
    
    def build_performance_benchmarking_prompt(
        self,
        deal_id: str,
        activities_text: str,
        rag_context: str = "",
        deal_metadata: Dict[str, Any] = None,
        benchmark_focus: str = "overall",
        **kwargs
    ) -> str:
        """
        Build prompt for performance benchmarking against successful deals
        
        Args:
            deal_id: Deal identifier
            activities_text: Combined activities text
            rag_context: Historical benchmarking data
            deal_metadata: Deal metadata
            benchmark_focus: Focus area for benchmarking
            **kwargs: Additional context
            
        Returns:
            Performance benchmarking prompt
        """
        
        deal_metadata = deal_metadata or {}
        
        context_vars = {
            'deal_id': deal_id,
            'activities_text': activities_text,
            'rag_context': rag_context,
            'current_date': datetime.now().strftime("%Y-%m-%d"),
            'benchmark_focus': benchmark_focus,
            
            # Performance context
            **self._prepare_deal_context(deal_metadata),
            **self._prepare_performance_context(deal_metadata),
            
            # Benchmarking context
            'performance_metrics': self._format_performance_metrics(deal_metadata),
            'benchmark_standards': self._extract_benchmark_standards_from_rag(rag_context),
            'peer_comparison': self._prepare_peer_comparison_context(rag_context),
            
            **kwargs
        }
        
        template = self.templates.get('performance_benchmarking', self._get_default_template('performance_benchmarking'))
        
        try:
            prompt = template.format(**context_vars)
            logger.debug(f"Built benchmarking prompt for deal {deal_id}")
            return prompt
            
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            return self._build_basic_benchmarking_prompt(deal_id, activities_text, rag_context, deal_metadata)
    
    def _prepare_deal_context(self, deal_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare deal context for salesperson analysis"""
        
        context = {}
        
        # Basic deal information
        context['deal_amount'] = self._format_currency(deal_metadata.get('deal_amount', 0))
        context['deal_stage'] = deal_metadata.get('deal_stage', 'Unknown')
        context['deal_type'] = deal_metadata.get('deal_type', 'Unknown')
        context['deal_probability'] = deal_metadata.get('deal_probability', 0)
        context['deal_age_days'] = deal_metadata.get('deal_age_days', 0)
        context['deal_outcome'] = deal_metadata.get('deal_outcome', 'open')
        context['deal_size_category'] = deal_metadata.get('deal_size_category', 'unknown')
        
        # Deal lifecycle context
        context['is_new_business'] = deal_metadata.get('is_new_business', False)
        context['is_closed'] = deal_metadata.get('is_closed', False)
        context['is_won'] = deal_metadata.get('is_won', False)
        context['is_lost'] = deal_metadata.get('is_lost', False)
        
        return context
    
    def _prepare_performance_context(self, deal_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare salesperson performance context"""
        
        context = {}
        
        # Activity metrics
        context['activities_count'] = deal_metadata.get('activities_count', 0)
        context['time_span_days'] = deal_metadata.get('time_span_days', 0)
        
        # Communication performance metrics
        response_metrics = deal_metadata.get('response_time_metrics', {})
        context['avg_response_time_hours'] = response_metrics.get('avg_response_time_hours', 0)
        context['fastest_response_hours'] = response_metrics.get('fastest_response_hours', 0)
        context['slowest_response_hours'] = response_metrics.get('slowest_response_hours', 0)
        context['response_count'] = response_metrics.get('response_count', 0)
        
        # Communication gaps and patterns
        context['communication_gaps_count'] = deal_metadata.get('communication_gaps_count', 0)
        context['business_hours_ratio'] = deal_metadata.get('business_hours_ratio', 0)
        context['weekend_activity_ratio'] = deal_metadata.get('weekend_activity_ratio', 0)
        
        # Activity breakdown
        activity_types = deal_metadata.get('activity_types', {})
        context['email_count'] = activity_types.get('email', 0)
        context['call_count'] = activity_types.get('call', 0)
        context['meeting_count'] = activity_types.get('meeting', 0)
        context['note_count'] = activity_types.get('note', 0)
        context['task_count'] = activity_types.get('task', 0)
        
        # Engagement metrics
        context['email_ratio'] = deal_metadata.get('email_ratio', 0)
        context['outgoing_emails'] = deal_metadata.get('outgoing_emails', 0)
        context['incoming_emails'] = deal_metadata.get('incoming_emails', 0)
        
        # Trend analysis
        context['activity_frequency_trend'] = deal_metadata.get('activity_frequency_trend', 'unknown')
        context['avg_time_between_activities_hours'] = deal_metadata.get('avg_time_between_activities_hours', 0)
        
        # Performance ratings
        context['response_performance'] = self._categorize_response_performance(context['avg_response_time_hours'])
        context['activity_frequency'] = self._categorize_activity_frequency(deal_metadata)
        context['engagement_level'] = self._assess_engagement_level(deal_metadata)
        context['communication_quality'] = self._assess_communication_quality(deal_metadata)
        
        return context
    
    def _prepare_benchmarking_context(self, rag_context: str, deal_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare benchmarking context from RAG data"""
        
        context = {}
        
        # Extract benchmark data from RAG context if available
        if rag_context:
            context['has_historical_benchmarks'] = True
            context['benchmark_source'] = "similar_historical_deals"
        else:
            context['has_historical_benchmarks'] = False
            context['benchmark_source'] = "general_standards"
        
        return context
    
    def _get_salesperson_focus_instructions(self, analysis_focus: str) -> str:
        """Get analysis instructions focused on salesperson behavior"""
        
        focus_instructions = {
            'comprehensive': """Provide comprehensive analysis of salesperson sentiment, behavior patterns, and performance against professional sales standards. Focus on communication effectiveness, deal progression activities, client engagement quality, and response patterns.""",
            
            'performance_gaps': """Focus on identifying specific performance gaps in the salesperson's approach. Analyze response times, communication consistency, proactive vs reactive behaviors, and missed opportunities for deal advancement.""",
            
            'coaching_opportunities': """Identify specific coaching opportunities and areas for improvement. Focus on actionable feedback about communication style, activity patterns, client relationship management, and deal progression strategies.""",
            
            'risk_assessment': """Focus on identifying behavioral risks that could jeopardize the deal. Analyze warning signs in communication patterns, activity frequency trends, client engagement levels, and competitive positioning.""",
            
            'deal_momentum': """Analyze deal momentum indicators from salesperson behavior. Focus on activity acceleration/deceleration, client response patterns, stage progression activities, and momentum-building actions.""",
            
            'benchmarking': """Compare salesperson performance against successful deal patterns and industry benchmarks. Focus on relative performance metrics, best practice alignment, and competitive positioning."""
        }
        
        return focus_instructions.get(analysis_focus, focus_instructions['comprehensive'])
    
    def _get_salesperson_evaluation_standards(self) -> str:
        """Get evaluation standards specific to salesperson performance"""
        
        return """
SALESPERSON EVALUATION STANDARDS:

**BASELINE PROFESSIONAL BEHAVIOR (Neutral: 0.0 to +0.3)**
- Standard follow-up emails and calls
- Basic CRM documentation and task completion
- Routine meeting scheduling and preparation
- Polite, professional communication tone
- Standard product presentations and demos
- Regular check-ins without clear advancement
- Administrative compliance

**POSITIVE PERFORMANCE INDICATORS (Required for +0.4 or higher)**
- Deal Progression Evidence: Clear advancement with documented client commitment
- Proactive Value Creation: Initiating solutions before client asks
- Strategic Relationship Building: Multi-stakeholder engagement and influence
- Competitive Intelligence: Gathering and acting on market insights
- Revenue Impact: Activities directly tied to revenue growth
- Client Success Orientation: Focus on client outcomes over product features
- Pattern-Breaking Excellence: Exceptional activities that stand out

**NEGATIVE PERFORMANCE INDICATORS**
- Reactive-Only Patterns: Only responding, never initiating strategic conversations
- Activity Without Outcomes: High volume with no measurable progress
- Client Avoidance Signals: Delayed responses, missed opportunities
- Proposal/Quote Delays: Slow turnaround on client requests
- Revenue Leakage: Deals shrinking or stagnating
- Communication Gaps: Extended periods without client contact
"""
    
    def _get_salesperson_response_format(self) -> str:
        """Get response format for salesperson sentiment analysis"""
        
        return """
Response must be valid JSON with this structure:
{
    "overall_sentiment": "exceptional_positive|positive|neutral|negative|critical_negative",
    "sentiment_score": 0.65,  // Float between -1.0 and 1.0
    "confidence": 0.85,       // Float between 0.0 and 1.0
    "activity_breakdown": {
        "email": {
            "sentiment": "positive|neutral|negative",
            "sentiment_score": 0.5,
            "key_indicators": ["Strategic value proposition", "Multi-stakeholder engagement"],
            "count": 3,
            "performance_rating": "excellent|good|fair|poor"
        },
        "call": { /* same structure */ },
        "meeting": { /* same structure */ },
        "note": { /* same structure */ },
        "task": { /* same structure */ }
    },
    "deal_momentum_indicators": {
        "stage_progression": "advancing|stagnant|regressing",
        "client_engagement_trend": "increasing|stable|decreasing", 
        "competitive_position": "strengthening|maintaining|weakening",
        "activity_velocity": "accelerating|stable|declining"
    },
    "performance_analysis": {
        "response_time_rating": "excellent|good|fair|poor",
        "communication_consistency": "excellent|good|fair|poor",
        "proactive_behavior": "high|medium|low",
        "client_relationship_quality": "strong|moderate|weak"
    },
    "reasoning": "Detailed explanation focusing on salesperson behavior analysis",
    "professional_gaps": ["Specific areas below professional standards"],
    "excellence_indicators": ["Areas of exceptional performance"],
    "risk_indicators": ["Behavioral patterns that could jeopardize the deal"],
    "coaching_opportunities": ["Specific improvement recommendations"],
    "temporal_trend": "accelerating|improving|stable|declining|deteriorating",
    "recommended_actions": ["Immediate actions to improve performance"],
    "benchmark_comparison": "above_average|average|below_average|insufficient_data"
}
"""
    
    def _get_coaching_response_format(self) -> str:
        """Get response format for coaching recommendations"""
        
        return """
Response must focus on actionable coaching recommendations:
{
    "coaching_priority": "high|medium|low",
    "primary_focus_areas": ["communication", "activity_management", "client_relationship"],
    "specific_recommendations": [
        {
            "area": "response_time",
            "current_performance": "fair", 
            "target_improvement": "Reduce average response time from 18h to <4h",
            "action_steps": ["Set up email alerts", "Block calendar time for responses"],
            "success_metrics": ["Response time <4h", "Client acknowledgment within 2h"]
        }
    ],
    "success_patterns_to_emulate": ["Pattern from successful similar deals"],
    "warning_signs_to_avoid": ["Pattern from failed similar deals"],
    "timeline_for_improvement": "immediate|short_term|medium_term",
    "coaching_session_topics": ["Specific topics for 1:1 coaching"]
}
"""
    
    def _categorize_response_performance(self, avg_response_hours: float) -> str:
        """Categorize response time performance"""
        if avg_response_hours <= 4:
            return 'excellent'
        elif avg_response_hours <= 12:
            return 'good' 
        elif avg_response_hours <= 24:
            return 'fair'
        else:
            return 'poor'
    
    def _categorize_activity_frequency(self, deal_metadata: Dict[str, Any]) -> str:
        """Categorize activity frequency"""
        activities_count = deal_metadata.get('activities_count', 0)
        time_span_days = deal_metadata.get('time_span_days', 1)
        
        if time_span_days == 0:
            return 'single_day'
        
        frequency = activities_count / time_span_days
        
        if frequency >= 1.0:
            return 'very_high'
        elif frequency >= 0.5:
            return 'high'
        elif frequency >= 0.25:
            return 'moderate'
        elif frequency >= 0.1:
            return 'low'
        else:
            return 'very_low'
    
    def _assess_engagement_level(self, deal_metadata: Dict[str, Any]) -> str:
        """Assess salesperson engagement level"""
        
        email_ratio = deal_metadata.get('email_ratio', 0)
        business_hours_ratio = deal_metadata.get('business_hours_ratio', 0)
        activities_count = deal_metadata.get('activities_count', 0)
        
        engagement_score = 0
        
        # Proactive email behavior
        if email_ratio > 1.0:
            engagement_score += 2
        elif email_ratio > 0.5:
            engagement_score += 1
        
        # Professional timing
        if business_hours_ratio > 0.8:
            engagement_score += 2
        elif business_hours_ratio > 0.6:
            engagement_score += 1
        
        # Activity volume
        if activities_count > 15:
            engagement_score += 2
        elif activities_count > 8:
            engagement_score += 1
        
        if engagement_score >= 5:
            return 'high'
        elif engagement_score >= 3:
            return 'moderate'
        else:
            return 'low'
    
    def _assess_communication_quality(self, deal_metadata: Dict[str, Any]) -> str:
        """Assess communication quality"""
        
        avg_response_time = deal_metadata.get('response_time_metrics', {}).get('avg_response_time_hours', 0)
        communication_gaps = deal_metadata.get('communication_gaps_count', 0)
        business_hours_ratio = deal_metadata.get('business_hours_ratio', 0)
        
        quality_score = 5  # Start with perfect score
        
        # Response time impact
        if avg_response_time > 24:
            quality_score -= 2
        elif avg_response_time > 12:
            quality_score -= 1
        
        # Communication gaps impact
        quality_score -= min(communication_gaps, 3)
        
        # Professional timing impact
        if business_hours_ratio < 0.5:
            quality_score -= 1
        
        if quality_score >= 4:
            return 'excellent'
        elif quality_score >= 3:
            return 'good'
        elif quality_score >= 2:
            return 'fair'
        else:
            return 'poor'
    
    def _format_currency(self, amount: float) -> str:
        """Format currency amount"""
        if amount == 0:
            return "$0"
        return f"${amount:,.0f}"
    
    def _format_risk_factors(self, risk_factors: List[str]) -> str:
        """Format risk factors for prompt"""
        if not risk_factors:
            return "No specific risk factors identified from previous analysis."
        
        return "Previously Identified Risk Factors:\n" + "\n".join(f"- {factor}" for factor in risk_factors)
    
    def _format_performance_gaps(self, performance_gaps: List[str]) -> str:
        """Format performance gaps for coaching prompt"""
        if not performance_gaps:
            return "No specific performance gaps identified from previous analysis."
        
        return "Identified Performance Gaps:\n" + "\n".join(f"- {gap}" for gap in performance_gaps)
    
    def _format_performance_metrics(self, deal_metadata: Dict[str, Any]) -> str:
        """Format performance metrics for benchmarking"""
        
        metrics = []
        
        # Response time metrics
        avg_response = deal_metadata.get('response_time_metrics', {}).get('avg_response_time_hours', 0)
        metrics.append(f"Average Response Time: {avg_response:.1f} hours")
        
        # Activity metrics
        activities_count = deal_metadata.get('activities_count', 0)
        time_span = deal_metadata.get('time_span_days', 0)
        metrics.append(f"Activity Count: {activities_count} over {time_span} days")
        
        # Communication metrics
        gaps = deal_metadata.get('communication_gaps_count', 0)
        metrics.append(f"Communication Gaps: {gaps}")
        
        bh_ratio = deal_metadata.get('business_hours_ratio', 0)
        metrics.append(f"Business Hours Activity: {bh_ratio:.0%}")
        
        return "Current Performance Metrics:\n" + "\n".join(f"- {metric}" for metric in metrics)
    
    def _identify_behavioral_risk_indicators(self, deal_metadata: Dict[str, Any]) -> str:
        """Identify behavioral risk indicators"""
        
        risks = []
        
        # Response time risks
        avg_response = deal_metadata.get('response_time_metrics', {}).get('avg_response_time_hours', 0)
        if avg_response > 24:
            risks.append("Slow response times (>24 hours)")
        
        # Communication gap risks
        gaps = deal_metadata.get('communication_gaps_count', 0)
        if gaps > 1:
            risks.append(f"Multiple communication gaps ({gaps} gaps)")
        
        # Activity trend risks
        trend = deal_metadata.get('activity_frequency_trend', '')
        if trend == 'declining':
            risks.append("Declining activity frequency trend")
        
        # Engagement risks
        email_ratio = deal_metadata.get('email_ratio', 0)
        if email_ratio < 0.5:
            risks.append("Low proactive email engagement")
        
        if not risks:
            return "No significant behavioral risk indicators detected."
        
        return "Behavioral Risk Indicators:\n" + "\n".join(f"- {risk}" for risk in risks)
    
    def _identify_improvement_opportunities(self, deal_metadata: Dict[str, Any]) -> str:
        """Identify improvement opportunities"""
        
        opportunities = []
        
        # Response time opportunities
        avg_response = deal_metadata.get('response_time_metrics', {}).get('avg_response_time_hours', 0)
        if avg_response > 4:
            opportunities.append("Improve response time to <4 hours for better client experience")
        
        # Activity frequency opportunities
        activities_count = deal_metadata.get('activities_count', 0)
        time_span = deal_metadata.get('time_span_days', 1)
        if activities_count / time_span < 0.3:
            opportunities.append("Increase activity frequency to maintain deal momentum")
        
        # Communication consistency opportunities
        gaps = deal_metadata.get('communication_gaps_count', 0)
        if gaps > 0:
            opportunities.append("Maintain more consistent communication cadence")
        
        # Professional timing opportunities
        bh_ratio = deal_metadata.get('business_hours_ratio', 0)
        if bh_ratio < 0.7:
            opportunities.append("Focus more activities during business hours")
        
        if not opportunities:
            return "Performance appears strong with limited improvement opportunities identified."
        
        return "Improvement Opportunities:\n" + "\n".join(f"- {opp}" for opp in opportunities)
    
    def _extract_risk_patterns_from_rag(self, rag_context: str) -> str:
        """Extract risk patterns from RAG context"""
        if not rag_context:
            return "No historical risk pattern data available."
        
        # Extract relevant risk information from RAG context
        if "FAILED DEAL PATTERNS" in rag_context:
            return "Historical risk patterns available in RAG context for comparison."
        else:
            return "Limited historical risk pattern data in context."
    
    def _extract_success_patterns_from_rag(self, rag_context: str) -> str:
        """Extract success patterns from RAG context"""
        if not rag_context:
            return "No historical success pattern data available."
        
        # Extract relevant success information from RAG context
        if "SUCCESSFUL DEAL PATTERNS" in rag_context:
            return "Historical success patterns available in RAG context for coaching reference."
        else:
            return "Limited historical success pattern data in context."
    
    def _extract_benchmark_standards_from_rag(self, rag_context: str) -> str:
        """Extract benchmark standards from RAG context"""
        if not rag_context:
            return "No historical benchmark data available."
        
        # Extract benchmark information from RAG context
        if "PERFORMANCE BENCHMARKS" in rag_context:
            return "Historical performance benchmarks available for comparison."
        else:
            return "Limited benchmark data in historical context."
    
    def _prepare_peer_comparison_context(self, rag_context: str) -> str:
        """Prepare peer comparison context"""
        if not rag_context:
            return "No peer comparison data available."
        
        return "Peer performance data available from similar deals for comparison."
    
    def _get_default_template(self, template_type: str) -> str:
        """Get default template for given type"""
        
        default_templates = {
            'salesperson_sentiment': """
You are an expert sales psychology analyst specializing in salesperson behavior and sentiment analysis. Analyze the salesperson's performance, communication patterns, and deal management approach using STRICT PROFESSIONAL STANDARDS.

**CRITICAL**: Apply professional salesperson standards where normal business communication is BASELINE (neutral), not positive. Only exceptional performance should be rated as positive.

## DEAL CONTEXT
Deal ID: {deal_id}
Deal Amount: {deal_amount} ({deal_size_category})
Deal Stage: {deal_stage} | Type: {deal_type} | Age: {deal_age_days} days
Probability: {deal_probability}% | Outcome: {deal_outcome}

## SALESPERSON PERFORMANCE METRICS
Total Activities: {activities_count} over {time_span_days} days
Response Performance: {response_performance} (avg: {avg_response_time_hours:.1f}h)
Communication Quality: {communication_quality}
Engagement Level: {engagement_level}
Activity Frequency: {activity_frequency}
Business Hours Ratio: {business_hours_ratio:.0%}
Communication Gaps: {communication_gaps_count}
Email Engagement Ratio: {email_ratio:.2f}

## ACTIVITY ANALYSIS
Recent Activity Frequency: {activity_frequency} activities
Total Activities Analyzed: {total_activities}

{activities_text}

## HISTORICAL CONTEXT & BENCHMARKING
{rag_context}

## ANALYSIS FOCUS
{focus_instructions}

## EVALUATION STANDARDS
{evaluation_standards}

## RESPONSE FORMAT
{response_format}

Analyze the salesperson's sentiment and performance using these professional standards and provide your assessment in the required JSON format.
""",
            
            'deal_risk_assessment': """
You are a sales risk assessment expert analyzing salesperson behavior patterns that could jeopardize deal success.

DEAL CONTEXT:
Deal ID: {deal_id} | Amount: {deal_amount} | Stage: {deal_stage}
Performance Rating: {response_performance} | Communication: {communication_quality}

ACTIVITIES:
{activities_text}

HISTORICAL RISK PATTERNS:
{rag_context}

CURRENT RISK INDICATORS:
{risk_indicators}

{identified_risks}

Focus on identifying behavioral risks in:
1. Communication response patterns and gaps
2. Activity frequency and consistency trends  
3. Client engagement deterioration signals
4. Deal progression stagnation indicators
5. Competitive positioning weaknesses

Provide detailed risk assessment with mitigation strategies in JSON format.
""",
            
            'coaching_recommendations': """
You are a sales coach providing actionable performance improvement recommendations for a salesperson.

SALESPERSON PERFORMANCE PROFILE:
Deal ID: {deal_id} | Current Performance: {engagement_level}
Response Time: {response_performance} | Communication: {communication_quality}

CURRENT ACTIVITIES:
{activities_text}

SUCCESS PATTERNS TO EMULATE:
{success_patterns}

IMPROVEMENT OPPORTUNITIES:
{improvement_opportunities}

PERFORMANCE GAPS:
{performance_gaps}

COACHING FOCUS: {coaching_focus}

Provide specific, actionable coaching recommendations focusing on:
1. Communication effectiveness improvement
2. Activity optimization and time management
3. Client relationship strengthening strategies
4. Deal progression and momentum building
5. Professional development priorities

{coaching_recommendations_format}
""",
            
            'performance_benchmarking': """
You are a sales performance analyst comparing salesperson performance against successful deal benchmarks.

CURRENT PERFORMANCE:
Deal ID: {deal_id}
{performance_metrics}

BENCHMARK DATA:
{benchmark_standards}

PEER COMPARISON:
{peer_comparison}

HISTORICAL CONTEXT:
{rag_context}

Compare current performance against:
1. Successful deal patterns from similar deals
2. Industry performance benchmarks
3. Peer performance metrics
4. Best practice standards

Provide detailed performance benchmarking analysis in JSON format.
"""
        }
        
        return default_templates.get(template_type, default_templates['salesperson_sentiment'])
    
    def _build_basic_salesperson_prompt(self, deal_id: str, activities_text: str, rag_context: str, deal_metadata: Dict[str, Any]) -> str:
        """Build basic salesperson sentiment prompt as fallback"""
        
        return f"""
Analyze the salesperson sentiment and behavior patterns from the following deal activities using professional sales standards:

Deal ID: {deal_id}
Activities: {activities_text}
Historical Context: {rag_context}

Focus on salesperson performance, communication effectiveness, and deal management approach.
Provide analysis in JSON format with sentiment score, performance ratings, and coaching recommendations.
"""
    
    def _build_basic_risk_prompt(self, deal_id: str, activities_text: str, rag_context: str, deal_metadata: Dict[str, Any]) -> str:
        """Build basic risk assessment prompt as fallback"""
        
        return f"""
Assess behavioral risks in the salesperson's approach for the following deal:

Deal ID: {deal_id}
Activities: {activities_text}
Historical Context: {rag_context}

Identify risks in communication patterns, activity management, and client relationship management.
Provide risk assessment with mitigation strategies in JSON format.
"""
    
    def _build_basic_coaching_prompt(self, deal_id: str, activities_text: str, rag_context: str, deal_metadata: Dict[str, Any]) -> str:
        """Build basic coaching prompt as fallback"""
        
        return f"""
Provide coaching recommendations for the salesperson managing this deal:

Deal ID: {deal_id}
Activities: {activities_text}
Historical Success Patterns: {rag_context}

Focus on specific, actionable recommendations for performance improvement.
Provide coaching recommendations in JSON format.
"""
    
    def _build_basic_benchmarking_prompt(self, deal_id: str, activities_text: str, rag_context: str, deal_metadata: Dict[str, Any]) -> str:
        """Build basic benchmarking prompt as fallback"""
        
        return f"""
Compare salesperson performance against successful deal benchmarks:

Deal ID: {deal_id}
Activities: {activities_text}
Benchmark Data: {rag_context}

Provide performance comparison and recommendations in JSON format.
"""


# Factory function
def create_prompt_builder(template_dir: str = "prompts/") -> PromptBuilder:
    """Create prompt builder instance for salesperson sentiment analysis"""
    return PromptBuilder(template_dir=template_dir)


# Example usage and testing
def test_prompt_builder():
    """Test the prompt builder with salesperson-focused sample data"""
    
    builder = create_prompt_builder()
    
    # Sample deal metadata with salesperson performance metrics
    sample_deal_metadata = {
        'deal_amount': 85000,
        'deal_stage': 'proposal',
        'deal_type': 'newbusiness',
        'deal_probability': 65,
        'deal_age_days': 52,
        'deal_outcome': 'open',
        'deal_size_category': 'medium',
        'activities_count': 18,
        'time_span_days': 42,
        'response_time_metrics': {
            'avg_response_time_hours': 8.5,
            'fastest_response_hours': 1.2,
            'slowest_response_hours': 24.5,
            'response_count': 12
        },
        'communication_gaps_count': 2,
        'business_hours_ratio': 0.75,
        'weekend_activity_ratio': 0.15,
        'activity_types': {'email': 10, 'call': 4, 'meeting': 2, 'note': 2, 'task': 0},
        'email_ratio': 1.25,
        'outgoing_emails': 10,
        'incoming_emails': 8,
        'activity_frequency_trend': 'stable',
        'avg_time_between_activities_hours': 48.5
    }
    
    sample_activities = """
    [2024-01-15] EMAIL: Subject: Proposal timeline discussion
    Content: Following up on our conversation about implementation timeline and next steps.
    
    [2024-01-18] CALL: Duration: 45min - Proposal walkthrough
    Notes: Detailed discussion of proposal components, addressed technical questions.
    
    [2024-01-22] EMAIL: Subject: Re: Technical requirements
    Content: Thanks for clarifying the requirements. I'll coordinate with our technical team.
    """
    
    sample_rag_context = """
    SUCCESSFUL DEAL PATTERNS (3 won deals):
    • Average Response Time: 4.2 hours
    • Average Activities: 22 over 35 days
    • Communication Gaps: 0.3 avg
    • Business Hours Activity: 85%
    
    FAILED DEAL PATTERNS (2 lost deals):
    • Average Response Time: 18.5 hours
    • Communication Gaps: 3.5 avg
    • Common Warning Signs: Slow response times, Multiple communication gaps
    """
    
    try:
        # Test salesperson sentiment analysis prompt
        print("Building salesperson sentiment analysis prompt...")
        sentiment_prompt = builder.build_salesperson_sentiment_prompt(
            deal_id="SALES_001",
            activities_text=sample_activities,
            rag_context=sample_rag_context,
            deal_metadata=sample_deal_metadata,
            activity_frequency=5,
            total_activities=18,
            analysis_focus="comprehensive"
        )
        
        print("Salesperson Sentiment Analysis Prompt:")
        print("=" * 60)
        print(sentiment_prompt[:800] + "...")
        
        # Test coaching recommendations prompt
        print("\nBuilding coaching recommendations prompt...")
        coaching_prompt = builder.build_coaching_prompt(
            deal_id="SALES_001",
            activities_text=sample_activities,
            rag_context=sample_rag_context,
            deal_metadata=sample_deal_metadata,
            performance_gaps=["Response time above benchmark", "Communication gaps detected"],
            coaching_focus="response_time_improvement"
        )
        
        print("Coaching Recommendations Prompt:")
        print("=" * 60)
        print(coaching_prompt[:600] + "...")
        
        print("\n✅ Salesperson-focused prompt builder test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_prompt_builder()