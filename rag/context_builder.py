import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class DealContext:
    """Container for deal context information"""
    deal_id: str
    activities: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    similar_deals: List[Dict[str, Any]] = None
    
class ContextComponent(ABC):
    """Abstract base class for context components"""
    
    @abstractmethod
    def build_context(self, deal_context: DealContext) -> str:
        """Build context section for this component"""
        pass
    
    @abstractmethod
    def get_component_name(self) -> str:
        """Get component name for logging"""
        pass
    
    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if component is enabled in settings"""
        pass

class SimilarDealsContext(ContextComponent):
    """Context component for similar deals"""
    
    def build_context(self, deal_context: DealContext) -> str:
        """Build similar deals context"""
        if not deal_context.similar_deals:
            return ""
        
        context_parts = ["## SIMILAR DEALS CONTEXT"]
        
        for i, similar_deal in enumerate(deal_context.similar_deals[:3], 1):
            deal_id = similar_deal.get('deal_id', f'Deal_{i}')
            outcome = similar_deal.get('metadata', {}).get('outcome', 'unknown')
            sentiment = similar_deal.get('metadata', {}).get('sentiment', 'neutral')
            
            context_parts.append(f"### Similar Deal {i} (ID: {deal_id})")
            context_parts.append(f"- **Outcome**: {outcome}")
            context_parts.append(f"- **Sentiment**: {sentiment}")
            
            # Add key activities
            activities = similar_deal.get('activities', [])
            if activities:
                context_parts.append("- **Key Activities**:")
                for activity in activities[:2]:  # Show first 2 activities
                    activity_type = activity.get('activity_type', 'unknown')
                    content = activity.get('content', '')[:100] + "..."
                    context_parts.append(f"  - {activity_type.upper()}: {content}")
            
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_component_name(self) -> str:
        return "SimilarDeals"
    
    @property
    def is_enabled(self) -> bool:
        return settings.CONTEXT_INCLUDE_SIMILAR_DEALS

class SentimentPatternsContext(ContextComponent):
    """Context component for sentiment patterns"""
    
    def build_context(self, deal_context: DealContext) -> str:
        """Build sentiment patterns context"""
        if not deal_context.similar_deals:
            return ""
        
        context_parts = ["## SENTIMENT PATTERNS ANALYSIS"]
        
        # Analyze sentiment patterns from similar deals
        won_deals = [d for d in deal_context.similar_deals if d.get('metadata', {}).get('outcome') == 'won']
        lost_deals = [d for d in deal_context.similar_deals if d.get('metadata', {}).get('outcome') == 'lost']
        
        if won_deals:
            context_parts.append("### Successful Deal Sentiment Patterns:")
            sentiment_indicators = self._extract_sentiment_indicators(won_deals)
            for indicator in sentiment_indicators:
                context_parts.append(f"- {indicator}")
        
        if lost_deals:
            context_parts.append("### Failed Deal Sentiment Patterns:")
            sentiment_indicators = self._extract_sentiment_indicators(lost_deals)
            for indicator in sentiment_indicators:
                context_parts.append(f"- {indicator}")
        
        return "\n".join(context_parts)
    
    def _extract_sentiment_indicators(self, deals: List[Dict[str, Any]]) -> List[str]:
        """Extract sentiment indicators from deals"""
        indicators = []
        
        for deal in deals:
            metadata = deal.get('metadata', {})
            sentiment = metadata.get('sentiment', 'neutral')
            
            # Extract common sentiment patterns
            if sentiment == 'positive':
                indicators.append("Proactive communication and quick response times")
            elif sentiment == 'negative':
                indicators.append("Delayed responses and reactive communication")
            
            # Add activity-based indicators
            activities = deal.get('activities', [])
            email_count = len([a for a in activities if a.get('activity_type') == 'email'])
            call_count = len([a for a in activities if a.get('activity_type') == 'call'])
            
            if email_count > call_count * 2:
                indicators.append("Email-heavy communication pattern")
            elif call_count > email_count:
                indicators.append("Call-focused engagement approach")
        
        return list(set(indicators))  # Remove duplicates
    
    def get_component_name(self) -> str:
        return "SentimentPatterns"
    
    @property
    def is_enabled(self) -> bool:
        return settings.CONTEXT_INCLUDE_SENTIMENT_PATTERNS

class LanguageToneContext(ContextComponent):
    """Context component for language tone analysis"""
    
    def build_context(self, deal_context: DealContext) -> str:
        """Build language tone context"""
        if not deal_context.similar_deals:
            return ""
        
        context_parts = ["## LANGUAGE TONE ANALYSIS"]
        
        # Analyze language patterns
        tone_patterns = self._analyze_language_tone(deal_context.similar_deals)
        
        context_parts.append("### Common Language Patterns:")
        for pattern in tone_patterns:
            context_parts.append(f"- {pattern}")
        
        return "\n".join(context_parts)
    
    def _analyze_language_tone(self, deals: List[Dict[str, Any]]) -> List[str]:
        """Analyze language tone from similar deals"""
        patterns = []
        
        for deal in deals:
            activities = deal.get('activities', [])
            
            # Analyze email activities for tone
            email_activities = [a for a in activities if a.get('activity_type') == 'email']
            
            for email in email_activities:
                content = email.get('content', '').lower()
                
                # Detect tone patterns
                if 'thanks' in content or 'appreciate' in content:
                    patterns.append("Appreciative and courteous tone")
                
                if 'urgent' in content or 'asap' in content:
                    patterns.append("Urgent communication style")
                
                if 'follow up' in content or 'following up' in content:
                    patterns.append("Proactive follow-up approach")
                
                if '?' in content:
                    patterns.append("Question-based engagement")
        
        return list(set(patterns))
    
    def get_component_name(self) -> str:
        return "LanguageTone"
    
    @property
    def is_enabled(self) -> bool:
        return settings.CONTEXT_INCLUDE_LANGUAGE_TONE

class DealProgressionContext(ContextComponent):
    """Context component for deal progression analysis"""
    
    def build_context(self, deal_context: DealContext) -> str:
        """Build deal progression context"""
        if not deal_context.similar_deals:
            return ""
        
        context_parts = ["## DEAL PROGRESSION PATTERNS"]
        
        # Analyze progression patterns
        progression_insights = self._analyze_deal_progression(deal_context.similar_deals)
        
        context_parts.append("### Successful Progression Patterns:")
        for insight in progression_insights['successful']:
            context_parts.append(f"- {insight}")
        
        context_parts.append("### Warning Signs:")
        for insight in progression_insights['warning']:
            context_parts.append(f"- {insight}")
        
        return "\n".join(context_parts)
    
    def _analyze_deal_progression(self, deals: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analyze deal progression patterns"""
        successful_patterns = []
        warning_patterns = []
        
        for deal in deals:
            metadata = deal.get('metadata', {})
            outcome = metadata.get('outcome', 'unknown')
            activities = deal.get('activities', [])
            
            # Analyze activity frequency
            if len(activities) > 10:
                if outcome == 'won':
                    successful_patterns.append("High activity engagement (10+ activities)")
                else:
                    warning_patterns.append("High activity without conversion")
            
            # Analyze activity types
            activity_types = [a.get('activity_type') for a in activities]
            meeting_count = activity_types.count('meeting')
            
            if meeting_count > 0:
                if outcome == 'won':
                    successful_patterns.append("In-person/virtual meetings scheduled")
                else:
                    warning_patterns.append("Meetings held but deal not closed")
            
            # Analyze time span
            if activities:
                time_span = metadata.get('time_span_days', 0)
                if time_span > 60:
                    if outcome == 'won':
                        successful_patterns.append("Extended engagement period leading to close")
                    else:
                        warning_patterns.append("Prolonged sales cycle without closure")
        
        return {
            'successful': list(set(successful_patterns)),
            'warning': list(set(warning_patterns))
        }
    
    def get_component_name(self) -> str:
        return "DealProgression"
    
    @property
    def is_enabled(self) -> bool:
        return settings.CONTEXT_INCLUDE_DEAL_PROGRESSION

class ClientBehaviorContext(ContextComponent):
    """Context component for client behavior analysis"""
    
    def build_context(self, deal_context: DealContext) -> str:
        """Build client behavior context"""
        if not deal_context.similar_deals:
            return ""
        
        context_parts = ["## CLIENT BEHAVIOR PATTERNS"]
        
        # Analyze client behavior
        behavior_insights = self._analyze_client_behavior(deal_context.similar_deals)
        
        context_parts.append("### Positive Client Behaviors:")
        for insight in behavior_insights['positive']:
            context_parts.append(f"- {insight}")
        
        context_parts.append("### Concerning Client Behaviors:")
        for insight in behavior_insights['concerning']:
            context_parts.append(f"- {insight}")
        
        return "\n".join(context_parts)
    
    def _analyze_client_behavior(self, deals: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analyze client behavior patterns"""
        positive_behaviors = []
        concerning_behaviors = []
        
        for deal in deals:
            metadata = deal.get('metadata', {})
            outcome = metadata.get('outcome', 'unknown')
            activities = deal.get('activities', [])
            
            # Analyze response patterns
            incoming_emails = [a for a in activities if a.get('activity_type') == 'email' and a.get('direction') == 'incoming']
            outgoing_emails = [a for a in activities if a.get('activity_type') == 'email' and a.get('direction') == 'outgoing']
            
            if len(incoming_emails) > 0:
                if outcome == 'won':
                    positive_behaviors.append("Client actively responds to communications")
                else:
                    concerning_behaviors.append("Client communication but no conversion")
            
            # Analyze meeting participation
            meetings = [a for a in activities if a.get('activity_type') == 'meeting']
            if meetings:
                if outcome == 'won':
                    positive_behaviors.append("Client willing to schedule meetings")
                else:
                    concerning_behaviors.append("Meetings scheduled but deal stalled")
            
            # Analyze communication gaps
            gaps = metadata.get('communication_gaps_count', 0)
            if gaps > 2:
                concerning_behaviors.append("Multiple communication gaps detected")
        
        return {
            'positive': list(set(positive_behaviors)),
            'concerning': list(set(concerning_behaviors))
        }
    
    def get_component_name(self) -> str:
        return "ClientBehavior"
    
    @property
    def is_enabled(self) -> bool:
        return settings.CONTEXT_INCLUDE_CLIENT_BEHAVIOR

class RAGContextBuilder:
    """
    Modular context builder that combines multiple context components
    Each component can be enabled/disabled and easily modified
    """
    
    def __init__(self):
        """Initialize context builder with all available components"""
        self.components = [
            SimilarDealsContext(),
            SentimentPatternsContext(),
            LanguageToneContext(),
            DealProgressionContext(),
            ClientBehaviorContext()
        ]
        
        # Filter only enabled components
        self.enabled_components = [c for c in self.components if c.is_enabled]
        
        logger.info(f"Context builder initialized with {len(self.enabled_components)} enabled components")
        for component in self.enabled_components:
            logger.debug(f"Enabled component: {component.get_component_name()}")
    
    def build_context(
        self,
        deal_id: str,
        activities: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        similar_deals: List[Dict[str, Any]] = None
    ) -> str:
        """
        Build comprehensive context from enabled components
        
        Args:
            deal_id: Current deal identifier
            activities: Deal activities
            metadata: Deal metadata
            similar_deals: Similar deals from RAG retrieval
            
        Returns:
            Formatted context string
        """
        
        if not similar_deals:
            return "## NO HISTORICAL CONTEXT AVAILABLE\nNo similar deals found for contextual analysis."
        
        deal_context = DealContext(
            deal_id=deal_id,
            activities=activities,
            metadata=metadata,
            similar_deals=similar_deals
        )
        
        context_parts = ["# HISTORICAL CONTEXT FOR SENTIMENT ANALYSIS"]
        
        # Build context from each enabled component
        for component in self.enabled_components:
            try:
                component_context = component.build_context(deal_context)
                if component_context.strip():
                    context_parts.append(component_context)
                    logger.debug(f"Added context from {component.get_component_name()}")
            except Exception as e:
                logger.error(f"Error building context for {component.get_component_name()}: {e}")
                continue
        
        context_parts.append("---")
        context_parts.append("Use this historical context to inform your sentiment analysis of the current deal.")
        
        return "\n\n".join(context_parts)
    
    def add_component(self, component: ContextComponent):
        """Add a new context component"""
        if component.is_enabled:
            self.enabled_components.append(component)
            logger.info(f"Added context component: {component.get_component_name()}")
    
    def remove_component(self, component_name: str):
        """Remove a context component by name"""
        self.enabled_components = [
            c for c in self.enabled_components 
            if c.get_component_name() != component_name
        ]
        logger.info(f"Removed context component: {component_name}")
    
    def get_enabled_components(self) -> List[str]:
        """Get list of enabled component names"""
        return [c.get_component_name() for c in self.enabled_components]

# Factory function
def create_context_builder() -> RAGContextBuilder:
    """Create context builder instance"""
    return RAGContextBuilder()