import json
import logging
import os
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
from datetime import datetime
import time
import re
from pathlib import Path
import openai
import anthropic
from groq import Groq

from config.settings import settings

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI LLM Provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        if not openai:
            raise ImportError("openai library not installed. Run: pip install openai")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized OpenAI provider with model: {model}")
    
    def generate_response(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate response using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert sales sentiment analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def get_provider_name(self) -> str:
        return f"OpenAI ({self.model})"

class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM Provider"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        if not anthropic:
            raise ImportError("anthropic library not installed. Run: pip install anthropic")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        logger.info(f"Initialized Anthropic provider with model: {model}")
    
    def generate_response(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate response using Anthropic Claude"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def get_provider_name(self) -> str:
        return f"Anthropic ({self.model})"

class GroqProvider(LLMProvider):
    """Groq LLM Provider"""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile"):
        if not Groq:
            raise ImportError("groq library not installed. Run: pip install groq")
        
        self.client = Groq(api_key=api_key)
        self.model = model
        logger.info(f"Initialized Groq provider with model: {model}")
    
    def generate_response(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate response using Groq"""
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert sales sentiment analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            return chat_completion.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise
    
    def get_provider_name(self) -> str:
        return f"Groq ({self.model})"

class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI LLM Provider"""
    
    def __init__(self, api_key: str, endpoint: str, deployment_name: str, api_version: str = "2024-02-15-preview"):
        if not openai:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        if not api_key:
            raise ValueError("Azure OpenAI API key is required")
        if not endpoint:
            raise ValueError("Azure OpenAI endpoint is required")
        if not deployment_name:
            raise ValueError("Azure OpenAI deployment name is required")
        
        self.client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        self.deployment_name = deployment_name
        self.endpoint = endpoint
        logger.info(f"Initialized Azure OpenAI provider with deployment: {deployment_name}")
    
    def generate_response(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate response using Azure OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert sales sentiment analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            raise
    
    def get_provider_name(self) -> str:
        return f"Azure OpenAI ({self.deployment_name})"

class PromptManager:
    """Manages prompt templates for sentiment analysis"""
    
    def __init__(self, prompt_file_path: str = "prompts/sentiment_analysis_prompt.txt"):
        self.prompt_file_path = prompt_file_path
        self.template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """Load prompt template from file"""
        try:
            prompt_path = Path(self.prompt_file_path)
            if not prompt_path.exists():
                logger.warning(f"Prompt file not found at {self.prompt_file_path}, using default")
                return self._get_default_prompt_template()
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            logger.info(f"Loaded prompt template from {self.prompt_file_path}")
            return template
            
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}")
            return self._get_default_prompt_template()
    
    def _get_default_prompt_template(self) -> str:
        """Fallback default prompt template"""
        return """You are an expert sales psychology analyst. Analyze the salesperson sentiment from the following activities:

Deal ID: {deal_id}
Activities: {activities_text}
RAG Context: {rag_context}

Provide analysis in JSON format with sentiment score, reasoning, and recommendations."""
    
    def format_prompt(
        self,
        deal_id: str,
        activities_text: str,
        rag_context: str = "",
        activity_frequency: int = 0,
        total_activities: int = 0,
        **kwargs
    ) -> str:
        """Format the prompt template with provided data"""
        
        try:
            formatted_prompt = self.template.format(
                deal_id=deal_id,
                activities_text=activities_text,
                rag_context=rag_context,
                activity_frequency=activity_frequency,
                total_activities=total_activities,
                **kwargs
            )
            
            return formatted_prompt
            
        except KeyError as e:
            logger.error(f"Missing required prompt parameter: {e}")
            raise ValueError(f"Missing required prompt parameter: {e}")
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            raise

class LLMClient:
    """Multi-provider LLM client for sentiment analysis"""
    
    def __init__(
        self, 
        provider: LLMProvider,
        prompt_file_path: str = "prompts\prompt_version_3.txt",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize LLM client
        
        Args:
            provider: LLM provider instance
            prompt_file_path: Path to prompt template file
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.provider = provider
        self.prompt_manager = PromptManager(prompt_file_path)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(f"Initialized LLM client with provider: {provider.get_provider_name()}")
    
    def analyze_sentiment(
        self,
        deal_id: str,
        activities_text: str,
        rag_context: str = "",
        activity_frequency: int = 0,
        total_activities: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze salesperson sentiment using RAG context and activities
        
        Args:
            deal_id: Unique deal identifier
            activities_text: Combined text of deal activities
            rag_context: Historical context from RAG system
            activity_frequency: Recent activity frequency
            total_activities: Total number of activities
            **kwargs: Additional parameters for prompt formatting
            
        Returns:
            Structured sentiment analysis result
        """
        
        # Format the prompt using the template
        prompt = self.prompt_manager.format_prompt(
            deal_id=deal_id,
            activities_text=activities_text,
            rag_context=rag_context,
            activity_frequency=activity_frequency,
            total_activities=total_activities,
            **kwargs
        )
        
        # Generate response with retries
        response_text = self._generate_with_retries(prompt)
        
        # Parse and validate response
        result = self._parse_and_validate_response(response_text, deal_id)
        
        # Add metadata
        result['analysis_metadata'] = {
            'provider': self.provider.get_provider_name(),
            'timestamp': datetime.utcnow().isoformat(),
            'deal_id': deal_id,
            'rag_context_length': len(rag_context),
            'activities_length': len(activities_text)
        }
        
        logger.info(f"Completed sentiment analysis for deal {deal_id}")
        return result
    
    def _generate_with_retries(self, prompt: str) -> str:
        """Generate response with retry logic"""
        
        for attempt in range(self.max_retries):
            try:
                response = self.provider.generate_response(prompt)
                
                if not response or not response.strip():
                    raise ValueError("Empty response from LLM")
                
                return response
                
            except Exception as e:
                logger.warning(f"LLM generation attempt {attempt + 1} failed: {e}")
                
                if attempt == self.max_retries - 1:
                    logger.error(f"All {self.max_retries} attempts failed")
                    raise
                
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        raise RuntimeError("Failed to generate response after all retries")
    
    def _parse_and_validate_response(self, response_text: str, deal_id: str) -> Dict[str, Any]:
        """Parse and validate the LLM response"""
        
        try:
            # Clean response text (remove markdown formatting if present)
            cleaned_response = self._clean_response_text(response_text)
            
            # Parse JSON
            result = json.loads(cleaned_response)
            
            # Validate required fields
            self._validate_response_structure(result)
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response for deal {deal_id}: {e}")
            logger.debug(f"Raw response: {response_text}")
            raise ValueError(f"Invalid JSON response: {e}")
        
        except Exception as e:
            logger.error(f"Response validation failed for deal {deal_id}: {e}")
            raise
    
    def _clean_response_text(self, response_text: str) -> str:
        """Clean response text to extract JSON"""
        
        # Remove markdown code blocks
        if response_text.startswith('```'):
            response_text = re.sub(r'^```(?:json)?\s*\n', '', response_text)
            response_text = re.sub(r'\n```\s*$', '', response_text)
        
        # Extract JSON if wrapped in other text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        return response_text.strip()
    
    def _validate_response_structure(self, result: Dict[str, Any]) -> None:
        """Validate that response has required structure"""
        
        required_fields = [
            'overall_sentiment',
            'sentiment_score',
            'confidence',
            'activity_breakdown',
            'deal_momentum_indicators',
            'reasoning'
        ]
        
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate sentiment score range
        score = result.get('sentiment_score')
        if not isinstance(score, (int, float)) or not (-1.0 <= score <= 1.0):
            raise ValueError(f"Invalid sentiment_score: {score}. Must be between -1.0 and 1.0")
        
        # Validate confidence range
        confidence = result.get('confidence')
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Invalid confidence: {confidence}. Must be between 0.0 and 1.0")

def create_llm_client(provider_name: str, **provider_kwargs) -> LLMClient:
    """
    Factory function to create LLM client with specified provider
    
    Args:
        provider_name: Name of provider ('openai', 'anthropic', 'groq', 'azure')
        **provider_kwargs: Provider-specific configuration
        
    Returns:
        Configured LLM client
    """
    
    provider_name = provider_name.lower()
    
    if provider_name == 'openai':
        provider = OpenAIProvider(**provider_kwargs)
    elif provider_name == 'anthropic':
        provider = AnthropicProvider(**provider_kwargs)
    elif provider_name == 'groq':
        provider = GroqProvider(**provider_kwargs)
    elif provider_name == 'azure':
        provider = AzureOpenAIProvider(**provider_kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
    
    return LLMClient(provider)

# Example usage and testing
def test_llm_client():
    """Test the LLM client with sample data"""
    
    # Test data
    sample_rag_context = """
    HISTORICAL DEAL ANALYSIS:
    - 3 similar won deals averaged 4.2 hour response times
    - 2 similar lost deals had communication gaps > 7 days
    - Success pattern: proactive follow-ups and meeting scheduling
    """
    
    sample_activities = """
    [2024-01-15] EMAIL: Subject: Follow-up on pricing discussion
    [2024-01-16] CALL: Discussed implementation timeline with client
    [2024-01-17] TASK: Prepare detailed proposal for client review
    """
    
    try:
        # Initialize with OpenAI (example)
        client = create_llm_client(
            'openai',
            api_key=os.getenv('OPENAI_API_KEY', 'your-openai-api-key'),
            model='gpt-4-turbo-preview'
        )
        
        # Analyze sentiment
        result = client.analyze_sentiment(
            deal_id="TEST_001",
            activities_text=sample_activities,
            rag_context=sample_rag_context,
            activity_frequency=5,
            total_activities=12
        )
        
        print("Sentiment Analysis Result:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_llm_client()