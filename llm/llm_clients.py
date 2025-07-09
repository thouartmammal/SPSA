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
    
    def generate_response(self, prompt: str, max_tokens: int = 3000) -> str:
        """Generate response using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert sales sentiment analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            logger.info(f"Sending prompt to LLM - Length: {len(prompt)} characters")
            logger.info(f"Full prompt sent to LLM:\n{prompt}")
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
            logger.info(f"Sending prompt to LLM - Length: {len(prompt)} characters")
            logger.info(f"Full prompt sent to LLM:\n{prompt}")
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
    
    def generate_response(self, prompt: str, max_tokens: int = 3000) -> str:
        """Generate response using Groq"""
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert sales sentiment analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            logger.info(f"Sending prompt to LLM - Length: {len(prompt)} characters")
            logger.info(f"Full prompt sent to LLM:\n{prompt}")
            return chat_completion.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise
    
    def get_provider_name(self) -> str:
        return f"Groq ({self.model})"

class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI LLM Provider using responses API"""
    
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
    
    def _extract_content_safely(self, response_dict: dict) -> str:
        """Safely extract content from Azure OpenAI response with proper error handling"""
        try:
            # Log the response structure for debugging
            logger.info(f"Response structure keys: {list(response_dict.keys())}")
            
            # Check if 'output' exists
            if 'output' not in response_dict:
                raise ValueError("Response missing 'output' field")
            
            output = response_dict['output']
            if not isinstance(output, list):
                raise ValueError(f"Expected 'output' to be a list, got {type(output)}")
            
            if len(output) == 0:
                raise ValueError("Response 'output' list is empty")
            
            # Find messages with type="message"
            messages = [item for item in output if isinstance(item, dict) and item.get('type') == "message"]
            
            if len(messages) == 0:
                # Fallback: try to find any item with 'content'
                messages = [item for item in output if isinstance(item, dict) and 'content' in item]
                if len(messages) == 0:
                    raise ValueError("No messages found with type='message' or 'content' field")
            
            # Get the first message
            message = messages[0]
            
            # Check if message has 'content'
            if 'content' not in message:
                raise ValueError("Message missing 'content' field")
            
            content = message['content']
            if not isinstance(content, list):
                raise ValueError(f"Expected 'content' to be a list, got {type(content)}")
            
            if len(content) == 0:
                raise ValueError("Message 'content' list is empty")
            
            # Get the first content item
            content_item = content[0]
            if not isinstance(content_item, dict):
                raise ValueError(f"Expected content item to be a dict, got {type(content_item)}")
            
            # Check if content item has 'text'
            if 'text' not in content_item:
                raise ValueError("Content item missing 'text' field")
            
            text = content_item['text']
            if not isinstance(text, str):
                raise ValueError(f"Expected 'text' to be a string, got {type(text)}")
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            logger.error(f"Response structure: {response_dict}")
            raise
    
    def _clean_json_response(self, raw_content: str) -> str:
        """
        Clean Azure OpenAI response to extract pure JSON.
        """
        import re
        import json
        
        if not raw_content or not raw_content.strip():
            raise ValueError("Empty or whitespace-only response")
        
        # Log the raw content for debugging
        logger.info(f"Raw Azure response : {raw_content}\n{type(raw_content)}")
        
        # Step 1: Basic cleanup - remove leading/trailing whitespace
        content = raw_content.strip()
        
        # Step 2: If it's already valid JSON, return as-is
        try:
            json.loads(content)
            logger.debug("Content is already valid JSON")
            return content
        except json.JSONDecodeError:
            logger.debug("Content is not valid JSON, attempting to clean")
        
        # Step 3: Remove markdown code blocks if present
        markdown_pattern = r'```(?:json)?\s*(.*?)\s*```'
        markdown_match = re.search(markdown_pattern, content, re.DOTALL | re.IGNORECASE)
        if markdown_match:
            content = markdown_match.group(1).strip()
            logger.debug("Removed markdown code block")
            
            # Check if it's valid JSON after markdown removal
            try:
                json.loads(content)
                return content
            except json.JSONDecodeError:
                pass
        
        # Step 4: Look for JSON pattern in the text
        json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
        json_matches = re.findall(json_pattern, content, re.DOTALL)
        
        for match in json_matches:
            try:
                # Test if this match is valid JSON
                json.loads(match)
                logger.debug("Found valid JSON using regex pattern")
                return match
            except json.JSONDecodeError:
                continue
        
        # Step 5: Try to find JSON by looking for lines that start and end with braces
        lines = content.split('\n')
        json_lines = []
        in_json = False
        brace_count = 0
        
        for line in lines:
            stripped_line = line.strip()
            if not in_json and stripped_line.startswith('{'):
                in_json = True
                json_lines = [line]
                brace_count = stripped_line.count('{') - stripped_line.count('}')
            elif in_json:
                json_lines.append(line)
                brace_count += stripped_line.count('{') - stripped_line.count('}')
                if brace_count == 0:
                    # Found complete JSON object
                    potential_json = '\n'.join(json_lines)
                    try:
                        json.loads(potential_json)
                        logger.debug("Found valid JSON using line-by-line parsing")
                        return potential_json
                    except json.JSONDecodeError:
                        pass
                    in_json = False
                    json_lines = []
                    brace_count = 0
        
        # Step 6: Last resort - try to extract any string that looks like JSON
        if '"overall_sentiment"' in content or '"sentiment_score"' in content:
            start_brace = content.find('{')
            end_brace = content.rfind('}')
            
            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                potential_json = content[start_brace:end_brace + 1]
                try:
                    json.loads(potential_json)
                    logger.debug("Found valid JSON using brace extraction")
                    return potential_json
                except json.JSONDecodeError:
                    pass
        
        # If all cleaning attempts fail, log the content and raise an error
        logger.error(f"Failed to extract valid JSON from Azure response")
        logger.error(f"Full response content: {raw_content}")
        raise ValueError(f"Could not extract valid JSON from Azure OpenAI response. Content starts with: {raw_content[:100]}")
    
    def generate_response(self, prompt: str, max_tokens: int = 4000) -> str:
        """Generate response using Azure OpenAI responses API"""
        try:
            # Add stronger JSON instruction to the prompt for Azure
            enhanced_prompt = f"""{prompt}

CRITICAL: Your response must be valid JSON only. Do not include any explanatory text, markdown formatting, or other content outside the JSON structure. Return only the JSON object."""
            
            response = self.client.responses.create(
                model=self.deployment_name,
                input=enhanced_prompt,
                max_output_tokens=max_tokens
            )
            
            # Convert response to dict for processing
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
            
            # Extract content safely
            raw_content = self._extract_content_safely(response_dict)
            
            # Clean and extract JSON
            cleaned_content = self._clean_json_response(raw_content)
            
            return cleaned_content
            
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
        prompt_file_path: str = "prompts/prompt_version_3.txt",
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