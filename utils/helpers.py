import logging
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from email.utils import parseaddr
import phonenumbers
from urllib.parse import urlparse
import math

logger = logging.getLogger(__name__)

# Data validation helpers
def validate_deal_data(deal_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate deal data structure and content
    
    Args:
        deal_data: Deal data dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    
    errors = []
    
    # Check required fields
    required_fields = ['deal_id', 'activities']
    for field in required_fields:
        if field not in deal_data:
            errors.append(f"Missing required field: {field}")
    
    # Validate deal_id
    if 'deal_id' in deal_data:
        if not deal_data['deal_id'] or not str(deal_data['deal_id']).strip():
            errors.append("deal_id cannot be empty")
    
    # Validate activities
    if 'activities' in deal_data:
        activities = deal_data['activities']
        if not isinstance(activities, list):
            errors.append("activities must be a list")
        elif len(activities) == 0:
            errors.append("activities list cannot be empty")
        else:
            # Validate each activity
            for i, activity in enumerate(activities):
                if not isinstance(activity, dict):
                    errors.append(f"Activity {i} must be a dictionary")
                    continue
                
                if 'activity_type' not in activity:
                    errors.append(f"Activity {i} missing activity_type")
                
                valid_types = ['email', 'call', 'meeting', 'note', 'task']
                if activity.get('activity_type') not in valid_types:
                    errors.append(f"Activity {i} has invalid activity_type: {activity.get('activity_type')}")
    
    # Validate deal amounts if present
    if 'amount' in deal_data:
        try:
            amount = float(deal_data['amount'])
            if amount < 0:
                errors.append("Deal amount cannot be negative")
        except (ValueError, TypeError):
            errors.append("Deal amount must be a valid number")
    
    # Validate probability if present
    if 'deal_stage_probability' in deal_data:
        try:
            prob = float(deal_data['deal_stage_probability'])
            if not 0 <= prob <= 100:
                errors.append("Deal probability must be between 0 and 100")
        except (ValueError, TypeError):
            errors.append("Deal probability must be a valid number")
    
    return len(errors) == 0, errors

def validate_search_parameters(
    query_embedding: List[float],
    search_results: List[Any]
) -> bool:
    """
    Validate search parameters
    
    Args:
        query_embedding: Query embedding vector
        search_results: Search results list
        
    Returns:
        True if valid, False otherwise
    """
    
    try:
        # Validate embedding
        if not isinstance(query_embedding, list):
            return False
        
        if len(query_embedding) == 0:
            return False
        
        # Check if all elements are numeric
        for val in query_embedding:
            if not isinstance(val, (int, float)):
                return False
        
        # Validate search results
        if not isinstance(search_results, list):
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Search parameter validation error: {e}")
        return False

def validate_email_address(email: str) -> bool:
    """
    Validate email address format
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    
    if not email or not isinstance(email, str):
        return False
    
    try:
        name, addr = parseaddr(email)
        return '@' in addr and '.' in addr.split('@')[1]
    except Exception:
        return False

def validate_phone_number(phone: str, country_code: str = 'US') -> bool:
    """
    Validate phone number format
    
    Args:
        phone: Phone number to validate
        country_code: Country code for validation
        
    Returns:
        True if valid, False otherwise
    """
    
    if not phone or not isinstance(phone, str):
        return False
    
    try:
        parsed = phonenumbers.parse(phone, country_code)
        return phonenumbers.is_valid_number(parsed)
    except Exception:
        return False

def validate_url(url: str) -> bool:
    """
    Validate URL format
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    
    if not url or not isinstance(url, str):
        return False
    
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

# Data processing helpers
def clean_text(text: str, remove_extra_spaces: bool = True) -> str:
    """
    Clean and normalize text content
    
    Args:
        text: Text to clean
        remove_extra_spaces: Whether to remove extra spaces
        
    Returns:
        Cleaned text
    """
    
    if not text or not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove excessive whitespace
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_email_addresses(text: str) -> List[str]:
    """
    Extract email addresses from text
    
    Args:
        text: Text to search
        
    Returns:
        List of email addresses found
    """
    
    if not text:
        return []
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    
    # Validate and filter emails
    valid_emails = [email for email in emails if validate_email_address(email)]
    
    return valid_emails

def extract_phone_numbers(text: str) -> List[str]:
    """
    Extract phone numbers from text
    
    Args:
        text: Text to search
        
    Returns:
        List of phone numbers found
    """
    
    if not text:
        return []
    
    # Common phone number patterns
    phone_patterns = [
        r'\b\d{3}-\d{3}-\d{4}\b',  # 123-456-7890
        r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',  # (123) 456-7890
        r'\b\d{3}\.\d{3}\.\d{4}\b',  # 123.456.7890
        r'\b\d{10}\b'  # 1234567890
    ]
    
    phones = []
    for pattern in phone_patterns:
        phones.extend(re.findall(pattern, text))
    
    return phones

def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text
    
    Args:
        text: Text to search
        
    Returns:
        List of URLs found
    """
    
    if not text:
        return []
    
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    
    # Validate URLs
    valid_urls = [url for url in urls if validate_url(url)]
    
    return valid_urls

def normalize_text_for_embedding(text: str) -> str:
    """
    Normalize text for embedding generation
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    
    if not text:
        return ""
    
    # Clean text
    text = clean_text(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep spaces and punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Mathematical and statistical helpers
def calculate_similarity_score(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    
    if not vector1 or not vector2 or len(vector1) != len(vector2):
        return 0.0
    
    try:
        # Convert to numpy arrays
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norms == 0:
            return 0.0
        
        similarity = dot_product / norms
        
        # Ensure result is between 0 and 1
        return max(0.0, min(1.0, similarity))
        
    except Exception as e:
        logger.error(f"Similarity calculation error: {e}")
        return 0.0

def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize scores to 0-1 range
    
    Args:
        scores: List of scores to normalize
        
    Returns:
        Normalized scores
    """
    
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    range_score = max_score - min_score
    return [(score - min_score) / range_score for score in scores]

def calculate_percentile(values: List[float], percentile: float) -> float:
    """
    Calculate percentile of values
    
    Args:
        values: List of values
        percentile: Percentile to calculate (0-100)
        
    Returns:
        Percentile value
    """
    
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    
    if f == c:
        return sorted_values[int(k)]
    
    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    
    return d0 + d1

def calculate_standard_deviation(values: List[float]) -> float:
    """
    Calculate standard deviation of values
    
    Args:
        values: List of values
        
    Returns:
        Standard deviation
    """
    
    if len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    
    return math.sqrt(variance)

# File and data helpers
def calculate_data_hash(file_path: str) -> str:
    """
    Calculate hash of file contents
    
    Args:
        file_path: Path to file
        
    Returns:
        MD5 hash of file contents
    """
    
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        return hashlib.md5(file_content).hexdigest()
        
    except Exception as e:
        logger.error(f"Error calculating file hash: {e}")
        return ""

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string
    
    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default

def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """
    Safely serialize object to JSON
    
    Args:
        obj: Object to serialize
        default: Default value if serialization fails
        
    Returns:
        JSON string or default value
    """
    
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return default

# Date and time helpers
def parse_datetime_string(date_str: str) -> Optional[datetime]:
    """
    Parse datetime string with multiple format support
    
    Args:
        date_str: Date string to parse
        
    Returns:
        Parsed datetime or None
    """
    
    if not date_str:
        return None
    
    # Common datetime formats
    formats = [
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Try ISO format parsing
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except ValueError:
        pass
    
    logger.debug(f"Could not parse datetime string: {date_str}")
    return None

def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"

def is_business_hours(dt: datetime, business_start: int = 9, business_end: int = 17) -> bool:
    """
    Check if datetime is within business hours
    
    Args:
        dt: Datetime to check
        business_start: Business start hour (24-hour format)
        business_end: Business end hour (24-hour format)
        
    Returns:
        True if within business hours, False otherwise
    """
    
    # Check if weekday (Monday = 0, Sunday = 6)
    if dt.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check if within business hours
    return business_start <= dt.hour < business_end

# Text analysis helpers
def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text
    
    Args:
        text: Text to analyze
        min_length: Minimum keyword length
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    
    if not text:
        return []
    
    # Clean and tokenize text
    cleaned_text = clean_text(text.lower())
    words = re.findall(r'\b[a-zA-Z]+\b', cleaned_text)
    
    # Filter by length and common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those'}
    
    keywords = [word for word in words 
                if len(word) >= min_length and word not in stop_words]
    
    # Count frequency and return most common
    from collections import Counter
    word_counts = Counter(keywords)
    
    return [word for word, count in word_counts.most_common(max_keywords)]

def calculate_text_sentiment_indicators(text: str) -> Dict[str, Any]:
    """
    Calculate basic sentiment indicators from text
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of sentiment indicators
    """
    
    if not text:
        return {'positive_words': 0, 'negative_words': 0, 'exclamations': 0, 'questions': 0}
    
    text_lower = text.lower()
    
    # Simple positive/negative word lists
    positive_words = ['good', 'great', 'excellent', 'amazing', 'fantastic', 'wonderful', 'perfect', 'love', 'like', 'happy', 'excited', 'pleased', 'satisfied', 'success', 'successful', 'win', 'won', 'best', 'better', 'improve', 'improvement', 'opportunity', 'opportunities']
    
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'angry', 'frustrated', 'disappointed', 'concerned', 'worried', 'problem', 'problems', 'issue', 'issues', 'fail', 'failed', 'failure', 'lose', 'lost', 'worst', 'worse', 'decline', 'decrease', 'risk', 'risks']
    
    # Count indicators
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    return {
        'positive_words': positive_count,
        'negative_words': negative_count,
        'exclamations': exclamation_count,
        'questions': question_count,
        'word_count': len(text.split()),
        'character_count': len(text)
    }

# Performance monitoring helpers
def measure_execution_time(func):
    """
    Decorator to measure function execution time
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function with timing
    """
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        # Add execution time to result if it's a dictionary
        if isinstance(result, dict):
            result['execution_time_seconds'] = execution_time
        
        return result
    
    return wrapper

class PerformanceTimer:
    """Context manager for measuring execution time"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.debug(f"{self.description} completed in {duration:.4f} seconds")
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time"""
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time or time.time()
        return end_time - self.start_time

# Error handling helpers
def safe_execute(func, default_return=None, log_errors=True):
    """
    Safely execute function with error handling
    
    Args:
        func: Function to execute
        default_return: Value to return on error
        log_errors: Whether to log errors
        
    Returns:
        Function result or default value
    """
    
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger.error(f"Safe execution failed: {e}")
        return default_return

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Decorator to retry function on failure
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff_factor: Multiplier for delay on each retry
        
    Returns:
        Decorated function
    """
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator


# Example usage and testing
def test_helpers():
    """Test helper functions"""
    
    try:
        print("Testing validation helpers...")
        
        # Test deal data validation
        valid_deal = {
            'deal_id': 'test_001',
            'activities': [
                {'activity_type': 'email', 'content': 'test email'}
            ],
            'amount': 50000,
            'deal_stage_probability': 75
        }
        
        is_valid, errors = validate_deal_data(valid_deal)
        print(f"Deal validation: {'Valid' if is_valid else 'Invalid'} - {errors}")
        
        # Test text processing
        text = "  This is a TEST email with   extra spaces!  "
        cleaned = clean_text(text)
        print(f"Text cleaning: '{text}' -> '{cleaned}'")
        
        # Test similarity calculation
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        similarity = calculate_similarity_score(v1, v2)
        print(f"Similarity calculation: {similarity}")
        
        # Test performance timer
        with PerformanceTimer("Test operation") as timer:
            time.sleep(0.1)
        print(f"Timer test: {timer.elapsed_time:.4f}s")
        
        print("✅ Helper functions test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    import time
    test_helpers()