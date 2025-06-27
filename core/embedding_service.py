
import logging
from typing import List, Union
from abc import ABC, abstractmethod
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch


from config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingService(ABC):
    """Abstract base class for embedding services"""
    
    @abstractmethod
    def encode(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass

class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI embedding service"""
    
    def __init__(self):
        if not openai:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_EMBEDDING_MODEL
        logger.info(f"Initialized OpenAI embedding service with model: {self.model}")
    
    def encode(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using OpenAI"""
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float"
        )
        
        embeddings = [data.embedding for data in response.data]
        return embeddings[0] if is_single else embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension for OpenAI models"""
        if "3-large" in self.model:
            return 3072
        elif "3-small" in self.model:
            return 1536
        else:
            return 1536  # Default

class SentenceTransformerEmbeddingService(EmbeddingService):
    """Sentence Transformers embedding service (free, local)"""
    
    def __init__(self):
        if not SentenceTransformer:
            raise ImportError("sentence-transformers library not installed. Run: pip install sentence-transformers")
        
        self.model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Initialized SentenceTransformer with model: {settings.SENTENCE_TRANSFORMER_MODEL}")
    
    def encode(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using SentenceTransformers"""
        embeddings = self.model.encode(text)
        
        if isinstance(text, str):
            return embeddings.tolist()
        else:
            return [emb.tolist() for emb in embeddings]
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension

class HuggingFaceEmbeddingService(EmbeddingService):
    """Hugging Face embedding service"""
    
    def __init__(self):
        if not (AutoTokenizer and AutoModel and torch):
            raise ImportError("transformers and torch not installed. Run: pip install transformers torch")
        
        self.tokenizer = AutoTokenizer.from_pretrained(settings.HUGGINGFACE_MODEL)
        self.model = AutoModel.from_pretrained(settings.HUGGINGFACE_MODEL)
        self.model.eval()
        logger.info(f"Initialized HuggingFace embedding service with model: {settings.HUGGINGFACE_MODEL}")
    
    def encode(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using HuggingFace"""
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        embeddings = []
        for txt in texts:
            inputs = self.tokenizer(txt, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                embeddings.append(embedding)
        
        return embeddings[0] if is_single else embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.config.hidden_size

def get_embedding_service() -> EmbeddingService:
    """Factory function to get the configured embedding service"""
    service_type = settings.EMBEDDING_SERVICE.lower()
    
    if service_type == "openai":
        return OpenAIEmbeddingService()
    elif service_type == "sentence_transformers":
        return SentenceTransformerEmbeddingService()
    elif service_type == "huggingface":
        return HuggingFaceEmbeddingService()
    else:
        raise ValueError(f"Unknown embedding service: {service_type}")
