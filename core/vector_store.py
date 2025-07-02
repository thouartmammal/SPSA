import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import json
import chromadb
from chromadb.config import Settings as ChromaSettings
import pinecone
from config.settings import settings
from models.schemas import DealPattern, VectorSearchResult

logger = logging.getLogger(__name__)

class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def store_patterns(self, patterns: List[DealPattern]) -> None:
        """Store deal patterns in vector database"""
        pass
    
    @abstractmethod
    def search_similar(self, query_embedding: List[float], top_k: int = 10) -> List[VectorSearchResult]:
        """Search for similar patterns"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        pass

class ChromaDBVectorStore(VectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(self):
        if not chromadb:
            raise ImportError("chromadb not installed. Run: pip install chromadb")
        
        # Initialize ChromaDB client with telemetry disabled
        self.client = chromadb.PersistentClient(
            path=settings.VECTOR_DB_PATH,
            settings=chromadb.Settings(
                anonymized_telemetry=False,  # Disable telemetry
                allow_reset=True
            )
        )
        self.collection_name = settings.CHROMADB_COLLECTION_NAME
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Using existing ChromaDB collection: {self.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Sales deal patterns for sentiment analysis"}
            )
            logger.info(f"Created new ChromaDB collection: {self.collection_name}")
    
    def store_patterns(self, patterns: List[DealPattern]) -> None:
        """Store deal patterns in ChromaDB"""
        if not patterns:
            logger.warning("No patterns to store")
            return
        
        # Prepare data for ChromaDB
        ids = [pattern.deal_id for pattern in patterns]
        embeddings = [pattern.embedding for pattern in patterns]
        documents = [pattern.combined_text for pattern in patterns]
        metadatas = []
        
        for pattern in patterns:
            # ChromaDB metadata must be JSON serializable
            metadata = {
                'activities_count': pattern.activities_count,
                'time_span_days': pattern.time_span_days,
                'activity_types': json.dumps(pattern.activity_types),
                **{k: v for k, v in pattern.metadata.items() 
                   if isinstance(v, (str, int, float, bool)) or v is None}
            }
            metadatas.append(metadata)
        
        # Store in collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Stored {len(patterns)} patterns in ChromaDB")
    
    def search_similar(self, query_embedding: List[float], top_k: int = 10) -> List[VectorSearchResult]:
        """Search for similar patterns in ChromaDB"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Convert to VectorSearchResult objects
        search_results = []
        for i, deal_id in enumerate(results['ids'][0]):
            result = VectorSearchResult(
                deal_id=deal_id,
                similarity_score=1 - results['distances'][0][i],  # Convert distance to similarity
                metadata=results['metadatas'][0][i],
                combined_text=results['documents'][0][i]
            )
            search_results.append(result)

        # logger.info(f"ChromaDB distances: {results['distances'][0]}")
        # logger.info(f"Converted similarities: {[1 - d for d in results['distances'][0]]}")
        
        logger.info(f"Found {len(search_results)} similar patterns")
        return search_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics"""
        try:
            count = self.collection.count()
            return {
                'total_patterns': count,
                'collection_name': self.collection_name,
                'database_type': 'chromadb'
            }
        except Exception as e:
            logger.error(f"Error getting ChromaDB stats: {e}")
            return {}

class PineconeVectorStore(VectorStore):
    """Pinecone vector store implementation"""
    
    def __init__(self):
        if not pinecone:
            raise ImportError("pinecone-client not installed. Run: pip install pinecone-client")
        
        if not settings.PINECONE_API_KEY or not settings.PINECONE_ENVIRONMENT:
            raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT are required")
        
        # Initialize Pinecone
        pinecone.init(
            api_key=settings.PINECONE_API_KEY,
            environment=settings.PINECONE_ENVIRONMENT
        )
        
        self.index_name = settings.PINECONE_INDEX_NAME
        
        # Create index if it doesn't exist
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                self.index_name,
                dimension=settings.VECTOR_DIMENSION,
                metric="cosine"
            )
            logger.info(f"Created new Pinecone index: {self.index_name}")
        
        self.index = pinecone.Index(self.index_name)
        logger.info(f"Connected to Pinecone index: {self.index_name}")
    
    def store_patterns(self, patterns: List[DealPattern]) -> None:
        """Store deal patterns in Pinecone"""
        if not patterns:
            logger.warning("No patterns to store")
            return
        
        # Prepare vectors for Pinecone
        vectors = []
        for pattern in patterns:
            # Pinecone metadata size limit
            metadata = {
                'activities_count': pattern.activities_count,
                'time_span_days': pattern.time_span_days,
                'combined_text': pattern.combined_text[:1000],  # Truncate for metadata
                **{k: v for k, v in pattern.metadata.items() 
                   if isinstance(v, (str, int, float, bool)) and len(str(v)) < 100}
            }
            
            vectors.append({
                'id': pattern.deal_id,
                'values': pattern.embedding,
                'metadata': metadata
            })
        
        # Upsert vectors
        self.index.upsert(vectors)
        logger.info(f"Stored {len(patterns)} patterns in Pinecone")
    
    def search_similar(self, query_embedding: List[float], top_k: int = 10) -> List[VectorSearchResult]:
        """Search for similar patterns in Pinecone"""
        response = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Convert to VectorSearchResult objects
        search_results = []
        for match in response.matches:
            result = VectorSearchResult(
                deal_id=match.id,
                similarity_score=match.score,
                metadata=match.metadata,
                combined_text=match.metadata.get('combined_text', '')
            )
            search_results.append(result)
        
        logger.info(f"Found {len(search_results)} similar patterns")
        return search_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_patterns': stats.total_vector_count,
                'index_name': self.index_name,
                'database_type': 'pinecone',
                'dimension': stats.dimension
            }
        except Exception as e:
            logger.error(f"Error getting Pinecone stats: {e}")
            return {}

def get_vector_store() -> VectorStore:
    """Factory function to get the configured vector store"""
    store_type = settings.VECTOR_DB.lower()
    
    if store_type == "chromadb":
        return ChromaDBVectorStore()
    elif store_type == "pinecone":
        return PineconeVectorStore()
    else:
        raise ValueError(f"Unknown vector store: {store_type}")
