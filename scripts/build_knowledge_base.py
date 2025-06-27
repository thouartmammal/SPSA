import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_processor import DealDataProcessor
from core.embedding_service import get_embedding_service
from core.vector_store import get_vector_store
from config.settings import settings
from utils.logging import setup_logging

def main():
    """Build knowledge base from deal data"""
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting knowledge base build process")
    
    try:
        # Initialize services
        logger.info("Initializing services...")
        embedding_service = get_embedding_service()
        vector_store = get_vector_store()
        
        # Initialize data processor with embedding service
        processor = DealDataProcessor(embedding_service=embedding_service)
        
        # Validate configuration
        settings.validate()
        
        # Process deals
        logger.info(f"Processing deals from: {settings.DATA_PATH}")
        processed_deals = processor.process_all_deals(settings.DATA_PATH)
        
        if not processed_deals:
            logger.error("No deals were processed successfully")
            return False
        
        # Store in vector database
        logger.info(f"Storing {len(processed_deals)} deals in vector database...")
        vector_store.store_patterns(processed_deals)
        
        # Get statistics
        stats = vector_store.get_stats()
        logger.info("Knowledge base build completed successfully", extra=stats)
        
        # Print summary
        print(f"\n{'='*60}")
        print("KNOWLEDGE BASE BUILD SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successfully processed: {len(processed_deals)} deals")
        print(f"📊 Vector database: {stats.get('database_type', 'unknown')}")
        print(f"📈 Total patterns stored: {stats.get('total_patterns', 0)}")
        print(f"🎯 Embedding service: {settings.EMBEDDING_SERVICE}")
        print(f"📁 Data source: {settings.DATA_PATH}")
        
        return True
        
    except Exception as e:
        logger.error(f"Knowledge base build failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)