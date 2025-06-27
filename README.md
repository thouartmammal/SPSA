    sales_sentiment_rag/
    ├── config/
    │   ├── __init__.py
    │   └── settings.py              # Configuration and environment variables
    ├── models/
    │   ├── __init__.py
    │   ├── schemas.py               # Pydantic models for data validation
    │   └── deal_activity.py         # Deal activity data models
    ├── core/
    │   ├── __init__.py
    │   ├── embedding_service.py     # Embedding generation (OpenAI, Sentence Transformers)
    │   ├── vector_store.py          # Vector database operations (Pinecone/Chroma)
    │   ├── data_processor.py        # Process deal activities into embeddings
    │   └── similarity_search.py     # Search similar patterns
    ├── rag/
    │   ├── __init__.py
    │   ├── context_builder.py       # Build context from similar cases
    │   ├── retriever.py             # Retrieve relevant historical patterns
    │   └── knowledge_base.py        # Manage knowledge base operations
    ├── llm/
    │   ├── __init__.py
    │   ├── llm_client.py           # Abstract LLM client (OpenAI, Anthropic, etc.)
    │   ├── prompt_builder.py       # Build prompts with RAG context
    │   └── sentiment_analyzer.py   # Main sentiment analysis engine
    ├── api/
    │   ├── __init__.py
    │   ├── main.py                 # FastAPI application
    │   ├── routes.py               # API endpoints
    │   └── middleware.py           # Authentication, logging, etc.
    ├── utils/
    │   ├── __init__.py
    │   ├── cache.py                # Redis caching
    │   ├── logging.py              # Structured logging
    │   └── helpers.py              # Utility functions
    ├── scripts/
    │   ├── build_knowledge_base.py # Script to populate vector DB
    │   ├── test_system.py          # System testing
    │   └── migrate_data.py         # Data migration utilities
    ├── data/
    │   ├── sample_deals.json       # Your sample data
    │   └── processed/              # Processed data files
    ├── tests/
    │   ├── __init__.py
    │   ├── test_embedding.py
    │   ├── test_vector_store.py
    │   └── test_rag.py
    ├── requirements.txt
    ├── .env
    ├── docker-compose.yml
    └── README.md