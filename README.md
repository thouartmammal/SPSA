# Sales Sentiment RAG System

An advanced sales sentiment analysis system powered by Retrieval-Augmented Generation (RAG) that analyzes salesperson performance and deal sentiment using historical sales data and activity patterns.

## 🎯 Overview

The Sales Sentiment RAG system provides intelligent analysis of sales activities and deal progression by combining:

- **Historical Pattern Recognition**: Analyzes past deals to identify success/failure patterns
- **Real-time Sentiment Analysis**: Evaluates salesperson performance using LLM-powered analysis
- **Contextual Intelligence**: Uses RAG to provide context-aware insights from similar deals
- **Performance Benchmarking**: Compares current performance against historical data

## 🏗️ Architecture

```
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
│   ├── vector_store.py          # Vector database operations (ChromaDB, Pinecone)
│   ├── data_processor.py        # Process deal activities into embeddings
│   └── similarity_search.py     # Search similar patterns
├── rag/
│   ├── __init__.py
│   ├── context_builder.py       # Build context from similar cases
│   ├── retriever.py             # Retrieve relevant historical patterns
│   └── knowledge_base.py        # Manage knowledge base operations
├── llm/
│   ├── __init__.py
│   ├── llm_clients.py           # LLM client (OpenAI, Anthropic, Groq, Azure)
│   ├── prompt_builder.py        # Build prompts with RAG context
│   └── sentiment_analyzer.py    # Main sentiment analysis engine
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   ├── routes.py                # API endpoints
│   └── middleware.py            # Authentication, logging, rate limiting
├── utils/
│   ├── __init__.py
│   ├── cache.py                 # Redis caching (optional)
│   ├── logging_config.py        # Structured logging
│   └── helpers.py               # Utility functions
├── scripts/
│   ├── build_knowledge_base.py  # Script to populate vector DB
│   ├── test_system.py           # System testing
│   └── migrate_data.py          # Data migration utilities
├── prompts/
│   ├── prompt_version_3.txt     # Advanced sentiment analysis prompts
│   ├── salesperson_sentiment_prompt.txt
│   └── coaching_recommendations_prompt.txt
├── data/
│   ├── final_deal_details.json  # Deal data
│   └── processed/               # Processed data files
├── tests/
│   ├── __init__.py
│   ├── test_embedding.py
│   ├── test_vector_store.py
│   └── test_rag.py
├── requirements.txt
├── .env
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Redis (optional, for caching)
- API keys for LLM services (OpenAI, Anthropic, Groq, or Azure)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd sales_sentiment_rag
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Configure your environment**
```bash
# Core configuration
LLM_PROVIDER=azure                      # or openai, anthropic, groq
EMBEDDING_SERVICE=sentence_transformers # or openai, huggingface
VECTOR_DB=chromadb                      # or pinecone
DATA_PATH=data/final_deal_details.json

# LLM Provider Configuration (choose one)
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment
AZURE_OPENAI_API_VERSION=2025-03-01-preview

# Alternative providers
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GROQ_API_KEY=your_groq_key

# Optional for enhanced functionality
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env
REDIS_URL=redis://localhost:6379
```

5. **Build the knowledge base**
```bash
python scripts/build_knowledge_base.py
```

6. **Start the API server**
```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## 🔧 Configuration

### Supported LLM Providers

- **Azure OpenAI** (`azure`) - Recommended for enterprise
- **OpenAI** (`openai`) - GPT-4 and GPT-3.5
- **Anthropic** (`anthropic`) - Claude models
- **Groq** (`groq`) - Fast inference with Llama models

### Supported Embedding Services

- **Sentence Transformers** (`sentence_transformers`) - Local embeddings (default)
- **OpenAI** (`openai`) - text-embedding-3-small/large
- **HuggingFace** (`huggingface`) - Transformer models

### Supported Vector Databases

- **ChromaDB** (`chromadb`) - Local vector database (default)
- **Pinecone** (`pinecone`) - Cloud vector database

## 📊 Usage

### Building Knowledge Base

```bash
# Process deal data and build vector database
python scripts/build_knowledge_base.py
```

### API Usage

**Analyze Deal Sentiment:**
```bash
curl -X POST "http://localhost:8000/api/v1/analyze/deal-sentiment" \
     -H "Content-Type: application/json" \
     -d '{
       "deal_id": "12345",
       "activities": [
         {
           "activity_type": "email",
           "sent_at": "2024-01-15T09:00:00Z",
           "subject": "Proposal follow-up",
           "body": "Following up on our proposal discussion",
           "direction": "outgoing"
         }
       ],
       "metadata": {
         "amount": 50000,
         "dealstage": "proposal",
         "dealtype": "newbusiness"
       }
     }'
```

**Health Check:**
```bash
curl "http://localhost:8000/health"
```

### Python API

```python
from llm.sentiment_analyzer import create_sentiment_analyzer

# Initialize analyzer
analyzer = create_sentiment_analyzer(
    provider_name="azure",
    provider_config={
        'api_key': 'your_azure_key',
        'endpoint': 'your_endpoint',
        'deployment_name': 'your_deployment'
    }
)

# Analyze deal sentiment
deal_data = {
    'deal_id': '12345',
    'activities': [...],  # List of activity dictionaries
    'metadata': {...}     # Deal metadata
}

result = analyzer.analyze_deal_sentiment(deal_data)
print(f"Sentiment Score: {result['sentiment_score']}")
print(f"Overall Sentiment: {result['overall_sentiment']}")
print(f"Risk Level: {result['risk_assessment']['risk_level']}")
```

## 🧪 Testing

### Run System Tests
```bash
python scripts/test_system.py
```

### Run Unit Tests
```bash
pytest tests/ -v
```

### Test Individual Components
```bash
# Test embeddings
python -m core.embedding_service

# Test vector store
python -m core.vector_store

# Test data processor
python -m core.data_processor
```

## 📈 Features

### Core Capabilities

- **Deal Activity Processing**: Extracts and processes sales activities (emails, calls, meetings, notes, tasks)
- **Similarity Search**: Finds similar historical deals and patterns using vector embeddings
- **Sentiment Analysis**: Evaluates salesperson performance using advanced LLM prompts
- **RAG Context**: Provides intelligent context from historical successful/failed deals
- **Performance Metrics**: Tracks response times, activity frequency, and engagement patterns

### Analysis Features

- **Sentiment Scoring**: Numerical sentiment scores from -1.0 to +1.0
- **Risk Assessment**: Identifies deal risks and behavioral patterns
- **Coaching Recommendations**: Provides actionable improvement suggestions
- **Performance Benchmarking**: Compares against successful deal patterns
- **Activity Breakdown**: Detailed analysis of email, call, meeting, and task activities

### Technical Features

- **Multi-Provider LLM Support**: Works with Azure OpenAI, OpenAI, Anthropic, Groq
- **Flexible Embeddings**: Support for local and cloud embedding services
- **Vector Database Options**: ChromaDB for local, Pinecone for cloud
- **Optional Redis Caching**: High-performance caching for embeddings and results
- **Comprehensive Logging**: Structured logging for debugging and monitoring

## 🛠️ Data Migration

### HubSpot Integration
```bash
# Migrate HubSpot CRM export to standard format
python scripts/migrate_data.py hubspot-to-standard \
    --input data/hubspot_export.json \
    --output data/final_deal_details.json
```

### Custom Data Format
```bash
# Convert custom CRM format
python scripts/migrate_data.py custom-format \
    --input data/custom_deals.csv \
    --output data/final_deal_details.json
```

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export LLM_PROVIDER=azure
export EMBEDDING_SERVICE=sentence_transformers
export VECTOR_DB=chromadb

# Run development server
python -m uvicorn api.main:app --reload
```

### Production Deployment
```bash
# Run with gunicorn
pip install gunicorn
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📋 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/api/v1/analyze/deal-sentiment` | POST | Analyze deal sentiment |
| `/api/v1/search/similar-deals` | POST | Find similar deals |
| `/api/v1/knowledge-base/build` | POST | Rebuild knowledge base |
| `/api/v1/knowledge-base/stats` | GET | Knowledge base statistics |

## 🔍 Monitoring

### Logging
- Structured JSON logging with configurable levels
- Component-specific loggers for debugging
- Performance metrics tracking
- Error tracking and alerting

### Health Checks
- Service availability monitoring
- Database connection health
- Vector database status
- LLM provider connectivity

## 🆘 Troubleshooting

### Common Issues

**Knowledge Base Build Fails:**
```bash
# Check data file exists and is valid JSON
python -m json.tool data/final_deal_details.json

# Check logs for specific errors
tail -f logs/app.log
```

**LLM Provider Issues:**
```bash
# Test Azure OpenAI connection
python -c "
from llm.llm_clients import create_llm_client
client = create_llm_client('azure')
print('Connection successful')
"
```

**Vector Database Issues:**
```bash
# For ChromaDB - check disk space
df -h data/vector_db/

# For Pinecone - verify API key
python -c "
import pinecone
pinecone.init(api_key='your-key')
print('Pinecone connection successful')
"
```

### Performance Optimization

- Use Redis caching for better performance
- Consider Pinecone for production vector storage
- Monitor memory usage with large embedding models
- Use Azure OpenAI for enterprise-grade reliability

## 📚 Environment Variables Reference

### Required Variables
```bash
LLM_PROVIDER=azure                      # LLM service provider
EMBEDDING_SERVICE=sentence_transformers # Embedding service
VECTOR_DB=chromadb                      # Vector database
DATA_PATH=data/final_deal_details.json  # Deal data file path
```

### LLM Provider Keys (choose one)
```bash
# Azure OpenAI (recommended)
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment

# OpenAI
OPENAI_API_KEY=your_openai_key

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key

# Groq
GROQ_API_KEY=your_groq_key
```

### Optional Variables
```bash
PINECONE_API_KEY=your_pinecone_key      # For Pinecone vector DB
REDIS_URL=redis://localhost:6379        # For caching
LOG_LEVEL=INFO                          # Logging level
API_PORT=8000                           # API server port
```

---