#!/usr/bin/env python3
"""
Simple RAG Pipeline Test
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.embedding_service import get_embedding_service
from core.vector_store import get_vector_store
from rag.context_builder import RAGContextBuilder

# Initialize services
print("Initializing RAG services...")
embedding_service = get_embedding_service()
vector_store = get_vector_store()
context_builder = RAGContextBuilder()

# Test scenario
current_deal = "Client wants pricing, scheduled call for tomorrow"
print(f"\nTest Query: '{current_deal}'")
print("\n" + "="*50)

# Get embedding and search for similar deals
print("🔍 Searching for similar deals...")
current_deal_embedding = embedding_service.encode(current_deal)
similar_deals = vector_store.search_similar(current_deal_embedding, top_k=5)

print(f"Found {len(similar_deals)} similar deals")

if similar_deals:
    print("Similar deals:")
    for i, deal in enumerate(similar_deals):
        print(f"  {i+1}. Deal {deal.deal_id} (similarity: {deal.similarity_score:.3f})")

# Build RAG context with synthetic metadata
print("\n🧠 Building RAG context...")

# Create synthetic current deal metadata for testing
current_deal_metadata = {
    'deal_amount': 8000,
    'deal_size_category': 'medium',
    'deal_type': 'newbusiness',
    'deal_stage': 'proposal',
    'deal_probability': 65,
    'probability_category': 'medium',
    'deal_outcome': 'open',
    'activities_count': 11,
    'deal_age_days': 115,
    'is_new_business': True,
    'has_amount': True
}

context = context_builder.build_context(similar_deals, current_deal_metadata)

print("\n📋 Generated RAG Context:")
print("="*70)
print(context)
print("="*70)

print("\n✅ RAG Pipeline Test Complete!")