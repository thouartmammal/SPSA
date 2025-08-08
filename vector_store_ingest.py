from core.vector_store import get_vector_store
from models.schemas import DealPattern

def ingest_sample_pattern():
    store = get_vector_store()
    
    # Example embedding vector — replace with your actual embedding values
    example_embedding = [0.01, 0.02, 0.03, 0.04, 0.05]  # must match your embedding size
    
    pattern = DealPattern(
        deal_id="test_deal_001",
        embedding=example_embedding,
        combined_text="Customer interested in product X.",
        metadata={"source": "manual_ingestion"},
        activities_count=1,
        activity_types=["email"],
        time_span_days=10
    )
    
    success = store.add_patterns([pattern])
    print("Ingestion success:", success)

if __name__ == "__main__":
    ingest_sample_pattern()
