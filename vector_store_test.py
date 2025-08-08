# vector_store_test.py
from core.vector_store import get_vector_store

def test_vector_store():
    store = get_vector_store()
    stats = store.get_stats()
    print("Vector store stats:", stats)

if __name__ == "__main__":
    test_vector_store()
