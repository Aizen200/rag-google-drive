from search.vector_store import VectorStore

def test_vector_search():
    # 1. Initialize Store
    print("Loading embedding model...")
    store = VectorStore()
    
    # 2. Mock Chunks (from previous layer)
    mock_chunks = [
        {
            "chunk_text": "The capital of France is Paris. It is known for the Eiffel Tower.",
            "doc_id": "doc_1",
            "file_name": "france.txt",
            "source": "gdrive"
        },
        {
            "chunk_text": "The capital of Japan is Tokyo. It is famous for its busy Shibuya Crossing.",
            "doc_id": "doc_2",
            "file_name": "japan.txt",
            "source": "gdrive"
        },
        {
            "chunk_text": "Python is a versatile programming language used for data science and AI.",
            "doc_id": "doc_3",
            "file_name": "python.txt",
            "source": "gdrive"
        }
    ]
    
    # 3. Add to Store
    store.add_chunks(mock_chunks)
    
    # 4. Search
    query = "Where is the Eiffel Tower?"
    print(f"\nSearching for: '{query}'")
    
    results = store.search(query, top_k=1)
    
    if results:
        match = results[0]
        print(f"Match found in: {match['file_name']}")
        print(f"Text: {match['chunk_text']}")
        print(f"Score (lower is better): {match['score']:.4f}")

if __name__ == "__main__":
    test_vector_search()
