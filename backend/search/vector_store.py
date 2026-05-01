import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model and FAISS index.
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index (Inner Product for Cosine Similarity if vectors are normalized)
        # Using IndexFlatL2 for simplicity here (Euclidean distance)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # To store metadata mapped to vector indices
        self.metadata = []

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Generate embeddings for chunks in batch and add to FAISS.
        """
        if not chunks:
            return

        texts = [chunk["chunk_text"] for chunk in chunks]
        
        # Generate embeddings (batch processing)
        # model.encode returns a numpy array
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Ensure embeddings are float32 (required by FAISS)
        embeddings = embeddings.astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata.extend(chunks)
        print(f"Added {len(chunks)} chunks to vector store.")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for the most similar chunks for a given query.
        """
        # Embed query
        query_vector = self.model.encode([query], convert_to_numpy=True).astype('float32')
        
        # Search FAISS index
        # distances: list of distances, indices: indices of the matching vectors
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                res = self.metadata[idx].copy()
                res["score"] = float(distances[0][i])
                results.append(res)
                
        return results

    def save(self, index_path: str, meta_path: str):
        """Save index and metadata to disk."""
        faiss.write_index(self.index, index_path)
        import pickle
        with open(meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def load(self, index_path: str, meta_path: str):
        """Load index and metadata from disk."""
        self.index = faiss.read_index(index_path)
        import pickle
        with open(meta_path, 'rb') as f:
            self.metadata = pickle.load(f)
