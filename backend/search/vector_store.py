import faiss
import numpy as np
import os
import pickle
from embedding.embedder import Embedder
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", storage_dir: str = "storage"):
        """
        Initialize the FAISS index and the embedding layer with persistence.
        """
        self.embedder = Embedder(model_name)
        self.embedding_dim = self.embedder.embedding_dim
        self.storage_dir = storage_dir
        self.index_path = os.path.join(storage_dir, "faiss_index.bin")
        self.meta_path = os.path.join(storage_dir, "metadata.pkl")
        
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

        # Inner Product (IP) index for Cosine Similarity (requires normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata = []
        
        # Auto-load existing data
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.load(self.index_path, self.meta_path)
                print(f"Loaded {len(self.metadata)} chunks from persistence.")
            except Exception as e:
                print(f"Error loading vector store: {e}")

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Generate embeddings for chunks and add to FAISS index.
        """
        if not chunks:
            return

        texts = [chunk["chunk_text"] for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)
        
        # Normalize vectors for Cosine Similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS and store metadata
        self.index.add(embeddings)
        self.metadata.extend(chunks)
        
        # Auto-save after adding
        self.save(self.index_path, self.meta_path)
        print(f"Added {len(chunks)} chunks and saved to disk.")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for the most relevant chunks using Cosine Similarity.
        """
        query_vector = self.embedder.embed_query(query)
        
        # Normalize query vector
        faiss.normalize_L2(query_vector)
        
        # Search index
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                res = self.metadata[idx].copy()
                # Score is inner product (cosine similarity) which ranges from -1 to 1 (usually 0-1 for text)
                res["score"] = float(distances[0][i])
                results.append(res)
                
        return results

    def save(self, index_path: str, meta_path: str):
        """Save index and metadata to disk."""
        faiss.write_index(self.index, index_path)
        with open(meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def load(self, index_path: str, meta_path: str):
        """Load index and metadata from disk."""
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            self.metadata = pickle.load(f)

    def get_indexed_doc_ids(self) -> set:
        """Return a set of doc_ids already indexed."""
        return {m["doc_id"] for m in self.metadata}
