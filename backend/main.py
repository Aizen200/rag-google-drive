from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from services.rag_service import RagService
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = FastAPI(title="Google Drive RAG API")

# Initialize the RAG Service
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")
rag_service = RagService(CREDENTIALS_PATH)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class ChunkResponse(BaseModel):
    chunk_text: str
    file_name: str
    score: float

class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str]

# --- Endpoints ---

@app.post("/sync", tags=["Ingestion"])
async def sync_data(folder_id: Optional[str] = None):
    """
    Downloads files from Google Drive, processes them, and stores them in the vector database.
    """
    try:
        count = rag_service.sync_and_index(folder_id)
        return {"status": "success", "chunks_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=List[ChunkResponse], tags=["Retrieval"])
async def query_rag(request: QueryRequest):
    """
    Given a question, returns the most relevant chunks from the indexed documents.
    """
    try:
        results = rag_service.query(request.question, top_k=request.top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask_question(request: AskRequest):
    """
    Finds relevant context and generates an answer using an LLM.
    """
    try:
        # 1. Retrieve relevant chunks
        chunks = rag_service.query(request.query, top_k=5)
        
        # 2. Generate answer using LLM
        result = rag_service.generate_answer(request.query, chunks)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
