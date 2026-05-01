from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from services.rag_service import RagService
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Google Drive RAG API")

# Initialize the RAG Service
# Prioritize service_account.json in the backend directory
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "service_account.json")

# Fallback check
if not os.path.exists(CREDENTIALS_PATH):
    print(f"Warning: {CREDENTIALS_PATH} not found. Checking GOOGLE_APPLICATION_CREDENTIALS env var.")
    CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")

try:
    rag_service = RagService(CREDENTIALS_PATH)
except Exception as e:
    print(f"Critical Error: Could not initialize RagService: {e}")
    # We don't raise here to allow the app to start, but endpoints will fail
    rag_service = None

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

@app.get("/", tags=["Health"])
async def health_check():
    return {
        "status": "online",
        "service_account_found": os.path.exists(CREDENTIALS_PATH) if CREDENTIALS_PATH else False,
        "rag_service_initialized": rag_service is not None
    }

@app.post("/sync-drive", tags=["Ingestion"])
async def sync_data(folder_id: Optional[str] = None):
    """
    Downloads files from Google Drive, processes them, and stores them in the vector database.
    """
    if not rag_service:
        raise HTTPException(status_code=500, detail="RagService not initialized. Check credentials.")
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
    if not rag_service:
        raise HTTPException(status_code=500, detail="RagService not initialized.")
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
    if not rag_service:
        raise HTTPException(status_code=500, detail="RagService not initialized.")
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
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
