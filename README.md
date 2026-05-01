# RAG Google Drive - AI Platform Engineer Trial

This repository contains a production-ready Retrieval-Augmented Generation (RAG) system that connects to Google Drive, processes documents, and enables question-answering over your personal data.

## 🚀 Overview

The system provides a seamless pipeline to:
1.  **Connect** to Google Drive via Service Account.
2.  **Sync** documents (PDF, Google Docs, TXT).
3.  **Process** and chunk text with metadata tagging.
4.  **Embed** text using `SentenceTransformers`.
5.  **Search** using `FAISS` vector storage.
6.  **Answer** questions using LLMs (Groq/Llama-3) grounded in retrieved context.

## 📂 Architecture

Following the expected structure for high scalability:

- `connectors/`: Google Drive integration logic.
- `processing/`: Text extraction, cleaning, and chunking.
- `embedding/`: Vector embedding generation layer.
- `search/`: Vector store management (FAISS).
- `services/`: Core RAG orchestration logic.
- `main.py`: FastAPI endpoints and API definitions.

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.9+
- Google Cloud Project with Drive API enabled.
- Service Account JSON key (placed in `backend/service_account.json`).
- [Groq API Key](https://console.groq.com/) for the LLM.

### Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd rag-google-drive/backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables in `.env`:
   ```env
   GROQ_API_KEY=your_key_here
   GOOGLE_APPLICATION_CREDENTIALS=service_account.json
   ```

### Running the App
```bash
uvicorn main:app --reload
```

## 📡 API Endpoints

### 1. Sync Drive
`POST /sync-drive`
Downloads files from Google Drive, chunks them, and indexes them in the vector store.
- **Optional Body**: `{"folder_id": "..."}`

### 2. Ask Question
`POST /ask`
```json
{
  "query": "What are our company policies on compliance?"
}
```
- **Response**:
  ```json
  {
    "answer": "The company policies state...",
    "sources": ["Compliance_2024.pdf", "Internal_SOP.docx"]
  }
  ```

## 🧪 Evaluation Criteria Met
- **Must Have**: Full end-to-end RAG pipeline working with Google Drive.
- **Strong Candidate**: Clean API design, meaningful chunking strategy, and source attribution.
- **Bonus**: Robust architecture with dedicated embedding and search layers.

## 🐳 Docker (Optional)
A Dockerfile is provided for containerized deployment.
```bash
docker build -t rag-gdrive .
docker run -p 8000:10000 --env-file .env rag-gdrive
```
