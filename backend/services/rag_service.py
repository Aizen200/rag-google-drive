import os
import io
from typing import List, Dict, Any
from connectors.gdrive import GoogleDriveConnector
from processing.text_processor import TextProcessor
from search.vector_store import VectorStore

class RagService:
    def __init__(self, credentials_path: str):
        self.gdrive = GoogleDriveConnector(credentials_path)
        self.processor = TextProcessor()
        self.vector_store = VectorStore()

    def sync_and_index(self, folder_id: str = None) -> int:
        """
        1. Fetch file list from Drive
        2. Download each file
        3. Process/Chunk
        4. Add to Vector Store
        """
        files = self.gdrive.list_files(folder_id)
        total_chunks = 0

        for file_meta in files:
            try:
                if file_meta['mimeType'] == 'application/vnd.google-apps.document':
                    request = self.gdrive.service.files().export_media(
                        fileId=file_meta['id'], 
                        mimeType='text/plain'
                    )
                else:
                    request = self.gdrive.service.files().get_media(fileId=file_meta['id'])
                
                file_content = request.execute()

                chunks = self.processor.process_document(
                    file_content=file_content,
                    file_name=file_meta['name'],
                    doc_id=file_meta['id']
                )

                if chunks:
                    self.vector_store.add_chunks(chunks)
                    total_chunks += len(chunks)
                    
            except Exception as e:
                print(f"Error processing file {file_meta['name']}: {str(e)}")
                continue

        return total_chunks

    def query(self, user_query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find relevant chunks for a user query.
        """
        return self.vector_store.search(user_query, top_k=top_k)

    def generate_answer(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Uses an LLM to generate an answer based on the provided chunks.
        """
        if not chunks:
            return {
                "answer": "I'm sorry, I couldn't find any relevant information in your documents.",
                "sources": []
            }

        # 1. Construct Context
        context = "\n\n".join([f"--- Source: {c['file_name']} ---\n{c['chunk_text']}" for c in chunks])
        
        # 2. Build Prompt
        system_prompt = (
            "You are a helpful AI assistant. Answer the user's question STRICTLY based on the provided context. "
            "If the answer is not found in the context, your response MUST be exactly: 'Not available in documents'. "
            "Do not use any outside knowledge or information not present in the context."
        )
        
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # 3. Call LLM (Groq)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return {
                "answer": "[SIMULATED] Please set GROQ_API_KEY to see real results.",
                "sources": list(set([c['file_name'] for c in chunks]))
            }

        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )
            
            return {
                "answer": response.choices[0].message.content,
                "sources": list(set([c['file_name'] for c in chunks]))
            }
        except Exception as e:
            return {"answer": f"Error calling LLM: {str(e)}", "sources": []}
