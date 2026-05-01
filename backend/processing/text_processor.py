import re
import io
from typing import List, Dict, Any
from pypdf import PdfReader

class TextProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the processor.
        :param chunk_size: Target number of words per chunk.
        :param chunk_overlap: Number of words to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text(self, file_content: bytes, file_name: str) -> str:
        """
        Extract text from various file formats.
        """
        extension = file_name.split('.')[-1].lower()
        
        if extension == 'pdf':
            return self._extract_from_pdf(file_content)
        elif extension in ['txt', 'text']:
            return file_content.decode('utf-8', errors='ignore')
        else:
            # Fallback for Google Docs (passed as plain text) or unknown types
            try:
                return file_content.decode('utf-8')
            except:
                return ""

    def _extract_from_pdf(self, file_content: bytes) -> str:
        pdf = PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        return text

    def clean_text(self, text: str) -> str:
        """
        Normalize encoding, remove extra whitespaces, and clean special characters.
        """
        # Normalize whitespace (replace multiple spaces/newlines with single ones)
        text = re.sub(r'\s+', ' ', text)
        # Remove unwanted control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        return text.strip()

    def chunk_text(self, text: str, doc_id: str, file_name: str) -> List[Dict[str, Any]]:
        """
        Splits text into overlapping chunks and attaches metadata.
        """
        words = text.split()
        chunks = []
        
        if not words:
            return []

        # Sliding window chunking
        i = 0
        while i < len(words):
            # Take a slice of words
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "chunk_text": chunk_text,
                "doc_id": doc_id,
                "file_name": file_name,
                "source": "gdrive"
            })
            
            # Move index forward by (chunk_size - overlap)
            # If we reached the end, break
            if i + self.chunk_size >= len(words):
                break
                
            i += (self.chunk_size - self.chunk_overlap)
            
        return chunks

    def process_document(self, file_content: bytes, file_name: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Full pipeline: Extract -> Clean -> Chunk
        """
        raw_text = self.extract_text(file_content, file_name)
        cleaned_text = self.clean_text(raw_text)
        return self.chunk_text(cleaned_text, doc_id, file_name)
