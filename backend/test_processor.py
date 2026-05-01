from processing.text_processor import TextProcessor

def test_chunking():
    processor = TextProcessor(chunk_size=50, chunk_overlap=10)
    
    # Sample long text
    sample_text = (
        "This is a long piece of text designed to test the chunking capability "
        "of our Document Processing layer. We want to make sure that the sliding "
        "window approach works correctly, that the metadata is attached, and that "
        "the overlap preserves context between chunks. In a real RAG system, "
        "this text would come from a PDF or a Google Doc downloaded from Drive. "
        "The processor cleans the text first, removing extra spaces and newlines, "
        "and then splits it into words to create the chunks."
    )
    
    # Mock data
    doc_id = "test_123"
    file_name = "test_document.txt"
    
    # Process
    cleaned = processor.clean_text(sample_text)
    chunks = processor.chunk_text(cleaned, doc_id, file_name)
    
    print(f"Total chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Text: {chunk['chunk_text'][:100]}...")
        print(f"Metadata: doc_id={chunk['doc_id']}, file_name={chunk['file_name']}")

if __name__ == "__main__":
    test_chunking()
