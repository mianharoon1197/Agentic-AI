import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# 1. Initialize a local Chroma database folder
chroma_client = chromadb.PersistentClient(path="./local_rag_db")

# 2. Use local open-source embedding model 
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# 3. Create a collection (table) for your documents
collection = chroma_client.get_or_create_collection(
    name="national_ai_policiy", 
    embedding_function=sentence_transformer_ef
)

pdf_file_path = "National AI Policy Consultation Draft V1.pdf" 
print(f"Reading PDF from: {pdf_file_path}...")

reader = PdfReader(pdf_file_path)

valid_chunks = []
metadatas = []
ids = []
chunk_id = 0

# 4. Extract and chunk page by page to keep track of Metadata
for page_num, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        # Clean up the text by removing excessive newlines and spaces
        text = " ".join(text.split())
        chunk_size = 1000
        overlap = 200
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if len(chunk.strip()) > 50: # Only add if the chunk has meaningful content
                valid_chunks.append(chunk)
                # Add Metadata!
                metadatas.append({"source": pdf_file_path, "page": page_num + 1})
                ids.append(f"chunk_{chunk_id}")
                chunk_id += 1
                
            start += (chunk_size - overlap)

# 6. Add to the database with metadatas
collection.add(
    documents=valid_chunks,
    metadatas=metadatas,
    ids=ids
)
print(f"Successfully ingested {len(valid_chunks)} chunks with metadata into the local vector store!")