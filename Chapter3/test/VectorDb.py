import os
import pdfplumber
import chromadb
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, SQLiteSession, function_tool

load_dotenv()
set_tracing_disabled(disabled=True)

client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

PDF_PATH = "document.pdf"

def load_pdf_chunks(path: str, chunk_size: int = 500) -> list[str]:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    
    chunks = []
    words = text.split()
    current_chunk = []
    current_len = 0
    
    for word in words:
        current_chunk.append(word)
        current_len += len(word) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

print("Reading PDF and creating chunks...")
chunks = load_pdf_chunks(PDF_PATH)
print(f" Created {len(chunks)} chunks")

chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    chroma_client.delete_collection("pdf_collection")
except:
    pass

collection = chroma_client.create_collection("pdf_collection")

print("Storing chunks in ChromaDB...")
for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        ids=[f"chunk_{i}"]
    )
    print(f"  Stored chunk {i+1}/{len(chunks)}")

print("Vector store ready!\n")

@function_tool
def search_document(query: str) -> str:
    """
    Searches the PDF vector store for content relevant to the query.
    Args: query (str) - The question to search for
    Returns: str - Most relevant chunks from the document
    """
    results = collection.query(
        query_texts=[query],  
        n_results=3
    )
    return "\n\n".join(results["documents"][0])

agent = Agent(
    name="PDFReaderTool",
    instructions="""You are an AI agent that answers questions ONLY from the document provided by the search_document tool.
ALWAYS call search_document first before answering.
Never use your own knowledge. Answer in one sentence.""",
    model=model,
    tools=[search_document]
)

session = SQLiteSession("first_session")

print("Ask questions about your document (Ctrl+C to quit)\n")
while True:
    question = input("You: ")
    result = Runner.run_sync(agent, question, session=session)
    print("Agent: ", result.final_output)