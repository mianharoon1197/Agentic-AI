import os
import asyncio
from dotenv import load_dotenv, find_dotenv
import chromadb
from chromadb.utils import embedding_functions
from agents import (
    Agent, 
    Runner, 
    OpenAIChatCompletionsModel, 
    AsyncOpenAI, 
    set_tracing_disabled,
    function_tool
)
load_dotenv(find_dotenv())
set_tracing_disabled(disabled=True)

# 1. Connect to the local Vector Database
chroma_client = chromadb.PersistentClient(path="./local_rag_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_collection(name="national_ai_policiy", embedding_function=sentence_transformer_ef)

# 2. Custom retrieval tool 
@function_tool
def search_local_documents(query: str) -> str:
    """Searches the local knowledge base for relevant information to answer user questions."""
    results = collection.query(
        query_texts=[query],
        n_results=5 # Increased to 5 chunks for better context
    )
    
    if not results['documents'][0]:
        return "No relevant information found in the documents."
        
    retrieved_context = ""
    for i in range(len(results['documents'][0])):
        chunk_text = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        retrieved_context += f"[Source: {metadata['source']}, Page: {metadata['page']}]\n{chunk_text}\n---\n"
        
    return f"Retrieved Context:\n{retrieved_context}"

external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-3.1-flash-lite-preview",
    openai_client=external_client
)

async def main():
    print("Initializing Local RAG Agent...")
    rag_agent = Agent(
        name="LocalDocAssistant",
        instructions="""You are an AI assistant helping a user understand a document. 
        Answer the user's questions based ONLY on the information retrieved from the search_local_documents tool. 
        If the answer is not in the retrieved context, say you don't know. 
        IMPORTANT: Output strictly in plain text. Do NOT use markdown formatting (like asterisks for bolding, bullet points, or hashtags for headers). Write in clear, readable paragraphs. You can also cite the page number if it is helpful.""",
        model=llm_model,
        tools=[search_local_documents]
    )

    print("\nAgent is ready! Type 'exit' or 'quit' to stop.")
    while True:
        question = input("\nYou: ")
        
        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not question.strip():
            continue
            
        print("Agent is thinking...")
        result = await Runner.run(rag_agent, question)
        
        print(f"\nAgent: {result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())