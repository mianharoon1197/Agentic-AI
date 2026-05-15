import os
from dotenv import load_dotenv

from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    SQLiteSession,
    OpenAIChatCompletionsModel,
    set_tracing_disabled
)

# Load .env
load_dotenv()

# Disable tracing
set_tracing_disabled(True)

# Gemini client
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Gemini model
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash-lite",
    openai_client=client
)

# Agent
agent = Agent(
    name="QuestionAnswerer",
    instructions="You are a helpful assistant.",
    model=model
)

# Session memory
session = SQLiteSession("first_session")

# Chat loop
while True:
    question = input("Ask a question (or 'exit' to quit): ")

    if question.lower() == "exit":
        break

    result = Runner.run_sync(
        agent,
        question,
        session=session
    )

    print(f"\nAnswer: {result.final_output}\n")