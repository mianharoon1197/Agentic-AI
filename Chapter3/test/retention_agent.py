import os
from dotenv import load_dotenv
from agents import Agent, AsyncOpenAI, Runner, OpenAIChatCompletionsModel

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

retention_agent = Agent(
    name="RetentionAgent",
    instructions="""
        You are a retention specialist.
        Your job is to convince the customer to stay.
        You may offer:
        - 10% discount
        - Faster delivery
        - Free support upgrade
    """,
    model=model
)
