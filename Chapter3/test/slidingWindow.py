import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from collections import deque

# Load environment variables from the .env file
load_dotenv(find_dotenv())
# set_tracing_disabled(disabled=True)
# Which LLM Provider to use? -> Groq Free API
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# 2. Which LLM Model to use?
llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="llama-3.1-8b-instant",
    openai_client=external_client
)
# Create the agent
agent = Agent(
    name="QuestionAnswer",
    instructions="You are an AI agent that answers questions.",
    model=llm_model
)

# Sliding window size (keep only the most recent 5 messages)
WINDOW_SIZE = 5
messages = deque(maxlen=WINDOW_SIZE)

while True:
    question = input("You: ")
    messages.append({"role": "user", "content": question})

    # Run the agent with only the most recent N messages
    result = Runner.run_sync(agent, list(messages))
    print("Agent:", result.final_output)

    messages.append({"role": "assistant", "content": result.final_output})
