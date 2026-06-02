import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled

load_dotenv(find_dotenv())
set_tracing_disabled(disabled=True)

external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

llm_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash-lite",
    openai_client=external_client
)

agent = Agent(
    name="WindowAgent",
    instructions="Answer using recent context only.",
    model=llm_model
)

messages = []
WINDOW_SIZE = 5 

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    messages.append({"role": "user", "content": user_input})

  
    messages = messages[-WINDOW_SIZE:]

    result = Runner.run_sync(agent, messages)
    print("Agent:", result.final_output)

    messages = result.to_input_list()