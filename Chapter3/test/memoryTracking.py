import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool, ModelSettings

load_dotenv(find_dotenv())

external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="llama-3.1-8b-instant",
    openai_client=external_client
)
agent = Agent(
    name="QuestionAnswer",
    instructions="You are an AI agent that answers questions.",
    model=llm_model
)

messages = []

messages.append({"role": "user", "content": "How hot is the sun?"})

result = Runner.run_sync(agent, messages)
print(result.final_output)

messages.append({"role": "assistant", "content": result.final_output})

messages.append({"role": "user", "content": "How big is it?"})

result = Runner.run_sync(agent, messages)
print(result.final_output)
