import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool
import json

load_dotenv(find_dotenv())

external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"), 
    base_url="https://api.groq.com/openai/v1",
)

llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="llama-3.1-8b-instant",
    openai_client=external_client
)

FILENAME = 'memory.json'
memory_default = {
    "user_profile": [],
    "order_preferences": [],
    "other": []
}

if not os.path.exists(FILENAME):
    with open(FILENAME, 'w') as f:
        json.dump(memory_default, f, indent=4)
    print(f"Created '{FILENAME}' with default data.")
else:
    print(f"'{FILENAME}' already exists.")


@function_tool
def save_memory(memory_type: str, memory: str) -> str:
    """
    Saves a memory to a memory store.

    Args:
        memory_type: the type of memory to store. Choose between user_profile, order_preferences, or other.
        memory: the memory to save
    """
    with open(FILENAME, 'r') as f:
        data = json.load(f)
    data[memory_type].append(memory)

    with open(FILENAME, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Memory ({memory}) saved")
    return f"Memory ({memory}) saved"


@function_tool
def load_memory(memory_type: str) -> str:
    """
    Loads a set of memory from a memory store.

    Args:
        memory_type: the type of memory to load. Choose between user_profile, order_preferences, or other.
    """
    with open(FILENAME, 'r') as f:
        data = json.load(f)
    return "|".join(data[memory_type])


agent = Agent(
    name="QuestionAnswer",
    instructions=(
        "You are an AI agent that answers questions using only the knowledge you have. "
        "You have access to two tools: save_memory and load_memory. "
        "Use save_memory to store important facts you learn about the user. "
        "Use load_memory to retrieve previously saved information about the user. "
        "You do NOT have access to web search or any other external tools. "
        "Answer questions based on your training knowledge and loaded memories only."
    ),
    model=llm_model,
    tools=[save_memory, load_memory]
)

while True:
    question = input("You: ")
    result = Runner.run_sync(agent, question)
    print("Agent: ", result.final_output)
