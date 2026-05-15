import os
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI,set_tracing_disabled

load_dotenv()  # load .env file
set_tracing_disabled(disabled=True)
# Access the API key
api_key = os.getenv("GEMINI_API_KEY")

# Create client for Gemini (OpenAI format)
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Select model
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

# Create simple agent
agent = Agent(
    name="HelloAgent",
    instructions="Just reply with: Setup successful",
    model=model
)




# Run agent
#result = Runner.run_sync(agent, "Hello")
#print("Agent Output:", result.final_output)
# Ask user for input
user_message = input("Enter your message for the agent: ")

# Run agent
result = Runner.run_sync(agent, user_message)

# Print final result
print("Agent Output:", result.final_output)