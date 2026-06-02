from agents import Agent, Runner, function_tool, trace, handoff,SQLiteSession
from pydantic import BaseModel
import os

complaint_agent = Agent(
    name="ComplaintAgent",
    instructions="You are a customer service agent. Handle any customer complaints with empathy and clear the next step."
)

sales_agent = Agent(
    name="SalesAgent",
    instructions="You are a sales agent. Handle any customer inquiries about products and provide clear information."
)

triage_agent = Agent(
    name="TriageAgent",
    instructions="You are a triage agent. Determine the appropriate agent to handle each customer inquiry.",
    handoffs = [sales_agent, complaint_agent]
)

while True:
    user_message = input("Enter your message: ")
    result = Runner.run_sync(triage_agent, user_message)
    print("Agent Output:", result.final_output)