# ---------------------------------------------------
# tools.py
# This file defines tools that the agent can use.
# A "tool" is simply a Python function that is made
# available to the AI agent through a decorator.
# ---------------------------------------------------

from agents import function_tool

# ---------------------------------------------------
# Tool 1: get_order_status
# ---------------------------------------------------
# This tool pretends to look up an order in a database.
# The agent can call this tool whenever user asks for
# order status like: "Check my order status for order 3"
#
# @function_tool tells the Agents SDK:
#   -> "This function is a tool the AI can use"
#
# When the agent needs this tool, it automatically:
#   - prepares the input
#   - calls the function
#   - reads the output
#   - uses the output in the final answer
# ---------------------------------------------------

@function_tool
def get_order_status(order_id: int) -> str:
    """
    Returns the status of an order.

    Parameters:
        order_id (int): The ID of the order the user wants to check.

    Returns:
        str: The status of the order such as:
             Delivered / Shipped / Processing / Cancelled / Not found.
    """

    # Fake database of orders
    # In a real business, this would come from an actual database.
    fake_db = {
        1: "Delivered",
        2: "Shipped",
        3: "Processing",
        4: "Cancelled"
    }

    # Return the status for the given order ID.
    # If the order does not exist, return "Order not found".
    return fake_db.get(order_id, "Order not found")
