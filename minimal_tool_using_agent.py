"""
A minimal AI agent built from scratch — no agent frameworks.
Just: an LLM + tools + a loop that lets the LLM take actions.

Requires: pip install anthropic --break-system-packages
Set your API key as an environment variable: ANTHROPIC_API_KEY
"""

import json
import os
from datetime import datetime
import ollama

MODEL = "qwen2.5"


#client = Anthropic()  # reads ANTHROPIC_API_KEY from env
#MODEL = "claude-sonnet-4-5"

# ----------------------------------------------------------------------
# 1. TOOLS — the actions the agent is allowed to take.
#    Each tool = a schema (so the model knows it exists) + real Python code.
# ----------------------------------------------------------------------

def tool_calculator(expression: str) -> str:
    """Safely evaluate a basic math expression."""
    try:
        # Only allow digits, operators, parentheses — no arbitrary code exec
        allowed = "0123456789+-*/(). "
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


def tool_get_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Registry mapping tool name -> Python function
TOOL_FUNCTIONS = {
    "calculator": tool_calculator,
    "get_time": tool_get_time,
}

# Schemas the model sees — this is how it knows what tools exist and
# what arguments each one needs.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Example: 2 + 2 * 3"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


# ----------------------------------------------------------------------
# 2. MEMORY — conversation history is just a growing list of messages.
#    This IS the agent's "memory" for the session (short-term memory).
# ----------------------------------------------------------------------

conversation_history = []


# ----------------------------------------------------------------------
# 3. TOOL EXECUTION — dispatch a tool call the model requested,
#    run the real code, and package the result to send back.
# ----------------------------------------------------------------------

def execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name not in TOOL_FUNCTIONS:
        return f"Error: unknown tool '{tool_name}'"
    fn = TOOL_FUNCTIONS[tool_name]
    return fn(**tool_input)


# ----------------------------------------------------------------------
# 4. THE AGENT LOOP — this is the actual "agentic" part.
#    Call the model -> if it wants a tool, run it and feed the result
#    back -> repeat until the model gives a final text answer.
# ----------------------------------------------------------------------

def run_agent(user_message: str, max_steps: int = 6):

    conversation_history.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    for step in range(max_steps):

        response = ollama.chat(
            model=MODEL,
            messages=conversation_history,
            tools=TOOL_SCHEMAS,
        )

        message = response["message"]

        conversation_history.append(message)

        tool_calls = message.get("tool_calls", [])


        if not tool_calls:
            return message["content"]


        for call in tool_calls:

            name = call["function"]["name"]

            arguments = call["function"]["arguments"]


            print(
                f"[agent calling tool: {name} {arguments}]"
            )


            result = execute_tool(
                name,
                arguments
            )


            conversation_history.append(
                {
                    "role": "tool",
                    "content": str(result)
                }
            )


    return "Max steps reached without a final answer."

 

# ----------------------------------------------------------------------
# 5. RUN IT
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Simple from-scratch agent. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break
        answer = run_agent(user_input)
        print(f"Agent: {answer}\n")