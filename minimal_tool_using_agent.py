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
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "e.g. '2 + 2 * 3'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current date and time.",
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

def run_agent(user_message: str, max_steps: int = 6) -> str:
    conversation_history.append({"role": "user", "content": user_message})

    for step in range(max_steps):
        response = ollama.chat(
                   model=MODEL,
                   messages=conversation_history,
                   tools=TOOL_SCHEMAS,
               )

        message = response["message"]
        
    """  response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system="You are a helpful assistant with access to tools. "
                   "Use them when needed to answer accurately.",
            messages=conversation_history,
            tools=TOOL_SCHEMAS,
        ) """

           


        # Save the assistant's turn (may contain text + tool_use blocks)
    conversation_history.append(
            {"role": "assistant", "content": response.content}
        )

        # Did the model ask to use a tool?
       # tool_calls = [b for b in response.content if b.type == "tool_use"]

    tool_calls = response["message"].get("tool_calls", [])


    if not tool_calls:
            # No tool call -> model gave its final answer, we're done
            final_text = "".join(
                b.text for b in response.content if b.type == "text"
            )
            return final_text

        # Execute every requested tool call and feed results back
    tool_results = []
    for call in tool_calls:
                name = call["function"]["name"]
                arguments = call["function"]["arguments"]

                print(f"calling tool: {name}({arguments})")

                result = execute_tool(name, arguments)

    """ for call in tool_calls:
            print(f"  [agent is calling tool: {call.name}({call.input})]")
            result = execute_tool(call.name, call.input) """


    tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": str(result),
                }
            )

    conversation_history.append({"role": "user", "content": tool_results})
        # loop continues -> model sees tool results, decides next step

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