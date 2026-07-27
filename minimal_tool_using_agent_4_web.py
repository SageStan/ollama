"""
A minimal AI agent built from scratch — using OLLAMA (local models)
instead of the Anthropic API. Same concept as before: an LLM + tools +
a loop that lets the model take actions — just running fully locally,
no API key, no internet required for inference.

Setup:
    1. Install Ollama: https://ollama.com
    2. Pull a model that supports tool calling, e.g.:
         ollama pull llama3.1
       (qwen2.5, mistral-nemo, and firefunction-v2 also support tools;
       plain llama3/llama2/gemma do NOT — check a model's page on
       ollama.com for a "tools" capability tag before using it here)
    3. pip install ollama --break-system-packages
    4. Make sure the Ollama server is running (it starts automatically
       after install, or run `ollama serve`)

Key difference from the Anthropic version:
    - No API key, everything runs on your machine
    - Uses ollama.chat(..., tools=[...]) instead of client.messages.create
    - Tool call results come back as response['message']['tool_calls']
      instead of content blocks with type "tool_use"
    - No server-side tools (like Anthropic's hosted web_search) — every
      tool, including "search", must be implemented locally by you
"""

import json
from datetime import datetime
import ollama

MODEL = "llama3.1"  # must be a tool-calling-capable model you've pulled

# ----------------------------------------------------------------------
# 1. TOOLS — identical concept to the Anthropic version: real Python
#    functions plus a schema describing them to the model.
# ----------------------------------------------------------------------

def tool_calculator(expression: str) -> str:
    try:
        allowed = "0123456789+-*/(). "
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


def tool_get_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOL_FUNCTIONS = {
    "calculator": tool_calculator,
    "get_time": tool_get_time,
}

# Ollama uses the OpenAI-style function-calling schema shape:
# {"type": "function", "function": {name, description, parameters}}
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
                        "description": "e.g. '2 + 2 * 3'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name not in TOOL_FUNCTIONS:
        return f"Error: unknown tool '{tool_name}'"
    return TOOL_FUNCTIONS[tool_name](**tool_input)


# ----------------------------------------------------------------------
# 2. MEMORY — same idea: a growing list of messages is the memory.
# ----------------------------------------------------------------------

conversation_history = []


# ----------------------------------------------------------------------
# 3 & 4. THE AGENT LOOP — call the model, check for tool_calls, execute
# them, feed results back, repeat until the model answers in plain text.
# ----------------------------------------------------------------------

def run_agent(user_message: str, max_steps: int = 6) -> str:
    conversation_history.append({"role": "user", "content": user_message})

    for step in range(max_steps):
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to tools. "
                               "Use them when needed to answer accurately.",
                },
                *conversation_history,
            ],
            tools=TOOL_SCHEMAS,
        )

        message = response["message"]
        conversation_history.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            # No tool call -> model gave its final answer
            return message.get("content", "")

        # Execute every requested tool call and feed results back
        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            # Ollama may give args as a dict already, or as a JSON string
            if isinstance(args, str):
                args = json.loads(args)

            print(f"  [agent is calling tool: {name}({args})]")
            result = execute_tool(name, args)

            conversation_history.append(
                {
                    "role": "tool",
                    "content": str(result),
                }
            )
        # loop continues -> model sees tool results, decides next step

    return "Max steps reached without a final answer."


# ----------------------------------------------------------------------
# 5. RUN IT
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Local agent running on Ollama model: {MODEL}")
    print("Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break
        answer = run_agent(user_input)
        print(f"Agent: {answer}\n")