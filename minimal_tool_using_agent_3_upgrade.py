"""
LEVEL 3 AGENT — Planning + Task Decomposition, built from scratch.

Builds directly on the Level 2 agent (tools + loop). What's new:

  1. A PLANNING step: before doing anything, the model breaks the goal
     into an ordered list of subtasks.
  2. A TASK TRACKER: subtask status (pending / done / failed) is kept
     as explicit state, not just buried in chat history.
  3. EXECUTION per subtask: each subtask runs its own Level-2 tool loop.
  4. REPLANNING: if a subtask fails, the agent revises the remaining
     plan instead of blindly continuing.
  5.  Read your files

  6. Access your XAMPP/MySQL database

  7. Browse Google  pip install duckduckgo-search

Requires: Requires: python -m pip install ollama   
ollama pull qwen2.5   

create virtual env.  with python -m venv .venv

.\.venv\Scripts\Activate.ps1

This activates the virtual environment in Windows PowerShell.
"""


import json
import os
from datetime import datetime
import mysql.connector
#from duckduckgo_search import DDGS
from ddgs import DDGS


import ollama

MODEL = "qwen2.5"



# ----------------------------------------------------------------------
# TOOLS (same as Level 2 — the agent's available actions)
# ----------------------------------------------------------------------

BASE = "C:/xampp/htdocs/gitReps"
#read the folder gitReps only
def tool_list_files(path="."):
    base = os.path.abspath(BASE)
    folder = os.path.abspath(os.path.join(base, path))

    if not folder.startswith(base):
        return "Access denied."

    try:
        return "\n".join(os.listdir(folder))
    except Exception as e:
        return str(e)


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


def tool_search_notes(query: str) -> str:
    """Fake 'knowledge base' search, stands in for a real tool (e.g. web search, DB)."""
    fake_notes = {
        "budget": "Q3 budget is $50,000, already spent $32,000.",
        "deadline": "Project deadline is August 15.",
        "team": "Team has 4 engineers and 1 designer.",
    }
    for key, val in fake_notes.items():
        if key in query.lower():
            return val
    return "No matching notes found."


def tool_mysql(query):
    if not query.lower().startswith("select"):
        return "Only SELECT queries are allowed."
    
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="mydatabase"
        )

        cursor = db.cursor()
        cursor.execute(query)

        rows = cursor.fetchall()

        cursor.close()
        db.close()

        return str(rows)

    except Exception as e:
        return str(e)
    
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'db' in locals():
            db.close()


def tool_list_tables():
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="mydatabase"
        )

        cursor = db.cursor()
        cursor.execute("SHOW TABLES")

        tables = [row[0] for row in cursor.fetchall()]

        cursor.close()
        db.close()

        return str(tables)

    except Exception as e:
        return str(e)


def tool_describe_table(table):
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="mydatabase"
        )

        cursor = db.cursor()
        cursor.execute(f"SHOW COLUMNS FROM `{table}`")

        columns = cursor.fetchall()

        cursor.close()
        db.close()

        return str(columns)

    except Exception as e:
        return str(e)




def tool_search_web(query: str) -> str:
    results = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(f"{r['title']}: {r['body']}")

        if not results:
            return "No results found."

        return "\n".join(results)

    except Exception as e:
        return f"Web search failed: {e}"


    return "\n".join(results)



TOOL_FUNCTIONS = {
    "calculator": tool_calculator,
    "get_time": tool_get_time,
    "search_notes": tool_search_notes,
    "list_files": tool_list_files,
    #"read_file": tool_read_file,
    "mysql": tool_mysql,
    "search_web": tool_search_web,

    "list_tables": tool_list_tables,
    "describe_table": tool_describe_table,
}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description":
                "Search the internal project notes. Use this for budget, deadlines, team information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string"
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
            "description": "Get current date and time.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Search internal notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": ["query"]
            }
        }
    },

    {
    "type": "function",
    "function": {
        "name": "read_file",
       "description":"Read a UTF-8 text file from the project directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string"
                }
            },
            "required": ["filename"]
        }
    }
},


    {
    "type": "function",
    "function": {
        "name": "mysql",
        "description": "Execute a SELECT query on the MySQL database.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string"
                }
            },
            "required": ["query"]
        }
    }
},

{
    "type": "function",
    "function": {
        "name": "list_tables",
        "description": "List every table in the MySQL database.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
},

{
    "type": "function",
    "function": {
        "name": "describe_table",
        "description": "Show the columns of a MySQL table.",
        "parameters": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string"
                }
            },
            "required": ["table"]
        }
    }
},


    {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the internet for current information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    }
}

]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name not in TOOL_FUNCTIONS:
        return f"Error: unknown tool '{tool_name}'"
    return TOOL_FUNCTIONS[tool_name](**tool_input)


# ----------------------------------------------------------------------
# LEVEL 2 CORE: run one subtask through a tool-use loop until it
# produces a final text answer (or hits max_steps).
# ----------------------------------------------------------------------


def run_subtask(subtask: str, context: str, max_steps=10):

    messages = [
        {
            "role": "user",
            "content":
            f"""
Overall context:
{context}

Current subtask:
{subtask}

Available tools:

Available tools:

- calculator
- get_time
- search_notes
- read_file
- mysql
- list_tables
- describe_table
- search_web

You are executing ONE subtask.

Use tools whenever information is needed.

Do not guess.

If a tool can answer the question, call the tool.

After receiving tool results, continue until the task is complete.

"""
        }
    ]


    for _ in range(max_steps):

        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS
        )


        message = response["message"]

        messages.append(message)


        tool_calls = message.get(
            "tool_calls",
            []
        )


        if not tool_calls:

            return {
                "status": "done",
                "result": message["content"]
            }


        for call in tool_calls:

            name = call["function"]["name"]

            args = call["function"]["arguments"]


            print(
                f"    [tool call: {name}({args})]"
            )


            result = execute_tool(
                name,
                args
            )


            messages.append(
                {
                    "role":"tool",
                    "content":str(result)
                }
            )


    return {
        "status":"failed",
        "result":"max steps reached"
    }

# ----------------------------------------------------------------------
# PLANNING: ask the model to decompose the goal into ordered subtasks.
# Uses structured JSON output so we get a real task list, not prose.
# ----------------------------------------------------------------------

def make_plan(goal):

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content":
                """
You can use these tools:

- calculator
- get_time
- search_notes
- read_file
- mysql
- search_web

Break the user's goal into the smallest useful ordered tasks.

Choose tasks that can be completed using these tools:

- calculator
- get_time
- search_notes
- read_file
- mysql
- search_web

Return ONLY a JSON array.

Example:

[
"Read config.json",
"Search the web for the latest PHP version",
"Compare with config"
]

"""
            },
            {
                "role": "user",
                "content": goal
            }
        ]
    )

    text = response["message"]["content"]

    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    try:
        return json.loads(text)
    except:
        return [goal]



# ----------------------------------------------------------------------
# THE LEVEL 3 AGENT LOOP: plan -> execute subtasks -> replan on failure
# -> synthesize final answer once all subtasks are handled.
# ----------------------------------------------------------------------

def run_agent(goal: str) -> str:
    print(f"\nGOAL: {goal}")
    plan = make_plan(goal)
    print(f"PLAN: {plan}\n")

    completed = []
    context = ""
    i = 0

    while i < len(plan):
        subtask = plan[i]
        print(f"  -> Executing subtask {i+1}/{len(plan)}: {subtask}")
        result = run_subtask(subtask, context)
        completed.append({"subtask": subtask, "result": result})
        print(f"     status: {result['status']}")

        if result["status"] == "failed":
            remaining = plan[i + 1:]
            print("     Subtask failed — replanning remaining steps...")
            new_remaining = replan(goal, completed, remaining, completed[-1])
            plan = plan[: i + 1] + new_remaining
            print(f"     Revised plan: {plan}")

        context += f"\n[{subtask}] -> {result['result']}"
        i += 1

    # Final synthesis: turn all subtask results into one coherent answer
    synthesis = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content":
                "Summarize the results of a multi-step task into one clear, direct answer."
            },
            {
                "role": "user",
                "content":
                f"""
        Goal:
        {goal}

        Subtask results:
        {context}

        Give the final answer.
        """
                    }
                ]
            )

    return synthesis["message"]["content"]

    # ----------------------------------------------------------------------
# THE LEVEL 3 AGENT LOOP: plan -> execute subtasks -> replan on failure
# -> synthesize final answer once all subtasks are handled.
# ----------------------------------------------------------------------
def replan(goal, completed, remaining, failure):

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content":
                """
You are a replanning module.

A task failed.
Create a new list of tasks.

Return ONLY a JSON array.
"""
            },
            {
                "role": "user",
                "content":
                f"""
Goal:
{goal}

Failed task:
{failure['subtask']}

Failure:
{failure['result']['result']}

Remaining tasks:
{remaining}
"""
            }
        ]
    )


    text = response["message"]["content"]

    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()


    try:
        return json.loads(text)

    except:
        return remaining


# ----------------------------------------------------------------------
# RUN IT
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Level 3 planning agent. Type 'quit' to exit.\n")
    while True:
        goal = input("Goal: ")
        if goal.lower() in ("quit", "exit"):
            break
        answer = run_agent(goal)
        print(f"\nFINAL ANSWER: {answer}\n")