"""
LEVEL 4 MULTI-AGENT SYSTEM — OLLAMA VERSION (local models).

This is the direct Ollama equivalent of agent_level4_multiagent_real_tools.py
— same architecture, same feature set, different backend:

    Feature                 | Anthropic version      | This (Ollama) version
    -------------------------------------------------------------------------
    Specialist agents       | researcher/file_mgr/    | same 4 roles
                             | analyst/developer       |
    Planning + dependencies | yes                     | yes
    Parallel execution      | yes (ThreadPoolExecutor)| yes (same)
    Reviewer critique loop  | yes                     | yes
    Persistent memory       | JSON file               | same
    Web search              | Anthropic hosted tool   | DuckDuckGo API (free,
                             |                         | no key — since Ollama
                             |                         | has no hosted search)
    File tools               | sandboxed to workspace/ | same
    Docker shell tool        | yes                     | same
    Cost / privacy           | cloud, per-token cost   | fully local, free,
                             |                         | private (except the
                             |                         | search tool call)

Setup:
    ollama pull llama3.1        # or qwen2.5 / mistral-nemo — tool-capable
    pip install ollama requests --break-system-packages
    (Docker required for the developer agent's sandbox, same as before)
"""

import json
import os
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import ollama
import requests

MODEL = "qwen2.5"     #"llama3.1"
MEMORY_FILE = "agent_memory_ollama.json"
WORKSPACE = Path("./workspace").resolve()
WORKSPACE.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# REAL TOOL: WEB SEARCH — Ollama has no hosted search tool, so this is
# a real local function hitting DuckDuckGo's free Instant Answer API.
# For more thorough results, swap this for Tavily/SerpAPI/Brave Search
# (all offer a free tier and a proper results list, unlike DDG's IA API).
# ----------------------------------------------------------------------

def tool_web_search(query: str) -> str:
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=10,
        )
        data = resp.json()
        pieces = []
        if data.get("AbstractText"):
            pieces.append(data["AbstractText"])
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                pieces.append(topic["Text"])
        return "\n".join(pieces) if pieces else "No results found."
    except Exception as e:
        return f"Error: {e}"


# ----------------------------------------------------------------------
# REAL TOOL: FILE SYSTEM (identical sandbox logic to the Anthropic version)
# ----------------------------------------------------------------------

def _safe_path(filename: str) -> Path:
    target = (WORKSPACE / filename).resolve()
    if WORKSPACE not in target.parents and target != WORKSPACE:
        raise ValueError("Path escapes the sandboxed workspace — blocked.")
    return target


def tool_read_file(filename: str) -> str:
    try:
        path = _safe_path(filename)
        if not path.exists():
            return f"Error: '{filename}' does not exist in workspace."
        return path.read_text()
    except Exception as e:
        return f"Error: {e}"


def tool_write_file(filename: str, content: str) -> str:
    try:
        path = _safe_path(filename)
        path.write_text(content)
        return f"Wrote {len(content)} characters to '{filename}'."
    except Exception as e:
        return f"Error: {e}"


def tool_list_directory() -> str:
    files = [f.name for f in WORKSPACE.iterdir()]
    return json.dumps(files) if files else "(workspace is empty)"


# ----------------------------------------------------------------------
# REAL TOOL: CALCULATOR
# ----------------------------------------------------------------------

def tool_calculator(expression: str) -> str:
    try:
        allowed = "0123456789+-*/(). "
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# ----------------------------------------------------------------------
# REAL TOOL: DOCKER SANDBOX (identical isolation to the Anthropic version)
# ----------------------------------------------------------------------

def tool_run_shell(command: str, timeout_seconds: int = 15) -> str:
    if len(command) > 2000:
        return "Error: command too long, refusing."
    docker_cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--memory", "256m",
        "--cpus", "0.5",
        "--read-only",
        "--tmpfs", "/tmp",
        "-v", f"{WORKSPACE}:/workspace",
        "-w", "/workspace",
        "--user", "nobody",
        "python:3.12-alpine",
        "sh", "-c", command,
    ]
    try:
        result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=timeout_seconds)
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout_seconds}s"
    except FileNotFoundError:
        return "Error: Docker is not installed or not on PATH."
    except Exception as e:
        return f"Error: {e}"


ALL_TOOL_FUNCTIONS = {
    "web_search": tool_web_search,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "list_directory": tool_list_directory,
    "calculator": tool_calculator,
    "run_shell": tool_run_shell,
}


def execute_tool(name: str, args: dict) -> str:
    if name not in ALL_TOOL_FUNCTIONS:
        return f"Error: unknown tool '{name}'"
    return ALL_TOOL_FUNCTIONS[name](**args)


# ----------------------------------------------------------------------
# SPECIALIZED AGENTS — same 4 roles as the Anthropic version, schemas
# rewritten in Ollama's OpenAI-style function-calling format.
# ----------------------------------------------------------------------

AGENTS = {
    "researcher": {
        "system": "You are a research agent. Use web_search to find current, "
                  "accurate information.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
    },
    "file_manager": {
        "system": "You are a file-management agent. Use read_file, write_file, "
                  "and list_directory. Only operate within the workspace.",
        "tools": [
            {"type": "function", "function": {
                "name": "read_file", "description": "Read a file from the workspace.",
                "parameters": {"type": "object", "properties": {"filename": {"type": "string"}},
                                "required": ["filename"]}}},
            {"type": "function", "function": {
                "name": "write_file", "description": "Write content to a file in the workspace.",
                "parameters": {"type": "object", "properties": {
                    "filename": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["filename", "content"]}}},
            {"type": "function", "function": {
                "name": "list_directory", "description": "List files in the workspace.",
                "parameters": {"type": "object", "properties": {}}}},
        ],
    },
    "analyst": {
        "system": "You are an analysis agent. Use calculator for arithmetic.",
        "tools": [
            {"type": "function", "function": {
                "name": "calculator", "description": "Evaluate an arithmetic expression.",
                "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                                "required": ["expression"]}}},
        ],
    },
    "developer": {
        "system": "You are a developer agent. Use run_shell to execute commands "
                  "inside an isolated, network-disabled container.",
        "tools": [
            {"type": "function", "function": {
                "name": "run_shell", "description": "Run a shell command in a sandboxed container.",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                                "required": ["command"]}}},
        ],
    },
}


# ----------------------------------------------------------------------
# SUBTASK EXECUTION — same tool loop shape as the Anthropic version,
# adapted to Ollama's message/response format.
# ----------------------------------------------------------------------

def run_subtask(agent_role: str, subtask: str, context: str, max_steps: int = 5) -> dict:
    agent = AGENTS[agent_role]
    messages = [
        {"role": "system", "content": agent["system"]},
        {"role": "user", "content": f"Context so far:\n{context}\n\nYour subtask: {subtask}\n"
                                     f"Complete it. If you cannot, say so clearly."},
    ]

    for _ in range(max_steps):
        response = ollama.chat(model=MODEL, messages=messages, tools=agent["tools"])
        message = response["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            final_text = message.get("content", "")
            failed = "cannot" in final_text.lower() or "unable" in final_text.lower()
            return {"status": "failed" if failed else "done", "result": final_text}

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            print(f"      [{agent_role} tool: {name}({args})]")
            result = execute_tool(name, args)
            messages.append({"role": "tool", "content": str(result)})

    return {"status": "failed", "result": "Max steps reached without completion."}


# ----------------------------------------------------------------------
# REVIEWER AGENT — same critique-and-retry pattern as the Anthropic version
# ----------------------------------------------------------------------

def review_result(subtask: str, result: dict) -> dict:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a strict reviewer. Respond with ONLY JSON: "
                                          '{"approved": true/false, "feedback": "..."}. '
                                          "Approve unless there's a real, specific deficiency."},
            {"role": "user", "content": f"Subtask: {subtask}\n\nResult: {result['result']}"},
        ],
    )
    text = response["message"].get("content", "").strip().strip("```json").strip("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"approved": True, "feedback": ""}


def run_subtask_with_review(agent_role: str, subtask: str, context: str, max_retries: int = 2) -> dict:
    feedback = ""
    for _ in range(max_retries + 1):
        task_prompt = subtask if not feedback else f"{subtask}\n\n(Reviewer feedback: {feedback})"
        result = run_subtask(agent_role, task_prompt, context)
        if result["status"] == "failed":
            return result
        review = review_result(subtask, result)
        if review["approved"]:
            return result
        print(f"      [reviewer rejected: {review['feedback']}]")
        feedback = review["feedback"]
    result["status"] = "done"
    result["result"] += "\n(Note: accepted after max review retries.)"
    return result


# ----------------------------------------------------------------------
# PERSISTENT MEMORY
# ----------------------------------------------------------------------

def load_memory() -> list[dict]:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_memory(entry: dict):
    history = load_memory()
    history.append(entry)
    with open(MEMORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def memory_summary(history: list[dict], limit: int = 5) -> str:
    if not history:
        return "(no past history)"
    return "\n".join(f"- {h['goal']} -> {h['answer'][:150]}" for h in history[-limit:])


# ----------------------------------------------------------------------
# PLANNING — assigns each subtask to a specialist + tracks dependencies
# ----------------------------------------------------------------------

def make_plan(goal: str, history_summary: str) -> list[dict]:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": (
                "You are an orchestrator. Break the goal into concrete subtasks. "
                "For each subtask specify: 'task', 'agent' (one of 'researcher', "
                "'file_manager', 'analyst', 'developer'), and 'depends_on' (0-based "
                "indices, empty if independent). Respond with ONLY a JSON array: "
                '[{"task": "...", "agent": "...", "depends_on": []}, ...]'
            )},
            {"role": "user", "content": f"Past history:\n{history_summary}\n\nGoal: {goal}"},
        ],
    )
    text = response["message"].get("content", "").strip().strip("```json").strip("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [{"task": goal, "agent": "analyst", "depends_on": []}]


# ----------------------------------------------------------------------
# PARALLEL EXECUTION (identical wave logic to the Anthropic version)
# ----------------------------------------------------------------------

def run_plan_parallel(plan: list[dict]) -> list[dict]:
    results, context_pieces = {}, {}
    pending = set(range(len(plan)))

    while pending:
        ready = [i for i in pending if all(d in results for d in plan[i]["depends_on"])]
        if not ready:
            for i in pending:
                results[i] = {"status": "failed", "result": "Unresolved dependency."}
            break

        print(f"  Parallel wave: {[(plan[i]['agent'], plan[i]['task']) for i in ready]}")
        with ThreadPoolExecutor(max_workers=len(ready)) as pool:
            futures = {}
            for i in ready:
                dep_context = "\n".join(context_pieces.get(d, "") for d in plan[i]["depends_on"])
                futures[pool.submit(run_subtask_with_review, plan[i]["agent"], plan[i]["task"], dep_context)] = i
            for future in as_completed(futures):
                i = futures[future]
                result = future.result()
                results[i] = result
                context_pieces[i] = f"[{plan[i]['task']}] -> {result['result']}"
                print(f"    done: {plan[i]['task']} -> {result['status']}")

        pending -= set(ready)

    return [{"subtask": plan[i]["task"], "agent": plan[i]["agent"], "result": results[i]} for i in range(len(plan))]


# ----------------------------------------------------------------------
# TOP-LEVEL AGENT LOOP
# ----------------------------------------------------------------------

def run_agent(goal: str) -> str:
    history = load_memory()
    hist_summary = memory_summary(history)

    print(f"\nGOAL: {goal}")
    plan = make_plan(goal, hist_summary)
    print(f"PLAN: {plan}\n")

    completed = run_plan_parallel(plan)
    context = "\n".join(f"[{c['agent']}: {c['subtask']}] -> {c['result']['result']}" for c in completed)

    synthesis = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Summarize multi-agent subtask results into one clear final answer."},
            {"role": "user", "content": f"Goal: {goal}\n\nResults:\n{context}\n\nFinal answer:"},
        ],
    )
    answer = synthesis["message"].get("content", "")

    save_memory({"goal": goal, "answer": answer, "timestamp": datetime.now().isoformat()})
    return answer


if __name__ == "__main__":
    print(f"Level 4 multi-agent system running on Ollama ({MODEL})")
    print(f"Workspace sandbox: {WORKSPACE}")
    print("Type 'quit' to exit, 'history' for past goals.\n")
    while True:
        goal = input("Goal: ")
        if goal.lower() in ("quit", "exit"):
            break
        if goal.lower() == "history":
            for h in load_memory():
                print(f"- [{h['timestamp']}] {h['goal']} -> {h['answer'][:150]}")
            continue
        answer = run_agent(goal)
        print(f"\nFINAL ANSWER: {answer}\n")
