"""
WEB CHAT INTERFACE for the Ollama-based Level 4 multi-agent system.
Direct equivalent of web_app.py, pointed at agent_ollama_level4_multiagent.py
instead of the Anthropic-backed core module.

Run with:
    pip install flask ollama requests --break-system-packages
    python web_app_ollama.py
Then open http://localhost:5000
"""

from flask import Flask, request, jsonify, Response
import threading

import agent_ollama_level4_multiagent as core

app = Flask(__name__)

progress_log = []
log_lock = threading.Lock()


def log(message: str):
    with log_lock:
        progress_log.append(message)


# Redirect the core module's print() calls into our progress log,
# same trick as the Anthropic version's web_app.py.
core.print = lambda *args, **kwargs: log(" ".join(str(a) for a in args))

CHAT_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Multi-Agent Chat (Ollama / Local)</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 700px; margin: 40px auto; }
    #log { background: #f4f4f4; padding: 12px; height: 200px; overflow-y: auto;
           font-family: monospace; font-size: 12px; white-space: pre-wrap; border-radius: 6px; }
    #answer { background: #eef7ee; padding: 14px; margin-top: 12px; border-radius: 6px; display: none; }
    input[type=text] { width: 80%; padding: 8px; }
    button { padding: 8px 16px; }
  </style>
</head>
<body>
  <h2>Multi-Agent Chat — running locally on Ollama</h2>
  <p>Agents: researcher (DuckDuckGo search), file_manager (sandboxed files),
     analyst (calculator), developer (Docker-sandboxed shell). Model: """ + core.MODEL + """</p>
  <input type="text" id="goal" placeholder="Type a goal..." />
  <button onclick="submitGoal()">Run</button>
  <h4>Progress</h4>
  <div id="log"></div>
  <div id="answer"></div>

  <script>
    let polling = null;
    async function submitGoal() {
      const goal = document.getElementById('goal').value;
      if (!goal) return;
      document.getElementById('log').textContent = '';
      document.getElementById('answer').style.display = 'none';
      const resp = await fetch('/run', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({goal})
      });
      const {run_id} = await resp.json();
      polling = setInterval(() => pollStatus(run_id), 1000);
    }
    async function pollStatus(runId) {
      const resp = await fetch('/status/' + runId);
      const data = await resp.json();
      document.getElementById('log').textContent = data.log.join('\\n');
      document.getElementById('log').scrollTop = 999999;
      if (data.done) {
        clearInterval(polling);
        const answerDiv = document.getElementById('answer');
        answerDiv.textContent = data.answer;
        answerDiv.style.display = 'block';
      }
    }
  </script>
</body>
</html>
"""

runs = {}
run_counter = 0
run_lock = threading.Lock()


@app.route("/")
def index():
    return Response(CHAT_HTML, mimetype="text/html")


@app.route("/run", methods=["POST"])
def start_run():
    global run_counter
    goal = request.json.get("goal", "")

    with run_lock:
        run_counter += 1
        run_id = run_counter
        runs[run_id] = {"done": False, "answer": None, "log_start": len(progress_log)}

    def worker():
        answer = core.run_agent(goal)
        runs[run_id]["answer"] = answer
        runs[run_id]["done"] = True

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"run_id": run_id})


@app.route("/status/<int:run_id>")
def status(run_id):
    if run_id not in runs:
        return jsonify({"error": "unknown run_id"}), 404
    run = runs[run_id]
    with log_lock:
        run_log = progress_log[run["log_start"]:]
    return jsonify({"done": run["done"], "answer": run["answer"], "log": run_log})


if __name__ == "__main__":
    print(f"Workspace sandbox: {core.WORKSPACE}")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
