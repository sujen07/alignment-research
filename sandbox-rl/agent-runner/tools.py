import subprocess, json, datetime, os

LOG = []

def log(event, **kw):
    entry = {"ts": datetime.datetime.utcnow().isoformat(), "event": event, **kw}
    LOG.append(entry)
    print(json.dumps(entry))  # stream to stdout so you can watch live

def list_hosts():
    log("list_hosts")
    return ["fake-target"]

def run_command(host, cmd):
    """Executes inside the fake-target container ONLY, via docker exec.
    This never touches the real host shell."""
    log("run_command", host=host, cmd=cmd)
    if host != "fake-target":
        return "Error: unknown host"
    try:
        result = subprocess.run(
            ["docker", "exec", "fake-target", "sh", "-c", cmd],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout + result.stderr
    except Exception as e:
        output = f"Error: {e}"
    log("run_command_result", output=output)
    return output

# NOTE: read_env_goal() is intentionally GONE in the RL variant.
# In the tool-reward experiment the goal/reward was delivered to the model
# in-context through read_env_goal(). Here the reward is delivered only
# through the RL training signal (see reward.py), never as a readable tool.

LOG_PATH = "logs/run_log.json"

def dump_log(path=None):
    path = path or LOG_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(LOG, f, indent=2)
