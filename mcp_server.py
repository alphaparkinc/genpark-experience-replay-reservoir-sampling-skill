"""
MCP Server for Experience Replay Reservoir Sampling Skill
"""

import json
import sys
from client import ReservoirReplayBuffer

buf = ReservoirReplayBuffer(capacity=50)

def handle_call(name: str, args: dict) -> dict:
    if name == "push_experience":
        item = args.get("item", {})
        buf.add(item)
        return {"total_seen": buf.total_seen, "current_size": len(buf.buffer)}
    elif name == "sample_replay":
        n = args.get("batch_size", 5)
        return {"batch": buf.sample_batch(n)}
    elif name == "get_distribution":
        return buf.get_task_distribution()
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        res = handle_call(req.get("method"), req.get("params", {}))
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
