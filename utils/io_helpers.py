import os
import json
import datetime
import subprocess

def save_metadata(output_dir, args, config_name):
    metadata = {
        "model": args.model,
        "dataset": args.data,
        "task": args.task if hasattr(args, "task") else "unspecified",
        "config": config_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": subprocess.getoutput("git rev-parse HEAD"),
    }
    with open(os.path.join(output_dir, "info.json"), "w") as f:
        json.dump(metadata, f, indent=2)
