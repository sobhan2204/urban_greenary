"""Subprocess entry point for running the ML pipeline.

Called by the Flask app to isolate GEE/tkinter from the web server process.
Reads a JSON request file, runs the pipeline, writes a JSON response.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Must be the FIRST import — initializes GEE before anything else
import ee
ee.Initialize(project="urban-green-mapping")


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: run_pipeline_subprocess.py <request.json> <response.json>")
        sys.exit(1)

    req_path = Path(sys.argv[1])
    resp_path = Path(sys.argv[2])

    with req_path.open() as f:
        req = json.load(f)

    try:
        from utils.config_loader import load_config, resolve_config_path
        from main import run_pipeline_for_point

        config = load_config(resolve_config_path(None))
        result = run_pipeline_for_point(
            lat=req["lat"],
            lng=req["lng"],
            config=config,
            target_date=req.get("date"),
        )
        resp = {"status": "ok", **result}
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        resp = {"status": "error", "message": str(exc), "traceback": tb}

    with resp_path.open("w") as f:
        json.dump(resp, f)


if __name__ == "__main__":
    main()
