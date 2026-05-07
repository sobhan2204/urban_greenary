from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ── Prevent tkinter/tcl crashes ────────────────────────────────────────────
# GEE creates Tcl/Tk objects with C-level __del__ that crash when GC runs
# from a non-main thread. The pipeline runs in a subprocess (see below),
# so no Tcl objects are created in the Flask process. This env var suppresses
# any remaining deprecation warnings.
os.environ["TK_SILENCE_DEPRECATION"] = "1"


# ── Flask app ──────────────────────────────────────────────────────────────
import flask

app = flask.Flask(
    __name__,
    template_folder="templates",
    static_folder=".",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent


def check_deps():
    """Auto-install missing required packages at startup."""
    required = {"geopy": "geopy"}
    missing = []
    for pkg, import_name in required.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.info("Installing missing packages: %s", missing)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + missing,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for pkg in missing:
            logger.info("Installed %s", pkg)


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return flask.render_template("index.html")


@app.route("/api/run", methods=["POST"])
def run_pipeline():
    """Run the full pipeline in a subprocess (isolates GEE/tkinter)."""
    data = flask.request.get_json(force=True)
    lat = float(data.get("lat", 0))
    lng = float(data.get("lng", 0))
    target_date = data.get("date")

    req = {"lat": lat, "lng": lng}
    if target_date:
        req["date"] = target_date

    # Write request / response to temp files, run subprocess
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as req_f:
        json.dump(req, req_f)
        req_path = req_f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as resp_f:
        resp_path = resp_f.name

    try:
        script = _PROJECT_ROOT / "run_pipeline_subprocess.py"
        result = subprocess.run(
            [sys.executable, str(script), req_path, resp_path],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(_PROJECT_ROOT),
        )

        if result.returncode != 0:
            logger.error("Subprocess stderr: %s", result.stderr)
            return flask.jsonify({
                "status": "error",
                "message": "Pipeline subprocess failed. Check server logs.",
            }), 500

        with open(resp_path) as f:
            resp = json.load(f)

        if resp.get("status") == "ok":
            return flask.jsonify(resp)
        else:
            return flask.jsonify(resp), 500

    except subprocess.TimeoutExpired:
        return flask.jsonify({
            "status": "error",
            "message": "Pipeline timed out after 10 minutes.",
        }), 500
    except Exception as exc:
        logger.exception("Pipeline failed")
        return flask.jsonify({
            "status": "error",
            "message": str(exc),
        }), 500
    finally:
        for p in (req_path, resp_path):
            try:
                os.unlink(p)
            except OSError:
                pass


@app.route("/api/geocode", methods=["POST"])
def geocode():
    """Reverse geocode a coordinate."""
    data = flask.request.get_json(force=True)
    lat = float(data["lat"])
    lng = float(data["lng"])

    try:
        from utils.interactive_collection import reverse_geocode
        loc = reverse_geocode(lat, lng)
    except Exception as exc:
        logger.warning("Geocoding unavailable: %s", exc)
        loc = {
            "location_name": "N/A",
            "city": "N/A",
            "country": "N/A",
            "full_address": f"{lat}, {lng}",
        }
    return flask.jsonify(loc)


@app.route("/data/output/final/<path:filename>")
def serve_output(filename):
    """Serve generated map images from data/output/final/."""
    output_dir = _PROJECT_ROOT / "data" / "output" / "final"
    return flask.send_from_directory(output_dir, filename)


@app.route("/data/output/reports/<path:filename>")
def serve_report(filename):
    """Serve generated report files from data/output/reports/."""
    report_dir = _PROJECT_ROOT / "data" / "output" / "reports"
    return flask.send_from_directory(report_dir, filename)


# ── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_deps()
    app.run(host="0.0.0.0", port=5000, debug=False)


