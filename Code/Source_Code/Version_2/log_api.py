import json
import os
from datetime import datetime

LOG_FILEPATH = "training_log.json"

# Module-level variable to hold the log data in memory
_log_data = None

def _initialize_log_if_missing(filepath=LOG_FILEPATH):
    """Create an empty log file if it does not exist."""
    if not os.path.exists(filepath):
        empty_log = {
            "trials": [],
            "best": None
        }
        with open(filepath, "w") as f:
            json.dump(empty_log, f, indent=4)
        print(f"[log_api] Created new empty log at {filepath}")

def _load_log(filepath=LOG_FILEPATH):
    """Load the log from file to the module-level _log_data."""
    global _log_data
    if not os.path.exists(filepath):
        _initialize_log_if_missing(filepath)
    with open(filepath, "r") as f:
        _log_data = json.load(f)

def _save_log(filepath=LOG_FILEPATH):
    """Save the in-memory _log_data to the JSON file."""
    global _log_data
    with open(filepath, "w") as f:
        json.dump(_log_data, f, indent=4)

def _ensure_loaded():
    """Ensure _log_data is loaded in memory before any operation."""
    global _log_data
    if _log_data is None:
        _load_log()

def save_hyper_train(result, params):
    """Append a new trial to the in-memory log, save to file, and update best if improved.
    Returns True if the best result was updated, else False.
    """
    _ensure_loaded()
    global _log_data
    trial_record = {
        "result": result,
        "params": params,
        "timestamp": datetime.utcnow().isoformat()
    }
    _log_data.setdefault("trials", []).append(trial_record)
    updated = False
    best = _log_data.get("best")
    if best is None or result > best.get("result", float("-inf")):
        _log_data["best"] = {
            "result": result,
            "params": params,
            "timestamp": datetime.utcnow().isoformat()
        }
        updated = True
    _save_log()
    return updated

def read_best_hyperparameter():
    """Return the best trial's (result, params), or (None, None) if none."""
    _ensure_loaded()
    best = _log_data.get("best")
    if best is None:
        return None, None
    return best.get("result"), best.get("params")

def update_best(result, params):
    """Update best trial if current result is better. Returns True if updated."""
    _ensure_loaded()
    global _log_data
    best = _log_data.get("best")
    if best is None or result > best.get("result", float("-inf")):
        _log_data["best"] = {
            "result": result,
            "params": params,
            "timestamp": datetime.utcnow().isoformat()
        }
        _save_log()
        return True
    return False

def load_log():
    """Return the full in-memory log data."""
    _ensure_loaded()
    return _log_data

def save_log():
    """Save the current in-memory log data to file."""
    _ensure_loaded()
    _save_log()
