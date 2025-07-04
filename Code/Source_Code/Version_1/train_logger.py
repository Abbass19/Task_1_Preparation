import json
import os

LOG_FILE = "training_log.json"


def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try:
                return json.load(f)
            except:
                return {"trials": [], "best": None}
    else:
        return {"trials": [], "best": None}


def save_log(log_data):
    with open(LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=2)


def append_trial(log_data, trial_result, params):
    log_data["trials"].append({
        "result": trial_result,
        "params": params
    })


def update_best(log_data, trial_result, params):
    best = log_data.get("best")
    if best is None or trial_result > best["result"]:
        log_data["best"] = {
            "result": trial_result,
            "params": params
        }
        return True
    return False


def objective(trial, load_best=True):
    log_data = load_log()

    # Optional: load best result on start
    if load_best and log_data["best"] is not None:
        best_result = log_data["best"]["result"]
        best_params = log_data["best"]["params"]
        print(f"Starting from best known result: {best_result} with params: {best_params}")
        # You can choose to use these to initialize your model or training here

    # Run your trial and get the result (reward, loss, accuracy...)
    trial_result = run_training_with_params(trial)  # your training logic here

    # Append trial info to log
    append_trial(log_data, trial_result, trial.params)

    # Update best if this trial is better
    if update_best(log_data, trial_result, trial.params):
        print(f"New best result: {trial_result}")

    save_log(log_data)

    return trial_result
