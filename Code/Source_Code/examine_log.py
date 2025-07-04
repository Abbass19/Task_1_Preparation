import json
import numpy as np
import matplotlib.pyplot as plt


def load_log_from_file(filepath):
    with open(filepath, 'r') as f:
        log = json.load(f)
    return log


def summary_statistics(log):
    results = [trial["result"] for trial in log["trials"]]
    print(f"Number of trials: {len(results)}")
    print(f"Max reward: {np.max(results):.2f}")
    print(f"Min reward: {np.min(results):.2f}")
    print(f"Mean reward: {np.mean(results):.2f}")
    print(f"Median reward: {np.median(results):.2f}")
    print(f"Std deviation: {np.std(results):.2f}")


def get_best_trial(log):
    best_trial = max(log["trials"], key=lambda t: t["result"])
    return best_trial["result"], best_trial["params"]


def analyze_log_subplot_grid(log):
    results = [trial["result"] for trial in log["trials"]]
    trials = list(range(len(results)))

    print("=== Log Summary Statistics ===")
    summary_statistics(log)

    best_result, best_params = get_best_trial(log)
    print(f"\nBest trial result: {best_result:.2f}")
    print(f"Best trial params: {best_params}")

    param_names = ["lr", "gamma", "clip_epsilon", "rollout_len", "update_epochs"]
    param_data = {param: [trial["params"][param] for trial in log["trials"]] for param in param_names}

    # Create subplots: 2 rows x 3 columns (1 for reward, 5 for each param)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()  # Make it 1D for easier indexing

    # Subplot 1: reward over trials
    axes[0].plot(trials, results, marker='o', color='black')
    axes[0].set_title("Reward over Trials")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Reward")

    # Subplots 2–6: hyperparameter vs result
    for i, param in enumerate(param_names):
        axes[i + 1].scatter(param_data[param], results, alpha=0.7)
        axes[i + 1].set_title(f"{param} vs Reward")
        axes[i + 1].set_xlabel(param)
        axes[i + 1].set_ylabel("Reward")

    # Hide unused subplot if any (e.g. if grid is 3x2 = 6 and we use only 6)
    for j in range(len(param_names) + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Training Log Analysis: Rewards and Hyperparameters", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    log_path = "training_log.json"
    log = load_log_from_file(log_path)
    analyze_log_subplot_grid(log)
