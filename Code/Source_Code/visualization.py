import matplotlib.pyplot as plt
import numpy as np


def plot_training_returns(returns_list):
    """
    Plot total return (reward) over episodes.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(returns_list, label="Total Return")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward / Return")
    plt.title("PPO Training Return per Episode")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_trading_actions(prices, actions):
    """
    Plot the stock price with Buy/Sell actions.
    Actions: 0 = Hold, 1 = Buy, 2 = Sell
    """
    plt.figure(figsize=(12, 5))
    plt.plot(prices, label="Stock Price", color='black')

    buys = [i for i, a in enumerate(actions) if a == 1]
    sells = [i for i, a in enumerate(actions) if a == 2]

    plt.scatter(buys, [prices[i] for i in buys], marker='^', color='green', label='Buy', s=80)
    plt.scatter(sells, [prices[i] for i in sells], marker='v', color='red', label='Sell', s=80)

    plt.title("Trading Actions Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def extract_metrics(log):
    """Extract results and parameters from log dict."""
    results = [trial["result"] for trial in log["trials"]]
    params = [trial["params"] for trial in log["trials"]]
    return results, params

def plot_results_over_trials(log):
    """Plot total reward vs trial index."""
    results, _ = extract_metrics(log)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(results) + 1), results, marker='o')
    plt.title("Total Reward over Trials")
    plt.xlabel("Trial Number")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()

def plot_hyperparam_vs_result(log, hyperparam_name):
    """Scatter plot of a hyperparameter vs total reward."""
    results, params = extract_metrics(log)
    values = []
    for p in params:
        val = p.get(hyperparam_name, None)
        if val is not None:
            values.append(val)
        else:
            values.append(np.nan)  # Handle missing values gracefully

    plt.figure(figsize=(8, 5))
    plt.scatter(values, results, c='blue', alpha=0.6)
    plt.title(f"{hyperparam_name} vs Total Reward")
    plt.xlabel(hyperparam_name)
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()

def get_best_trial(log):
    """Return the trial with the highest result and its params."""
    best_trial = max(log["trials"], key=lambda t: t["result"])
    return best_trial["result"], best_trial["params"]

def summary_statistics(log):
    """Print summary statistics of the results."""
    results, _ = extract_metrics(log)
    print(f"Number of trials: {len(results)}")
    print(f"Max reward: {np.max(results):.2f}")
    print(f"Min reward: {np.min(results):.2f}")
    print(f"Mean reward: {np.mean(results):.2f}")
    print(f"Median reward: {np.median(results):.2f}")
    print(f"Std deviation: {np.std(results):.2f}")