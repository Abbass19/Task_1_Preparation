# metrics.py
import numpy
import numpy as np
from collections import Counter
import math

import pandas as pd


def compute_stability_metrics(profits):
    """
    Compute stability-related metrics from a sequence of profit values.

    Args:
        profits (list or np.ndarray): Array-like sequence of numeric profit values
            recorded at each step or episode. Example: [0.5, -0.2, 0.1, 0.7]

    Returns:
        tuple:
            var_profit (float): Variance of profit values.
            std_profit (float): Standard deviation of profit values.
            cv_profit (float): Coefficient of Variation (std / mean), 0 if mean=0.
            iqr_profit (float): Interquartile range (75th percentile - 25th percentile).
    """
    profits = np.array(profits)
    var_profit = np.var(profits)
    std_profit = np.std(profits)
    mean_profit = np.mean(profits)
    cv_profit = std_profit / mean_profit if mean_profit != 0 else 0
    iqr_profit = np.percentile(profits, 75) - np.percentile(profits, 25)
    return var_profit, std_profit, cv_profit, iqr_profit

def compute_generalization_metrics(train_profit, val_profit):
    """
    Compute generalization metrics comparing training and validation profits.

    Args:
        train_profit (float): Aggregate profit metric from training data.
        val_profit (float): Aggregate profit metric from validation data.

    Returns:
        tuple:
            fitting_ratio (float): Ratio val_profit / train_profit (0 if train_profit=0).
            delta_profit (float): Difference val_profit - train_profit.
    """
    fitting_ratio = val_profit / train_profit if train_profit != 0 else 0
    delta_profit = val_profit - train_profit
    return fitting_ratio, delta_profit

def compute_diversity_metrics(actions):
    """
    Compute diversity-related metrics based on agent actions.

    Args:
        actions (list or np.ndarray): Sequence of discrete actions taken by the agent.
            Each action is typically an integer representing a class, e.g.:
            0 = Hold, 1 = Buy, 2 = Sell

    Returns:
        tuple:
            entropy (float): Shannon entropy of the action distribution.
            diversity_ratio (float): Number of unique actions divided by total actions.
    """
    counts = Counter(actions)
    total = sum(counts.values())
    probs = [count / total for count in counts.values()]

    entropy = -sum(p * math.log(p, 2) for p in probs if p > 0)
    diversity_ratio = len(counts) / total if total > 0 else 0

    return entropy, diversity_ratio

def compute_all_metrics(phase, episode_returns, episode_actions, val_episode_return=None):
    """
    Compute stability, diversity, and optionally generalization metrics for an episode.

    Args:
        phase (str): "Training", "Validation", or "Testing"
        episode_returns (list or np.ndarray): rewards/profits during episode
        episode_actions (list): actions taken during episode
        val_episode_return (float or None): validation return to compare for generalization

    Returns:
        dict: Combined metrics dictionary with keys matching Excel columns
    """

    var_profit, std_profit, cv_profit, iqr_profit = compute_stability_metrics(episode_returns)
    entropy_action, diversity_action_ratio = compute_diversity_metrics(episode_actions)

    # Generalization metrics only make sense comparing train vs val
    if phase == "Validation" and val_episode_return is not None:
        fitting_ratio, delta_profit = compute_generalization_metrics(val_episode_return, np.mean(episode_returns))
    else:
        fitting_ratio, delta_profit = None, None

    metrics = {
        "var_profit": var_profit,
        "std_profit": std_profit,
        "cv_profit": cv_profit,
        "iqr_profit": iqr_profit,
        "entropy_action": entropy_action,
        "diversity_action_ratio": diversity_action_ratio,
        "fitting_ratio": fitting_ratio,
        "delta_profit": delta_profit,
        # add placeholders for any other micro metrics you want here
    }
    return metrics

# Additional metric functions can be added here with similar detailed docstrings.
def save_episode_metrics(df, batch_id, agent_id, phase, episode_number, rewards, actions, values, advantages, info, total_return):
    step_count = len(rewards)
    avg_reward = total_return / step_count if step_count else 0
    var_profit, std_profit, cv_profit, iqr_profit = compute_stability_metrics(rewards)
    entropy_action, diversity_ratio = compute_diversity_metrics(actions)

    buy_count = actions.count(1)
    sell_count = actions.count(2)
    hold_count = actions.count(0)
    advantages_np = advantages.detach().cpu().numpy()

    row = {
        "batch_ID": batch_id,
        "agent_id": agent_id,
        "phase": phase,
        "episode_number": episode_number,
        "episode_return": total_return,
        "episode_length": step_count,
        "avg_reward_per_step": avg_reward,
        "max_reward": max(rewards) if rewards else 0,
        "min_reward": min(rewards) if rewards else 0,
        "value_mean": np.mean(values) if values else 0,
        "value_std": np.std(values) if values else 0,
        "value_last": values[-1] if values else 0,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "action_entropy": entropy_action,
        "final_inventory": info.get("inventory", 0),
        "final_cash": info.get("cash", 0),
        "profit_change_total": total_return,
        "advantage_mean": np.mean(advantages_np),
        "advantage_std": np.std(advantages_np),
        "advantage_max": np.max(advantages_np),
        "advantage_min": np.min(advantages_np),
        "returns_mean": np.mean(rewards) if rewards else 0,
        "returns_std": np.std(rewards) if rewards else 0,
        "Unnamed: 24": np.nan,  # placeholder for that unnamed column
        "var_profit": var_profit,
        "std_profit": std_profit,
        "cv_profit": cv_profit,
        "iqr_profit": iqr_profit,
        "delta_profit": 0,
        "return_gap_loss": 0,
        "fitting_ratio": 0,
        "delta_entropy": 0,
        "policy_div_spread": 0,
        "entropy_action": entropy_action,
        "action_dist_std": 0,
        "kl_train_test": 0,
        "diversity_action_ratio": diversity_ratio,
        "Buy": buy_count,
        "Sell": sell_count,
        "Hold": hold_count,
        "generalization_index": 0,
        "Stability W1": 0,
        "Stability W2": 0,
        "Stability W3": 0,
        "Stability W4": 0,
        "General W1": 0,
        "General W2": 0,
        "General W3": 0,
        "General W4": 0,
        "General W5": 0,
        "Diversity W1": 0,
        "Diversity W2": 0,
        "Diversity W3": 0,
        "Diversity W4": 0,
        "Diversity W5": 0,
        "Stability": 0,
        "General": 0,
        "Diversity": 0,
    }

    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

