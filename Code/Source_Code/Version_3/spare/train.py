import collections
import os

import optuna
import numpy as np
import pandas as pd
import torch
from log_api import *
from visualization import display_optuna_trials,display_agents_performance
from ppo_agent import PPOAgent
from environment import Environment
from model_manager import *
import collections

csv_path = os.path.join(os.path.dirname(__file__), "models", "my_data.csv")
data = pd.read_csv(csv_path)

obs_dim = 1
act_dim = 3


def hyperparameter_search(load_best_value=False, n_trials=25, num_episodes=20, visualization=True):
    import optuna
    all_agents_returns = []

    def trial_train(trial):
        # Only load best params if explicitly requested (like for final eval), not during search
        if load_best_value:
            best_params = load_best_hyperparameter()
            if best_params is None:
                best_params = None
        else:
            best_params = None

        if best_params is not None and load_best_value:
            # Use loaded best parameters (rare case)
            lr = best_params.get("lr")
            gamma = best_params.get("gamma")
            clip_epsilon = best_params.get("clip_epsilon")
            rollout_len = best_params.get("rollout_len")
            update_epochs = best_params.get("update_epochs")
        else:
            # Sample new hyperparameters for this trial
            lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
            gamma = trial.suggest_float("gamma", 0.90, 0.999)
            clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.3)
            rollout_len = trial.suggest_categorical("rollout_len", [128, 256, 512])
            update_epochs = trial.suggest_int("update_epochs", 5, 15)

        agent = PPOAgent(obs_dim, act_dim, gamma=gamma, clip_epsilon=clip_epsilon, lr=lr)
        all_returns = []

        for episode in range(num_episodes):
            env = Environment(data)
            obs = env.reset()
            observations, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

            for step in range(rollout_len):
                action, log_prob, _ = agent.get_action(obs)
                observations.append(obs)
                actions.append(action)
                log_probs_old.append(log_prob.detach())
                obs, reward, done, _ = env.step(action)
                rewards.append(reward)
                dones.append(done)

                with torch.no_grad():
                    _, value = agent.model(torch.FloatTensor(obs).unsqueeze(0))
                values.append(value.item())

                if done:
                    break

            if len(rewards) == 0:
                print("Warning: No steps taken, skipping trial.")
                return -float("inf")

            with torch.no_grad():
                _, last_value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

            returns = agent.compute_returns(rewards, dones, last_value.item())
            values_tensor = torch.tensor(values, dtype=torch.float32)
            observations_tensor = torch.FloatTensor(np.array(observations)).float()
            actions_tensor = torch.tensor(actions, dtype=torch.long)

            try:
                log_probs_old_tensor = torch.stack(log_probs_old).float()
            except Exception as e:
                print(f"Error stacking log_probs_old: {e}")
                return -float("inf")

            returns_tensor = torch.tensor(np.array(returns), dtype=torch.float32)
            advantages = returns_tensor - values_tensor
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            if not (len(observations_tensor) == len(actions_tensor) == len(log_probs_old_tensor) == len(returns_tensor) == len(advantages)):
                print("Mismatch in batch sizes, skipping trial")
                return -float("inf")

            agent.update(observations_tensor, actions_tensor, log_probs_old_tensor, returns_tensor, advantages, epochs=update_epochs)

            total_return = returns_tensor.sum().item()
            all_returns.append(total_return)

            print(f"Episode {episode+1}/{num_episodes} completed, total return: {total_return:.2f}")

        all_agents_returns.append(all_returns)

        avg_return = np.mean(all_returns)
        print(f"Trial average return: {avg_return:.2f}")

        save_hyperparameter_result({
            "result": avg_return,
            "params": {
                "lr": lr,
                "gamma": gamma,
                "clip_epsilon": clip_epsilon,
                "rollout_len": rollout_len,
                "update_epochs": update_epochs
            }
        })

        if visualization:
            display_optuna_trials(all_returns[:9])

        return avg_return

    study = optuna.create_study(direction="maximize")
    study.optimize(trial_train, n_trials=n_trials)

    print(f"Best trial parameters: {study.best_trial.params}")
    print(f"Best trial value: {study.best_trial.value}")

    return study.best_trial.params, study.best_trial.value


def train_one_agent(report_number= "", no_episodes=60 ):
    """
    Train a single PPO agent using the best saved hyperparameters on the training environment.
    After training, evaluate it on validation and test environments, printing detailed actions and reports.
    Returns: [train_profit_changes, val_profit_changes, test_profit_changes]
    """

    # Load best hyperparameters
    best_params = load_best_hyperparameter()
    if best_params is None:
        print("⚠️ No best hyperparameters found. Using default.")
        best_params = {
            "lr": 1e-4,
            "gamma": 0.99,
            "clip_epsilon": 0.2,
            "rollout_len": 256,
            "update_epochs": 10
        }

    # Set environment
    training_env, validation_env, testing_env = Environment.with_splits_time_series(data)
    print(f"[INFO] Training env length: {len(training_env.data)}")
    print(f"[INFO] Validation env length: {len(validation_env.data)}")
    print(f"[INFO] Testing env length: {len(testing_env.data)}")

    # Initialize agent
    agent = PPOAgent(
        obs_dim=1,
        act_dim=3,
        gamma=best_params["gamma"],
        clip_epsilon=best_params["clip_epsilon"],
        lr=best_params["lr"]
    )

    print(f"🧠 Training PPO agent for {no_episodes} episodes using:")
    print(best_params)

    # Training Loop
    for episode in range(no_episodes):
        obs = training_env.reset()
        done = False
        total_return = 0

        observations, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

        for step in range(best_params["rollout_len"]):
            action, log_prob, _ = agent.get_action(obs)
            next_obs, reward, done, _ = training_env.step(action)

            with torch.no_grad():
                _, value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

            # Store trajectory
            observations.append(obs)
            actions.append(action)
            log_probs_old.append(log_prob.detach())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())

            total_return += reward
            obs = next_obs

            if done:
                break

        with torch.no_grad():
            _, last_value = agent.model(torch.FloatTensor(obs).unsqueeze(0))
        returns = agent.compute_returns(rewards, dones, last_value.item())

        # Prepare tensors
        obs_tensor = torch.FloatTensor(np.array(observations))
        actions_tensor = torch.tensor(actions)
        log_probs_tensor = torch.stack(log_probs_old).float()
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        values_tensor = torch.tensor(values, dtype=torch.float32)

        advantages = returns_tensor - values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update model
        agent.update(obs_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages,
                     epochs=best_params["update_epochs"])

        print(f"Train_One_Agent Speaking : Episode {episode + 1} finished. Total return: {total_return}")

    print("✅ Training completed. Evaluating agent...")

    def evaluate(env, label="Unknown"):
        print(f"\n📊 Evaluating agent on {label}...")
        obs = env.reset()
        done = False
        step_count = 0
        profit_changes = []

        while not done:
            with torch.no_grad():
                action, _, _ = agent.get_action(obs)
            obs, reward, done, info = env.step(action)

            profit_change = info.get("profit_change", 0)
            action_str = ["Hold", "Buy", "Sell"][action]
            print(f"Step {step_count + 1}: Action: {action_str}, Profit change: {profit_change}, Inventory: {info.get('inventory', 'N/A')}")
            profit_changes.append(profit_change)
            step_count += 1

        total_profit = sum(profit_changes)
        print(f"    Train_One_Agent Speaking : Finished {label}. Total profit: {total_profit} over {step_count} steps.")
        return profit_changes

    train_profit_changes = evaluate(training_env, label="Training")
    val_profit_changes = evaluate(validation_env, label="Validation")
    test_profit_changes = evaluate(testing_env, label="Testing")

    # === Summary Report ===
    report_lines = []
    report_lines.append("\n==================== AGENT PERFORMANCE REPORT ====================")
    report_lines.append(f"Used Hyperparameters: {best_params}")

    def summarize(label, profits, action_counts, steps, profit_stats, inv, cash):
        report_lines.append(f"\n--- {label} Phase ---")
        report_lines.append(f"Total Profit: {sum(profits)}")
        report_lines.append(f"Avg Profit/Step: {sum(profits)/(steps or 1)}")
        report_lines.append(f"Max Gain: {max(profit_stats, default=0)}")
        report_lines.append(f"Max Loss: {min(profit_stats, default=0)}")
        report_lines.append(f"Steps: {steps}")
        report_lines.append(f"Positive Steps: {sum(p > 0 for p in profit_stats)}")
        report_lines.append(f"Negative Steps: {sum(p < 0 for p in profit_stats)}")
        report_lines.append(f"Neutral Steps: {sum(p == 0 for p in profit_stats)}")
        for act in ["Buy", "Sell", "Hold"]:
            count = action_counts.get(act, 0)
            percent = 100 * count / steps if steps else 0
            report_lines.append(f"{act}: {count} times ({percent}%)")
        report_lines.append(f"Final Inventory: {inv}")
        report_lines.append(f"Final Cash: {cash}")

    def silent_eval(env):
        obs = env.reset()
        done = False
        action_counts = collections.Counter()
        profit_stats = []
        step_count = 0
        inventory = 0
        cash = 0

        while not done:
            with torch.no_grad():
                action, _, _ = agent.get_action(obs)
            obs, reward, done, info = env.step(action)

            profit_change = info.get("profit_change", 0)
            action_str = ["Hold", "Buy", "Sell"][action]
            action_counts[action_str] += 1
            profit_stats.append(profit_change)
            step_count += 1
            inventory = info.get("inventory", 0)
            cash = info.get("cash", 0)

        return profit_stats, action_counts, step_count, inventory, cash

    train_stats = silent_eval(training_env)
    val_stats = silent_eval(validation_env)
    test_stats = silent_eval(testing_env)

    train_profit, train_actions, train_steps, train_inventory, train_cash = train_stats
    val_profit, val_actions, val_steps, val_inventory, val_cash = val_stats
    test_profit, test_actions, test_steps, test_inventory, test_cash = test_stats

    # Corrected call
    summarize("Training", train_profit_changes, train_stats[1], train_stats[2], train_stats[0], train_stats[3],
              train_stats[4])
    summarize("Validation", val_profit_changes, val_stats[1], val_stats[2], val_stats[0], val_stats[3], val_stats[4])
    summarize("Testing", test_profit_changes, test_stats[1], test_stats[2], test_stats[0], test_stats[3], test_stats[4])

    # Save report
    final_report = "\n".join(report_lines)
    print(final_report)

    document_name = os.path.join("agent_reports", f"agent_report_{report_number}.txt")  # Change filename as needed

    # Write the file
    with open(document_name, "w", encoding="utf-8") as f:
        f.write(final_report)

    print("\n📄 Report saved as 'agent_report.txt'. You can open it in Notepad.")

    # Save performance for tracking
    performance_entry = {
        "train_profit": train_profit_changes,
        "val_profit": val_profit_changes,
        "test_profit": test_profit_changes,
        "hyperparameters": best_params
    }
    save_agent_result(performance_entry)

    return [train_profit_changes, val_profit_changes, test_profit_changes]


def train_multiple_agents(num_agents, episodes_per_agent):
    """
    Trains multiple agents sequentially by calling train_one_agent,
    collects their profit change data, and then visualizes all agents' performance.

    Args:
        num_agents (int): Number of agents to train.
        episodes_per_agent (int): Number of episodes per agent training.

    Returns:
        List of tuples (train_profit_changes, val_profit_changes, test_profit_changes) for each agent.
    """

    all_agents_data = []

    for i in range(num_agents):
        print(f"Training agent {i+1}/{num_agents}...")
        agent_data = train_one_agent(no_episodes=episodes_per_agent)
        all_agents_data.append(agent_data)

    display_agents_performance(all_agents_data)  # Assuming you have this visualization function

    return all_agents_data

