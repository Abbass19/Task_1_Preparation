import os

import optuna
import numpy as np
import pandas as pd
import torch
from log_api import load_log, save_log, save_hyper_train, update_best, read_best_hyperparameter
from visualization import display_optuna_trials,display_agents_performance
from ppo_agent import PPOAgent
from environment import Environment


csv_path = os.path.join(os.path.dirname(__file__), "my_data.csv")
data = pd.read_csv(csv_path)

obs_dim = 1
act_dim = 3

def hyperparameter_search(load_best_value=True, n_trials=25, num_episodes=20, visualization=True):
    log_data = load_log()

    best_result = -float("inf")
    best_params = None

    def trial_train(trial):
        nonlocal best_result, best_params

        # Sample hyperparameters or load best from log
        if load_best_value and log_data.get("best") is not None:
            params = log_data["best"]["params"]
            lr = params["lr"]
            gamma = params["gamma"]
            clip_epsilon = params["clip_epsilon"]
            rollout_len = params["rollout_len"]
            update_epochs = params["update_epochs"]
        else:
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

        avg_return = np.mean(all_returns)

        save_hyper_train(result=avg_return, params={
            "lr": lr,
            "gamma": gamma,
            "clip_epsilon": clip_epsilon,
            "rollout_len": rollout_len,
            "update_epochs": update_epochs
        })

        if update_best(avg_return, {
            "lr": lr,
            "gamma": gamma,
            "clip_epsilon": clip_epsilon,
            "rollout_len": rollout_len,
            "update_epochs": update_epochs
        }):
            print(f"New best result: {avg_return:.2f}")
            best_result = avg_return
            best_params = {
                "lr": lr,
                "gamma": gamma,
                "clip_epsilon": clip_epsilon,
                "rollout_len": rollout_len,
                "update_epochs": update_epochs
            }
            save_log(log_data)

        return avg_return

    study = optuna.create_study(direction="maximize")
    study.optimize(trial_train, n_trials=n_trials)

    print(f"Best trial value: {best_result}")
    print(f"Best parameters: {best_params}")

    if visualization:
        display_optuna_trials()

    return best_result, best_params




def train_one_agent(no_episodes=60):
    """
    Train a single agent with the best saved hyperparameters on the training environment,
    then evaluate on validation and testing environments.
    Returns profit change lists for train, val, and test environments.
    """

    # Load best hyperparameters
    best_result, best_params = read_best_hyperparameter()
    if best_params is None:
        print("No best hyperparameters found in log. Using default parameters.")
        # Define some default hyperparameters here:
        best_params = {
            "lr": 1e-4,
            "gamma": 0.99,
            "clip_epsilon": 0.2,
            "rollout_len": 256,
            "update_epochs": 10
        }

    # Create environments
    training_env , validation_env, testing_env= Environment.with_splits_time_series(data)

    # Initialize agent with best hyperparameters
    agent = PPOAgent(
        obs_dim=1,  # adjust as per your data
        act_dim=3,
        gamma=best_params["gamma"],
        clip_epsilon=best_params["clip_epsilon"],
        lr=best_params["lr"]
    )

    # Train agent on training_env for no_episodes
    for episode in range(no_episodes):
        obs = training_env.reset()
        done = False
        while not done:
            action, log_prob, _ = agent.get_action(obs)
            obs, reward, done, info = training_env.step(action)
            # Collect experience & update agent here as per your PPO implementation

    # Freeze agent (no more training)
    agent.model.eval()

    def evaluate(env):
        obs = env.reset()
        done = False
        profit_changes = []
        while not done:
            with torch.no_grad():
                action, _, _ = agent.get_action(obs)
            obs, reward, done, info = env.step(action)
            profit_changes.append(info.get("profit_change", 0))  # Adjust key if needed
        return profit_changes

    train_profit_changes = evaluate(training_env) # This is wrong
    val_profit_changes = evaluate(validation_env)
    test_profit_changes = evaluate(testing_env)


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
