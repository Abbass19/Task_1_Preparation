import optuna
import torch
from train import objective
from ppo_agent import PPOAgent
from train_logger import load_log
import pandas as pd
import matplotlib.pyplot as plt

from visualization import plot_training_returns

# Configuration
N_TRIALS = 125  # 5x longer than 25
BEST_MODEL_PATH = "Code/Source_Code/saved_model.pt"

def main():
    # Start Optuna Study
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial), n_trials=N_TRIALS)

    # Show best result
    print("\n========== BEST TRIAL ==========")
    print(study.best_trial)

    # Load the best agent
    best_params = study.best_trial.params
    print(f"\nBest Hyperparameters: {best_params}")

    obs_dim = 1
    act_dim = 3
    agent = PPOAgent(obs_dim, act_dim,
                     gamma=best_params["gamma"],
                     clip_epsilon=best_params["clip_epsilon"],
                     lr=best_params["lr"])

    # Save the model
    torch.save(agent.model.state_dict(), BEST_MODEL_PATH)
    print(f"Model saved to {BEST_MODEL_PATH}")

    # Evaluate the agent
    returns = evaluate_agent(agent)
    print(f"\n✅ Average profit over test run: {sum(returns):,.2f}")
    plot_training_returns(returns)

def evaluate_agent(agent, episodes=1):
    from stock_env import Environment
    data = pd.read_csv("my_data.csv")

    returns_all = []

    for _ in range(episodes):
        env = Environment(data)
        obs = env.reset()
        total_return = 0
        done = False

        while not done:
            action, _, _ = agent.get_action(obs)
            obs, reward, done, _ = env.step(action)
            total_return += reward

        returns_all.append(total_return)

    return returns_all

if __name__ == "__main__":
    main()
