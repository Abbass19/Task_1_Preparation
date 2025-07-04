import torch
import numpy as np
import pandas as pd
import optuna
from stock_env import Environment
from ppo_agent import PPOAgent
from torch.distributions import Categorical
from tqdm import trange
from visualization import plot_training_returns
import optuna.visualization as vis
from train_logger import load_log, save_log, append_trial, update_best


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Load your data (must contain 'Close')
data = pd.read_csv("my_data.csv")
obs_dim = 1
act_dim = 3


def objective(trial, load_best_value=False, num_episodes = 20):
    log_data = load_log()

    # Load or sample hyperparameters
    if load_best_value and log_data.get("best") is not None:
        best_params = log_data["best"]["params"]
        print(f"Using best saved params: {best_params}")

        lr = best_params["lr"]
        gamma = best_params["gamma"]
        clip_epsilon = best_params["clip_epsilon"]
        rollout_len = best_params["rollout_len"]
        update_epochs = best_params["update_epochs"]
    else:
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.999)
        clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.3)
        rollout_len = trial.suggest_categorical("rollout_len", [128, 256, 512])
        update_epochs = trial.suggest_int("update_epochs", 5, 15)

    agent = PPOAgent(obs_dim, act_dim, gamma=gamma, clip_epsilon=clip_epsilon, lr=lr)

    num_episodes = 20
    all_returns = []

    for episode in range(num_episodes):
        env = Environment(data)
        obs = env.reset()

        observations = []
        actions = []
        log_probs_old = []
        rewards = []
        dones = []
        values = []

        for step in range(rollout_len):
            # Get action and log probability from agent
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
                obs = env.reset()

        if len(rewards) == 0:
            print("Warning: No steps taken, skipping trial.")
            return -float("inf")

        with torch.no_grad():
            _, last_value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

        returns = agent.compute_returns(rewards, dones, last_value.item())
        values = torch.tensor(values, dtype=torch.float32)
        observations = torch.FloatTensor(np.array(observations)).float()
        actions = torch.tensor(actions, dtype=torch.long)

        # Convert collected old log_probs to tensor safely
        try:
            log_probs_old = torch.stack(log_probs_old).float()
        except Exception as e:
            print(f"Error converting log_probs_old: {e}")
            log_probs_old = torch.tensor([], dtype=torch.float32)

        returns = torch.tensor(np.array(returns), dtype=torch.float32)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Sanity check for matching batch sizes before update
        if (len(observations) == len(actions) == len(log_probs_old) == len(returns) == len(advantages)):
            agent.update(observations, actions, log_probs_old, returns, advantages, epochs=update_epochs)
        else:
            print(f"Mismatch in batch sizes! obs: {len(observations)}, actions: {len(actions)}, log_probs_old: {len(log_probs_old)}, returns: {len(returns)}, advantages: {len(advantages)}")
            return -float("inf")

        total_return = returns.sum().item()
        all_returns.append(total_return)

    avg_return = np.mean(all_returns)

    # Logging
    append_trial(log_data, avg_return, {
        "lr": lr,
        "gamma": gamma,
        "clip_epsilon": clip_epsilon,
        "rollout_len": rollout_len,
        "update_epochs": update_epochs
    })

    if update_best(log_data, avg_return, {
        "lr": lr,
        "gamma": gamma,
        "clip_epsilon": clip_epsilon,
        "rollout_len": rollout_len,
        "update_epochs": update_epochs
    }):
        print(f"New best result: {avg_return:.2f}")

    save_log(log_data)

    return avg_return



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_best_value", action="store_true", help="Load best parameters from log")
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, load_best_value=args.load_best_value), n_trials=25)

    print("Best trial:")
    print(study.best_trial)

    # Visualization
    fig = vis.plot_optimization_history(study)
    fig.show()

    fig2 = vis.plot_param_importances(study)
    fig2.show()

#Best trial:
#FrozenTrial(number=17, state=1, values=[1052121337.6], datetime_start=datetime.datetime(2025, 7, 4, 11, 0, 42, 69286), datetime_complete=datetime.datetime(2025, 7, 4, 11, 0, 50, 89474), params={'lr': 2.099427503938859e-05, 'gamma': 0.9986159260658843, 'clip_epsilon': 0.21613062115787807, 'rollout_len': 512, 'update_epochs': 5}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'lr': FloatDistribution(high=0.001, log=True, low=1e-05, step=None), 'gamma': FloatDistribution(high=0.999, log=False, low=0.9, step=None), 'clip_epsilon': FloatDistribution(high=0.3, log=False, low=0.1, step=None), 'rollout_len': CategoricalDistribution(choices=(128, 256, 512)), 'update_epochs': IntDistribution(high=15, log=False, low=5, step=1)}, trial_id=17, value=None)
#Install optuna[visualization] to see plots.