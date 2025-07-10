import pandas as pd

from Code.Source_Code.Version_4.metrics import save_episode_metrics
from log_api import *
from model_manager import *
import collections

import torch
import optuna
from environment import Environment
from ppo_agent import PPOAgent

#Some Storage Data
csv_path = os.path.join(os.path.dirname(__file__), "models", "my_data.csv")
data = pd.read_csv(csv_path)
EXCEL_PATH = r"C:\Users\Abbass Zahreddine\Documents\GitHub\Task_1_Preparation\Code\Source_Code\Version_4\agent_reports\Data-Analysis.xlsx"
SHEET_NAME = "Sheet1"

obs_dim = 1
act_dim = 3


#This does not use Optuna. It is an old number
def train_one_agent(report_number="", no_episodes=60):
    """
    Train a PPO agent with Optuna hyperparameter tuning inside this function.
    Uses train/validation/test splits.
    Reports agent performance similarly to the previous implementation.
    Saves report to 'agent_reports/agent_report_{report_number}.txt'.
    Returns: [train_profit_changes, val_profit_changes, test_profit_changes]
    """

    # Load data and split environments
    training_env, validation_env, testing_env = Environment.with_splits_time_series(data)
    print(f"[INFO] Training env length: {len(training_env.data)}")
    print(f"[INFO] Validation env length: {len(validation_env.data)}")
    print(f"[INFO] Testing env length: {len(testing_env.data)}")

    # Optuna objective function for hyperparameter tuning
    def objective(trial):
        # Suggest hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.999)
        clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.3)
        rollout_len = trial.suggest_categorical("rollout_len", [128, 256, 512])
        update_epochs = trial.suggest_int("update_epochs", 5, 15)

        # Initialize agent with trial params
        agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, gamma=gamma, clip_epsilon=clip_epsilon, lr=lr)

        for episode in range(no_episodes):
            obs = training_env.reset()
            done = False

            observations, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

            for step in range(rollout_len):
                action, log_prob, _ = agent.get_action(obs)
                next_obs, reward, done, _ = training_env.step(action)

                with torch.no_grad():
                    _, value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

                observations.append(obs)
                actions.append(action)
                log_probs_old.append(log_prob.detach())
                rewards.append(reward)
                dones.append(done)
                values.append(value.item())

                obs = next_obs

                if done:
                    break

            with torch.no_grad():
                _, last_value = agent.model(torch.FloatTensor(obs).unsqueeze(0))
            returns = agent.compute_returns(rewards, dones, last_value.item())

            obs_tensor = torch.FloatTensor(np.array(observations))
            actions_tensor = torch.tensor(actions)
            log_probs_tensor = torch.stack(log_probs_old).float()
            if isinstance(returns, torch.Tensor):
                returns_tensor = returns.detach().clone().float()
            else:
                returns_tensor = torch.tensor(returns, dtype=torch.float32)

            values_tensor = torch.tensor(values, dtype=torch.float32)

            advantages = returns_tensor - values_tensor
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            agent.update(obs_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages, epochs=update_epochs)

        # Validation phase evaluation for Optuna trial objective
        obs = validation_env.reset()
        done = False
        val_profit = 0
        while not done:
            with torch.no_grad():
                action, _, _ = agent.get_action(obs)
            obs, reward, done, _ = validation_env.step(action)
            val_profit += reward

        return val_profit

    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)

    best_params = study.best_trial.params
    print(f"[INFO] Best hyperparameters from Optuna: {best_params}")

    # Train final agent on training env with best hyperparameters
    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim,
                     gamma=best_params["gamma"],
                     clip_epsilon=best_params["clip_epsilon"],
                     lr=best_params["lr"])

    rollout_len = best_params["rollout_len"]
    update_epochs = best_params["update_epochs"]

    for episode in range(no_episodes):
        obs = training_env.reset()
        done = False
        total_return = 0

        observations, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

        for step in range(rollout_len):
            action, log_prob, _ = agent.get_action(obs)
            next_obs, reward, done, _ = training_env.step(action)

            with torch.no_grad():
                _, value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

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

        obs_tensor = torch.FloatTensor(np.array(observations))
        actions_tensor = torch.tensor(actions)
        log_probs_tensor = torch.stack(log_probs_old).float()
        if isinstance(returns, torch.Tensor):
            returns_tensor = returns.detach().clone().float()
        else:
            returns_tensor = torch.tensor(returns, dtype=torch.float32)

        values_tensor = torch.tensor(values, dtype=torch.float32)

        advantages = returns_tensor - values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        agent.update(obs_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages, epochs=update_epochs)

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
            report_lines.append(f"{act}: {count} times ({percent:.2f}%)")
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

    summarize("Training", train_profit_changes, train_actions, train_steps, train_profit, train_inventory, train_cash)
    summarize("Validation", val_profit_changes, val_actions, val_steps, val_profit, val_inventory, val_cash)
    summarize("Testing", test_profit_changes, test_actions, test_steps, test_profit, test_inventory, test_cash)

    final_report = "\n".join(report_lines)
    print(final_report)

    # Ensure directory exists
    os.makedirs("agent_reports", exist_ok=True)
    document_name = os.path.join("agent_reports", f"agent_report_{report_number}.txt")

    with open(document_name, "w", encoding="utf-8") as f:
        f.write(final_report)

    print(f"\n📄 Report saved as '{document_name}'. You can open it in Notepad or any text editor.")

    # Save performance data for tracking (if implemented)
    performance_entry = {
        "train_profit": train_profit_changes,
        "val_profit": val_profit_changes,
        "test_profit": test_profit_changes,
        "hyperparameters": best_params
    }
    save_agent_result(performance_entry)

    return [train_profit_changes, val_profit_changes, test_profit_changes]

#The first train with Optuna (Works with 30 episode in a good way )
def train_two_agent(report_number="", no_episodes=60):
    """
    Train a PPO agent with Optuna hyperparameter tuning inside this function.
    Uses train/validation/test splits.
    Reports agent performance similarly to the previous implementation.
    Returns: [train_profit_changes, val_profit_changes, test_profit_changes]
    """

    # Load data and split environments
    training_env, validation_env, testing_env = Environment.with_splits_time_series(data)
    print(f"[INFO] Training env length: {len(training_env.data)}")
    print(f"[INFO] Validation env length: {len(validation_env.data)}")
    print(f"[INFO] Testing env length: {len(testing_env.data)}")

    # Optuna objective function for hyperparameter tuning
    def objective(trial):
        # Suggest hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.999)
        clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.3)
        rollout_len = trial.suggest_categorical("rollout_len", [128, 256, 512])
        update_epochs = trial.suggest_int("update_epochs", 5, 15)

        # Initialize agent with trial params
        agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, gamma=gamma, clip_epsilon=clip_epsilon, lr=lr)

        all_returns = []

        for episode in range(no_episodes):
            obs = training_env.reset()
            done = False
            total_return = 0

            observations, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

            for step in range(rollout_len):
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
            if isinstance(returns, torch.Tensor):
                returns_tensor = returns.detach().clone().float()
            else:
                returns_tensor = torch.tensor(returns, dtype=torch.float32)

            values_tensor = torch.tensor(values, dtype=torch.float32)

            advantages = returns_tensor - values_tensor
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Update model
            agent.update(obs_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages, epochs=update_epochs)

            all_returns.append(total_return)

        # Use validation env to evaluate generalization for this trial
        obs = validation_env.reset()
        done = False
        val_profit = 0
        while not done:
            with torch.no_grad():
                action, _, _ = agent.get_action(obs)
            obs, reward, done, _ = validation_env.step(action)
            val_profit += reward

        # Optuna maximizes objective, so return validation total profit
        return val_profit

    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)

    best_params = study.best_trial.params
    print(f"[INFO] Best hyperparameters from Optuna: {best_params}")

    # Train final agent on training env with best hyperparameters
    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim,
                     gamma=best_params["gamma"],
                     clip_epsilon=best_params["clip_epsilon"],
                     lr=best_params["lr"])

    rollout_len = best_params["rollout_len"]
    update_epochs = best_params["update_epochs"]

    for episode in range(no_episodes):
        obs = training_env.reset()
        done = False
        total_return = 0

        observations, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

        for step in range(rollout_len):
            action, log_prob, _ = agent.get_action(obs)
            next_obs, reward, done, _ = training_env.step(action)

            with torch.no_grad():
                _, value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

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

        obs_tensor = torch.FloatTensor(np.array(observations))
        actions_tensor = torch.tensor(actions)
        log_probs_tensor = torch.stack(log_probs_old).float()
        if isinstance(returns, torch.Tensor):
            returns_tensor = returns.detach().clone().float()
        else:
            returns_tensor = torch.tensor(returns, dtype=torch.float32)

        values_tensor = torch.tensor(values, dtype=torch.float32)

        advantages = returns_tensor - values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        agent.update(obs_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages, epochs=update_epochs)

        print(f"Train_One_Agent Speaking : Episode {episode + 1} finished. Total return: {total_return}")

    print("✅ Training completed. Evaluating agent...")

    # Evaluation helper (same as original train_one_agent)
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

    # Same reporting code as original train_one_agent (not repeated here for brevity)
    # Please copy the summarize(), silent_eval(), and report writing parts from your existing function.

    # For now, just return the profits to keep consistent:
    return [train_profit_changes, val_profit_changes, test_profit_changes]

#Train with Optuna and txt Reporting (Works for 30 Plus episodes)
def train_two_with_report(report_number="", no_episodes=60):
    """
    Train a PPO agent with Optuna hyperparameter tuning.
    Also generates a detailed performance report.
    """

    # Load data and split environments
    training_env, validation_env, testing_env = Environment.with_splits_time_series(data)
    print(f"[INFO] Training env length: {len(training_env.data)}")
    print(f"[INFO] Validation env length: {len(validation_env.data)}")
    print(f"[INFO] Testing env length: {len(testing_env.data)}")

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.999)
        clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.3)
        rollout_len = trial.suggest_categorical("rollout_len", [128, 256, 512])
        update_epochs = trial.suggest_int("update_epochs", 5, 15)

        agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, gamma=gamma, clip_epsilon=clip_epsilon, lr=lr)

        for episode in range(no_episodes):
            obs = training_env.reset()
            done = False
            total_return = 0

            observations, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

            for step in range(rollout_len):
                action, log_prob, _ = agent.get_action(obs)
                next_obs, reward, done, _ = training_env.step(action)

                with torch.no_grad():
                    _, value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

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

            obs_tensor = torch.FloatTensor(np.array(observations))
            actions_tensor = torch.tensor(actions)
            log_probs_tensor = torch.stack(log_probs_old).float()
            if isinstance(returns, torch.Tensor):
                returns_tensor = returns.detach().clone().float()
            else:
                returns_tensor = torch.tensor(returns, dtype=torch.float32)

            values_tensor = torch.tensor(values, dtype=torch.float32)

            advantages = returns_tensor - values_tensor
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            agent.update(obs_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages, epochs=update_epochs)

        obs = validation_env.reset()
        done = False
        val_profit = 0
        while not done:
            with torch.no_grad():
                action, _, _ = agent.get_action(obs)
            obs, reward, done, _ = validation_env.step(action)
            val_profit += reward

        return val_profit

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)

    best_params = study.best_trial.params
    print(f"[INFO] Best hyperparameters from Optuna: {best_params}")

    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim,
                     gamma=best_params["gamma"],
                     clip_epsilon=best_params["clip_epsilon"],
                     lr=best_params["lr"])

    rollout_len = best_params["rollout_len"]
    update_epochs = best_params["update_epochs"]

    for episode in range(no_episodes):
        obs = training_env.reset()
        done = False
        total_return = 0

        observations, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

        for step in range(rollout_len):
            action, log_prob, _ = agent.get_action(obs)
            next_obs, reward, done, _ = training_env.step(action)

            with torch.no_grad():
                _, value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

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

        obs_tensor = torch.FloatTensor(np.array(observations))
        actions_tensor = torch.tensor(actions)
        log_probs_tensor = torch.stack(log_probs_old).float()
        if isinstance(returns, torch.Tensor):
            returns_tensor = returns.detach().clone().float()
        else:
            returns_tensor = torch.tensor(returns, dtype=torch.float32)

        values_tensor = torch.tensor(values, dtype=torch.float32)

        advantages = returns_tensor - values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        agent.update(obs_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages, epochs=update_epochs)
        print(f"Train_Two_Agent Speaking : Episode {episode + 1} finished. Total return: {total_return}")

    def evaluate(env, label="Unknown"):
        print(f"\n📊 Evaluating agent on {label}...")
        obs = env.reset()
        done = False
        step_count = 0
        profit_changes = []
        actions_taken = []
        final_info = {}

        while not done:
            with torch.no_grad():
                action, _, _ = agent.get_action(obs)
            obs, reward, done, info = env.step(action)

            profit_change = info.get("profit_change", 0)
            profit_changes.append(profit_change)
            actions_taken.append(action)
            final_info = info
            step_count += 1

        return {
            "label": label,
            "profits": profit_changes,
            "actions": actions_taken,
            "steps": step_count,
            "final_inventory": final_info.get("inventory", 0),
            "final_cash": final_info.get("cash", 0)
        }

    def summarize_performance(metrics):
        profits = np.array(metrics["profits"])
        actions = np.array(metrics["actions"])

        total_profit = np.sum(profits)
        avg_profit = np.mean(profits)
        max_gain = np.max(profits)
        max_loss = np.min(profits)
        pos_steps = np.sum(profits > 0)
        neg_steps = np.sum(profits < 0)
        neut_steps = np.sum(profits == 0)
        steps = metrics["steps"]

        buy = np.sum(actions == 1)
        sell = np.sum(actions == 2)
        hold = np.sum(actions == 0)

        buy_pct = 100 * buy / steps
        sell_pct = 100 * sell / steps
        hold_pct = 100 * hold / steps

        return f"""
--- {metrics['label']} Phase ---
Total Profit: [{total_profit:.6f}]
Avg Profit/Step: [{avg_profit:.8f}]
Max Gain: [{max_gain:.6f}]
Max Loss: [{max_loss:.6f}]
Steps: {steps}
Positive Steps: [{pos_steps}]
Negative Steps: [{neg_steps}]
Neutral Steps: [{neut_steps}]
Buy: {buy} times ({buy_pct:.2f}%)
Sell: {sell} times ({sell_pct:.2f}%)
Hold: {hold} times ({hold_pct:.2f}%)
Final Inventory: {metrics["final_inventory"]}
Final Cash: [{metrics["final_cash"]}]
""".strip()

    def write_report(best_params, train_metrics, val_metrics, test_metrics):
        report_dir = r"C:\Users\Abbass Zahreddine\Documents\GitHub\Task_1_Preparation\Code\Source_Code\Version_4\agent_reports"
        os.makedirs(report_dir, exist_ok=True)
        filename = f"agent_report_{report_number}.txt"
        path = os.path.join(report_dir, filename)

        report = f"""==================== AGENT PERFORMANCE REPORT ====================
Used Hyperparameters: {best_params}

{summarize_performance(train_metrics)}

{summarize_performance(val_metrics)}

{summarize_performance(test_metrics)}
"""
        with open(path, "w") as f:
            f.write(report)
        print(f"\n📄 Report saved to: {path}")

    train_metrics = evaluate(training_env, label="Training")
    val_metrics = evaluate(validation_env, label="Validation")
    test_metrics = evaluate(testing_env, label="Testing")

    write_report(best_params, train_metrics, val_metrics, test_metrics)

    return [train_metrics["profits"], val_metrics["profits"], test_metrics["profits"]]



#The naming system is fucked up right now no problem; The function named train_three_agent down here
# is the first to write the inner data in Excel and calculate the micro-metrics. And save them
# in this state I aim to set the boundaries for micro-metrics.



#Optuna with Excel data saving
#With number of episode 30 we will examine the results here ()

def train_three_agent(report_number="", no_episodes=60):
    training_env, validation_env, testing_env = Environment.with_splits_time_series(data)
    obs_dim, act_dim = 1, 3

    if os.path.exists(EXCEL_PATH):
        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    else:
        df = pd.DataFrame(columns=[
            "batch_ID", "agent_id", "phase", "episode_number", "episode_return", "episode_length",
            "avg_reward_per_step", "max_reward", "min_reward", "value_mean", "value_std", "value_last",
            "buy_count", "sell_count", "hold_count", "action_entropy", "final_inventory", "final_cash",
            "profit_change_total", "advantage_mean", "advantage_std", "advantage_max", "advantage_min",
            "returns_mean", "returns_std", "Unnamed: 24", "var_profit", "std_profit", "cv_profit", "iqr_profit",
            "delta_profit", "return_gap_loss", "fitting_ratio", "delta_entropy", "policy_div_spread",
            "entropy_action", "action_dist_std", "kl_train_test", "diversity_action_ratio",
            "Buy", "Sell", "Hold", "generalization_index", "Stability W1", "Stability W2", "Stability W3", "Stability W4",
            "General W1", "General W2", "General W3", "General W4", "General W5", "Diversity W1",
            "Diversity W2", "Diversity W3", "Diversity W4", "Diversity W5", "Stability", "General", "Diversity"
        ])

    batch_id = df["batch_ID"].max() + 1 if not df.empty else 1

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.999)
        clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.3)
        rollout_len = trial.suggest_categorical("rollout_len", [128, 256, 512])
        update_epochs = trial.suggest_int("update_epochs", 5, 15)

        agent = PPOAgent(obs_dim, act_dim, gamma, clip_epsilon, lr)

        for episode in range(no_episodes):
            obs = training_env.reset()
            done = False
            observations, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []
            total_return = 0

            for _ in range(rollout_len):
                action, log_prob, _ = agent.get_action(obs)
                next_obs, reward, done, info = training_env.step(action)
                with torch.no_grad():
                    _, value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

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

            obs_tensor = torch.FloatTensor(np.array(observations))
            actions_tensor = torch.tensor(actions)
            log_probs_tensor = torch.stack(log_probs_old).float()
            if isinstance(returns, torch.Tensor):
                returns_tensor = returns.detach().clone().float()
            else:
                returns_tensor = torch.tensor(returns, dtype=torch.float32)
            values_tensor = torch.tensor(values, dtype=torch.float32)

            advantages = returns_tensor - values_tensor
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            agent.update(obs_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages, epochs=update_epochs)

        obs = validation_env.reset()
        done = False
        val_profit = 0
        while not done:
            with torch.no_grad():
                action, _, _ = agent.get_action(obs)
            obs, reward, done, _ = validation_env.step(action)
            val_profit += reward
        return val_profit

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)
    best_params = study.best_trial.params

    agent = PPOAgent(obs_dim, act_dim,
                     gamma=best_params["gamma"],
                     clip_epsilon=best_params["clip_epsilon"],
                     lr=best_params["lr"])

    rollout_len = best_params["rollout_len"]
    update_epochs = best_params["update_epochs"]

    for episode in range(no_episodes):
        obs = training_env.reset()
        done = False
        observations, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []
        total_return = 0

        for _ in range(rollout_len):
            action, log_prob, _ = agent.get_action(obs)
            next_obs, reward, done, info = training_env.step(action)
            with torch.no_grad():
                _, value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

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

        obs_tensor = torch.FloatTensor(np.array(observations))
        actions_tensor = torch.tensor(actions)
        log_probs_tensor = torch.stack(log_probs_old).float()
        if isinstance(returns, torch.Tensor):
            returns_tensor = returns.detach().clone().float()
        else:
            returns_tensor = torch.tensor(returns, dtype=torch.float32)
        values_tensor = torch.tensor(values, dtype=torch.float32)

        advantages = returns_tensor - values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        agent.update(obs_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages, epochs=update_epochs)

        df = save_episode_metrics(df, batch_id, agent_id=1, phase="training_episode", episode_number=episode + 1,
                                  rewards=rewards, actions=actions, values=values, advantages=advantages, info=info, total_return=total_return)

    df = pd.concat([df, pd.DataFrame([{col: np.nan for col in df.columns}])], ignore_index=True)
    df.to_excel(EXCEL_PATH, index=False, sheet_name=SHEET_NAME)
    print(f"✅ Data saved to {EXCEL_PATH}")

    return df

#These are Working for Now:
def train_three_agent_test(report_number="", no_episodes=60):
    training_env, validation_env, testing_env = Environment.with_splits_time_series(data)
    obs_dim, act_dim = 1, 3

    if os.path.exists(EXCEL_PATH):
        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    else:
        df = pd.DataFrame(columns=[
            "batch_ID", "agent_id", "phase", "episode_number", "episode_return", "episode_length",
            "avg_reward_per_step", "max_reward", "min_reward", "value_mean", "value_std", "value_last",
            "buy_count", "sell_count", "hold_count", "action_entropy",
            "final_inventory", "final_cash", "profit_change_total", "advantage_mean", "advantage_std",
            "advantage_max", "advantage_min", "returns_mean", "returns_std", "Unnamed: 24", "var_profit",
            "std_profit", "cv_profit", "iqr_profit", "delta_profit", "return_gap_loss", "fitting_ratio",
            "delta_entropy", "policy_div_spread", "entropy_action", "action_dist_std", "kl_train_test",
            "diversity_action_ratio", "generalization_index", "Stability W1", "Stability W2", "Stability W3",
            "Stability W4", "General W1", "General W2", "General W3", "General W4", "General W5", "Diversity W1",
            "Diversity W2", "Diversity W3", "Diversity W4", "Diversity W5", "Stability", "General", "Diversity"
        ])

    batch_id = df["batch_ID"].max() + 1 if not df.empty else 1

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.999)
        clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.3)
        rollout_len = trial.suggest_categorical("rollout_len", [128, 256, 512])
        update_epochs = trial.suggest_int("update_epochs", 5, 15)

        agent = PPOAgent(obs_dim, act_dim, gamma, clip_epsilon, lr)

        for episode in range(no_episodes):
            obs = training_env.reset()
            done = False
            observations, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []
            total_return = 0

            for _ in range(rollout_len):
                action, log_prob, _ = agent.get_action(obs)
                next_obs, reward, done, info = training_env.step(action)
                with torch.no_grad():
                    _, value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

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

            obs_tensor = torch.FloatTensor(np.array(observations))
            actions_tensor = torch.tensor(actions)
            log_probs_tensor = torch.stack(log_probs_old).float()
            if isinstance(returns, torch.Tensor):
                returns_tensor = returns.detach().clone().float()
            else:
                returns_tensor = torch.tensor(returns, dtype=torch.float32)

            values_tensor = torch.tensor(values, dtype=torch.float32)

            advantages = returns_tensor - values_tensor
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            agent.update(obs_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages, epochs=update_epochs)

        obs = validation_env.reset()
        done = False
        val_profit = 0
        while not done:
            with torch.no_grad():
                action, _, _ = agent.get_action(obs)
            obs, reward, done, _ = validation_env.step(action)
            val_profit += reward
        return val_profit

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)
    best_params = study.best_trial.params

    agent = PPOAgent(obs_dim, act_dim,
                     gamma=best_params["gamma"],
                     clip_epsilon=best_params["clip_epsilon"],
                     lr=best_params["lr"])

    rollout_len = best_params["rollout_len"]
    update_epochs = best_params["update_epochs"]

    for episode in range(no_episodes):
        obs = training_env.reset()
        done = False
        observations, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []
        total_return = 0

        for _ in range(rollout_len):
            action, log_prob, _ = agent.get_action(obs)
            next_obs, reward, done, info = training_env.step(action)
            with torch.no_grad():
                _, value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

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

        obs_tensor = torch.FloatTensor(np.array(observations))
        actions_tensor = torch.tensor(actions)
        log_probs_tensor = torch.stack(log_probs_old).float()
        returns_tensor = returns.detach().clone().float()
        values_tensor = torch.tensor(values, dtype=torch.float32)

        advantages = returns_tensor - values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        df = save_episode_metrics(
            df,
            batch_id=batch_id,
            agent_id=1,
            phase="training_episode",
            episode_number=episode + 1,
            rewards=rewards,
            actions=actions,
            values=values,
            advantages=advantages,
            info=info,
            total_return=total_return
        )

    # Insert an empty row between training and eval phases
    df = pd.concat([df, pd.DataFrame([{col: np.nan for col in df.columns}])], ignore_index=True)

    df = evaluate_phase(agent, training_env, phase="training_eval", df=df, batch_id=batch_id, episode_number=999)
    df = evaluate_phase(agent, validation_env, phase="validation", df=df, batch_id=batch_id, episode_number=1000)
    df = evaluate_phase(agent, testing_env, phase="testing", df=df, batch_id=batch_id, episode_number=1001)

    df.to_excel(EXCEL_PATH, index=False, sheet_name=SHEET_NAME)
    print(f"✅ Data saved to {EXCEL_PATH}")

    return df
def evaluate_phase(agent, env, phase, df, batch_id, episode_number):
    obs = env.reset()
    done = False

    observations = []
    actions = []
    rewards = []
    dones = []
    values = []

    total_return = 0

    while not done:
        with torch.no_grad():
            action, _, _ = agent.get_action(obs)
            _, value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

        next_obs, reward, done, info = env.step(action)

        observations.append(obs)
        actions.append(int(action))  # ensure int
        rewards.append(reward)
        dones.append(done)
        values.append(value.item())

        total_return += reward
        obs = next_obs

    with torch.no_grad():
        _, last_value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

    returns = agent.compute_returns(rewards, dones, last_value.item())
    returns_tensor = returns.detach().clone().float()
    values_tensor = torch.tensor(values, dtype=torch.float32)

    advantages = returns_tensor - values_tensor
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    df = save_episode_metrics(
        df,
        batch_id=batch_id,
        agent_id=1,
        phase=phase,
        episode_number=episode_number,
        rewards=rewards,
        actions=actions,
        values=values,
        advantages=advantages,
        info=info,
        total_return=total_return
    )

    return df



#Now we are making the functions for testing things