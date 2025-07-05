from model_manager import ModelManager
from ppo_agent import PPOAgent
from log_api import initialize_log_if_missing, save_agent_result
from train import hyperparameter_search, train_one_agent, train_multiple_agents


def main():
    model_manager = ModelManager(model_class=PPOAgent)
    model_manager.display_info()
    initialize_log_if_missing()



    # Step 3: Display current saved models (if any)
    model_manager.display_info()
    # hyperparameter_search( n_trials=5, num_episodes=40,visualization=False)
    train_one_agent(4, no_episodes=20)




if __name__ == "__main__":
    main()