from model_manager import ModelManager
from ppo_agent import PPOAgent
from log_api import initialize_log_if_missing, save_agent_result
from train import hyperparameter_search, train_one_agent, train_multiple_agents


def main():
    model_manager = ModelManager(model_class=PPOAgent)
    model_manager.display_info()
    initialize_log_if_missing()
    train_one_agent("Test", no_episodes=10)



    # Step 3: Display current saved models (if any)
    # hyperparameter_search( n_trials=5, num_episodes=40,visualization=False)
    for k in range(20):
        print("-------------------------------------------------------------------------------------------")
        print(f"This is the {k}th Iterations")
        train_one_agent(str(k), no_episodes=10)




if __name__ == "__main__":


    main()