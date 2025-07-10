from model_manager import ModelManager
from ppo_agent import PPOAgent
from log_api import initialize_log_if_missing, save_agent_result
from train import  train_one_agent, train_two_agent, train_two_with_report, train_three_agent, train_three_agent_test

def main():
    model_manager = ModelManager(model_class=PPOAgent)
    model_manager.display_info()
    initialize_log_if_missing()
    train_two_with_report("Test", no_episodes=30)


    #
    # # Step 3: Display current saved models (if any)
    # for k in range(20):
    #     print("-------------------------------------------------------------------------------------------")
    #     print(f"This is the {k}th Iterations")
    #     train_two_with_report(str(k), no_episodes=10)
    #
    #


if __name__ == "__main__":


    main()