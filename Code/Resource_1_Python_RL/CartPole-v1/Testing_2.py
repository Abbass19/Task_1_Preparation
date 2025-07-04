
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from helper_methods import setup_training_directories, testing_function_2 # Ensure this is correct

# Preparing the loading Link:
log_path, model_path = setup_training_directories()
model_filename = "CartPole_20000"
full_model_path = os.path.join(model_path, model_filename)

# --- PHASE 1: Evaluation ---
print("--- Starting Evaluation ---")
eval_env = gym.make('CartPole-v1') # Set render_mode here
eval_env = Monitor(eval_env)
eval_env = DummyVecEnv([lambda: eval_env])

# Load the PPO model
model = PPO.load(full_model_path, env=eval_env)

# Evaluate the policy, with rendering
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=False)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# IMPORTANT: Close the environment used for evaluation before the next part.
eval_env.close()
print("--- Evaluation complete, environment closed ---")

# --- PHASE 2: Testing Function ---
print("\n--- Starting Testing Function ---")
# Create a *new* environment instance for testing_function_2
# It's crucial to create a new one, as the previous one was closed.
# Also ensure it's wrapped the same way (DummyVecEnv) if your testing_function_2 expects it.
test_env = gym.make('CartPole-v1', render_mode="human") # Create with human render_mode
test_env = Monitor(test_env) # Monitor is good practice
test_env = DummyVecEnv([lambda: test_env]) # Wrap in DummyVecEnv

# Call your testing function
results = testing_function_2(model, 2, environment=test_env, Display=True)
print("\nTesting function results summary:")
for res in results:
    print(f"  Episode {res['episode']}: Score = {res['score']:.2f}, Steps = {res['steps']}")

# Ensure the testing environment is closed after all operations are done
test_env.close()
print("--- Testing function complete, environment closed ---")