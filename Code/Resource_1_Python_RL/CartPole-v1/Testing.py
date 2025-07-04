import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import  matplotlib.pyplot as plt
from helper_methods import setup_training_directories,testing_function, testing_function_2



#Preparing the loading Link:
log_path, model_path = setup_training_directories()
model_filename = "CartPole_20000"
full_model_path = os.path.join(model_path, model_filename)

#Loaing the PPO model
env = gym.make('CartPole-v1') # <<< Set render_mode here

env = Monitor(env) # Wrap with Monitor
env = DummyVecEnv([lambda :env])
model = PPO.load(full_model_path, env= env)


# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

testing_function_2(model, 2, environment=env, Display=True)

env.close() # Always close the environment when done




