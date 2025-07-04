import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import  matplotlib.pyplot as plt
from helper_methods import setup_training_directories

# Load the environment
environment_name = 'CartPole-v1'

#Preparing the loading Link:
log_path, model_path = setup_training_directories()
model_filename = "CartPole_20000"
full_model_path = os.path.join(model_path, model_filename)

#train our model
env = gym.make('CartPole-v1') # <<< Set render_mode here
env = Monitor(env) # Wrap with Monitor
env = DummyVecEnv([lambda :env])

#Check if we Have previously trained model :
model = PPO.load(full_model_path, env=env)

model.learn(total_timesteps=200000)
model.save(full_model_path )

