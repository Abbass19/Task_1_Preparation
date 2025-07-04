import gym
import os
from stable_baselines3.common.env_util import make_atari_env

# Create the Atari environment (Breakout)
env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=0)

# Wrap it with frame stacker (for richer observations)
from stable_baselines3.common.vec_env import VecFrameStack
env = VecFrameStack(env, n_stack=4)

# Run episodes with random actions
episodes = 5
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()  # Might crash if no GUI available
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score += reward[0]  # because obs and reward are vectors now

    print(f'Episode {episode} — Score: {score}')

env.close()