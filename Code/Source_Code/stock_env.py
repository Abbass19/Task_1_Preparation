import gym
from gym import spaces
import numpy as np
import pandas as pd

class Environment(gym.Env):
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.current_step = 0

        # Actions: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

        # Observations: [current price]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.float32
        )

        self.inventory = 0
        self.initial_cash = 10000
        self.cash = self.initial_cash

    def reset(self):
        self.current_step = 0
        self.inventory = 0
        self.cash = self.initial_cash
        return self._get_observation()

    def _get_observation(self):
        price = self.data.loc[self.current_step, 'CPCP']
        return np.array([price], dtype=np.float32)

    def step(self, action):
        done = False
        price = self.data.loc[self.current_step, 'CPCP']

        # Apply action
        if action == 1:  # Buy
            if self.cash >= price:
                self.inventory += 1
                self.cash -= price
        elif action == 2:  # Sell
            if self.inventory > 0:
                self.inventory -= 1
                self.cash += price
        # Hold = do nothing

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        total_assets = self.cash + self.inventory * price
        reward = total_assets  # Simplified reward

        obs = self._get_observation()
        info = {
            'step': self.current_step,
            'inventory': self.inventory,
            'cash': self.cash,
            'price': price
        }

        return obs, reward, done, info

    def render(self, mode='human'):
        price = self.data.loc[self.current_step, 'Close']
        total_assets = self.cash + self.inventory * price
        print(f"Step: {self.current_step}")
        print(f"Cash: {self.cash:.2f}")
        print(f"Inventory: {self.inventory}")
        print(f"Price: {price:.2f}")
        print(f"Total Assets: {total_assets:.2f}")
