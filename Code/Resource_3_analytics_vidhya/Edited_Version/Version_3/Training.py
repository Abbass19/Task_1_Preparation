# train_v3.py

import sys
from Agent import Agent
from Basics import *

# Parameters
stock_name = input("Enter stock_name: ")
state_size = int(input("Enter window_size (state_size): "))
epoch = int(input("Enter number of episodes: "))
train_every_n_steps = int(input("Enter training frequency (train every N steps): "))

agent = Agent(state_size=state_size)
data = getStockDataVec(stock_name)
data_length = len(data) - 1
batch_size = 32

# Training process
for e in range(epoch + 1):
    print(f"Episode {e}/{epoch}")
    state = getState(data, 0, state_size + 1)
    total_profit = 0
    agent.inventory = []
    step_counter = 0

    for t in range(data_length):
        action = agent.act(state)

        next_state = getState(data, t + 1, state_size + 1)
        reward = 0

        if action == 1:  # buy
            agent.inventory.append(data[t])
            print(f"Step {t}: Agent chose to BUY at Rs.{data[t]:.2f}")
        elif action == 2 and len(agent.inventory) > 0:  # sell
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print(f"Step {t}: Agent chose to SELL at Rs.{data[t]:.2f} | Profit: Rs.{data[t] - bought_price:.2f}")
        else:
            print(f"Step {t}: Agent chose to HOLD")

        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        step_counter += 1

        # Train only every N steps and only if enough samples are collected
        if len(agent.memory) > batch_size and (step_counter % train_every_n_steps == 0):
            agent.Experience_Replay(batch_size)

        if done:
            print("--------------------------------")
            print(f"Total Profit this episode: Rs.{total_profit:.2f}")
            print("--------------------------------")

    # Save model every 10 episodes
    if e % 10 == 0:
        agent.model.save(f"model_episode_{e}.h5")
