import sys
from Agent import *
from Basics import *


#Parameters 1 :
stock_name = input("Enter stock_name, window_size, Episode_count")
state_size = input()
epoch = input()

#Verfiying Input (Try Catch)
stock_name = str(stock_name)
state_size = int(state_size)
epoch = int(epoch)

#Parameters 2:
agent = Agent(state_size= state_size)
data = getStockDataVec(stock_name)
data_length = len(data) - 1
batch_size = 32


#Training Process
for e in range(epoch + 1):
    print("Episode " + str(e) + "/" + str(epoch))
    state = getState(data, 0, state_size + 1)
    total_profit = 0
    agent.inventory = []
    for t in range(data_length):
        action = agent.act(state)
        # sit
        next_state = getState(data, t + 1, state_size + 1)
        reward = 0
        if action == 1: # buy
            agent.inventory.append(data[t])
            print("Buy: " + formatPrice(data[t]))
        elif action == 2 and len(agent.inventory) > 0: # sell
            bought_price  = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")
        if len(agent.memory) > batch_size:
            agent.Experience_Replay(batch_size)
    if e % 10 == 0:
        agent.model.save(str(e))