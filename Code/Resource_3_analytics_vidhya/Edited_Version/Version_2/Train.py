import matplotlib.pyplot as plt
from Agent import Agent
from Basics import *

stock_name = "hdfc_bank_5years"
state_size = 15
epoch = 15

agent = Agent(state_size=state_size)
data = getStockDataVec(stock_name)
data_length = len(data) - 1
batch_size = 32

total_profits = []

for e in range(epoch + 1):
    print(f"Episode {e}/{epoch}")
    state = getState(data, 0, state_size + 1)
    total_profit = 0
    agent.inventory = []

    for t in range(data_length):
        action = agent.act(state)
        next_state = getState(data, t + 1, state_size + 1)
        reward = 0

        if action == 1:  # buy
            agent.inventory.append(data[t])
            print(f"Step {t}: Agent chose to BUY at {formatPrice(data[t])}")
        elif action == 2 and len(agent.inventory) > 0:  # sell
            bought_price = agent.inventory.pop(0)
            relevant_change = data[t] - bought_price
            action_string = "+++ WON! +++ " if relevant_change > 0 else "--- LOST ---"
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print(f"Step {t}: Agent chose to SELL at {formatPrice(data[t])} | The agent has {action_string} this time with {relevant_change} making Cumulative Profit: {total_profit}")
        else:
            print(f"Step {t}: Agent chose to HOLD")

        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            print(f"Total Profit this episode: {formatPrice(total_profit)}")
            print("--------------------------------")
            total_profits.append(total_profit)

        if len(agent.memory) > batch_size:
            agent.Experience_Replay(batch_size)

    if e % 10 == 0:
        agent.model.save(f"model_episode_{e}.h5")

# Plotting total profits over episodes
plt.figure(figsize=(10, 6))
plt.plot(total_profits, label="Total Profit per Episode", color="green")
plt.xlabel("Episode")
plt.ylabel("Total Profit")
plt.title(f"Training Progress for {stock_name}")
plt.legend()
plt.grid(True)
plt.show()
