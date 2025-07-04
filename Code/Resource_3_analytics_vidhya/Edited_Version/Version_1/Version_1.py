import numpy as np
import math
import random
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam

# ---------------------------- Basics ----------------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def formatPrice(n):
    return ("-Rs." if n < 0 else "Rs.") + "{0:.2f}".format(abs(n))

def getStockDataVec(filename):
    vec = []
    lines = open(filename + ".csv", "r").read().splitlines()
    for line in lines[3:]:
        variable = float(line.split(",")[4])  # Open column
        vec.append(variable)
    return vec

def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])

# ---------------------------- Agent ----------------------------

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size
        self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = load_model(model_name) if is_eval else self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state, verbose=0)
        return np.argmax(options[0])

    def experience_replay(self, batch_size):
        mini_batch = list(self.memory)[-batch_size:]
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ---------------------------- Training Loop ----------------------------

if __name__ == "__main__":
    stock_name = ("hdfc_bank_5years")
    state_size = 15
    epoch = 15

    agent = Agent(state_size=state_size)
    data = getStockDataVec(stock_name)
    data_length = len(data) - 1
    batch_size = 32

    for e in range(epoch + 1):
        print("Episode " + str(e) + "/" + str(epoch))
        state = getState(data, 0, state_size + 1)
        total_profit = 0
        agent.inventory = []

        for t in range(data_length):
            action = agent.act(state)
            next_state = getState(data, t + 1, state_size + 1)
            reward = 0

            if action == 1:  # buy
                agent.inventory.append(data[t])
                print(f"Step {t}: Agent chose to BUY at price {formatPrice(data[t])}")

            elif action == 2 and len(agent.inventory) > 0:  # sell
                bought_price = agent.inventory.pop(0)
                reward = data[t] - bought_price
                total_profit += reward
                result = "won" if reward > 0 else "lost"
                print(
                    f"Step {t}: Agent chose to SELL at price {formatPrice(data[t])} | Profit: {formatPrice(reward)} -> Agent {result}")
                print(f"Cumulative profit so far: {formatPrice(total_profit)}")

            else:  # hold
                print(f"Step {t}: Agent chose to HOLD")

            done = (t == data_length - 1)
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("--------------------------------------------------")
                print(f"Training episode ended. Total Profit: {formatPrice(total_profit)}")
                print("--------------------------------------------------")

            if len(agent.memory) > batch_size:
                agent.experience_replay(batch_size)