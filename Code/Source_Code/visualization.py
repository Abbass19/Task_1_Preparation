import matplotlib.pyplot as plt


def plot_training_returns(returns_list):
    """
    Plot total return (reward) over episodes.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(returns_list, label="Total Return")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward / Return")
    plt.title("PPO Training Return per Episode")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_trading_actions(prices, actions):
    """
    Plot the stock price with Buy/Sell actions.
    Actions: 0 = Hold, 1 = Buy, 2 = Sell
    """
    plt.figure(figsize=(12, 5))
    plt.plot(prices, label="Stock Price", color='black')

    buys = [i for i, a in enumerate(actions) if a == 1]
    sells = [i for i, a in enumerate(actions) if a == 2]

    plt.scatter(buys, [prices[i] for i in buys], marker='^', color='green', label='Buy', s=80)
    plt.scatter(sells, [prices[i] for i in sells], marker='v', color='red', label='Sell', s=80)

    plt.title("Trading Actions Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
