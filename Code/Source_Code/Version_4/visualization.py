import math
import matplotlib.pyplot as plt

def get_grid_dimensions(n):
    n = min(n, 9)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    return rows, cols

def display_optuna_trials(profit_lists):
    n = len(profit_lists)
    rows, cols = get_grid_dimensions(n)

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)
    fig.suptitle("Cumulative Profit Curves of Top Optuna Trials", fontsize=16)

    for i, profits in enumerate(profit_lists):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        ax.plot(profits, label="Cumulative Profit", color='blue')
        ax.set_title(f"Trial #{i+1}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Profit")
        ax.grid(True)
        ax.legend()

    # Hide unused subplots
    for j in range(n, rows*cols):
        r = j // cols
        c = j % cols
        axes[r][c].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def display_agents_performance(data_lists):
    """
    data_lists: list of tuples/lists: (train_profit, val_profit, test_profit)
    """
    n = len(data_lists)
    rows, cols = get_grid_dimensions(n)

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), squeeze=False)
    fig.suptitle("Agent Performance: Train/Val/Test Profit Curves", fontsize=16)

    for i, (train_p, val_p, test_p) in enumerate(data_lists):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        ax.plot(train_p, label='Train', color='green')
        ax.plot(val_p, label='Validation', color='orange')
        ax.plot(test_p, label='Test', color='red')
        ax.set_title(f"Agent #{i+1}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Profit")
        ax.grid(True)
        ax.legend()

    # Hide unused subplots
    for j in range(n, rows*cols):
        r = j // cols
        c = j % cols
        axes[r][c].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


