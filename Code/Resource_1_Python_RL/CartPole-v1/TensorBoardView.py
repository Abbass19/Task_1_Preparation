import os
import webbrowser
from tensorboard import program

# Path to your logs
log_dir = r"/Code/Resource_1_Python_RL/CartPole-v1/Training/Logs/PPO_6"

# Start TensorBoard
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', log_dir])
url = tb.launch()

print(f"TensorBoard is running at {url}")
webbrowser.open(url)  # Opens TensorBoard in your default web browser