# testing_train_frequency.py

import csv
import subprocess

stock_name = "your_stock"  # put your csv file name without .csv
state_size = 32
epoch = 10
batch_sizes_to_test = [5, 10, 15, 20]

results = []

for train_freq in batch_sizes_to_test:
    print(f"Running training with train_every_n_steps = {train_freq}")

    # Run train_v3.py script with inputs using subprocess
    proc = subprocess.Popen(
        ['python', 'train_v3.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    # Send inputs: stock_name, state_size, epoch, train_freq
    inputs = f"{stock_name}\n{state_size}\n{epoch}\n{train_freq}\n"
    stdout, stderr = proc.communicate(input=inputs)

    # Parse total profits from output logs
    profits = []
    for line in stdout.split('\n'):
        if "Total Profit this episode:" in line:
            profit_str = line.split(": Rs.")[-1]
            profits.append(float(profit_str))

    avg_profit = sum(profits) / len(profits) if profits else 0
    print(f"Avg profit for train_every_n_steps={train_freq}: Rs.{avg_profit:.2f}")
    results.append((train_freq, avg_profit))

    # Optionally save output logs to file if you want
    with open(f"output_train_freq_{train_freq}.txt", "w") as f:
        f.write(stdout)

# Save results to CSV for easy analysis
with open('train_frequency_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['train_every_n_steps', 'average_profit'])
    writer.writerows(results)

print("Testing complete. Results saved to train_frequency_results.csv")
