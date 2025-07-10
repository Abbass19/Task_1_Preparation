import os
import re
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import json
import pprint
import matplotlib.pyplot as plt
import seaborn as sns

class BatchReportAnalyzer:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.reports_data = []

    def load_reports(self):
        report_files = sorted(
            [f for f in os.listdir(self.folder_path) if f.split('.')[0].isdigit()],
            key=lambda x: int(x.split('.')[0])
        )
        print(f"📂 Found {len(report_files)} report files.")

        for file in report_files:
            full_path = os.path.join(self.folder_path, file)
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                    parsed = self.parse_report(content)
                    if parsed:
                        self.reports_data.append(parsed)
            except Exception as e:
                print(f"❌ Failed to read file {file}: {e}")

        print(f"✅ Successfully parsed {len(self.reports_data)} reports.")

    def parse_report(self, content):
        try:
            result = {}
            sections = re.split(r'--- (Training|Validation|Testing) Phase ---', content)
            result['hyperparams'] = eval(re.search(r"Used Hyperparameters: ({.*})", content).group(1))

            for i in range(1, len(sections), 2):
                phase = sections[i].strip().lower()
                stats = sections[i+1]

                phase_data = {
                    'total_profit': float(re.search(r"Total Profit: \[?([\-\d\.]+)\]?", stats).group(1)),
                    'avg_profit_step': float(re.search(r"Avg Profit/Step: \[?([\-\d\.]+)\]?", stats).group(1)),
                    'max_gain': float(re.search(r"Max Gain: \[?([\-\d\.]+)\]?", stats).group(1)),
                    'max_loss': float(re.search(r"Max Loss: \[?([\-\d\.]+)\]?", stats).group(1)),
                    'steps': int(re.search(r"Steps: (\d+)", stats).group(1)),
                    'positive_steps': int(re.search(r"Positive Steps: \[?(\d+)\]?", stats).group(1)),
                    'negative_steps': int(re.search(r"Negative Steps: \[?(\d+)\]?", stats).group(1)),
                    'neutral_steps': int(re.search(r"Neutral Steps: \[?(\d+)\]?", stats).group(1)),
                    'buy': int(re.search(r"Buy: (\d+)", stats).group(1)),
                    'sell': int(re.search(r"Sell: (\d+)", stats).group(1)),
                    'hold': int(re.search(r"Hold: (\d+)", stats).group(1)),
                    'final_inventory': int(re.search(r"Final Inventory: (\d+)", stats).group(1)),
                    'final_cash': float(re.search(r"Final Cash: \[?([\-\d\.]+)\]?", stats).group(1)),
                }

                result[phase] = phase_data
            return result
        except Exception as e:
            print(f"❌ Failed to parse report content: {e}")
            return None

    def compute_metrics(self):
        metrics = defaultdict(lambda: defaultdict(list))

        for report in self.reports_data:
            for phase in ['training', 'validation', 'testing']:
                if phase in report:
                    data = report[phase]
                    metrics[phase]['total_profit'].append(data['total_profit'])
                    metrics[phase]['avg_profit_step'].append(data['avg_profit_step'])
                    metrics[phase]['max_gain'].append(data['max_gain'])
                    metrics[phase]['max_loss'].append(data['max_loss'])

                    buy, sell, hold = data['buy'], data['sell'], data['hold']
                    actions = ['buy'] * buy + ['sell'] * sell + ['hold'] * hold
                    total_actions = len(actions)

                    if total_actions > 0:
                        action_counts = Counter(actions)
                        entropy = -sum((count / total_actions) * np.log(count / total_actions + 1e-8)
                                       for count in action_counts.values())
                    else:
                        entropy = 0.0

                    metrics[phase]['action_entropy'].append(entropy)

        summary = {}
        for phase, phase_metrics in metrics.items():
            summary[phase] = {
                k: {
                    'mean': float(np.mean(v)),
                    'std': float(np.std(v)),
                    'min': float(np.min(v)),
                    'max': float(np.max(v))
                } for k, v in phase_metrics.items()
            }

        print("📊 Metrics summary computed.")
        return summary

    def generate_plots(self, save_dir):
        def plot_box(metric):
            data = []
            phases = []
            for phase in ['training', 'validation', 'testing']:
                if metric == 'action_entropy':
                    # Use precomputed values from compute_metrics
                    values = self.compute_metrics()[phase][metric]
                    values = [v for v in values.values() if isinstance(v, float)]  # just in case
                    data.extend(values)
                    phases.extend([phase.capitalize()] * len(values))
                else:
                    values = [
                        report[phase][metric]
                        for report in self.reports_data if phase in report and metric in report[phase]
                    ]
                    data.extend(values)
                    phases.extend([phase.capitalize()] * len(values))

            if not data:
                print(f"⚠️ No data found for metric: {metric}")
                return

            plt.figure(figsize=(10, 6))
            sns.boxplot(x=phases, y=data)
            plt.title(f"{metric.replace('_', ' ').title()} by Phase")
            plt.xlabel("Phase")
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True)
            plt.tight_layout()
            plot_path = save_dir / f"batch_report_{metric}_boxplot.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"📦 Saved plot: {plot_path}")

        metrics_to_plot = ['total_profit', 'avg_profit_step', 'max_gain', 'max_loss', 'action_entropy']
        for metric in metrics_to_plot:
            plot_box(metric)


if __name__ == "__main__":
    report_dir = Path(r"C:\Users\Abbass Zahreddine\Documents\GitHub\Task_1_Preparation\Code\Source_Code\Version_3\agent_reports")
    analyzer = BatchReportAnalyzer(report_dir)

    analyzer.load_reports()
    metrics_summary = analyzer.compute_metrics()

    # Save summary
    text_path = report_dir / "batch_report.txt"
    json_path = report_dir / "batch_report.json"
    with open(text_path, "w") as f:
        pprint.pprint(metrics_summary, stream=f)
    with open(json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"✅ Batch report saved as:\n- {text_path}\n- {json_path}")

    # Generate and save plots
    analyzer.generate_plots(report_dir)
