import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# Load JSON data
with open("results/final_report.json", "r") as f:
    data = json.load(f)

# Define metrics and their nested keys
metrics = {
    "honesty_score": ("honesty", "honesty_score"),
    # "honesty_time": ("honesty", "execution_time_seconds"),
    "calibration_score": ("calibration", "calibration_score"),
    # "calibration_time": ("calibration", "execution_time_seconds"),
    "bias_score": ("bias", "bias_score"),
    # "bias_time": ("bias", "execution_time_seconds"),
    "deception_score": ("deception", "resistance_score"),
    # "deception_time": ("deception", "execution_time_seconds"),
    "consistency_score": ("consistency", "consistency_score"),
    # "consistency_time": ("consistency", "execution_time_seconds"),
}

# Ensure results directory exists
os.makedirs("results", exist_ok=True)


def plot_reliability_diagram(confidences, correctness, num_bins=10, model_name="model"):
    confidences = np.array(confidences)
    correctness = np.array(correctness)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accuracies = []
    avg_confidences = []
    counts = []

    for i in range(num_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            avg_conf = np.mean(confidences[in_bin])
            acc = np.mean(correctness[in_bin])
        else:
            avg_conf = 0.0
            acc = 0.0

        avg_confidences.append(avg_conf)
        accuracies.append(acc)
        counts.append(bin_size)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.bar(bin_centers, accuracies, width=0.1, edgecolor='black', alpha=0.7, label='Model')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Reliability Diagram ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_reliability_diagram.png")
    plt.show()


def draw_all():
    # Prepare data
    models = list(data.keys())
    metric_names = list(metrics.keys())
    results = {metric: [] for metric in metric_names}

    for model in models:
        model_data = data[model]
        for metric, (section, key) in metrics.items():
            try:
                results[metric].append(model_data[section][key])
            except KeyError:
                results[metric].append(None)  # Use None for missing data

    # Plotting
    x = np.arange(len(models))
    width = 0.08  # Width of each bar

    fig, ax = plt.subplots(figsize=(16, 8))
    for i, metric in enumerate(metric_names):
        offset = (i - len(metric_names) / 2) * width
        values = results[metric]
        ax.bar(x + offset, values, width, label=metric.replace("_", " "))

    # Labeling
    ax.set_ylabel("Scores")
    ax.set_title("Model Evaluation Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig("results/model_metrics.png")


if __name__ == "__main__":

    # Generate separate chart for each model
    for model_name, model_data in data.items():
        metric_labels = []
        metric_values = []

        for label, (section, key) in metrics.items():
            try:
                value = model_data[section][key]
            except KeyError:
                value = None
            metric_labels.append(label.replace("_", " "))
            metric_values.append(value)

        # Plot
        x = np.arange(len(metric_labels))
        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.bar(x, metric_values, color='skyblue')

        # Annotate bars
        for bar, val in zip(bars, metric_values):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        ax.set_title(f"Metrics for {model_name}")
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=45, ha="right")
        ax.set_ylim([0, max(metric_values) * 1.2 if any(metric_values) else 1])

        plt.tight_layout()
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        output_path = f"results/{safe_model_name}_metrics.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")
        draw_all()

