import json
import matplotlib
from matplotlib import cm
import seaborn as sns

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


def draw_final_score():
    # Prepare data
    models = list(data.keys())
    results = {}

    for model in models:
        model_data = data[model]
        try:
            results[model_data["final_score"]["model"]] = model_data["final_score"]["final_safety_score"]
        except KeyError:
            results[model_data["final_score"]["model"]] = 0  # Use None for missing data

    # Sort data by score (descending)
    sorted_data = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

    models = list(sorted_data.keys())
    scores = list(sorted_data.values())

    # Plot
    plt.figure(figsize=(12, 7))
    bars = plt.bar(models, scores, color='cornflowerblue', edgecolor='black', linewidth=0.7)

    # colors = [, , , '#7CFC00', '#ADFF2F', '#D3D3D3']

    # # Highlight top performer
    bars[0].set_color('#006400')
    bars[0].set_edgecolor('black')

    bars[1].set_color('#228B22')
    bars[1].set_edgecolor('black')

    bars[2].set_color('#32CD32')
    bars[2].set_edgecolor('black')

    bars[3].set_color('#7CFC00')
    bars[3].set_edgecolor('black')

    bars[5].set_color('#D3D3D3')
    bars[5].set_edgecolor('black')

    # Add score labels on top
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Aesthetics
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Score Comparison & Ranking', fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig('results/model_comparison_chart.png', dpi=300)


def get_colors(n):
    cmap = matplotlib.colormaps.get_cmap('tab10')  # Or 'tab20', 'Set2', etc.
    return [cmap(i % 10) for i in range(n)]


def draw_metric():
    # Your data is assumed to be already loaded in the `data` variable
    # Define the metrics to extract and their corresponding score keys
    metrics_names = {
        "honesty": "honesty_score",
        "bias": "bias_score",
        "calibration": "calibration_score",
        "deception": "resistance_score",  # use resistance_score, not deception_rate
        "consistency": "consistency_score"
    }

    # Extract and plot each metric separately
    for metric, score_key in metrics_names.items():
        model_names = []
        scores = []

        for model_name, model_data in data.items():
            metric_data = model_data.get(metric, {})
            score = metric_data.get(score_key, None)

            if score is not None:
                model_names.append(model_name)
                scores.append(score)

        # Plotting
        plt.figure(figsize=(10, 5))
        x = np.arange(len(model_names))
        colors = get_colors(len(model_names))
        bars = plt.bar(model_names, scores, color=colors, width=0.4, edgecolor='black')

        # Add score labels on top
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{score:.3f}", ha='center', va='bottom', fontsize=9)

        plt.title(f"{metric.capitalize()} Score by Model")
        plt.xlabel("Models")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.xticks(rotation=30, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('results/'+metric+'.png', dpi=300)


def draw_each_model(output_dir="results"):
    sns.set_style("whitegrid")
    # plt.style.use("seaborn-whitegrid")

    for model_name, model_data in data.items():
        # Collect non-null metrics
        entries = [
            (label.replace("_", " "), model_data.get(sec, {}).get(key))
            for label, (sec, key) in metrics.items()
            if model_data.get(sec, {}).get(key) is not None
        ]
        if not entries:
            continue

        # Sort descending
        entries.sort(key=lambda x: x[1], reverse=True)
        labels, values = zip(*entries)
        y_pos = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0.3, 0.7, len(labels)))
        bars = ax.barh(y_pos, values, color=colors)
        ax.invert_yaxis()

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Score", fontsize=12)
        ax.set_title(f"Model Metrics: {model_name}", fontsize=14, fontweight="bold")

        # Annotate
        max_val = max(values)
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + max_val * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}",
                va="center",
                fontsize=9
            )

        plt.tight_layout()
        safe_name = model_name.replace("/", "_").replace(":", "_")
        path = f"{output_dir}/{safe_name}_metrics.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")


# def draw_each_model():
#     # Generate separate chart for each model
#     for model_name, model_data in data.items():
#         metric_labels = []
#         metric_values = []
#
#         for label, (section, key) in metrics.items():
#             try:
#                 value = model_data[section][key]
#             except KeyError:
#                 value = None
#             metric_labels.append(label.replace("_", " "))
#             metric_values.append(value)
#
#         # Plot
#         x = np.arange(len(metric_labels))
#         fig, ax = plt.subplots(figsize=(14, 6))
#         bars = ax.bar(x, metric_values, color='skyblue')
#
#         # Annotate bars
#         for bar, val in zip(bars, metric_values):
#             if val is not None:
#                 ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
#                         f'{val:.2f}', ha='center', va='bottom', fontsize=8)
#
#         ax.set_title(f"Metrics for {model_name}")
#         ax.set_ylabel("Score")
#         ax.set_xticks(x)
#         ax.set_xticklabels(metric_labels, rotation=45, ha="right")
#         ax.set_ylim([0, max(metric_values) * 1.2 if any(metric_values) else 1])
#
#         plt.tight_layout()
#         safe_model_name = model_name.replace("/", "_").replace(":", "_")
#         output_path = f"results/{safe_model_name}_metrics.png"
#         plt.savefig(output_path)
#         plt.close()
#         print(f"Saved: {output_path}")


def run_analysis():
    draw_each_model()
    draw_all()
    draw_final_score()
    draw_metric()


