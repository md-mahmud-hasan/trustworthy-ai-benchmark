import json
import os

from datasets import load_dataset, load_from_disk

from evaluations.utils import clean_string
from models.model_adapter import ModelAdapter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('Agg')


def evaluate_calibration(model_name, num_samples=5):
    """
    Evaluates the model's calibration using OpenBookQA.
    """

    # Define local directory
    local_dir = "data/openbookqa"

    # Check if dataset is already downloaded
    if not os.path.exists(local_dir):
        # Download and save dataset
        dataset = load_dataset("allenai/openbookqa")
        dataset.save_to_disk(local_dir)

    # Load dataset from local directory
    dataset = load_from_disk(local_dir)
    model = ModelAdapter(model_name)

    confidences = []
    correctness = []
    sample_dataset = dataset["test"][:num_samples]
    results = []
    for i, question_stem in enumerate(sample_dataset['question_stem']):
        index_performance = {}
        choices = sample_dataset["choices"][i]["text"]
        labels = sample_dataset["choices"][i]["label"]
        correct_answer_label = sample_dataset["answerKey"][i]
        correct_answer = [choices[idx] for idx, val in enumerate(labels) if val == correct_answer_label][0]

        # Prompting model for an answer with confidence estimation
        prompt = clean_string(
            f"{question_stem} (Please provide only one answer and your confidence from 0 to 1 from the choices e.g: your answer 0.5). " \
            f"your choices are: \n" + ", ".join(choices))
        response = clean_string(model.generate_response(prompt))
        index_performance['index'] = i
        index_performance['question'] = prompt
        index_performance['correct_answer'] = correct_answer
        # Extracting confidence and answer
        try:
            answer, confidence = response.rsplit(" ", 1)
            confidence = float(confidence)
        except ValueError:
            answer, confidence = response, 0  # Default confidence

        is_correct = answer == correct_answer
        index_performance['model_answer'] = answer
        index_performance['model_confidence'] = confidence
        index_performance['is_correct'] = is_correct

        correctness.append(is_correct)
        confidences.append(confidence)
        results.append(index_performance)

    # Expected Calibration Error (ECE)
    # bins = np.linspace(0, 1, 10)
    # bin_correctness = np.digitize(confidences, bins)
    ece = compute_ece(confidences, correctness)

    result = {"calibration_score": round(1 - ece, 4)}  # Higher is better
    plot_reliability_diagram(confidences, correctness, model_name=model_name)

    with open(f"results/{model_name}_calibration.json", "w") as f:
        json.dump(results, f, indent=4)

    return result


def compute_ece(confidences, correctness, num_bins=10):
    confidences = np.array(confidences)
    correctness = np.array(correctness)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        # Get indices of samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(correctness[in_bin])
            bin_ece = (bin_size / len(confidences)) * np.abs(avg_confidence - avg_accuracy)
            ece += bin_ece

    return ece


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


if __name__ == "__main__":
    print(evaluate_calibration("gpt-4o-mini"))
