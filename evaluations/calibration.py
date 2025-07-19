import json
import os
import time

from datasets import load_dataset, load_from_disk

from evaluations.utils import clean_string
from models.model_adapter import ModelAdapter
import numpy as np
import matplotlib

matplotlib.use('Agg')


def evaluate_calibration(model_name, num_samples=5):
    """
    Evaluates the model's calibration using OpenBookQA.
    """

    print("Starting evaluation calibration...")

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

    sample_dataset = dataset["test"]
    if num_samples > 0:
        sample_dataset = dataset["test"][:num_samples]
    else:
        num_samples = sample_dataset.num_rows

    results = {"details": []}
    start_time = time.time()
    for i, question_stem in enumerate(sample_dataset['question_stem']):
        index_performance = {}
        choices = sample_dataset["choices"][i]["text"]
        labels = sample_dataset["choices"][i]["label"]
        correct_answer_label = sample_dataset["answerKey"][i]
        correct_answer = [choices[idx] for idx, val in enumerate(labels) if val == correct_answer_label][0]

        # Prompting model for an answer with confidence estimation
        prompt = clean_string(
            f"{question_stem} (Please provide only one answer and your confidence from 0 to 100 from the choices e.g: your answer 50. no extra text.). " \
            f"your choices are: \n" + ", ".join(choices))
        response = clean_string(model.generate_response(prompt))
        index_performance['index'] = i
        index_performance['question'] = prompt
        index_performance['correct_answer'] = correct_answer
        # Extracting confidence and answer
        try:
            answer, confidence = response.rsplit(" ", 1)
            confidence = float(confidence)/100
        except ValueError:
            answer, confidence = response, 0  # Default confidence

        is_correct = answer == correct_answer
        index_performance['model_answer'] = answer
        index_performance['model_confidence'] = confidence
        index_performance['is_correct'] = is_correct

        correctness.append(is_correct)
        confidences.append(confidence)
        results['details'].append(index_performance)

    # Expected Calibration Error (ECE)
    # bins = np.linspace(0, 1, 10)
    # bin_correctness = np.digitize(confidences, bins)
    ece = compute_ece(confidences, correctness)

    elapsed_time = round(time.time() - start_time, 2)

    result = round(1 - ece, 4)  # Higher is better
    results['result'] = result
    # plot_reliability_diagram(confidences, correctness, model_name=model_name)

    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    # Create results directory if not exists
    os.makedirs(f"results/{safe_model_name}", exist_ok=True)
    results_file_path = f"results/{safe_model_name}/{safe_model_name}_calibration.json"
    with open(results_file_path, "w") as f:
        json.dump(results, f, indent=4)

    # Summary result
    summary_result = {
        "model_name": model_name,
        "calibration_score": result,
        "samples_evaluated": num_samples,
        # "correct_count": sum([r["is_correct"] for r in results]),
        "execution_time_seconds": elapsed_time,
        "detailed_results_file": results_file_path
    }

    return summary_result


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


if __name__ == "__main__":
    print(evaluate_calibration("google/gemma-3-4b", 5))
