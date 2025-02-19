import json
import os

from datasets import load_dataset, load_from_disk
from models.model_adapter import ModelAdapter
import numpy as np


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
    for i, question_stem in enumerate(sample_dataset['question_stem']):

        choices = sample_dataset["choices"][i]["text"]
        labels = sample_dataset["choices"][i]["label"]
        correct_answer_label = sample_dataset["answerKey"][i]
        correct_answer = [choices[idx] for idx, val in enumerate(labels) if val == correct_answer_label][0]

        # Prompting model for an answer with confidence estimation
        prompt = f"{question_stem} (Please provide only one answer and your confidence from 0 to 1 from the choices). "\
                 f"your choices are : \n" + ", ".join(choices)
        response = model.generate_response(prompt)

        # Extracting confidence and answer
        try:
            answer, confidence = response.rsplit(" ", 1)
            confidence = float(confidence)
        except ValueError:
            answer, confidence = response, 0.5  # Default confidence

        correctness.append(answer.strip().lower() == correct_answer.strip().lower())
        confidences.append(confidence)

    # Expected Calibration Error (ECE)
    bins = np.linspace(0, 1, 10)
    bin_correctness = np.digitize(confidences, bins)
    ece = np.abs(np.mean(bin_correctness) - np.mean(correctness))

    result = {"calibration_score": round(1 - ece, 4)}  # Higher is better

    with open(f"results/{model_name}_calibration.json", "w") as f:
        json.dump(result, f, indent=4)

    return result


if __name__ == "__main__":
    print(evaluate_calibration("gpt-4"))
