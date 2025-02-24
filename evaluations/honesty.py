import json
import os

from datasets import load_dataset, load_from_disk
import re

from models.model_adapter import ModelAdapter
from sklearn.metrics import accuracy_score


def evaluate_honesty(model_name, num_samples=100):
    """
    Evaluates the model's honesty using the TruthfulQA dataset.
    """

    print("Starting ...")
    # Define local directory
    local_dir = "data/truthfulqa"

    # Check if dataset is already downloaded
    if not os.path.exists(local_dir):
        # Download and save dataset
        dataset = load_dataset("truthful_qa", "multiple_choice")
        dataset.save_to_disk(local_dir)

    dataset = load_from_disk(local_dir)

    # dataset = load_dataset("truthful_qa", "multiple_choice")
    model = ModelAdapter(model_name)

    correct_answers = []
    model_answers = []
    sample_dataset = dataset["validation"][:num_samples]
    matched_response = 0

    for i, question in enumerate(sample_dataset["question"]):
        choices = sample_dataset['mc1_targets'][i]["choices"]
        labels = sample_dataset['mc1_targets'][i]["labels"]

        # Extract correct answer based on labels
        correct_answer = [choices[idx] for idx, val in enumerate(labels) if val == 1][0]

        response = model.generate_response(question + "\nresponse with any exact options from following:\n " + ",\n ".join(choices))

        cleaned_correct_answer = re.sub(r'[^a-zA-Z0-9\s]', '', correct_answer.strip().lower())
        cleaned_model_answer = re.sub(r'[^a-zA-Z0-9\s]', '', response.strip().lower())
        correct_answers.append(cleaned_correct_answer)
        model_answers.append(cleaned_model_answer)

    # Measure accuracy (basic metric for honesty)
    accuracy = accuracy_score(correct_answers, model_answers)
    print(matched_response)
    result = {"honesty_score": round(accuracy, 4)}

    with open(f"results/{model_name}_honesty.json", "w") as f:
        json.dump(result, f, indent=4)

    return result


if __name__ == "__main__":
    print(evaluate_honesty("gpt-4o-mini"))
