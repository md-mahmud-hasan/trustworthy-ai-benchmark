import json
import os

from datasets import load_dataset, load_dataset_builder, load_from_disk
from models.model_adapter import ModelAdapter
from sklearn.metrics import accuracy_score
from datasets import get_dataset_config_names


def evaluate_honesty(model_name, num_samples=2):
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

    # Load dataset from local directory
    dataset = load_from_disk(local_dir)

    # dataset = load_dataset("truthful_qa", "multiple_choice")
    model = ModelAdapter(model_name)

    correct_answers = []
    model_answers = []
    sample_dataset = dataset["validation"][:num_samples]

    for i, question in enumerate(sample_dataset["question"]):
        choices = sample_dataset['mc1_targets'][i]["choices"]
        labels = sample_dataset['mc1_targets'][i]["labels"]

        # Extract correct answer based on labels
        correct_answer = [choices[idx] for idx, val in enumerate(labels) if val == 1][0]

        response = model.generate_response_with_choices(question, choices)

        correct_answers.append(correct_answer)
        model_answers.append(response)

    # Measure accuracy (basic metric for honesty)
    accuracy = accuracy_score(correct_answers, model_answers)

    result = {"honesty_score": round(accuracy, 4)}

    with open(f"results/{model_name}_honesty.json", "w") as f:
        json.dump(result, f, indent=4)

    return result


if __name__ == "__main__":
    print(evaluate_honesty("gpt-4o-mini"))
