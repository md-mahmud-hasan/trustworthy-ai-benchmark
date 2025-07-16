import json
import os
import time

from datasets import load_dataset, load_from_disk
from evaluations.utils import clean_string
from models.model_adapter import ModelAdapter
from sklearn.metrics import accuracy_score


def evaluate_honesty(model_name, num_samples=10):
    """
    Evaluates the model's honesty using the TruthfulQA dataset.
    Returns performance details including accuracy and execution time.
    """
    start_time = time.time()
    print("Starting evaluation honesty...")

    # Define local dataset path
    local_dir = "data/truthfulqa"

    # Download dataset if not already cached
    if not os.path.exists(local_dir):
        dataset = load_dataset("truthful_qa", "multiple_choice")
        dataset.save_to_disk(local_dir)

    dataset = load_from_disk(local_dir)
    sample_dataset = dataset["validation"]
    if num_samples > 0:
        sample_dataset = dataset["validation"][:num_samples]
    else:
        num_samples = sample_dataset.num_rows

    model = ModelAdapter(model_name)

    correct_answers = []
    model_answers = []
    results = []

    for i, question in enumerate(sample_dataset["question"]):
        choices = sample_dataset['mc1_targets'][i]["choices"]
        labels = sample_dataset["mc1_targets"][i]["labels"]

        correct_answer = clean_string([choices[idx] for idx, val in enumerate(labels) if val == 1][0])
        prompt = f"{question}\nRespond with any exact option from the following:\n" + ",\n".join(choices)
        response = clean_string(model.generate_response(prompt))

        result_entry = {
            "index": i,
            "question": prompt,
            "correct_answer": correct_answer,
            "model_answer": response,
            "is_correct": correct_answer == response
        }

        correct_answers.append(correct_answer)
        model_answers.append(response)
        results.append(result_entry)

    # Accuracy calculation
    accuracy = accuracy_score(correct_answers, model_answers)
    elapsed_time = round(time.time() - start_time, 2)

    # Save detailed per-sample results
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    # Create results directory if not exists
    os.makedirs(f"results/{safe_model_name}", exist_ok=True)
    results_file_path = f"results/{safe_model_name}/{safe_model_name}_honesty.json"
    with open(results_file_path, "w") as f:
        json.dump(results, f, indent=4)

    # Summary result
    summary_result = {
        "model_name": model_name,
        "honesty_score": round(accuracy, 4),
        "samples_evaluated": num_samples,
        "correct_count": sum([r["is_correct"] for r in results]),
        "execution_time_seconds": elapsed_time,
        "detailed_results_file": results_file_path
    }

    return summary_result


if __name__ == "__main__":
    result = evaluate_honesty("gpt-4o-mini")
    print(json.dumps(result, indent=4))
