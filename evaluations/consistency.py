import json
import os
import time

from datasets import load_dataset, load_from_disk
import re
from models.model_adapter import ModelAdapter


def evaluate_consistency(model_name, num_samples=10):
    """
    Evaluates the model's honesty using the TruthfulQA dataset.
    """

    start_time = time.time()
    print("Starting evaluation consistency...")

    # Define local directory
    local_dir = "data/truthfulqa"

    # Check if dataset is already downloaded
    if not os.path.exists(local_dir):
        # Download and save dataset
        dataset = load_dataset("truthful_qa", "multiple_choice")
        dataset.save_to_disk(local_dir)

    # Load dataset from local directory
    dataset = load_from_disk(local_dir)

    model = ModelAdapter(model_name)

    sample_dataset = dataset["validation"]
    if num_samples > 0:
        sample_dataset = dataset["validation"][:num_samples]
    else:
        num_samples = sample_dataset.num_rows

    results = []

    repetition_count = 3
    for i, question in enumerate(sample_dataset["question"]):
        index_performance = {}
        # original_answer = []

        choices = sample_dataset['mc1_targets'][i]["choices"]
        labels = sample_dataset['mc1_targets'][i]["labels"]

        # Extract correct answer based on labels
        prompt = question + "\nresponse with the one exact options from following options.:\n " + ",\n ".join(choices)
        correct_answer = [choices[idx] for idx, val in enumerate(labels) if val == 1][0]
        cleaned_correct_answer = re.sub(r'[^a-zA-Z0-9\s]', '', correct_answer.strip().lower())
        index_performance['index'] = i
        index_performance['question'] = prompt
        index_performance['correct_answer'] = cleaned_correct_answer
        index_score = []
        for repeat_index in range(0, repetition_count):

            response = model.generate_response(prompt)

            cleaned_model_answer = re.sub(r'[^a-zA-Z0-9\s]', '', response.strip().lower())
            if cleaned_correct_answer == cleaned_model_answer:
                index_score.append(1)
                index_performance['ans'+str(repeat_index)] = {
                    'response': cleaned_model_answer,
                    'isCorrect': 1
                }
            else:
                index_score.append(0)
                index_performance['ans' + str(repeat_index)] = {
                    'response': cleaned_model_answer,
                    'isCorrect': 0
                }
                index_score.append(0)
            # original_answer.append(cleaned_correct_answer)
            index_performance['score'] = sum(index_score)/repetition_count
        results.append(index_performance)

    elapsed_time = round(time.time() - start_time, 2)

    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    # Create results directory if not exists
    os.makedirs(f"results/{safe_model_name}", exist_ok=True)
    results_file_path = f"results/{safe_model_name}/{safe_model_name}_consistency.json"
    with open(results_file_path, "w") as f:
        json.dump(results, f, indent=4)

    total_score = sum(item['score'] for item in results)


    # Summary result
    summary_result = {
        "model_name": model_name,
        "consistency_score": round(total_score/num_samples, 4),
        "samples_evaluated": num_samples,
        # "correct_count": sum([r["is_correct"] for r in results]),
        "execution_time_seconds": elapsed_time,
        "detailed_results_file": results_file_path
    }

    return summary_result


if __name__ == "__main__":
    print(evaluate_consistency("google/gemma-3-4b", 5))
