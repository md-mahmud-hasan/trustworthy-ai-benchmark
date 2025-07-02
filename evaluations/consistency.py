import json
import os
from datasets import load_dataset, load_from_disk
import re
from models.model_adapter import ModelAdapter
# Ensure results directory exists
os.makedirs("results", exist_ok=True)

def evaluate_consistency(model_name, num_samples=10):
    """
    Evaluates the model's honesty using the TruthfulQA dataset.
    """
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


    sample_dataset = dataset["validation"][:num_samples]

    results = []

    repetition_count = 1
    for i, question in enumerate(sample_dataset["question"]):
        index_performance = {}
        original_answer = []

        choices = sample_dataset['mc1_targets'][i]["choices"]
        labels = sample_dataset['mc1_targets'][i]["labels"]

        # Extract correct answer based on labels
        prompt = question + "\nresponse with any exact options from following:\n " + ",\n ".join(choices)
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
            original_answer.append(cleaned_correct_answer)
            index_performance['score'] = sum(index_score)/repetition_count
        results.append(index_performance)

    with open(f"results/{model_name}_consistency.json", "w") as f:
        json.dump(results, f, indent=4)

    total_score = sum(item['score'] for item in results)
    final_result = {"consistency_score": round(total_score/num_samples, 4)}
    return final_result


if __name__ == "__main__":
    print(evaluate_consistency("gpt-4o-mini"))
