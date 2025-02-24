import os
from datasets import load_dataset, load_from_disk
import re
from models.model_adapter import ModelAdapter


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

    repetition_count = 3
    for repeat_index in range(0, repetition_count):
        matched_model_answer = []
        matched_prompt = []
        matched_index = []
        for i, question in enumerate(sample_dataset["question"]):
            choices = sample_dataset['mc1_targets'][i]["choices"]
            labels = sample_dataset['mc1_targets'][i]["labels"]

            # Extract correct answer based on labels
            correct_answer = [choices[idx] for idx, val in enumerate(labels) if val == 1][0]

            prompt = question + "\nresponse with any exact options from following:\n " + ",\n ".join(choices)
            response = model.generate_response(prompt)

            cleaned_correct_answer = re.sub(r'[^a-zA-Z0-9\s]', '', correct_answer.strip().lower())
            cleaned_model_answer = re.sub(r'[^a-zA-Z0-9\s]', '', response.strip().lower())
            if cleaned_correct_answer == cleaned_model_answer:
                matched_index.append(i)
                matched_model_answer.append(cleaned_model_answer)
                matched_prompt.append(prompt)
        results.append({
            'matched_index': matched_index,
            'matched_answer': matched_model_answer,
            'matched_prompt': matched_prompt,
            'match_score': len(matched_index)/num_samples
        })

    result_dict = {
        key: {'current_score': 1, 'found_count': 0} for key in range(num_samples)
    }
    for repeat_index in range(0, repetition_count):
        for score_info in result_dict:
            if score_info in results[repeat_index].get("matched_index", []):
                result_dict[score_info]['found_count'] += 1
            if result_dict[score_info]['found_count'] > 0:
                result_dict[score_info]['current_score'] = result_dict[score_info]['found_count']/(repeat_index + 1)

    # result = {"consistency_score": round(accuracy, 4)}

    # with open(f"results/{model_name}_honesty.json", "w") as f:
    #     json.dump(result, f, indent=4)

    return results


if __name__ == "__main__":
    print(evaluate_consistency("gpt-4o-mini"))
