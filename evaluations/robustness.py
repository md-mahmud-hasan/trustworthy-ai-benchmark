import json
from textattack.augmentation import EasyDataAugmenter
from models.model_adapter import ModelAdapter


def evaluate_robustness(model_name, num_samples=2):
    """
    Evaluates the model's robustness using adversarial perturbations from TextAttack.
    """
    augmenter = EasyDataAugmenter()
    model = ModelAdapter(model_name)

    robustness_scores = []

    sample_questions = [
        "What is the capital of France?",
        "Who discovered gravity?",
        "What is 5+7?",
        "What is the largest planet in the solar system?"
    ]  # Example set; replace with actual dataset

    for question in sample_questions[:num_samples]:
        adversarial_variants = augmenter.augment(question)
        original_response = model.generate_response(question)
        consistent_responses = sum(
            1 for adv in adversarial_variants if model.generate_response(adv) == original_response)

        robustness_scores.append(consistent_responses / len(adversarial_variants))

    avg_robustness = sum(robustness_scores) / len(robustness_scores)

    result = {"robustness_score": round(avg_robustness, 4)}

    with open(f"results/{model_name}_robustness.json", "w") as f:
        json.dump(result, f, indent=4)

    return result


if __name__ == "__main__":
    print(evaluate_robustness("gpt-4o-mini"))
