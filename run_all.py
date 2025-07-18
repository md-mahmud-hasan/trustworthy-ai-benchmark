import json
import os
from evaluations.honesty import evaluate_honesty
from evaluations.consistency import evaluate_consistency
from evaluations.bias_mitigation import evaluate_bias
from evaluations.calibration import evaluate_calibration
from evaluations.deception_resistance import evaluate_deception_resistance
from score import aggregate_scores

# Ensure results directory exists
os.makedirs("results", exist_ok=True)


def run_all_benchmarks(model_name="gpt-4o-mini", sample_size=5):
    """
    Runs all AI safety benchmark evaluations and computes final safety score.
    """
    print(f"\n🚀 Running AI Safety Benchmark for Model: {model_name}...\n")

    # Load existing results if the file exists
    results_file = "results/final_report.json"
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = {}

    # Run individual evaluations
    honesty_result = evaluate_honesty(model_name, sample_size)
    print(f"✅ Honesty Score: {honesty_result}")

    bias_result = evaluate_bias(model_name, sample_size)
    print(f"✅ Bias Score: {bias_result}")

    calibration_result = evaluate_calibration(model_name, sample_size)
    print(f"✅ Calibration Score: {calibration_result}")

    deception_result = evaluate_deception_resistance(model_name, sample_size)
    print(f"✅ Deception Score: {deception_result}")

    consistency_result = evaluate_consistency(model_name, sample_size)
    print(f"✅ Consistency Score: {consistency_result}")

    scores = {
        "honesty": honesty_result['honesty_score'],
        "bias": bias_result['bias_score'],
        "calibration": calibration_result['calibration_score'],
        "deception": deception_result['resistance_score'],
        "consistency": consistency_result['consistency_score']
    }

    # Aggregate all scores into final benchmark evaluation
    final_score = aggregate_scores(model_name, scores)
    print(f"\n🎯 Final AI Safety Score for {model_name}: {final_score['final_safety_score']}\n")

    results[model_name] = {
        "honesty": honesty_result,
        "bias": bias_result,
        "calibration": calibration_result,
        "deception": deception_result,
        "consistency": consistency_result,
        "final_score": final_score
    }

    # Save the updated results
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AI Safety Benchmark")
    # parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name to evaluate")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Model name to evaluate")
    # parser.add_argument("--model", type=str, default="hermes-3-llama-3.2-3b", help="Model name to evaluate")
    # parser.add_argument("--model", type=str, default="DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf", help="Model name to evaluate")
    # parser.add_argument("--model", type=str, default="deepseek-chat", help="Model name to evaluate")
    # parser.add_argument("--model", type=str, default="google/gemma-3-4b", help="Model name to evaluate")

    args = parser.parse_args()
    run_all_benchmarks(args.model, -1)
