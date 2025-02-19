import json
import os
from evaluations.honesty import evaluate_honesty
from evaluations.robustness import evaluate_robustness
from evaluations.bias_mitigation import evaluate_bias
from evaluations.calibration import evaluate_calibration
from evaluations.deception import evaluate_deception
from score import aggregate_scores

# Ensure results directory exists
os.makedirs("results", exist_ok=True)


def run_all_benchmarks(model_name="gpt-4o-mini"):
    """
    Runs all AI safety benchmark evaluations and computes final safety score.
    """
    print(f"\n🚀 Running AI Safety Benchmark for Model: {model_name}...\n")
    #
    # Run individual evaluations
    honesty_result = evaluate_honesty(model_name)
    print(f"✅ Honesty Score: {honesty_result['honesty_score']}")

    robustness_result = evaluate_robustness(model_name)
    print(f"✅ Robustness Score: {robustness_result['robustness_score']}")

    bias_result = evaluate_bias(model_name)
    print(f"✅ Bias Score: {bias_result['bias_score']}")

    calibration_result = evaluate_calibration(model_name)
    print(f"✅ Calibration Score: {calibration_result['calibration_score']}")

    deception_result = evaluate_deception(model_name)
    print(f"✅ Deception Score: {deception_result['deception_score']}")

    # Aggregate all scores into final benchmark evaluation
    final_scores = aggregate_scores(model_name)

    print(f"\n🎯 Final AI Safety Score for {model_name}: {final_scores['final_safety_score']}\n")

    # Save results
    with open(f"results/{model_name}_final_report.json", "w") as f:
        json.dump(final_scores, f, indent=4)

    print(f"📁 Results saved in 'results/{model_name}_final_report.json'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AI Safety Benchmark")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name to evaluate")

    args = parser.parse_args()
    run_all_benchmarks(args.model)
