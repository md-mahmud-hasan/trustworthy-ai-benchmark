# main.py

import argparse

from result_analysis import run_analysis
from run_all import run_all_benchmarks


def main():
    parser = argparse.ArgumentParser(description="Process model name and data count.")
    parser.add_argument("model_name", type=str, help="The name of the model (required)")
    parser.add_argument("--data_count", type=int, default=-1, help="Number of data items (default: -1)")

    args = parser.parse_args()

    print(f"Model Name: {args.model_name}")
    print(f"Data Count: {args.data_count}")
    run_all_benchmarks(args.model_name, args.data_count)


if __name__ == "__main__":
    main()
    run_analysis()
