#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple


def load_class_names(json_path: str) -> Dict[str, str]:
    """
    Load class names from JSON file
    """
    try:
        with open(json_path, "r") as f:
            class_names = json.load(f)
        return class_names
    except Exception as e:
        print(f"Error loading class names: {str(e)}")
        return {}


def evaluate_predictions(
    csv_path: str, class_names: Dict[str, str]
) -> Tuple[Dict, float, int, Dict]:
    """
    Evaluates predictions in CSV file and calculates accuracy rates

    Args:
        csv_path: Path to CSV file
        class_names: Dictionary mapping class IDs to class names

    Returns:
        Tuple[Dict, float, int, Dict]: (Class-based success rates, overall success rate, total sample count, class-based error analyses)
    """
    # Dictionaries to hold class-based statistics
    class_counts = defaultdict(int)  # Total sample count for each class
    class_correct = defaultdict(int)  # Correct prediction count for each class
    class_errors = defaultdict(
        lambda: defaultdict(int)
    )  # Incorrect predictions made for each class

    total_count = 0  # Total sample count
    total_correct = 0  # Total correct prediction count
    unknown_count = 0  # Count of unknown predictions

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Check required fields
                if "true_class_id" not in row or "predicted_class_id" not in row:
                    print(f"WARNING: Required columns missing in CSV file: {csv_path}")
                    continue

                # Get actual and predicted class IDs
                true_class_id = row["true_class_id"]
                predicted_class_id = row["predicted_class_id"]

                # Increment total sample count
                total_count += 1

                # Update class-based statistics
                class_counts[true_class_id] += 1

                # Unknown predictions
                if predicted_class_id == "Unknown":
                    unknown_count += 1
                    # We also record unknown predictions as errors
                    class_errors[true_class_id]["Unknown"] += 1
                    continue

                # Calculate correct predictions
                if true_class_id == predicted_class_id:
                    total_correct += 1
                    class_correct[true_class_id] += 1
                else:
                    # Record incorrect predictions
                    class_errors[true_class_id][predicted_class_id] += 1

        # Calculate class-based success rates
        class_accuracy = {}
        for class_id, count in class_counts.items():
            if count > 0:
                accuracy = (class_correct[class_id] / count) * 100
                class_name = class_names.get(class_id, f"Class {class_id}")
                class_accuracy[class_id] = {
                    "name": class_name,
                    "accuracy": accuracy,
                    "correct": class_correct[class_id],
                    "total": count,
                }

        # Calculate overall success rate
        overall_accuracy = (total_correct / total_count) * 100 if total_count > 0 else 0

        # Organize error analysis for each class
        class_error_analysis = {}
        for class_id, errors in class_errors.items():
            # Sort most common error types (highest frequency first)
            sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)
            # Take up to 5 error types
            top_errors = sorted_errors[:5]
            # Save error analysis information for class
            class_error_analysis[class_id] = top_errors

        return class_accuracy, overall_accuracy, total_count, class_error_analysis

    except Exception as e:
        print(f"ERROR: An error occurred while processing CSV file: {str(e)}")
        return {}, 0, 0, {}


def print_evaluation_results(
    class_accuracy: Dict, overall_accuracy: float, total_count: int
) -> None:
    """
    Prints evaluation results in formatted way to screen
    """
    print("\n" + "=" * 80)
    print(f"GENERAL EVALUATION RESULTS:")
    print("=" * 80)
    print(f"Total Sample Count: {total_count}")
    print(f"Overall Success Rate: {overall_accuracy:.2f}%")
    print("=" * 80)

    print("\nCLASS-BASED SUCCESS RATES:")
    print("-" * 80)
    print(
        f"{'Class ID':<10} {'Class Name':<30} {'Success Rate':<15} {'Correct/Total':<15}"
    )
    print("-" * 80)

    # Sort classes by success rate
    sorted_classes = sorted(
        class_accuracy.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )

    for class_id, data in sorted_classes:
        print(
            f"{class_id:<10} {data['name']:<30} {data['accuracy']:.2f}% {data['correct']}/{data['total']:<15}"
        )

    print("-" * 80)


def save_results_to_csv(
    output_path: str,
    class_accuracy: Dict,
    overall_accuracy: float,
    total_count: int,
    class_error_analysis: Dict,
    class_names: Dict,
) -> None:
    """
    Saves evaluation results to CSV file
    """
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header row
            writer.writerow(
                [
                    "Class ID",
                    "Class Name",
                    "Success Rate (%)",
                    "Correct Predictions",
                    "Total Samples",
                    "Most Common Errors (Top 5)",
                ]
            )

            # Class-based results
            for class_id, data in class_accuracy.items():
                # Prepare error analysis information
                error_text = ""
                if class_id in class_error_analysis and class_error_analysis[class_id]:
                    error_list = []
                    for error_class_id, error_count in class_error_analysis[class_id]:
                        error_class_name = class_names.get(
                            error_class_id, f"Class {error_class_id}"
                        )
                        error_list.append(f"{error_class_name} ({error_count})")
                    error_text = ", ".join(error_list)

                writer.writerow(
                    [
                        class_id,
                        data["name"],
                        f"{data['accuracy']:.2f}",
                        data["correct"],
                        data["total"],
                        error_text,
                    ]
                )

            # Overall results
            writer.writerow([])
            writer.writerow(
                ["OVERALL", "", f"{overall_accuracy:.2f}", "", total_count, ""]
            )

        print(f"\nResults saved to CSV file: {output_path}")

    except Exception as e:
        print(f"ERROR: An error occurred while saving results to CSV file: {str(e)}")


def process_folder(
    folder_path: str, class_names: Dict[str, str], save_csv: bool = False
) -> None:
    """
    Evaluates the model_output.csv file in the specified folder
    """
    csv_path = os.path.join(folder_path, "model_output.csv")

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        return

    print(f"Evaluating: {csv_path}")

    # Evaluate CSV file
    class_accuracy, overall_accuracy, total_count, class_error_analysis = (
        evaluate_predictions(csv_path, class_names)
    )

    # Print results to screen
    print_evaluation_results(class_accuracy, overall_accuracy, total_count)

    # Save results to CSV file if requested
    if save_csv:
        output_filename = os.path.join(folder_path, "evaluation_results.csv")
        save_results_to_csv(
            output_filename,
            class_accuracy,
            overall_accuracy,
            total_count,
            class_error_analysis,
            class_names,
        )


def process_all_folders(
    output_dir: str, class_json_path: str, save_csv: bool = False
) -> None:
    """
    Evaluates all folders under output folder
    """
    # Load class names
    class_names = load_class_names(class_json_path)
    if not class_names:
        print(f"ERROR: Failed to load class names from {class_json_path}.")
        return

    # Find all folders under the output directory
    try:
        folders = [
            folder
            for folder in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, folder))
        ]
    except Exception as e:
        print(f"ERROR: An error occurred while listing files in {output_dir}: {str(e)}")
        return

    if not folders:
        print(f"WARNING: No folders found under {output_dir}.")
        return

    # Evaluate each folder
    for folder in folders:
        folder_path = os.path.join(output_dir, folder)
        process_folder(folder_path, class_names, save_csv)


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(
        description="Evaluate yoga pose predictions and calculate accuracy rates"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Main folder containing model outputs to be evaluated (default: output)",
    )

    parser.add_argument(
        "--class_json",
        type=str,
        default="Yoga-82/class_82.json",
        help="Path to the JSON file containing class names (default: Yoga-82/class_82.json)",
    )

    parser.add_argument(
        "--specific_folder",
        type=str,
        default="",
        help="Evaluate only a specific folder (if not specified, all folders are evaluated)",
    )

    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Save evaluation results to a CSV file",
    )

    args = parser.parse_args()

    # Load class names
    class_names = load_class_names(args.class_json)
    if not class_names:
        return

    # If a specific folder is specified, evaluate only that folder
    if args.specific_folder:
        folder_path = os.path.join(args.output_dir, args.specific_folder)
        if os.path.isdir(folder_path):
            process_folder(folder_path, class_names, args.save_csv)
        else:
            print(f"ERROR: {folder_path} is not a folder or could not be found.")
    else:
        # Evaluate all folders
        process_all_folders(args.output_dir, args.class_json, args.save_csv)


if __name__ == "__main__":
    main()
