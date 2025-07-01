#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
import re
import argparse
import glob
from typing import Dict, List


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


def get_class_id_mapping(class_names: Dict[str, str]) -> Dict[str, str]:
    """
    Create a dictionary mapping class names to IDs
    """
    class_id_mapping = {}
    for class_id, class_name in class_names.items():
        class_id_mapping[class_name.lower()] = class_id
    return class_id_mapping


def extract_predicted_class(
    raw_output: str, class_id_mapping: Dict[str, str], class_names: Dict[str, str]
) -> tuple:
    """
    Extracts the predicted class from model output and returns matching ID and name

    Supports "Predicted Yoga Pose: X" or "Predicted Yoga Pose: X: Y" formats

    Returns:
        tuple: (predicted_class_id, predicted_class_name)
    """
    # Searches for "Predicted Yoga Pose: X" format
    pattern = r"Predicted Yoga Pose:\s*(.*?)(?:\s*\(.*?\))?\s*$"
    matches = re.findall(pattern, raw_output, re.MULTILINE)

    if not matches:
        return "Unknown", "Unknown"

    predicted_class_text = matches[0].strip()

    # Handle "40: Low Lunge Pose" format - remove number and following ":" character
    if re.match(r"^\d+\s*:\s*", predicted_class_text):
        predicted_class_text = re.sub(r"^\d+\s*:\s*", "", predicted_class_text)

    # Remove numbers in parentheses
    predicted_class_text = re.sub(r"\(\d+\)", "", predicted_class_text).strip()

    # Check first word
    first_word = predicted_class_text.split()[0].lower() if predicted_class_text else ""

    # Search for matching
    for class_name, class_id in class_id_mapping.items():
        if predicted_class_text.lower() == class_name:
            return class_id, class_names.get(class_id, "Unknown")

        # If no exact match found, check by first word
        if first_word and class_name.lower().startswith(first_word):
            return class_id, class_names.get(class_id, "Unknown")

    # If no match found, return "Unknown" or special value
    return "Unknown", "Unknown"


def process_csv_file(
    csv_path: str, class_id_mapping: Dict[str, str], class_names: Dict[str, str]
) -> None:
    """
    Processes the given CSV file and adds predicted_class_id and predicted_class_name columns
    If these columns already exist, removes them first
    """
    temp_path = csv_path + ".temp"

    try:
        # Create a temporary file
        with open(csv_path, "r", newline="", encoding="utf-8") as input_file, open(
            temp_path, "w", newline="", encoding="utf-8"
        ) as output_file:

            reader = csv.reader(input_file)
            writer = csv.writer(output_file)

            # Read header row
            header = next(reader)

            # Find the index of raw_output column
            raw_output_index = (
                header.index("raw_output") if "raw_output" in header else -1
            )

            if raw_output_index == -1:
                print(f"ERROR: 'raw_output' column not found in {csv_path} file.")
                return

            # Find indexes of existing predicted_class_id and predicted_class_name columns
            predicted_class_id_index = -1
            predicted_class_name_index = -1

            if "predicted_class_id" in header:
                predicted_class_id_index = header.index("predicted_class_id")

            if "predicted_class_name" in header:
                predicted_class_name_index = header.index("predicted_class_name")

            # If these columns already exist, remove them from header
            new_header = []
            for i, col in enumerate(header):
                if i != predicted_class_id_index and i != predicted_class_name_index:
                    new_header.append(col)

            # Now add predicted_class_id and predicted_class_name columns to new header
            new_header.append("predicted_class_id")
            new_header.append("predicted_class_name")
            writer.writerow(new_header)

            # Process each row
            for row in reader:
                if len(row) <= raw_output_index:
                    # Skip faulty rows
                    continue

                # If columns already exist, remove them from row
                new_row = []
                for i, value in enumerate(row):
                    if (
                        i != predicted_class_id_index
                        and i != predicted_class_name_index
                    ):
                        new_row.append(value)

                # Get raw_output value (index may have changed in new row)
                raw_output_new_index = (
                    new_header.index("raw_output") if "raw_output" in new_header else -1
                )
                raw_output = (
                    new_row[raw_output_new_index]
                    if raw_output_new_index != -1
                    else row[raw_output_index]
                )

                # Find predicted class ID and name
                predicted_class_id, predicted_class_name = extract_predicted_class(
                    raw_output, class_id_mapping, class_names
                )

                # Add predicted_class_id and predicted_class_name to row
                new_row.append(predicted_class_id)
                new_row.append(predicted_class_name)
                writer.writerow(new_row)

        # Replace temporary file with original file
        os.replace(temp_path, csv_path)
        print(f"Processed: {csv_path}")

    except Exception as e:
        print(f"ERROR: An error occurred while processing {csv_path}: {str(e)}")
        # Clean up temporary file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)


def process_all_output_folders(output_dir: str, class_json_path: str) -> None:
    """
    Processes all model_output.csv files under the output folder
    """
    # Load class names
    class_names = load_class_names(class_json_path)
    if not class_names:
        print(f"ERROR: Could not load class names from {class_json_path} file.")
        return

    # Create dictionary mapping class names to IDs
    class_id_mapping = get_class_id_mapping(class_names)

    # Find all subfolders under output folder
    output_folders = [
        f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))
    ]

    if not output_folders:
        print(f"WARNING: No folders found under {output_dir}.")
        return

    # For each output folder
    for folder in output_folders:
        folder_path = os.path.join(output_dir, folder)
        csv_path = os.path.join(folder_path, "model_output.csv")

        # Check if CSV file exists
        if os.path.exists(csv_path) and os.path.isfile(csv_path):
            print(f"Processing: {csv_path}")
            process_csv_file(csv_path, class_id_mapping, class_names)
        else:
            print(f"WARNING: {csv_path} not found.")


def main():
    parser = argparse.ArgumentParser(
        description="Add predicted_class_id column to CSV files"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Path to output folder to be processed (default: output)",
    )

    parser.add_argument(
        "--class_json",
        type=str,
        default="Yoga-82/class_82.json",
        help="Path to JSON file containing class names (default: Yoga-82/class_82.json)",
    )

    parser.add_argument(
        "--specific_folder",
        type=str,
        default="",
        help="Process only a specific folder (if not specified, all folders are processed)",
    )

    args = parser.parse_args()

    # If a specific folder is specified, process only that one
    if args.specific_folder:
        folder_path = os.path.join(args.output_dir, args.specific_folder)
        csv_path = os.path.join(folder_path, "model_output.csv")

        if os.path.exists(csv_path) and os.path.isfile(csv_path):
            print(f"Processing: {csv_path}")
            class_names = load_class_names(args.class_json)
            class_id_mapping = get_class_id_mapping(class_names)
            process_csv_file(csv_path, class_id_mapping, class_names)
        else:
            print(f"ERROR: {csv_path} not found.")
    else:
        # Process all output folders
        process_all_output_folders(args.output_dir, args.class_json)

    print("Processing completed.")


if __name__ == "__main__":
    main()
