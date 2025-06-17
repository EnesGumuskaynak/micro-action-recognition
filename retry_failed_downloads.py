#!/usr/bin/env python3
import os
import glob
import time
from tqdm import tqdm
from download_yoga_images import download_image


def retry_failed_downloads(failures_file):
    """Retry downloading images from a failures log file."""
    print(f"Processing failed downloads from {failures_file}")

    # Read the failures file
    with open(failures_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    successful = 0
    failed = 0

    # Process each line
    for line in tqdm(lines, desc=os.path.basename(failures_file)):
        try:
            url, save_path = line.split("\t")

            # Skip if file now exists
            if os.path.exists(save_path):
                continue

            # Try downloading with more retries and longer timeout
            if download_image(url, save_path, max_retries=5):
                successful += 1
            else:
                failed += 1

            # Longer delay between retries to be extra nice
            time.sleep(0.5)

        except Exception as e:
            print(f"Error processing line '{line}': {e}")
            failed += 1

    print(f"Retry complete: {successful} successful, {failed} still failing")
    return successful, failed


def main():
    # Find all failure log files
    failure_files = glob.glob("*_failures.txt")

    if not failure_files:
        print("No failure logs found!")
        return

    print(f"Found {len(failure_files)} failure logs to process")

    total_successful = 0
    total_failed = 0

    for failure_file in failure_files:
        successful, failed = retry_failed_downloads(failure_file)
        total_successful += successful
        total_failed += failed

    print(
        f"\nRetry complete. Total: {total_successful} now successful, {total_failed} still failing"
    )


if __name__ == "__main__":
    main()
