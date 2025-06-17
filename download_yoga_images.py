#!/usr/bin/env python3
import os
import glob
import requests
from tqdm import tqdm
import time
import logging
import concurrent.futures
from urllib3.exceptions import NameResolutionError
from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout

# Setup logging
logging.basicConfig(
    filename="yoga_download_errors.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Common User-Agent to avoid 403 errors
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"


def download_image(url, save_path, max_retries=2):
    """Download an image from URL and save it to the specified path."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "image/jpeg,image/png,image/*",
        "Referer": url,  # Some sites check referer
    }

    retries = 0
    while retries < max_retries:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Download the image with proper headers
            response = requests.get(url, stream=True, timeout=30, headers=headers)
            response.raise_for_status()  # Raise exception for HTTP errors

            # Save the image
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True
        except HTTPError as e:
            if e.response.status_code == 403:
                # Special handling for 403 Forbidden
                logging.error(f"403 Forbidden: {url} - This site may block scrapers")
                return False  # No retry for 403s, likely policy-based
            elif e.response.status_code == 404:
                # File not found, no need to retry
                logging.error(f"404 Not Found: {url}")
                return False
            # For other HTTP errors, retry
            logging.warning(
                f"HTTP error {e.response.status_code} for {url}, retrying ({retries+1}/{max_retries})"
            )
        except NameResolutionError:
            # DNS resolution failed, likely domain no longer exists
            logging.error(
                f"DNS resolution failed for {url} - Domain may no longer exist"
            )
            return False  # No retry for non-existent domains
        except (ConnectionError, Timeout) as e:
            # Connection issues might be temporary
            logging.warning(
                f"Connection error for {url}, retrying ({retries+1}/{max_retries}): {e}"
            )
        except Exception as e:
            logging.error(f"Error downloading {url} to {save_path}: {e}")

        # Wait before retry, with exponential backoff
        # time.sleep(2 ** retries)
        time.sleep(retries)
        retries += 1

    if retries == max_retries:
        logging.error(f"Failed after {max_retries} attempts: {url}")

    return False


def process_line(line, base_dir):
    """Process a single line containing an image path and URL."""
    try:
        # Split line into path and URL
        parts = line.split("\t")
        if len(parts) != 2:
            print(f"Invalid line format: {line}")
            return (0, 1, 0, [])  # failed

        rel_path, url = parts
        save_path = os.path.join(base_dir, rel_path)

        # Skip if file already exists
        if os.path.exists(save_path):
            return (0, 0, 1, [])  # skipped

        # Download the image
        if download_image(url, save_path):
            return (1, 0, 0, [])  # successful
        else:
            return (0, 1, 0, [(url, save_path)])  # failed

    except Exception as e:
        print(f"Error processing line '{line}': {e}")
        logging.error(f"Error processing line '{line}': {e}")
        return (0, 1, 0, [])  # failed


def process_txt_file(txt_file, base_dir, max_workers=10):
    """Process a single txt file containing image paths and URLs with parallel downloading."""
    print(f"Processing {txt_file}")

    # Read the file
    with open(txt_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    successful = 0
    failed = 0
    skipped = 0

    # Keep track of failures for potential later retry
    failures = []

    # Create a progress bar that will update from multiple threads
    pbar = tqdm(total=len(lines), desc=os.path.basename(txt_file))

    # Process lines in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_line = {
            executor.submit(process_line, line, base_dir): line for line in lines
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_line):
            line = future_to_line[future]
            try:
                s, f, sk, fail_items = future.result()
                successful += s
                failed += f
                skipped += sk
                failures.extend(fail_items)
                pbar.update(1)
            except Exception as e:
                print(f"Exception processing {line}: {e}")
                logging.error(f"Exception processing {line}: {e}")
                failed += 1
                pbar.update(1)

    pbar.close()

    print(
        f"Completed {txt_file}: {successful} downloaded, {failed} failed, {skipped} already existed"
    )

    # # Write failures to a retry log for this specific file
    # if failures:
    #     retry_log = os.path.splitext(os.path.basename(txt_file))[0] + "_failures.txt"
    #     with open(retry_log, "w") as f:
    #         for url, path in failures:
    #             rel_path = os.path.relpath(path, base_dir)
    #             f.write(f"{rel_path}\t{url}\n")
    #     print(f"Recorded {len(failures)} failures in {retry_log} for later retry")

    return successful, failed, skipped


def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download yoga images in parallel")
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of worker threads for parallel downloading (default: 10)",
    )
    args = parser.parse_args()

    # Base directory where images will be saved
    base_dir = "./Yoga-82/dataset"

    # Find all txt files in the yoga_dataset_links directory
    txt_files = glob.glob("./Yoga-82/yoga_dataset_links/*.txt")

    if not txt_files:
        print("No txt files found!")
        return

    print(f"Found {len(txt_files)} txt files to process")
    print(f"Using {args.workers} worker threads for parallel downloading")

    # Process each txt file
    total_successful = 0
    total_failed = 0
    total_skipped = 0

    for txt_file in txt_files:
        successful, failed, skipped = process_txt_file(txt_file, base_dir, args.workers)
        total_successful += successful
        total_failed += failed
        total_skipped += skipped

    print(
        f"\nDownload complete. Total: {total_successful} successful, {total_failed} failed, {total_skipped} skipped"
    )
    print(f"Check yoga_download_errors.log for detailed error information")


if __name__ == "__main__":
    main()
