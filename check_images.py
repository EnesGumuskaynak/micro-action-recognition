#!/usr/bin/env python3
import os

def check_image_exists(dataset_path, image_line):
    """Check if an image exists in the dataset_path"""
    parts = image_line.strip().split(',')
    if not parts:
        return False
    
    image_path = os.path.join(dataset_path, parts[0])
    return os.path.exists(image_path)

def filter_file(input_file, output_file, dataset_path):
    """Filter lines based on image existence"""
    valid_lines = 0
    invalid_lines = 0
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if check_image_exists(dataset_path, line):
                outfile.write(line)
                valid_lines += 1
            else:
                invalid_lines += 1
    
    return valid_lines, invalid_lines

if __name__ == "__main__":
    base_path = "./Yoga-82"
    dataset_path = os.path.join(base_path, "dataset")
    
    # Process train file
    train_valid, train_invalid = filter_file(
        os.path.join(base_path, "yoga_train.txt"),
        os.path.join(base_path, "yoga_train_filtered.txt"),
        dataset_path
    )
    
    # Process test file
    test_valid, test_invalid = filter_file(
        os.path.join(base_path, "yoga_test.txt"),
        os.path.join(base_path, "yoga_test_filtered.txt"),
        dataset_path
    )
    
    print(f"Train dosyası: {train_valid} geçerli, {train_invalid} geçersiz görüntü")
    print(f"Test dosyası: {test_valid} geçerli, {test_invalid} geçersiz görüntü")
