# Yoga Pose Recognition System

This project provides an artificial intelligence system that automatically recognizes yoga poses from images. Using the InternVL3-8B model, it can identify and classify 82 different yoga poses.

![Yoga Pose Recognition](https://img.shields.io/badge/Yoga-Pose%20Recognition-orange)
![Model Accuracy](https://img.shields.io/badge/Top--1%20Accuracy-18.49%25-blue)

## Project Summary

This system uses InternVL3-8B, an advanced vision-language model, to identify yoga poses in images. The project aims to accurately identify and classify yoga poses.

### Features

- Recognition of 82 different yoga poses
- Output of predicted pose name and ID
- Calculation of accuracy rate for each pose
- Analysis of the most common misclassifications for each pose

## Installation

Follow these steps to run the project:

1. Clone the repository:
```bash
git clone https://gitlab.aurorabilisim.com/enes.gumuskaynak/micro-action-recognition.git
cd micro-action-recognition
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv env_name
source env_name/bin/activate
pip install -r requirements.txt
```

3. Download the Yoga-82 dataset (if not already present):
```bash
python download_yoga_images.py
```

## Usage

The system consists of three main components:

### 1. Pose Prediction - intervlm3-8B.py

To make a yoga pose prediction on a specific image:

```bash
python intervlm3-8B.py --input_image path/to/yoga_image.jpg
```

To process all images in a folder:

```bash
python intervlm3-8B.py --input_file path/to/image_list.txt --output_dir output/my_results
```

### 2. Adding Prediction Class - add_predicted_class.py

To process model outputs and add predicted pose ID and name to the CSV file:

```bash
python add_predicted_class.py --output_dir output --class_json Yoga-82/class_82.json
```

### 3. Result Evaluation - evaluate_predictions.py

To evaluate the model's performance and create a detailed analysis report:

```bash
python evaluate_predictions.py --output_dir output --class_json Yoga-82/class_82.json --save_csv
```

## Performance and Analysis

The current model has been evaluated on the Yoga-82 dataset, which contains 82 different yoga poses.

### Overall Performance

- **Overall Accuracy Rate:** 18.49%
- **Total Evaluated Samples:** 1,368
- **Most Successful Classes:** 
  - Handstand Pose (88.89%)
  - Plank Pose (85.71%)
  - Corpse Pose (81.48%)

### Error Analysis

The system analyzes the most common errors made for each class. For example:

- "Cobra Pose" is most frequently confused with "Camel Pose"
- "Eagle Pose" is most often misidentified as "Tree Pose"
- "Extended Side Angle Pose" is commonly confused with "Extended Puppy Pose"

Detailed analysis reports can be found in the `evaluation_results.csv` file.

## Project Structure

```
micro-action-recognition/
├── intervlm3-8B.py         # Main model script
├── add_predicted_class.py  # Script for adding predictions to CSV
├── evaluate_predictions.py # Performance evaluation script
├── download_yoga_images.py # Dataset download script
├── Yoga-82/                # Yoga-82 dataset 
│   ├── class_82.json      # Class ID-name mappings
└── output/                # Model outputs
```

## Future Improvements

- Increasing the accuracy rate of the model through fine-tuning
- Collecting more data for the most frequently confused poses
- Mobile application integration
- Real-time video analysis support

## Authors and Contact

- **Maintainer:** [Project Manager]
- **Contact:** [Contact Information]

## License

This project is licensed under [license name]. All rights reserved.

---

**Note:** This document reflects the current status and usage of the project. If you have any questions or suggestions, please contact us.
