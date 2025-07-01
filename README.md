# Micro Action Recognition - Yoga Pose Recognition System

This project is an advanced AI system that automatically recognizes yoga poses from images. Using multi-model approaches, it can identify and classify 82 different yoga poses with high accuracy.

![Yoga Pose Recognition](https://img.shields.io/badge/Yoga-Pose%20Recognition-orange)
![Multi Model](https://img.shields.io/badge/Multi--Model-Approach-green)
![Hierarchical](https://img.shields.io/badge/Hierarchical-Classification-blue)

## Project Overview

This system recognizes yoga poses using different vision-language models (InternVL3-8B, MiniCPM-V, Qwen2.5-VL). The project aims to achieve high accuracy rates through hierarchical classification and multi-model approaches.

### Features

- Recognition of 82 different yoga poses
- Multi-model support (InternVL3-8B, MiniCPM-V, Qwen2.5-VL)
- Hierarchical classification approach
- Multi-stage model combinations
- Predicted pose name and ID output
- Accuracy calculation for each pose
- Analysis of most common misclassifications for each pose
- Image validation and error checking tools

## Installation

To run the project, follow these steps:

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

Alternatively, for installation with CUDA 12.8 support:

```bash
chmod +x install.sh
./install.sh
```

3. Request access and prepare the Yoga-82 dataset:

   - Visit the official [Yoga-82 dataset page](https://sites.google.com/view/yoga-82/home)
   - Fill out the request form to obtain permission to download the dataset.
   - After receiving access, download the dataset files provided by the authors.
   - Move all downloaded files and folders into the `Yoga-82` directory in your project root (so that image links and train/test splits are under `Yoga-82/`).

4. Download the Yoga-82 dataset (if not already available):

```bash
python download_yoga_images.py
```

## Usage

The system consists of several main components:

### 1. Main Yoga Model - run_yoga_models.py

For yoga pose prediction with multi-model support:

```bash
python run_yoga_models.py --model internvl3-8b --input_file Yoga-82/yoga_test_filtered.txt
```

Supported models:

- internvl3-8b
- internvl3-14b
- minicpmo
- minicpmv
- qwen2.5-vl7B

### 2. Hierarchical Multi-Model Recognition - hierarchical_yoga_recognition_multi_model.py

With advanced hierarchical approach:

```bash
python hierarchical_yoga_recognition_multi_model.py --model internvl3-8b --input_file Yoga-82/yoga_test_filtered.txt
```

### 3. Multi-Stage Model Combinations - hierarchical_yoga_recognition_multi_stage_models.py

Using different models at different stages:

```bash
python hierarchical_yoga_recognition_multi_stage_models.py --stage1_model internvl3-8b --stage2_model qwen2.5-vl7B --stage3_model internvl3-8b --input_file Yoga-82/yoga_test_filtered.txt
```

### 4. Add Predicted Class - add_predicted_class.py

To process model outputs and add predicted pose ID and name to CSV file:

```bash
python add_predicted_class.py --output_dir output --class_json Yoga-82/class_82.json
```

### 5. Evaluate Predictions - evaluate_predictions.py

To evaluate model performance and generate detailed analysis report:

```bash
python evaluate_predictions.py --output_dir output --class_json Yoga-82/class_82.json --save_csv
```

### 6. Utility Tools

**Image Checking:**

```bash
python check_images.py
```

**Retry Failed Downloads:**

```bash
python retry_failed_downloads.py
```

**VLLM Example Usage:**

```bash
python vllm_example.py
```

**Caption Utilities:**

```bash
python caption_utils.py
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

```text
micro-action-recognition/
├── run_yoga_models.py                              # Main model script
├── hierarchical_yoga_recognition_multi_model.py    # Hierarchical multi-model approach
├── hierarchical_yoga_recognition_multi_stage_models.py # Multi-stage model combinations
├── add_predicted_class.py                          # Script for adding predictions to CSV
├── evaluate_predictions.py                         # Performance evaluation script
├── download_yoga_images.py                         # Dataset download script
├── retry_failed_downloads.py                       # Retry failed downloads
├── check_images.py                                 # Image validation tool
├── caption_utils.py                                # Caption processing utilities
├── vllm_example.py                                 # VLLM usage example
├── yoga_dataset_analysis.ipynb                     # Dataset analysis notebook
├── requirements.txt                                # Python dependencies
├── install.sh                                     # Installation script
├── Yoga-82/                                       # Yoga-82 dataset
│   ├── class_82.json                             # Class ID-name mappings
│   └── Images/                                   # Yoga pose images
└── output/                                       # Model outputs and results
    └── [timestamp_model_name]/                   # Timestamped output directories
```

## Dataset

This project uses the **Yoga-82** dataset, which contains:

- 82 different yoga pose classes
- High-quality images of various yoga poses
- Class mappings in JSON format

## Models and Approaches

### Supported Models

- **InternVL3-8B**: Advanced vision-language model with 8B parameters
- **MiniCPM-V**: Compact vision model optimized for efficiency
- **Qwen2.5-VL**: Latest Qwen vision-language model

### Recognition Approaches

1. **Single Model**: Direct classification using one model
2. **Hierarchical Multi-Model**: Multiple models working together hierarchically
3. **Multi-Stage**: Different models at different classification stages

## Analysis and Evaluation

The project includes comprehensive evaluation tools:

- **Accuracy Metrics**: Per-class and overall accuracy calculation
- **Confusion Analysis**: Most common misclassifications
- **Performance Reports**: Detailed CSV reports with statistics
- **Visual Analysis**: Jupyter notebook for dataset exploration

## Future Improvements

- Increasing the accuracy rate of the model through fine-tuning
- Collecting more data for the most frequently confused poses
- Mobile application integration
- Real-time video analysis support
- Enhanced multi-modal fusion techniques

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
