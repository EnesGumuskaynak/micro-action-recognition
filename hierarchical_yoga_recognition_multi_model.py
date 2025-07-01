import os
import argparse
import datetime
import csv
import json
from dataclasses import asdict
from PIL import Image
from transformers import AutoTokenizer
import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig

# type: ignore
from vllm import LLM, EngineArgs  # type: ignore

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_class_names(json_path):
    """Load class names from JSON file"""
    try:
        with open(json_path, "r") as f:
            class_names = json.load(f)
        return class_names
    except Exception as e:
        print(f"Error loading class names: {str(e)}")
        return {}


# Model functions (from run_yoga_models.py)
def run_minicpm_base(questions: list[str], modality: str, model_name: str):
    assert modality == "image"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=1024,  # Reduce memory usage
        max_num_seqs=1,
        trust_remote_code=True,
        limit_mm_per_prompt={modality: 1},
        gpu_memory_utilization=0.85,  # Limit GPU memory usage
        swap_space=4,  # Swap space
        enforce_eager=True,
    )
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in stop_tokens]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"(<image>./</image>)\n{q}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]
    return engine_args, prompts, stop_token_ids


def run_minicpmo(questions: list[str], modality: str):
    return run_minicpm_base(questions, modality, "openbmb/MiniCPM-o-2_6")


def run_minicpmv(questions: list[str], modality: str):
    return run_minicpm_base(questions, modality, "openbmb/MiniCPM-V-2_6")


def run_internvl_8b(questions: list[str], modality: str):
    model_name = "OpenGVLab/InternVL3-8B"
    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,  # Increase token limit
        limit_mm_per_prompt={modality: 1},
        dtype="bfloat16",
        enforce_eager=True,
        max_num_seqs=1,
        gpu_memory_utilization=0.8,  # Limit GPU memory usage
        swap_space=4,  # Swap space
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    placeholder = "<image>" if modality == "image" else "<video>"
    messages = [[{"role": "user", "content": f"{placeholder}\n{q}"}] for q in questions]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(s) for s in stop_tokens]
    stop_token_ids = [s for s in stop_token_ids if s is not None]
    return engine_args, prompts, stop_token_ids


def run_internvl_14b(questions: list[str], modality: str):
    model_name = "OpenGVLab/InternVL3-14B"
    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={modality: 1},
        dtype="bfloat16",
        enforce_eager=True,
        max_num_seqs=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    placeholder = "<image>" if modality == "image" else "<video>"
    messages = [[{"role": "user", "content": f"{placeholder}\n{q}"}] for q in questions]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(s) for s in stop_tokens]
    stop_token_ids = [s for s in stop_token_ids if s is not None]
    return engine_args, prompts, stop_token_ids


def run_qwen2_5_vl(questions: list[str], modality: str):
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=1,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )
    placeholder = "<|image_pad|>" if modality == "image" else "<|video_pad|>"
    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>{q}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for q in questions
    ]
    return engine_args, prompts, None


MODEL_FUNCS = {
    "minicpmo": run_minicpmo,
    "minicpmv": run_minicpmv,
    "internvl3-8b": run_internvl_8b,
    "internvl3-14b": run_internvl_14b,
    "qwen2.5_vl7B": run_qwen2_5_vl,
}


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def extract_from_response(response, field_name):
    """Extract a specific field from LLM response"""
    if not response:
        return None

    lines = response.strip().split("\n")

    # Different format patterns to search
    search_patterns = [
        f"{field_name}:",
        f"{field_name.lower()}:",
        f"{field_name.replace(' ', '')}:",
        f"{field_name.replace(' ', '').lower()}:",
    ]

    for line in lines:
        line = line.strip()
        for pattern in search_patterns:
            if line.lower().startswith(pattern.lower()):
                result = line.split(":", 1)[1].strip()
                return result if result else None

    # If exact format is not found, search within the line
    for line in lines:
        for pattern in search_patterns:
            if pattern.lower() in line.lower():
                # Take the part after the :
                if ":" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        result = ":".join(parts[1:]).strip()
                        return result if result else None

    return None


def get_subcategory_prompt(main_category, class_20):
    """Prepare subcategories based on the main category"""
    subcategories = {}
    subcategory_text = "Subcategories for {}:\n".format(main_category)

    if "Standing" in main_category:
        subcategories = {
            "0": "Standing - Straight",
            "1": "Standing - Forward bend",
            "2": "Standing - Side bend",
            "3": "Standing - Others",
        }
    elif "Sitting" in main_category:
        subcategories = {
            "4": "Sitting - Normal1 (legs in front)",
            "5": "Sitting - Normal2 (legs behind)",
            "6": "Sitting - Split",
            "7": "Sitting - Forward bend",
            "8": "Sitting - Twist",
        }
    elif "Balancing" in main_category:
        subcategories = {"9": "Balancing - Front", "10": "Balancing - Side"}
    elif "Inverted" in main_category:
        subcategories = {
            "11": "Inverted - Legs straight up",
            "12": "Inverted - Legs bend",
        }
    elif "Reclining" in main_category:
        subcategories = {
            "13": "Reclining - Up-facing",
            "14": "Reclining - Down-facing",
            "15": "Reclining - Side facing",
            "16": "Reclining - Plank balance",
        }
    elif "Wheel" in main_category:
        subcategories = {
            "17": "Wheel - Up-facing",
            "18": "Wheel - Down-facing",
            "19": "Wheel - Others",
        }

    for key, value in subcategories.items():
        subcategory_text += "{}. {}\n".format(key, value)

    return subcategories, subcategory_text


def get_specific_poses(main_category, subcategory, yoga_diagram):
    """Prepare poses under the subcategory for transition from level 2 to level 3"""
    specific_poses = {}
    specific_poses_text = "Specific poses for {} - {}:\n".format(
        main_category, subcategory
    )

    subcategory_number = None
    for key, value in yoga_diagram["level2"].items():
        if value == subcategory:
            subcategory_number = key
            break

    if subcategory_number:
        pose_ids = yoga_diagram["level2_to_level3"].get(subcategory_number, [])
        for pose_id in pose_ids:
            pose_name = yoga_diagram["level3"].get(pose_id)
            if pose_name:
                specific_poses[pose_id] = pose_name
                specific_poses_text += "{}. {}\n".format(pose_id, pose_name)

    return specific_poses, specific_poses_text


def prepare_yoga_diagram():
    """Prepare the yoga diagram (establish the relationship between levels 1, 2, and 3)"""
    # Load class names for levels 1, 2, and 3
    class_6 = load_class_names("Yoga-82/class_6.json")
    class_20 = load_class_names("Yoga-82/class_20.json")
    class_82 = load_class_names("Yoga-82/class_82.json")

    # Define the diagram structure
    yoga_diagram = {
        "level1": class_6,  # 6 main categories
        "level2": class_20,  # 20 subcategories
        "level3": class_82,  # 82 specific yoga poses
        # Level 1 to level 2 mappings
        "level1_to_level2": {
            "0": ["0", "1", "2", "3"],  # Standing -> Standing subcategories
            "1": ["4", "5", "6", "7", "8"],  # Sitting -> Sitting subcategories
            "2": ["9", "10"],  # Balancing -> Balancing subcategories
            "3": ["11", "12"],  # Inverted -> Inverted subcategories
            "4": ["13", "14", "15", "16"],  # Reclining -> Reclining subcategories
            "5": ["17", "18", "19"],  # Wheel -> Wheel subcategories
        },
        # Level 2 to level 3 mappings
        "level2_to_level3": {
            # Standing subcategories
            "0": ["8", "18", "68"],  # Standing - Straight
            "1": ["16", "17", "36", "60", "77"],  # Standing - Forward bend
            "2": ["21", "22", "29", "31", "40", "75", "81"],  # Standing - Side bend
            "3": ["39", "61", "62", "73", "74"],  # Standing - Others
            # Sitting subcategories
            "4": ["3", "28", "41", "57", "59"],  # Sitting - Normal1
            "5": ["1", "13", "30", "72"],  # Sitting - Normal2
            "6": ["58", "76"],  # Sitting - Split
            "7": ["34", "49", "52", "67"],  # Sitting - Forward bend
            "8": ["0", "35", "47"],  # Sitting - Twist
            # Balancing subcategories
            "9": ["11", "14", "24", "50", "53"],  # Balancing - Front
            "10": ["19", "46", "55"],  # Balancing - Side
            # Inverted subcategories
            "11": ["23", "32", "37", "63", "64"],  # Inverted - Legs straight up
            "12": ["45", "51"],  # Inverted - Legs bend
            # Reclining subcategories
            "13": [
                "12",
                "25",
                "33",
                "48",
                "65",
                "66",
                "79",
                "80",
            ],  # Reclining - Up-facing
            "14": ["9", "10", "20", "27", "38"],  # Reclining - Down-facing
            "15": ["54", "56"],  # Reclining - Side facing
            "16": ["15", "26", "42", "44"],  # Reclining - Plank balance
            # Wheel subcategories
            "17": ["5", "6", "43", "69", "70", "71", "78"],  # Wheel - Up-facing
            "18": ["7"],  # Wheel - Down-facing
            "19": ["2", "4"],  # Wheel - Others
        },
    }

    return yoga_diagram


def process_with_vllm_model(
    llm,
    model_name,
    image_path,
    formatted_prompts,
    stop_token_ids,
    class_6,
    class_20,
    class_82,
    yoga_diagram,
):
    """Function that performs hierarchical classification using VLLM"""
    try:
        # Load the image
        image = Image.open(image_path).convert("RGB")

        results = {}

        # LEVEL 1: Main category
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            # temperature=0.3,  # Lower temperature for more stable results
            top_p=0.9,
            max_tokens=512,  # Reduce the number of tokens
        )

        # Add stop tokens
        if stop_token_ids:
            sampling_params.stop_token_ids = stop_token_ids

        level1_input = [
            {"prompt": formatted_prompts[0], "multi_modal_data": {"image": image}}
        ]
        level1_outputs = llm.generate(level1_input, sampling_params)  # type: ignore
        level1_response = level1_outputs[0].outputs[0].text.strip()

        main_category_text = extract_from_response(level1_response, "Main Category")
        main_category_id = None
        main_category = "Unknown"

        if main_category_text:
            for key, value in class_6.items():
                if value.lower() in main_category_text.lower():
                    main_category_id = key
                    main_category = value
                    break

        if not main_category_id:
            main_category = main_category_text if main_category_text else "Unknown"
            main_category_id = "Unknown"

        results["level1"] = {
            "response": level1_response,
            "category": main_category,
            "category_id": main_category_id,
        }

        # Memory cleanup
        del level1_outputs, level1_input
        import gc

        gc.collect()

        # LEVEL 2: Subcategory - Format the prompt here
        subcategories, subcategory_text = get_subcategory_prompt(
            main_category, class_20
        )
        level2_prompt_text = f"""
You've identified this as a {main_category} yoga pose.
Now determine the specific subcategory within {main_category} poses.

Respond in this format:
Description: [Describe the specific body position]
Subcategory: [One of the subcategories below]

{subcategory_text}

Only return one of these subcategories in your Subcategory response.
"""

        # Model-specific formatting for token ID's and format
        if model_name.startswith("internvl"):
            level2_formatted_prompt = f"<image>\n{level2_prompt_text}"
        elif model_name.startswith("minicpm"):
            level2_formatted_prompt = f"(<image>./</image>)\n{level2_prompt_text}"
        elif model_name == "qwen2.5_vl":
            level2_formatted_prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{level2_prompt_text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            level2_formatted_prompt = f"<image>\n{level2_prompt_text}"

        level2_input = [
            {
                "prompt": level2_formatted_prompt,
                "multi_modal_data": {"image": image},
            }
        ]
        level2_outputs = llm.generate(level2_input, sampling_params)  # type: ignore
        level2_response = level2_outputs[0].outputs[0].text.strip()

        subcategory_text_extracted = extract_from_response(
            level2_response, "Subcategory"
        )
        subcategory_id = None
        subcategory = "Unknown"

        if subcategory_text_extracted:
            for key, value in class_20.items():
                if value.lower() in subcategory_text_extracted.lower():
                    subcategory_id = key
                    subcategory = value
                    break

        if not subcategory_id:
            subcategory = (
                subcategory_text_extracted if subcategory_text_extracted else "Unknown"
            )
            subcategory_id = "Unknown"

        results["level2"] = {
            "response": level2_response,
            "subcategory": subcategory,
            "subcategory_id": subcategory_id,
        }

        # Memory cleanup
        del level2_outputs, level2_input
        gc.collect()

        # LEVEL 3: Specific pose
        specific_poses, specific_poses_text = get_specific_poses(
            main_category, subcategory, yoga_diagram
        )
        level3_prompt_text = f"""
You've identified this as a {main_category} - {subcategory} yoga pose.
Now determine the specific yoga pose name.

Respond in this format:
Description: [Describe the specific yoga pose]
Specific Pose: [One of the poses below]

{specific_poses_text}

Only return one of these specific poses in your Specific Pose response.
"""

        # Model-specific formatting
        if model_name.startswith("internvl"):
            level3_formatted_prompt = f"<image>\n{level3_prompt_text}"
        elif model_name.startswith("minicpm"):
            level3_formatted_prompt = f"(<image>./</image>)\n{level3_prompt_text}"
        elif model_name == "qwen2.5_vl":
            level3_formatted_prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{level3_prompt_text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            level3_formatted_prompt = f"<image>\n{level3_prompt_text}"

        level3_input = [
            {
                "prompt": level3_formatted_prompt,
                "multi_modal_data": {"image": image},
            }
        ]
        level3_outputs = llm.generate(level3_input, sampling_params)  # type: ignore
        level3_response = level3_outputs[0].outputs[0].text.strip()

        specific_pose_text = extract_from_response(level3_response, "Specific Pose")
        specific_pose_id = None
        specific_pose = "Unknown"

        if specific_pose_text:
            for key, value in class_82.items():
                if value.lower() in specific_pose_text.lower():
                    specific_pose_id = key
                    specific_pose = value
                    break

        if not specific_pose_id:
            specific_pose = specific_pose_text if specific_pose_text else "Unknown"
            specific_pose_id = "Unknown"

        results["level3"] = {
            "response": level3_response,
            "pose": specific_pose,
            "pose_id": specific_pose_id,
        }

        # Memory cleanup
        del level3_outputs, level3_input, image
        gc.collect()

        return results

    except Exception as e:
        print(f"Error for model {model_name}: {str(e)}")
        # Memory cleanup
        import gc

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return {
            "level1": {
                "response": f"ERROR: {str(e)}",
                "category": "",
                "category_id": "",
            },
            "level2": {"response": "", "subcategory": "", "subcategory_id": ""},
            "level3": {"response": "", "pose": "", "pose_id": ""},
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hierarchical yoga pose classification with a multi-VL model"
    )

    # Level 1 prompt (6 main categories)
    level1_prompt = """
Look at this image carefully and identify the yoga pose category.

You MUST respond in this exact format:
Description: [Describe what you see]
Main Category: [Choose exactly one: Standing, Sitting, Balancing, Inverted, Reclining, or Wheel]

Main Yoga Pose Categories:
0. Standing - Person is upright, weight primarily on feet
1. Sitting - Person is seated on the ground/mat  
2. Balancing - Person is balancing on hands, feet, or other body parts
3. Inverted - Person is upside down or head is below heart
4. Reclining - Person is lying down on back, stomach or side
5. Wheel - Body forms a wheel or circular shape

Remember: You MUST include both "Description:" and "Main Category:" in your response.
"""

    # Level 2 prompt template
    level2_prompt_template = """
You've identified this as a {main_category} yoga pose.
Now determine the specific subcategory within {main_category} poses.

Respond in this format:
Description: [Describe the specific body position]
Subcategory: [One of the subcategories below]

{subcategories_text}

Only return one of these subcategories in your Subcategory response.
"""

    # Level 3 prompt template
    level3_prompt_template = """
You've identified this as a {main_category} - {subcategory} yoga pose.
Now determine the specific yoga pose name.

Respond in this format:
Description: [Describe the specific yoga pose]
Specific Pose: [One of the poses below]

{specific_poses_text}

Only return one of these specific poses in your Specific Pose response.
"""

    parser.add_argument(
        "--image_path",
        type=str,
        default="./0_25.jpg",
        help="Path to the yoga pose image to be classified",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="",
        help="Text file containing image paths",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="Yoga-82/dataset/",
        help="Root directory of the yoga dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=MODEL_FUNCS.keys(),
        default=list(MODEL_FUNCS.keys())[0],
        help="Model to use",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--level1_prompt", type=str, default=level1_prompt, help="Level 1 prompt"
    )
    parser.add_argument(
        "--level2_prompt_template",
        type=str,
        default=level2_prompt_template,
        help="Level 2 prompt template",
    )
    parser.add_argument(
        "--level3_prompt_template",
        type=str,
        default=level3_prompt_template,
        help="Level 3 prompt template",
    )

    args = parser.parse_args()
    args.models = [args.model]
    return args


def main():
    args = parse_args()

    # Set up GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(i) for i in range(args.tensor_parallel_size)
    )

    # Load class names
    class_6 = load_class_names("Yoga-82/class_6.json")
    class_20 = load_class_names("Yoga-82/class_20.json")
    class_82 = load_class_names("Yoga-82/class_82.json")

    # Prepare yoga diagram
    yoga_diagram = prepare_yoga_diagram()

    # Prepare prompts
    prompts = [
        args.level1_prompt,
        args.level2_prompt_template,
        args.level3_prompt_template,
    ]

    # Create output folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder_name = f"{timestamp}_hierarchical_{'_'.join(args.models)}"
    output_dir = os.path.join("output", output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save prompts
    prompts_path = os.path.join(output_dir, "prompts.txt")
    with open(prompts_path, "w", encoding="utf-8") as prompts_file:
        prompts_file.write("LEVEL 1 PROMPT:\n" + args.level1_prompt)
        prompts_file.write(
            "\n\nLEVEL 2 PROMPT TEMPLATE:\n" + args.level2_prompt_template
        )
        prompts_file.write(
            "\n\nLEVEL 3 PROMPT TEMPLATE:\n" + args.level3_prompt_template
        )

    # CSV file path
    csv_path = os.path.join(output_dir, "hierarchical_results.csv")

    # Create CSV headers (single model)
    # Simple column names for single model
    csv_headers = [
        "path",
        "level1_response",
        "main_category",
        "main_category_id",
        "level2_response",
        "subcategory",
        "subcategory_id",
        "level3_response",
        "specific_pose",
        "specific_pose_id",
        "true_class_id",
        "true_class_name",
    ]

    # Check if processing from text file or single image
    if args.input_file:
        # Read image paths from text file
        with open(args.input_file, "r") as f:
            lines = f.readlines()

        # Create CSV file and write headers
        with open(csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(csv_headers)

        # Process only for single model
        model_name = args.models[0]

        print(f"\n=== Starting model: {model_name} ===")

        # Initialize the model
        func = MODEL_FUNCS[model_name]
        engine_args, _, stop_token_ids = func([prompts[0]], "image")
        engine_args.tensor_parallel_size = args.tensor_parallel_size
        engine_args_dict = asdict(engine_args)

        # Memory cleanup
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        llm = LLM(**engine_args_dict)
        print(f"Model ready: {model_name}")

        # Load tokenizer to format Level 1 prompt (only once)
        # This is necessary for prompt formatting
        _, level1_formatted_prompts, _ = func([prompts[0]], "image")

        # Process all images
        results_for_model = []

        for i, line in enumerate(lines):
            if line.strip() == "":
                continue

            # Parse the line and get the image path
            parts = line.strip().split(",")
            image_path = os.path.join(args.dataset_root, parts[0])

            # Get class information
            class_id = parts[-1] if len(parts) > 1 else "Unknown"
            class_name = class_82.get(class_id, "Unknown")

            print(
                f"Processing: {image_path} ({i+1}/{len(lines)}) - Model: {model_name} - Class: {class_id} ({class_name})"
            )

            try:
                # Run the model
                results = process_with_vllm_model(
                    llm,
                    model_name,
                    image_path,
                    level1_formatted_prompts,
                    stop_token_ids,
                    class_6,
                    class_20,
                    class_82,
                    yoga_diagram,
                )

                # Save the results
                results_for_model.append(
                    {
                        "path": parts[0],
                        "class_id": class_id,
                        "class_name": class_name,
                        "results": results,
                    }
                )
                # Immediately write results to CSV
                result_data = results_for_model[-1]
                csv_row = [
                    result_data["path"],
                    result_data["results"]["level1"]["response"],
                    result_data["results"]["level1"]["category"],
                    result_data["results"]["level1"]["category_id"],
                    result_data["results"]["level2"]["response"],
                    result_data["results"]["level2"]["subcategory"],
                    result_data["results"]["level2"]["subcategory_id"],
                    result_data["results"]["level3"]["response"],
                    result_data["results"]["level3"]["pose"],
                    result_data["results"]["level3"]["pose_id"],
                    result_data["class_id"],
                    result_data["class_name"],
                ]
                with open(csv_path, "a", newline="") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(csv_row)

                print(
                    f"    Result: {results['level3']['pose']} (ID: {results['level3']['pose_id']})"
                )

            except Exception as e:
                print(f"    ERROR: {str(e)}")
                # In case of error, add empty results
                results_for_model.append(
                    {
                        "path": parts[0],
                        "class_id": class_id,
                        "class_name": class_name,
                        "results": {
                            "level1": {
                                "response": f"ERROR: {str(e)}",
                                "category": "",
                                "category_id": "",
                            },
                            "level2": {
                                "response": "",
                                "subcategory": "",
                                "subcategory_id": "",
                            },
                            "level3": {"response": "", "pose": "", "pose_id": ""},
                        },
                    }
                )
                # Immediately write results to CSV (in case of error)
                result_data = results_for_model[-1]
                csv_row = [
                    result_data["path"],
                    result_data["results"]["level1"]["response"],
                    result_data["results"]["level1"]["category"],
                    result_data["results"]["level1"]["category_id"],
                    result_data["results"]["level2"]["response"],
                    result_data["results"]["level2"]["subcategory"],
                    result_data["results"]["level2"]["subcategory_id"],
                    result_data["results"]["level3"]["response"],
                    result_data["results"]["level3"]["pose"],
                    result_data["results"]["level3"]["pose_id"],
                    result_data["class_id"],
                    result_data["class_name"],
                ]
                with open(csv_path, "a", newline="") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(csv_row)

                # Memory cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Release the model
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Model released: {model_name}")

    else:
        # Process with a single image
        class_id = "Unknown"
        class_name = "Unknown"

        print(f"Processing: {args.image_path}")

        # Create CSV file and write headers
        with open(csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(csv_headers)

        # Process separately for each model (to avoid memory issues)
        all_results = []

        for model_name in args.models:
            print(f"\n=== Starting model: {model_name} ===")

            # Initialize the model
            func = MODEL_FUNCS[model_name]
            engine_args, _, stop_token_ids = func([prompts[0]], "image")
            engine_args.tensor_parallel_size = args.tensor_parallel_size
            engine_args_dict = asdict(engine_args)

            # Memory cleanup
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            llm = LLM(**engine_args_dict)
            print(f"Model ready: {model_name}")

            print(f"  Model: {model_name}")

            # Format Level 1 prompt (only once)
            _, level1_formatted_prompts, _ = func([prompts[0]], "image")

            # Run the model
            results = process_with_vllm_model(
                llm,
                model_name,
                args.image_path,
                level1_formatted_prompts,
                stop_token_ids,
                class_6,
                class_20,
                class_82,
                yoga_diagram,
            )

            # Save the results
            all_results.append({"model_name": model_name, "results": results})

            print(
                f"    Result: {results['level3']['pose']} (ID: {results['level3']['pose_id']})"
            )

            # Release the model
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Model released: {model_name}")

        # Create CSV row
        if len(args.models) == 1:
            # Simple format for single model
            csv_row = [
                args.image_path,
                all_results[0]["results"]["level1"]["response"],
                all_results[0]["results"]["level1"]["category"],
                all_results[0]["results"]["level1"]["category_id"],
                all_results[0]["results"]["level2"]["response"],
                all_results[0]["results"]["level2"]["subcategory"],
                all_results[0]["results"]["level2"]["subcategory_id"],
                all_results[0]["results"]["level3"]["response"],
                all_results[0]["results"]["level3"]["pose"],
                all_results[0]["results"]["level3"]["pose_id"],
                class_id,
                class_name,
            ]
        else:
            # Model-prefix format for multiple models
            csv_row = [args.image_path, class_id, class_name]
            for result_data in all_results:
                csv_row.extend(
                    [
                        result_data["results"]["level1"]["response"],
                        result_data["results"]["level1"]["category"],
                        result_data["results"]["level1"]["category_id"],
                        result_data["results"]["level2"]["response"],
                        result_data["results"]["level2"]["subcategory"],
                        result_data["results"]["level2"]["subcategory_id"],
                        result_data["results"]["level3"]["response"],
                        result_data["results"]["level3"]["pose"],
                        result_data["results"]["level3"]["pose_id"],
                    ]
                )

        # Write to CSV
        with open(csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(csv_headers)
            csv_writer.writerow(csv_row)

    print(f"\nAll results saved to CSV file: {csv_path}")
    print(f"Prompts saved: {prompts_path}")


if __name__ == "__main__":
    main()
