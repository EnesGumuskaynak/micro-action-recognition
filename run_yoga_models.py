import os
import argparse
import datetime
import csv
import json
from dataclasses import asdict
from PIL import Image
from transformers import AutoTokenizer

# type: ignore
from vllm import LLM, EngineArgs  # type: ignore
from vllm import (
    SamplingParams,
)  # added: sampling parameters to extend model output

# Load class names


def load_class_names(json_path="Yoga-82/class_82.json"):
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading class names: {e}")
        return {}


# Model functions


def run_minicpm_base(questions: list[str], modality: str, model_name: str):
    assert modality == "image"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=1,
        trust_remote_code=True,
        limit_mm_per_prompt={modality: 1},
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
        max_model_len=8192,
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


def run_internvl_14b(questions: list[str], modality: str):
    model_name = "OpenGVLab/InternVL3-14B"
    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
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

# Argument parser


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run yoga pose classification with multiple VL models."
    )
    parser.add_argument(
        "--model", choices=MODEL_FUNCS.keys(), required=True, help="Model to use."
    )
    parser.add_argument(
        "--image_path", type=str, default=None, help="Single image path"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="",
        help="Text file with image paths and labels",
    )
    parser.add_argument(
        "--dataset_root", type=str, default="Yoga-82/dataset/", help="Dataset root"
    )
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
    parser.add_argument(
        "--class_json",
        type=str,
        default="Yoga-82/class_82.json",
        help="JSON with class names",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=2,
        help="Number of GPUs to use for tensor parallelism",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Ensure both GPUs are used
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(i) for i in range(args.tensor_parallel_size)
    )
    class_names = load_class_names(args.class_json)
    # Default prompt
    default_prompt = """
Analyze the image and describe in detail what you see in terms of body position, orientation, and any distinctive posture features.
Then, based on your description, choose the most appropriate yoga pose name from the following list of 82 classes.

Respond in this format:
Description : [Your detailed description of the pose]
Predicted Yoga Pose : [One of the pose names below]

Yoga Pose Classes:
 Akarna Dhanurasana
 Bharadvaja's Twist
 Boat Pose
 Bound Angle Pose
 Bow Pose
 Bridge Pose
 Camel Pose
 Cat Cow Pose
 Chair Pose
 Child Pose
 Cobra Pose
 Cockerel Pose
 Corpse Pose
 Cow Face Pose
 Crane (Crow) Pose
 Dolphin Plank Pose
 Dolphin Pose
 Downward-Facing Dog
 Eagle Pose
 Eight-Angle Pose
 Extended Puppy Pose
 Extended Side Angle Pose
 Extended Triangle Pose
 Feathered Peacock Pose
 Firefly Pose
 Fish Pose
 Four-Limbed Staff Pose
 Frog Pose
 Garland Pose
 Gate Pose
 Half Lord of the Fishes Pose
 Half Moon Pose
 Handstand Pose
 Happy Baby Pose
 Head-to-Knee Forward Bend
 Heron Pose
 Intense Side Stretch Pose
 Legs-Up-the-Wall Pose
 Locust Pose
 Lord of the Dance Pose
 Low Lunge Pose
 Noose Pose
 Peacock Pose
 Pigeon Pose
 Plank Pose
 Plow Pose
 Sage Koundinya Pose
 King Pigeon Pose
 Reclining Hand-to-Big-Toe Pose
 Revolved Head-to-Knee Pose
 Scale Pose
 Scorpion Pose
 Seated Forward Bend
 Shoulder-Pressing Pose
 Side-Reclining Leg Lift Pose
 Side Crane (Crow) Pose
 Side Plank Pose
 Easy Sitting Pose
 Split Pose
 Staff Pose
 Standing Forward Bend
 Standing Split Pose
 Standing Big-Toe Hold Pose
 Headstand Pose
 Shoulder Stand Pose
 Reclining Cobbler Pose
 Reclining Hero Pose
 Tortoise Pose
 Tree Pose
 Upward Bow (Wheel) Pose
 Upward Facing Two-Foot Staff Pose
 Upward Plank Pose
 Hero and Thunderbolt Pose
 Warrior III Pose
 Warrior II Pose
 Warrior I Pose
 Wide Angle Seated Forward Bend
 Wide-Legged Forward Bend
 Wild Thing Pose
 Wind Relieving Pose
 Yogic Sleep Pose
 Reverse Warrior Pose
"""
    prompt_text = args.prompt or default_prompt
    # Prepare output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("output", f"{timestamp}_{args.model}")
    os.makedirs(out_dir, exist_ok=True)
    # Save prompt
    with open(os.path.join(out_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt_text)
    # CSV header
    csv_path = os.path.join(out_dir, "model_output.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_path", "prediction", "true_class_id", "true_class_name"]
        )
    # Load entries
    if args.input_file:
        with open(args.input_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
    else:
        if not args.image_path:
            raise ValueError("Provide --image_path or --input_file")
        lines = [args.image_path]
    # Initialize model
    func = MODEL_FUNCS[args.model]
    # Prepare tokenization and engine once
    _, prompts, stop_ids = func([prompt_text], "image")
    engine_args, _, _ = func([prompt_text], "image") if False else (None, None, None)
    # Actually need engine_args
    engine_args, prompts, stop_ids = func([prompt_text], "image")
    # Enable multiple GPUs via tensor parallelism
    engine_args.tensor_parallel_size = args.tensor_parallel_size
    # Initialize LLM with updated engine args
    engine_args_dict = asdict(engine_args)
    llm = LLM(**engine_args_dict)
    # Process each image
    for line in lines:
        if args.input_file:
            parts = line.split(",")
            rel_path = parts[0]
            image_path = os.path.join(args.dataset_root, rel_path)
            class_id = parts[-1]
            class_name = class_names.get(class_id, "Unknown")
        else:
            image_path = line
            class_id = ""
            class_name = ""
        print(f"Processing: {image_path}")
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: couldn't open image {image_path}: {e}, skipping.")
            continue
        # Generate
        inputs = [{"prompt": prompts[0], "multi_modal_data": {"image": image}}]
        # Set sampling parameters for longer outputs
        sampling_params = SamplingParams(
            temperature=0.3,  # Stable but diverse results
            top_p=0.9,
            max_tokens=1024,  # Generate more tokens
        )
        # Stop token ID'leri varsa ekle
        if stop_ids:
            sampling_params.stop_token_ids = stop_ids
        # Use generate with sampling parameters
        outputs = llm.generate(inputs, sampling_params)  # type: ignore
        prediction = outputs[0].outputs[0].text.strip()
        # Save
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([image_path, prediction, class_id, class_name])
        print(f"Result: {prediction}\n")


if __name__ == "__main__":
    main()
