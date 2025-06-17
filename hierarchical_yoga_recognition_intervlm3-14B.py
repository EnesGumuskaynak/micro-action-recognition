import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
import argparse
import os
import datetime
import csv
import json

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_class_names(json_path):
    """
    JSON dosyasından sınıf adlarını yükle
    """
    try:
        with open(json_path, "r") as f:
            class_names = json.load(f)
        return class_names
    except Exception as e:
        print(f"Sınıf adları yüklenirken hata oluştu: {str(e)}")
        return {}


# Argüman ayrıştırıcıyı tanımla
def parse_args():
    parser = argparse.ArgumentParser(
        description="InternVL3-14B için hiyerarşik yoga pozu sınıflandırma"
    )

    # Seviye 1 prompt (6 ana kategori)
    level1_prompt = """
Analyze the image and identify which of the following 6 main yoga pose categories 
it belongs to based on the body position.

Respond in this format:
Description: [Describe the body position and orientation]
Main Category: [One of the categories below]

Main Yoga Pose Categories:
0. Standing - Person is upright, weight primarily on feet
1. Sitting - Person is seated on the ground/mat
2. Balancing - Person is balancing on hands, feet, or other body parts
3. Inverted - Person is upside down or head is below heart
4. Reclining - Person is lying down on back, stomach or side
5. Wheel - Body forms a wheel or circular shape

Only return one of these 6 categories in your Main Category response.
"""

    # Seviye 2 prompt şablonu (alt kategoriler için, dinamik olarak doldurulacak)
    level2_prompt_template = """
You've identified this as a {main_category} yoga pose.
Now determine the specific subcategory within {main_category} poses.

Respond in this format:
Description: [Describe the specific body position]
Subcategory: [One of the subcategories below]

{subcategories_text}

Only return one of these subcategories in your Subcategory response.
"""

    # Seviye 3 prompt şablonu (spesifik pozlar için, dinamik olarak doldurulacak)
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
        help="Sınıflandırılacak yoga pozu görselinin yolu",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="",
        help="Görüntü yollarını içeren metin dosyası (format: 'Akarna_Dhanurasana/16.jpg,1,8,0')",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="Yoga-82/dataset/",
        help="Yoga dataset kök dizini",
    )
    parser.add_argument(
        "--level1_prompt", type=str, default=level1_prompt, help="Seviye 1 prompt"
    )
    parser.add_argument(
        "--level2_prompt_template",
        type=str,
        default=level2_prompt_template,
        help="Seviye 2 prompt şablonu",
    )
    parser.add_argument(
        "--level3_prompt_template",
        type=str,
        default=level3_prompt_template,
        help="Seviye 3 prompt şablonu",
    )
    parser.add_argument("--max_num", type=int, default=12, help="Maksimum parça sayısı")

    args = parser.parse_args()
    return args


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


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


def get_subcategory_prompt(main_category, class_20):
    # Ana kategoriye göre alt kategorileri hazırla
    subcategories = {}
    subcategory_text = "Subcategories for {}:\n".format(main_category)

    if main_category == "Standing":
        subcategories = {
            "0": "Standing - Straight",
            "1": "Standing - Forward bend",
            "2": "Standing - Side bend",
            "3": "Standing - Others",
        }
    elif main_category == "Sitting":
        subcategories = {
            "4": "Sitting - Normal1 (legs in front)",
            "5": "Sitting - Normal2 (legs behind)",
            "6": "Sitting - Split",
            "7": "Sitting - Forward bend",
            "8": "Sitting - Twist",
        }
    elif main_category == "Balancing":
        subcategories = {"9": "Balancing - Front", "10": "Balancing - Side"}
    elif main_category == "Inverted":
        subcategories = {
            "11": "Inverted - Legs straight up",
            "12": "Inverted - Legs bend",
        }
    elif main_category == "Reclining":
        subcategories = {
            "13": "Reclining - Up-facing",
            "14": "Reclining - Down-facing",
            "15": "Reclining - Side facing",
            "16": "Reclining - Plank balance",
        }
    elif main_category == "Wheel":
        subcategories = {
            "17": "Wheel - Up-facing",
            "18": "Wheel - Down-facing",
            "19": "Wheel - Others",
        }

    for key, value in subcategories.items():
        subcategory_text += "{}. {}\n".format(key, value)

    return subcategories, subcategory_text


# Seviye 2'den seviye 3'e geçiş için alt kategori altındaki pozları hazırla
def get_specific_poses(main_category, subcategory, yoga_diagram):
    # Diyagramdan ilgili alt kategori altındaki pozları bul
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


def extract_from_response(response, field_name):
    """LLM yanıtından belirli bir alanı çıkartır"""
    lines = response.strip().split("\n")
    for line in lines:
        if line.lower().startswith(field_name.lower()):
            return line.split(":", 1)[1].strip()
    return None


def prepare_yoga_diagram():
    """Yoga diyagramını hazırla (Seviye 1, 2 ve 3 arasındaki ilişkiyi oluştur)"""
    # Seviye 1, 2 ve 3 için sınıf adlarını yükle
    class_6 = load_class_names("Yoga-82/class_6.json")
    class_20 = load_class_names("Yoga-82/class_20.json")
    class_82 = load_class_names("Yoga-82/class_82.json")

    # Diyagram yapısını tanımla
    yoga_diagram = {
        "level1": class_6,  # 6 ana kategori
        "level2": class_20,  # 20 alt kategori
        "level3": class_82,  # 82 spesifik yoga pozu
        # Seviye 1'den seviye 2'ye eşleşmeler
        "level1_to_level2": {
            "0": ["0", "1", "2", "3"],  # Standing -> Standing subcategories
            "1": ["4", "5", "6", "7", "8"],  # Sitting -> Sitting subcategories
            "2": ["9", "10"],  # Balancing -> Balancing subcategories
            "3": ["11", "12"],  # Inverted -> Inverted subcategories
            "4": ["13", "14", "15", "16"],  # Reclining -> Reclining subcategories
            "5": ["17", "18", "19"],  # Wheel -> Wheel subcategories
        },
        # Seviye 2'den seviye 3'e eşleşmeler (görüntüdeki diyagrama göre düzenlenmiş)
        # Her alt kategoride bulunan pozlar için JSON dosyasından doğru ID'leri eşleştiriyoruz
        "level2_to_level3": {
            # Standing subcategories
            "0": ["8", "18", "68"],  # Standing - Straight (Eagle, Tree, Chair)
            "1": ["16", "17", "36", "60", "77"],  # Standing - Forward bend
            "2": ["21", "22", "29", "31", "40", "75", "81"],  # Standing - Side bend
            "3": ["39", "61", "62", "73", "74"],  # Standing - Others
            # Sitting subcategories
            "4": ["3", "28", "41", "57", "59"],  # Sitting - Normal1 (legs in front)
            "5": ["1", "13", "30", "72"],  # Sitting - Normal2 (legs behind)
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

    # Level 3 pozları ile ilgili eşleşmeleri doğru şekilde ekle

    # Standing - Straight (0): Eagle, Tree, Chair
    for pose_id, pose_name in class_82.items():
        if pose_name in ["Eagle Pose", "Tree Pose", "Chair Pose"]:
            yoga_diagram["level2_to_level3"]["0"].append(pose_id)

    # Standing - Forward bend (1)
    for pose_id, pose_name in class_82.items():
        if pose_name in [
            "Bow Pose",
            "Standing Forward Bend",
            "Wide-Legged Forward Bend",
            "Bridge Pose",
            "Downward-Facing Dog",
            "Camel Pose",
            "Low Lunge Pose",
            "Dolphin Pose",
        ]:
            yoga_diagram["level2_to_level3"]["1"].append(pose_id)

    # Standing - Side bend (2)
    for pose_id, pose_name in class_82.items():
        if pose_name in [
            "Half Moon Pose",
            "Extended Side Angle Pose",
            "Extended Triangle Pose",
            "Gate Pose",
            "Corpse Pose",
            "Intense Side Stretch Pose",
            "Cow Face Pose",
            "Crane (Crow) Pose",
            "Dolphin Plank Pose",
        ]:
            yoga_diagram["level2_to_level3"]["2"].append(pose_id)

    # Standing - Others (3)
    for pose_id, pose_name in class_82.items():
        if pose_name in [
            "Warrior II Pose",
            "Warrior I Pose",
            "Lord of the Dance Pose",
            "Standing Big-Toe Hold Pose",
            "Standing Split Pose",
        ]:
            yoga_diagram["level2_to_level3"]["3"].append(pose_id)

    # Sitting - Normal1 (4) (legs in front)
    for pose_id, pose_name in class_82.items():
        if pose_name in [
            "Extended Side Angle Pose",
            "Easy Sitting Pose",
            "Extended Triangle Pose",
            "Bound Angle Pose",
            "Fish Pose",
        ]:
            yoga_diagram["level2_to_level3"]["4"].append(pose_id)

    # Sitting - Normal2 (5) (legs behind)
    for pose_id, pose_name in class_82.items():
        if pose_name in [
            "Four-Limbed Staff Pose",
            "Hero and Thunderbolt Pose",
            "Frog Pose",
            "Garland Pose",
            "Gate Pose",
        ]:
            yoga_diagram["level2_to_level3"]["5"].append(pose_id)

    # Sitting - Split (6)
    for pose_id, pose_name in class_82.items():
        if pose_name in [
            "Half Lord of the Fishes Pose",
            "Half Moon Pose",
            "Split Pose",
        ]:
            yoga_diagram["level2_to_level3"]["6"].append(pose_id)

    # Sitting - Forward bend (7)
    for pose_id, pose_name in class_82.items():
        if pose_name in [
            "Handstand Pose",
            "Revolved Head-to-Knee Pose",
            "Head-to-Knee Forward Bend",
            "Heron Pose",
            "Seated Forward Bend",
            "Wide Angle Seated Forward Bend",
            "Staff Pose",
        ]:
            yoga_diagram["level2_to_level3"]["7"].append(pose_id)

    # Sitting - Twist (8)
    for pose_id, pose_name in class_82.items():
        if pose_name in [
            "Intense Side Stretch Pose",
            "Legs-Up-the-Wall Pose",
            "Locust Pose",
            "Tortoise Pose",
        ]:
            yoga_diagram["level2_to_level3"]["8"].append(pose_id)

    # Balancing - Front (9)
    for pose_id, pose_name in class_82.items():
        if pose_name in [
            "Lord of the Dance Pose",
            "Crane (Crow) Pose",
            "Low Lunge Pose",
            "Scale Pose",
            "Peacock Pose",
            "Pigeon Pose",
        ]:
            yoga_diagram["level2_to_level3"]["9"].append(pose_id)

    # Balancing - Side (10)
    for pose_id, pose_name in class_82.items():
        if pose_name in [
            "Plank Pose",
            "Sage Koundinya Pose",
            "Plow Pose",
            "Side Crane (Crow) Pose",
        ]:
            yoga_diagram["level2_to_level3"]["10"].append(pose_id)

    # Inverted - Legs straight up (11)
    for pose_id, pose_name in class_82.items():
        if pose_name in [
            "King Pigeon Pose",
            "Headstand Pose",
            "Revolved Head-to-Knee Pose",
            "Scale Pose",
            "Scorpion Pose",
        ]:
            yoga_diagram["level2_to_level3"]["11"].append(pose_id)

    # Inverted - Legs bend (12)
    for pose_id, pose_name in class_82.items():
        if pose_name in ["Seated Forward Bend", "Plow Pose"]:
            yoga_diagram["level2_to_level3"]["12"].append(pose_id)

    # Reclining - Up-facing (13)
    for pose_id, pose_name in class_82.items():
        if pose_name in [
            "Shoulder-Pressing Pose",
            "Side-Reclining Leg Lift Pose",
            "Side Crane (Crow) Pose",
            "Side Plank Pose",
            "Easy Sitting Pose",
            "Split Pose",
            "Staff Pose",
        ]:
            yoga_diagram["level2_to_level3"]["13"].append(pose_id)

    # Reclining - Down-facing (14)
    for pose_id, pose_name in class_82.items():
        if pose_name in [
            "Standing Forward Bend",
            "Standing Split Pose",
            "Standing Big-Toe Hold Pose",
            "Headstand Pose",
            "Shoulder Stand Pose",
        ]:
            yoga_diagram["level2_to_level3"]["14"].append(pose_id)

    # Reclining - Side facing (15)
    for pose_id, pose_name in class_82.items():
        if pose_name in [
            "Reclining Cobbler Pose",
            "Reclining Hero Pose",
            "Tortoise Pose",
            "Tree Pose",
        ]:
            yoga_diagram["level2_to_level3"]["15"].append(pose_id)

    # Reclining - Plank balance (16)
    for pose_id, pose_name in class_82.items():
        if pose_name in [
            "Upward Bow (Wheel) Pose",
            "Upward Facing Two-Foot Staff Pose",
            "Upward Plank Pose",
            "Hero and Thunderbolt Pose",
        ]:
            yoga_diagram["level2_to_level3"]["16"].append(pose_id)

    # Wheel - Up-facing (17)
    for pose_id, pose_name in class_82.items():
        if pose_name in [
            "Warrior III Pose",
            "Warrior II Pose",
            "Warrior I Pose",
            "Wide Angle Seated Forward Bend",
            "Wide-Legged Forward Bend",
        ]:
            yoga_diagram["level2_to_level3"]["17"].append(pose_id)

    # Wheel - Down-facing (18)
    for pose_id, pose_name in class_82.items():
        if pose_name in ["Yogic Sleep Pose", "Reverse Warrior Pose"]:
            yoga_diagram["level2_to_level3"]["18"].append(pose_id)

    # Wheel - Others (19)
    for pose_id, pose_name in class_82.items():
        if pose_name in ["Wild Thing Pose", "Wind Relieving Pose"]:
            yoga_diagram["level2_to_level3"]["19"].append(pose_id)

    return yoga_diagram


def main():
    # Argümanları analiz et
    args = parse_args()

    # Yoga diyagramını hazırla
    yoga_diagram = prepare_yoga_diagram()

    # Sınıf adlarını yükle
    class_6 = load_class_names("Yoga-82/class_6.json")
    class_20 = load_class_names("Yoga-82/class_20.json")
    class_82 = load_class_names("Yoga-82/class_82.json")
    path = "OpenGVLab/InternVL3-14B"
    model_name = path.split("/")[-1]  # InternVL3-14B

    # Birden fazla GPU'ya yükleme yapalım
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        print(f"Kullanılabilir GPU sayısı: {world_size}")

        if world_size > 1:
            print(f"Model birden fazla GPU'ya dağıtılıyor...")
            # Her işlem öncesinde GPU önbelleğini temizle
            torch.cuda.empty_cache()

            # Device map'i oluştur - modeli iki GPU arasında paylaştır
            device_map = split_model(path)

            # Modeli birden fazla GPU'ya yükle
            model = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=device_map,  # Her kısmı ilgili GPU'ya yükle
            )
            model = model.eval()

            # İlk GPU'yu primary device olarak kullan
            device = torch.device("cuda:0")
        else:
            device = torch.device("cuda:0")
            print(f"Sadece 1 GPU mevcut. Model {device} cihazına yükleniyor...")

            # Her işlem öncesinde GPU önbelleğini temizle
            torch.cuda.empty_cache()

            # Modeli önce CPU'ya yükle, sonra GPU'ya taşı
            model = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",  # Otomatik dağıtım dene
            )
            model = model.eval()
    else:
        print("CUDA kullanılamıyor, model CPU'ya yükleniyor.")
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )

    # Output klasörünü oluştur
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder_name = f"{timestamp}_hierarchical_{model_name}"
    output_dir = os.path.join("output", output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # CSV dosyasının yolu
    csv_path = os.path.join(output_dir, "hierarchical_results.csv")

    # Kullanılan promptları kaydet
    prompts_path = os.path.join(output_dir, "prompts.txt")
    with open(prompts_path, "w", encoding="utf-8") as prompts_file:
        prompts_file.write("LEVEL 1 PROMPT:\n" + args.level1_prompt)
        prompts_file.write(
            "\n\nLEVEL 2 PROMPT TEMPLATE:\n" + args.level2_prompt_template
        )
        prompts_file.write(
            "\n\nLEVEL 3 PROMPT TEMPLATE:\n" + args.level3_prompt_template
        )
    print(f"Promptlar kaydedildi: {prompts_path}")

    # Text dosyasından mı yoksa tek bir görüntüden mi işlem yapılacak kontrol et
    if args.input_file:
        # CSV dosyasını oluştur ve başlıkları yaz
        with open(csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [
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
            )

        # Text dosyasından görüntü yollarını oku
        with open(args.input_file, "r") as f:
            lines = f.readlines()

        # Her satır için işlem yap
        results = []
        for i, line in enumerate(lines):
            if line.strip() == "":
                continue

            # Satırı ayrıştır ve görüntü yolunu al
            parts = line.strip().split(",")
            image_path = os.path.join(args.dataset_root, parts[0])

            # Sınıf bilgisini al (son virgülden sonraki değer)
            class_id = parts[-1] if len(parts) > 1 else "Bilinmiyor"

            # Sınıf adını al
            class_name = class_82.get(class_id, "Bilinmiyor")

            print(
                f"İşleniyor: {image_path} ({i+1}/{len(lines)}) - Sınıf: {class_id} ({class_name})"
            )

            try:
                # Görüntüyü yükle
                pixel_values = load_image(image_path, max_num=args.max_num)

                # Modelle aynı cihaz ve veri tipini kullanalım
                if torch.cuda.is_available():
                    pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
                else:
                    pixel_values = pixel_values.to(dtype=torch.bfloat16)

                generation_config = dict(max_new_tokens=512, do_sample=True)

                # SEVİYE 1: Ana kategoriyi belirle
                question_level1 = "<image>\n" + args.level1_prompt
                response_level1 = model.chat(
                    tokenizer, pixel_values, question_level1, generation_config
                )

                main_category_text = extract_from_response(
                    response_level1, "Main Category"
                )
                main_category_id = None

                # Ana kategori ID'sini bul
                for key, value in class_6.items():
                    if value.lower() in main_category_text.lower():
                        main_category_id = key
                        main_category = value
                        break

                if not main_category_id:
                    main_category = main_category_text
                    main_category_id = "Bilinmiyor"

                print(
                    f"SEVİYE 1: Ana Kategori: {main_category} (ID: {main_category_id})"
                )

                # SEVİYE 2: Alt kategoriyi belirle
                subcategories, subcategory_text = get_subcategory_prompt(
                    main_category, class_20
                )
                level2_prompt = args.level2_prompt_template.format(
                    main_category=main_category, subcategories_text=subcategory_text
                )

                question_level2 = "<image>\n" + level2_prompt
                response_level2 = model.chat(
                    tokenizer, pixel_values, question_level2, generation_config
                )

                subcategory_text = extract_from_response(response_level2, "Subcategory")
                subcategory_id = None

                # Alt kategori ID'sini bul
                for key, value in class_20.items():
                    if value.lower() in subcategory_text.lower():
                        subcategory_id = key
                        subcategory = value
                        break

                if not subcategory_id:
                    subcategory = subcategory_text
                    subcategory_id = "Bilinmiyor"

                print(f"SEVİYE 2: Alt Kategori: {subcategory} (ID: {subcategory_id})")

                # SEVİYE 3: Spesifik pozu belirle
                specific_poses, specific_poses_text = get_specific_poses(
                    main_category, subcategory, yoga_diagram
                )
                level3_prompt = args.level3_prompt_template.format(
                    main_category=main_category,
                    subcategory=subcategory,
                    specific_poses_text=specific_poses_text,
                )

                question_level3 = "<image>\n" + level3_prompt
                response_level3 = model.chat(
                    tokenizer, pixel_values, question_level3, generation_config
                )

                # Son işlemden sonra GPU önbelleğini temizle
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                specific_pose_text = extract_from_response(
                    response_level3, "Specific Pose"
                )
                specific_pose_id = None

                # Spesifik poz ID'sini bul
                for key, value in class_82.items():
                    if value.lower() in specific_pose_text.lower():
                        specific_pose_id = key
                        specific_pose = value
                        break

                if not specific_pose_id:
                    specific_pose = specific_pose_text
                    specific_pose_id = "Bilinmiyor"

                print(
                    f"SEVİYE 3: Spesifik Poz: {specific_pose} (ID: {specific_pose_id})"
                )

                # Sonuçları kaydet
                results.append(
                    {
                        "image_path": parts[0],
                        "level1_response": response_level1,
                        "main_category": main_category,
                        "main_category_id": main_category_id,
                        "level2_response": response_level2,
                        "subcategory": subcategory,
                        "subcategory_id": subcategory_id,
                        "level3_response": response_level3,
                        "specific_pose": specific_pose,
                        "specific_pose_id": specific_pose_id,
                        "class_id": class_id,
                        "class_name": class_name,
                    }
                )

                # CSV'ye yaz
                with open(csv_path, "a", newline="") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(
                        [
                            parts[0],
                            response_level1,
                            main_category,
                            main_category_id,
                            response_level2,
                            subcategory,
                            subcategory_id,
                            response_level3,
                            specific_pose,
                            specific_pose_id,
                            class_id,
                            class_name,
                        ]
                    )

                print(f"Sonuç: {specific_pose}\n")

            except Exception as e:
                print(f"Hata: {image_path} işlenemedi: {str(e)}")

                # Hata durumunda bile CSV'ye bilgi yaz
                with open(csv_path, "a", newline="") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(
                        [
                            parts[0],
                            f"HATA: {str(e)}",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            class_id,
                            class_name,
                        ]
                    )

        # Sonuçları göster
        print(f"\n--- SONUÇLAR --- (CSV: {csv_path})")
        for result in results:
            print(
                f"{result['image_path']}: {result['specific_pose']} (ID: {result['specific_pose_id']}) "
                f"(Gerçek: {result['class_id']} - {result['class_name']})"
            )

    else:
        # Tek bir görüntü ile işlem yap
        pixel_values = load_image(args.image_path, max_num=args.max_num)

        # Modelle aynı cihaz ve veri tipini kullanalım
        if torch.cuda.is_available():
            pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
        else:
            pixel_values = pixel_values.to(dtype=torch.bfloat16)

        generation_config = dict(max_new_tokens=512, do_sample=True)

        # SEVİYE 1: Ana kategoriyi belirle
        question_level1 = "<image>\n" + args.level1_prompt
        response_level1 = model.chat(
            tokenizer, pixel_values, question_level1, generation_config
        )

        main_category_text = extract_from_response(response_level1, "Main Category")
        main_category_id = None

        # Ana kategori ID'sini bul
        for key, value in class_6.items():
            if value.lower() in main_category_text.lower():
                main_category_id = key
                main_category = value
                break

        if not main_category_id:
            main_category = main_category_text
            main_category_id = "Bilinmiyor"

        print(f"SEVİYE 1: Ana Kategori: {main_category} (ID: {main_category_id})")

        # SEVİYE 2: Alt kategoriyi belirle
        subcategories, subcategory_text = get_subcategory_prompt(
            main_category, class_20
        )
        level2_prompt = args.level2_prompt_template.format(
            main_category=main_category, subcategories_text=subcategory_text
        )

        question_level2 = "<image>\n" + level2_prompt
        response_level2 = model.chat(
            tokenizer, pixel_values, question_level2, generation_config
        )

        subcategory_text = extract_from_response(response_level2, "Subcategory")
        subcategory_id = None

        # Alt kategori ID'sini bul
        for key, value in class_20.items():
            if value.lower() in subcategory_text.lower():
                subcategory_id = key
                subcategory = value
                break

        if not subcategory_id:
            subcategory = subcategory_text
            subcategory_id = "Bilinmiyor"

        print(f"SEVİYE 2: Alt Kategori: {subcategory} (ID: {subcategory_id})")

        # SEVİYE 3: Spesifik pozu belirle
        specific_poses, specific_poses_text = get_specific_poses(
            main_category, subcategory, yoga_diagram
        )
        level3_prompt = args.level3_prompt_template.format(
            main_category=main_category,
            subcategory=subcategory,
            specific_poses_text=specific_poses_text,
        )

        question_level3 = "<image>\n" + level3_prompt
        response_level3 = model.chat(
            tokenizer, pixel_values, question_level3, generation_config
        )

        specific_pose_text = extract_from_response(response_level3, "Specific Pose")
        specific_pose_id = None

        # Spesifik poz ID'sini bul
        for key, value in class_82.items():
            if value.lower() in specific_pose_text.lower():
                specific_pose_id = key
                specific_pose = value
                break

        if not specific_pose_id:
            specific_pose = specific_pose_text
            specific_pose_id = "Bilinmiyor"

        print(f"SEVİYE 3: Spesifik Poz: {specific_pose} (ID: {specific_pose_id})")

        # CSV dosyasını oluştur ve tek satır yaz
        with open(csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [
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
            )
            csv_writer.writerow(
                [
                    args.image_path,
                    response_level1,
                    main_category,
                    main_category_id,
                    response_level2,
                    subcategory,
                    subcategory_id,
                    response_level3,
                    specific_pose,
                    specific_pose_id,
                    "Bilinmiyor",
                    "Bilinmiyor",
                ]
            )

        print(f"\nHiyerarşik Sonuçlar:")
        print(f"Seviye 1 (Ana Kategori): {main_category} (ID: {main_category_id})")
        print(f"Seviye 2 (Alt Kategori): {subcategory} (ID: {subcategory_id})")
        print(f"Seviye 3 (Spesifik Poz): {specific_pose} (ID: {specific_pose_id})")
        print(f"\nSonuçlar CSV dosyasına kaydedildi: {csv_path}")


if __name__ == "__main__":
    main()
