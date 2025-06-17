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


def load_class_names(json_path="Yoga-82/class_82.json"):
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
        description="InternVL3-8B için yoga pozu sınıflandırma"
    )

    # Varsayılan prompt
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
        "--prompt", type=str, default=default_prompt, help="Modele gönderilecek prompt"
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


# If you set `load_in_8bit=True`, you will need two 80GB GPUs.
# If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
def main():
    # Argümanları analiz et
    args = parse_args()

    # Sınıf adlarını yükle
    class_names = load_class_names()

    path = "OpenGVLab/InternVL3-14B"
    model_name = path.split("/")[-1]  # InternVL3-14B
    device_map = split_model(path)
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )

    # Output klasörünü oluştur
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder_name = f"{timestamp}_{model_name}"
    output_dir = os.path.join("output", output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # CSV dosyasının yolu
    csv_path = os.path.join(output_dir, "model_output.csv")

    # Kullanılan prompt'u txt dosyasına kaydet
    prompt_path = os.path.join(output_dir, "prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as prompt_file:
        prompt_file.write(args.prompt)
    print(f"Prompt kaydedildi: {prompt_path}")

    # Text dosyasından mı yoksa tek bir görüntüden mi işlem yapılacak kontrol et
    if args.input_file:
        # CSV dosyasını oluştur ve başlıkları yaz
        with open(csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                ["path", "raw_output", "true_class_id", "true_class_name"]
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
            class_name = class_names.get(class_id, "Bilinmiyor")

            print(
                f"İşleniyor: {image_path} ({i+1}/{len(lines)}) - Sınıf: {class_id} ({class_name})"
            )

            try:
                # Görüntüyü yükle
                pixel_values = (
                    load_image(image_path, max_num=args.max_num)
                    .to(torch.bfloat16)
                    .cuda()
                )
                generation_config = dict(max_new_tokens=1024, do_sample=True)

                # Prompt ile sorgula
                question = "<image>\n" + args.prompt
                response = model.chat(
                    tokenizer, pixel_values, question, generation_config
                )

                # Sonuçları kaydet
                results.append(
                    {
                        "image_path": parts[0],
                        "prediction": response,
                        "class_id": class_id,
                        "class_name": class_name,
                        "original_line": line.strip(),
                    }
                )

                # CSV'ye yaz
                with open(csv_path, "a", newline="") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([parts[0], response, class_id, class_name])

                print(f"Sonuç: {response}\n")

            except Exception as e:
                print(f"Hata: {image_path} işlenemedi: {str(e)}")

                # Hata durumunda bile CSV'ye bilgi yaz
                with open(csv_path, "a", newline="") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(
                        [parts[0], f"HATA: {str(e)}", class_id, class_name]
                    )

        # Sonuçları göster
        print(f"\n--- SONUÇLAR --- (CSV: {csv_path})")
        for result in results:
            print(
                f"{result['image_path']}: {result['prediction']} (Gerçek: {result['class_id']} - {result['class_name']})"
            )

    else:
        # Tek bir görüntü ile işlem yap
        pixel_values = (
            load_image(args.image_path, max_num=args.max_num).to(torch.bfloat16).cuda()
        )
        generation_config = dict(max_new_tokens=1024, do_sample=True)

        # Argümanlardan gelen prompt'u kullan
        question = "<image>\n" + args.prompt
        response = model.chat(tokenizer, pixel_values, question, generation_config)

        # CSV dosyasını oluştur ve tek satır yaz
        with open(csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                ["path", "raw_output", "true_class_id", "true_class_name"]
            )
            csv_writer.writerow([args.image_path, response, "Bilinmiyor", "Bilinmiyor"])

        print(f"User: {question}\nAssistant: {response}")
        print(f"\nSonuç CSV dosyasına kaydedildi: {csv_path}")


# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ]
    )
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(
        bound, fps, max_frame, first_idx=0, num_segments=num_segments
    )
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(
            img, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


if __name__ == "__main__":
    main()
