import math
import numpy as np
import torch
import torchvision.transforms as T
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
        description="MiniCPM-V için yoga pozu sınıflandırma"
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
    parser.add_argument("--max_num", type=int, default=1, help="Maksimum parça sayısı")

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


def preprocess_image(image_path, input_size=448):
    """
    MiniCPM-V için görüntüyü ön işleme
    """
    image = Image.open(image_path).convert("RGB")
    return image


def main():
    # Argümanları analiz et
    args = parse_args()

    # Sınıf adlarını yükle
    class_names = load_class_names()

    model_path = "openbmb/MiniCPM-V"
    model_name = model_path.split("/")[-1]  # MiniCPM-V

    # MiniCPM-V modelini yükle
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = model.to(device="cuda", dtype=torch.bfloat16)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

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
                image = preprocess_image(image_path)

                # Prompt ile sorgula
                question = args.prompt

                # Görüntüyü içeren mesaj oluştur
                msgs = [{"role": "user", "content": question}]

                # MiniCPM-V modeli ile tahmin yap
                response, context, _ = model.chat(
                    image=image,
                    msgs=msgs,
                    context=None,
                    tokenizer=tokenizer,
                    sampling=True,
                    temperature=0.7,
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
        image = preprocess_image(args.image_path)

        # Prompt ile sorgula
        question = args.prompt

        # Görüntüyü içeren mesaj oluştur
        msgs = [{"role": "user", "content": question}]

        # MiniCPM-V modeli ile tahmin yap
        response, context, _ = model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7,
        )

        # CSV dosyasını oluştur ve tek satır yaz
        with open(csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                ["path", "raw_output", "true_class_id", "true_class_name"]
            )
            csv_writer.writerow([args.image_path, response, "Bilinmiyor", "Bilinmiyor"])

        print(f"User: {question}\nAssistant: {response}")
        print(f"\nSonuç CSV dosyasına kaydedildi: {csv_path}")


if __name__ == "__main__":
    main()
