#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
import re
import argparse
import glob
from typing import Dict, List


def load_class_names(json_path: str) -> Dict[str, str]:
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


def get_class_id_mapping(class_names: Dict[str, str]) -> Dict[str, str]:
    """
    Sınıf adlarını, ID'lere eşleyen bir sözlük oluştur
    """
    class_id_mapping = {}
    for class_id, class_name in class_names.items():
        class_id_mapping[class_name.lower()] = class_id
    return class_id_mapping


def extract_predicted_class(
    raw_output: str, class_id_mapping: Dict[str, str], class_names: Dict[str, str]
) -> tuple:
    """
    Model çıktısından tahmin edilen sınıfı çıkarır ve eşleşen ID ve adı döndürür

    "Predicted Yoga Pose: X" veya "Predicted Yoga Pose: X: Y" formatlarını destekler

    Returns:
        tuple: (predicted_class_id, predicted_class_name)
    """
    # "Predicted Yoga Pose: X" formatını arar
    pattern = r"Predicted Yoga Pose:\s*(.*?)(?:\s*\(.*?\))?\s*$"
    matches = re.findall(pattern, raw_output, re.MULTILINE)

    if not matches:
        return "Bilinmiyor", "Bilinmiyor"

    predicted_class_text = matches[0].strip()

    # "40: Low Lunge Pose" formatını işle - sayı ve sonrasındaki ":" karakterini kaldır
    if re.match(r"^\d+\s*:\s*", predicted_class_text):
        predicted_class_text = re.sub(r"^\d+\s*:\s*", "", predicted_class_text)

    # Parantez içindeki numaraları kaldır
    predicted_class_text = re.sub(r"\(\d+\)", "", predicted_class_text).strip()

    # İlk kelimeyi kontrol et
    first_word = predicted_class_text.split()[0].lower() if predicted_class_text else ""

    # Eşleşme arama
    for class_name, class_id in class_id_mapping.items():
        if predicted_class_text.lower() == class_name:
            return class_id, class_names.get(class_id, "Bilinmiyor")

        # Eğer tam eşleşme bulunamazsa, ilk kelimeye göre kontrol et
        if first_word and class_name.lower().startswith(first_word):
            return class_id, class_names.get(class_id, "Bilinmiyor")

    # Eğer eşleşme bulunamazsa, "None" veya özel bir değer döndür
    return "Bilinmiyor", "Bilinmiyor"


def process_csv_file(
    csv_path: str, class_id_mapping: Dict[str, str], class_names: Dict[str, str]
) -> None:
    """
    Verilen CSV dosyasını işler ve predicted_class_id ve predicted_class_name sütunlarını ekler
    Eğer bu sütunlar zaten varsa, önce onları siler
    """
    temp_path = csv_path + ".temp"

    try:
        # Geçici bir dosya oluştur
        with open(csv_path, "r", newline="", encoding="utf-8") as input_file, open(
            temp_path, "w", newline="", encoding="utf-8"
        ) as output_file:

            reader = csv.reader(input_file)
            writer = csv.writer(output_file)

            # Başlık satırını oku
            header = next(reader)

            # raw_output sütununun indeksini bul
            raw_output_index = (
                header.index("raw_output") if "raw_output" in header else -1
            )

            if raw_output_index == -1:
                print(f"HATA: {csv_path} dosyasında 'raw_output' sütunu bulunamadı.")
                return

            # Mevcut predicted_class_id ve predicted_class_name sütunlarının indekslerini bul
            predicted_class_id_index = -1
            predicted_class_name_index = -1

            if "predicted_class_id" in header:
                predicted_class_id_index = header.index("predicted_class_id")

            if "predicted_class_name" in header:
                predicted_class_name_index = header.index("predicted_class_name")

            # Eğer bu sütunlar zaten varsa, onları başlıktan çıkar
            new_header = []
            for i, col in enumerate(header):
                if i != predicted_class_id_index and i != predicted_class_name_index:
                    new_header.append(col)

            # Şimdi yeni başlığa predicted_class_id ve predicted_class_name sütunlarını ekle
            new_header.append("predicted_class_id")
            new_header.append("predicted_class_name")
            writer.writerow(new_header)

            # Her satırı işle
            for row in reader:
                if len(row) <= raw_output_index:
                    # Hatalı satırları atla
                    continue

                # Eğer sütunlar zaten varsa, onları satırdan çıkar
                new_row = []
                for i, value in enumerate(row):
                    if (
                        i != predicted_class_id_index
                        and i != predicted_class_name_index
                    ):
                        new_row.append(value)

                # raw_output değerini al (yeni satırda indeks değişmiş olabilir)
                raw_output_new_index = (
                    new_header.index("raw_output") if "raw_output" in new_header else -1
                )
                raw_output = (
                    new_row[raw_output_new_index]
                    if raw_output_new_index != -1
                    else row[raw_output_index]
                )

                # Tahmin edilen sınıf ID'sini ve adını bul
                predicted_class_id, predicted_class_name = extract_predicted_class(
                    raw_output, class_id_mapping, class_names
                )

                # Satıra predicted_class_id ve predicted_class_name ekle
                new_row.append(predicted_class_id)
                new_row.append(predicted_class_name)
                writer.writerow(new_row)

        # Geçici dosyayı orijinal dosya ile değiştir
        os.replace(temp_path, csv_path)
        print(f"İşlendi: {csv_path}")

    except Exception as e:
        print(f"HATA: {csv_path} işlenirken bir hata oluştu: {str(e)}")
        # Eğer geçici dosya varsa temizle
        if os.path.exists(temp_path):
            os.remove(temp_path)


def process_all_output_folders(output_dir: str, class_json_path: str) -> None:
    """
    output klasörü altındaki tüm model_output.csv dosyalarını işler
    """
    # Sınıf adlarını yükle
    class_names = load_class_names(class_json_path)
    if not class_names:
        print(f"HATA: {class_json_path} dosyasından sınıf adları yüklenemedi.")
        return

    # Sınıf adlarını ID'lere eşleyen sözlük oluştur
    class_id_mapping = get_class_id_mapping(class_names)

    # output klasörü altındaki tüm alt klasörleri bul
    output_folders = [
        f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))
    ]

    if not output_folders:
        print(f"UYARI: {output_dir} altında hiç klasör bulunamadı.")
        return

    # Her bir output klasörü için
    for folder in output_folders:
        folder_path = os.path.join(output_dir, folder)
        csv_path = os.path.join(folder_path, "model_output.csv")

        # CSV dosyası var mı kontrol et
        if os.path.exists(csv_path) and os.path.isfile(csv_path):
            print(f"İşleniyor: {csv_path}")
            process_csv_file(csv_path, class_id_mapping, class_names)
        else:
            print(f"UYARI: {csv_path} bulunamadı.")


def main():
    parser = argparse.ArgumentParser(
        description="CSV dosyalarına predicted_class_id sütunu ekle"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="İşlenecek output klasörünün yolu (varsayılan: output)",
    )

    parser.add_argument(
        "--class_json",
        type=str,
        default="Yoga-82/class_82.json",
        help="Sınıf adlarını içeren JSON dosyasının yolu (varsayılan: Yoga-82/class_82.json)",
    )

    parser.add_argument(
        "--specific_folder",
        type=str,
        default="",
        help="Sadece belirli bir klasörü işle (belirtilmezse tüm klasörler işlenir)",
    )

    args = parser.parse_args()

    # Belirli bir klasör belirtilmişse sadece onu işle
    if args.specific_folder:
        folder_path = os.path.join(args.output_dir, args.specific_folder)
        csv_path = os.path.join(folder_path, "model_output.csv")

        if os.path.exists(csv_path) and os.path.isfile(csv_path):
            print(f"İşleniyor: {csv_path}")
            class_names = load_class_names(args.class_json)
            class_id_mapping = get_class_id_mapping(class_names)
            process_csv_file(csv_path, class_id_mapping, class_names)
        else:
            print(f"HATA: {csv_path} bulunamadı.")
    else:
        # Tüm output klasörlerini işle
        process_all_output_folders(args.output_dir, args.class_json)

    print("İşlem tamamlandı.")


if __name__ == "__main__":
    main()
