#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple


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


def evaluate_predictions(
    csv_path: str, class_names: Dict[str, str]
) -> Tuple[Dict, float, int, Dict]:
    """
    CSV dosyasındaki tahminleri değerlendirir ve doğruluk oranlarını hesaplar

    Args:
        csv_path: CSV dosyasının yolu
        class_names: Sınıf ID'lerini sınıf adlarına eşleyen sözlük

    Returns:
        Tuple[Dict, float, int, Dict]: (Sınıf bazlı başarı oranları, genel başarı oranı, toplam örnek sayısı, sınıf bazlı hata analizleri)
    """
    # Sınıf bazlı istatistikleri tutacak sözlükler
    class_counts = defaultdict(int)  # Her sınıf için toplam örnek sayısı
    class_correct = defaultdict(int)  # Her sınıf için doğru tahmin sayısı
    class_errors = defaultdict(
        lambda: defaultdict(int)
    )  # Her sınıf için yapılan hatalı tahminler

    total_count = 0  # Toplam örnek sayısı
    total_correct = 0  # Toplam doğru tahmin sayısı
    unknown_count = 0  # Bilinmeyen tahminlerin sayısı

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Gerekli alanları kontrol et
                if "true_class_id" not in row or "predicted_class_id" not in row:
                    print(f"UYARI: CSV dosyasında gerekli sütunlar eksik: {csv_path}")
                    continue

                # Gerçek ve tahmin edilen sınıf ID'lerini al
                true_class_id = row["true_class_id"]
                predicted_class_id = row["predicted_class_id"]

                # Toplam örnek sayısını artır
                total_count += 1

                # Sınıf bazlı istatistikleri güncelle
                class_counts[true_class_id] += 1

                # Bilinmeyen tahminler
                if predicted_class_id == "Bilinmiyor":
                    unknown_count += 1
                    # Bilinmeyen tahminleri de hata olarak kaydediyoruz
                    class_errors[true_class_id]["Bilinmiyor"] += 1
                    continue

                # Doğru tahminleri hesapla
                if true_class_id == predicted_class_id:
                    total_correct += 1
                    class_correct[true_class_id] += 1
                else:
                    # Hatalı tahminleri kaydet
                    class_errors[true_class_id][predicted_class_id] += 1

        # Sınıf bazlı başarı oranlarını hesapla
        class_accuracy = {}
        for class_id, count in class_counts.items():
            if count > 0:
                accuracy = (class_correct[class_id] / count) * 100
                class_name = class_names.get(class_id, f"Sınıf {class_id}")
                class_accuracy[class_id] = {
                    "name": class_name,
                    "accuracy": accuracy,
                    "correct": class_correct[class_id],
                    "total": count,
                }

        # Genel başarı oranını hesapla
        overall_accuracy = (total_correct / total_count) * 100 if total_count > 0 else 0

        # Her sınıf için hata analizini düzenle
        class_error_analysis = {}
        for class_id, errors in class_errors.items():
            # En çok yapılan hata türlerini sırala (en yüksek frekans önce)
            sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)
            # En fazla 5 hata türü al
            top_errors = sorted_errors[:5]
            # Hata analiz bilgisini sınıf için kaydet
            class_error_analysis[class_id] = top_errors

        return class_accuracy, overall_accuracy, total_count, class_error_analysis

    except Exception as e:
        print(f"HATA: CSV dosyası işlenirken bir hata oluştu: {str(e)}")
        return {}, 0, 0


def print_evaluation_results(
    class_accuracy: Dict, overall_accuracy: float, total_count: int
) -> None:
    """
    Değerlendirme sonuçlarını formatlı bir şekilde ekrana yazdırır
    """
    print("\n" + "=" * 80)
    print(f"GENEL DEĞERLENDİRME SONUÇLARI:")
    print("=" * 80)
    print(f"Toplam Örnek Sayısı: {total_count}")
    print(f"Genel Başarı Oranı: {overall_accuracy:.2f}%")
    print("=" * 80)

    print("\nSINIF BAZLI BAŞARI ORANLARI:")
    print("-" * 80)
    print(
        f"{'Sınıf ID':<10} {'Sınıf Adı':<30} {'Başarı Oranı':<15} {'Doğru/Toplam':<15}"
    )
    print("-" * 80)

    # Sınıfları başarı oranına göre sırala
    sorted_classes = sorted(
        class_accuracy.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )

    for class_id, data in sorted_classes:
        print(
            f"{class_id:<10} {data['name']:<30} {data['accuracy']:.2f}% {data['correct']}/{data['total']:<15}"
        )

    print("-" * 80)


def save_results_to_csv(
    output_path: str,
    class_accuracy: Dict,
    overall_accuracy: float,
    total_count: int,
    class_error_analysis: Dict,
    class_names: Dict,
) -> None:
    """
    Değerlendirme sonuçlarını CSV dosyasına kaydeder
    """
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Başlık satırı
            writer.writerow(
                [
                    "Sınıf ID",
                    "Sınıf Adı",
                    "Başarı Oranı (%)",
                    "Doğru Tahmin",
                    "Toplam Örnek",
                    "En Sık Yapılan Hatalar (Top 5)",
                ]
            )

            # Sınıf bazlı sonuçlar
            for class_id, data in class_accuracy.items():
                # Hata analizi bilgilerini hazırla
                error_text = ""
                if class_id in class_error_analysis and class_error_analysis[class_id]:
                    error_list = []
                    for error_class_id, error_count in class_error_analysis[class_id]:
                        error_class_name = class_names.get(
                            error_class_id, f"Sınıf {error_class_id}"
                        )
                        error_list.append(f"{error_class_name} ({error_count})")
                    error_text = ", ".join(error_list)

                writer.writerow(
                    [
                        class_id,
                        data["name"],
                        f"{data['accuracy']:.2f}",
                        data["correct"],
                        data["total"],
                        error_text,
                    ]
                )

            # Genel sonuçlar
            writer.writerow([])
            writer.writerow(
                ["GENEL", "", f"{overall_accuracy:.2f}", "", total_count, ""]
            )

        print(f"\nSonuçlar CSV dosyasına kaydedildi: {output_path}")

    except Exception as e:
        print(f"HATA: Sonuçlar CSV dosyasına kaydedilirken bir hata oluştu: {str(e)}")


def process_folder(
    folder_path: str, class_names: Dict[str, str], save_csv: bool = False
) -> None:
    """
    Belirtilen klasördeki model_output.csv dosyasını değerlendirir
    """
    csv_path = os.path.join(folder_path, "model_output.csv")

    if not os.path.exists(csv_path):
        print(f"HATA: {csv_path} bulunamadı.")
        return

    print(f"Değerlendiriliyor: {csv_path}")

    # CSV dosyasını değerlendir
    class_accuracy, overall_accuracy, total_count, class_error_analysis = (
        evaluate_predictions(csv_path, class_names)
    )

    # Sonuçları ekrana yazdır
    print_evaluation_results(class_accuracy, overall_accuracy, total_count)

    # İstenirse sonuçları CSV dosyasına kaydet
    if save_csv:
        output_filename = os.path.join(folder_path, "evaluation_results.csv")
        save_results_to_csv(
            output_filename,
            class_accuracy,
            overall_accuracy,
            total_count,
            class_error_analysis,
            class_names,
        )


def process_all_folders(
    output_dir: str, class_json_path: str, save_csv: bool = False
) -> None:
    """
    Output klasörü altındaki tüm klasörleri değerlendirir
    """
    # Sınıf adlarını yükle
    class_names = load_class_names(class_json_path)
    if not class_names:
        print(f"HATA: {class_json_path} dosyasından sınıf adları yüklenemedi.")
        return

    # Output klasörü altındaki tüm klasörleri bul
    try:
        folders = [
            folder
            for folder in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, folder))
        ]
    except Exception as e:
        print(
            f"HATA: {output_dir} klasöründeki dosyalar listelenirken bir hata oluştu: {str(e)}"
        )
        return

    if not folders:
        print(f"UYARI: {output_dir} altında hiç klasör bulunamadı.")
        return

    # Her bir klasör için değerlendirme yap
    for folder in folders:
        folder_path = os.path.join(output_dir, folder)
        process_folder(folder_path, class_names, save_csv)


def main():
    """
    Ana işlev
    """
    parser = argparse.ArgumentParser(
        description="Yoga duruş tahminlerini değerlendir ve başarı oranlarını hesapla"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Değerlendirilecek model çıktılarının bulunduğu ana klasör (varsayılan: output)",
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
        help="Sadece belirli bir klasörü değerlendir (belirtilmezse tüm klasörler değerlendirilir)",
    )

    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Değerlendirme sonuçlarını CSV dosyasına kaydet",
    )

    args = parser.parse_args()

    # Sınıf adlarını yükle
    class_names = load_class_names(args.class_json)
    if not class_names:
        return

    # Belirli bir klasör belirtilmişse sadece onu değerlendir
    if args.specific_folder:
        folder_path = os.path.join(args.output_dir, args.specific_folder)
        if os.path.isdir(folder_path):
            process_folder(folder_path, class_names, args.save_csv)
        else:
            print(f"HATA: {folder_path} bir klasör değil veya bulunamadı.")
    else:
        # Tüm klasörleri değerlendir
        process_all_folders(args.output_dir, args.class_json, args.save_csv)


if __name__ == "__main__":
    main()
