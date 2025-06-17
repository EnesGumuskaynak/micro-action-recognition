#!/bin/bash

echo "PyTorch AI projesi için gerekli paketler kuruluyor..."

# PyTorch paketlerini CUDA 12.8 için yükleme
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Diğer gerekli paketleri yükleme
pip3 install -r requirements.txt

echo "Kurulum tamamlandı!"