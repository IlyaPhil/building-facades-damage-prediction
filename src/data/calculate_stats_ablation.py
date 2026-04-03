import os
import cv2
import numpy as np
from tqdm import tqdm
import torch

BASE_DIR = "/home/neuralist/HSE_AI_master/master-thesis/building-facades-damage-prediction"
TRAIN_MASKS_DIR = os.path.join(BASE_DIR, "data/processed/ablation_dataset_206/masks/train")

CLASS_MAPPING = {
    0: "background",
    1: "coating_deterioration",
    2: "cracks",
    3: "masonry_degradation",
    4: "moisture_bio_damage",
    5: "vandalism"
}

def main():
    print("Подсчет пикселей в абляционном датасете для расчета весов (Median Frequency Balancing)...")
    
    pixel_counts = {class_id: 0 for class_id in CLASS_MAPPING.keys()}
    total_pixels_in_dataset = 0
    
    mask_files = [f for f in os.listdir(TRAIN_MASKS_DIR) if f.endswith('.png')]
    for mask_file in tqdm(mask_files, desc="Сканирование масок"):
        mask_path = os.path.join(TRAIN_MASKS_DIR, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        unique, counts = np.unique(mask, return_counts=True)
        for cls_id, count in zip(unique, counts):
            if cls_id in pixel_counts:
                pixel_counts[cls_id] += count
                total_pixels_in_dataset += count

    print("\n--- Статистика пикселей в TRAIN (Абляция) ---")
    frequencies = {}
    for cls_id, count in pixel_counts.items():
        freq = count / total_pixels_in_dataset
        frequencies[cls_id] = freq
        print(f"Класс {cls_id} ({CLASS_MAPPING[cls_id]}): {count} пикселей ({freq*100:.4f}%)")

    # Формула: weight = median(frequencies) / frequency
    valid_freqs = [f for f in frequencies.values() if f > 0]
    median_freq = np.median(valid_freqs)
    
    weights = {}
    for cls_id, freq in frequencies.items():
        if freq > 0:
            weights[cls_id] = median_freq / freq
        else:
            weights[cls_id] = 0.0

    print("\nРассчитанные веса (Median Frequency Balancing):")
    for cls_id, w in weights.items():
        print(f"Класс {cls_id} ({CLASS_MAPPING[cls_id]}): {w:.4f}")

    weights_list = [weights[i] for i in range(len(CLASS_MAPPING))]
    print("\n--- Для вставки в Kaggle ноутбук ---")
    print(f"class_weights = torch.tensor([{', '.join([f'{w:.4f}' for w in weights_list])}], dtype=torch.float32).to(device)")

if __name__ == "__main__":
    main()
