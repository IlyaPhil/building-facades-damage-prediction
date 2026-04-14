import os
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path("/home/neuralist/HSE_AI_master/master-thesis/building-facades-damage-prediction")
TRAIN_MASKS_DIR = BASE_DIR / "data/processed/tiles_800x800_408-imgs/masks/train"

CLASS_NAMES = {
    0: "background",
    1: "coating_deterioration",
    2: "cracks",
    3: "masonry_degradation",
    4: "moisture_bio_damage",
    5: "vandalism"
}

def calculate_class_weights(masks_dir):
    print("Подсчет пикселей по классам...")
    
    pixel_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    total_pixels = 0
    total_images = 0
    
    for mask_file in os.listdir(masks_dir):
        if not mask_file.endswith('.png'):
            continue
            
        mask_path = masks_dir / mask_file
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Подсчет количества пикселей каждого класса
        unique, counts = np.unique(mask, return_counts=True)
        for val, count in zip(unique, counts):
            if val in pixel_counts:
                pixel_counts[val] += count
            else:
                print(f"Внимание: найден неизвестный класс {val} в файле {mask_file}")
                
        total_pixels += mask.size
        total_images += 1
        
    print(f"\nОбработано изображений: {total_images}")
    print("Статистика пикселей по классам:")
    for cls, count in pixel_counts.items():
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"Класс {cls} ({CLASS_NAMES[cls]}): {count} пикселей ({percentage:.4f}%)")
        
    # Расчет весов по методу Median Frequency Balancing (MFB)
    # frequency = count / total_pixels (считается только для присутствующих классов, чтобы не было деления на 0)
    
    valid_counts = [count for cls, count in pixel_counts.items() if count > 0]
    if not valid_counts:
        return
        
    valid_frequencies = [count / total_pixels for count in valid_counts]
    median_freq = np.median(valid_frequencies)
    
    weights = []
    print("\nРассчитанные веса (Median Frequency Balancing) для функции потерь (CrossEntropyLoss):")
    for cls in range(len(CLASS_NAMES)):
        count = pixel_counts[cls]
        if count == 0:
            weights.append(0.0)
            print(f"Класс {cls}: 0.0 (класс отсутствует)")
        else:
            freq = count / total_pixels
            w = median_freq / freq
            # Слегка сглаживаем, чтобы веса фона не падали почти до 0, а редких не улетали в космос (по желанию, 
            # но классический метод оставляет как есть, ограничим сверху скажем 50.0)
            w = min(w, 50.0)
            weights.append(w)
            print(f"Класс {cls} ({CLASS_NAMES[cls]}): {w:.4f}")
            
    # Форматированный вывод для вставки в ноутбук
    print("\n--- Для вставки в Kaggle ноутбук ---")
    weights_str = ", ".join([f"{w:.4f}" for w in weights])
    print(f"class_weights = torch.tensor([{weights_str}], dtype=torch.float32).to(device)")

if __name__ == "__main__":
    calculate_class_weights(TRAIN_MASKS_DIR)
