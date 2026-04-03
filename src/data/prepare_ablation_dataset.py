import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter

BASE_DIR = Path("/home/neuralist/HSE_AI_master/master-thesis/building-facades-damage-prediction")
DS1_RAW = BASE_DIR / "data/raw/screenshots_206_imgs"
DS1_MASKS = BASE_DIR / "data/processed/cvat_2026_03_18_18_24_25_segmentation mask 1.1/SegmentationClass"

OUT_DIR = BASE_DIR / "data/processed/ablation_dataset_206"
OUT_IMAGES = OUT_DIR / "images"
OUT_MASKS = OUT_DIR / "masks"

# Датасет 1 (206 патчей) BGR:
BGR_COLOR_MAP_1 = {
    (0, 0, 0): 0,          # background
    (217, 21, 54): 1,      # coating_deterioration
    (183, 50, 250): 2,     # cracks
    (209, 125, 42): 3,     # masonry_degradation
    (98, 221, 38): 4,      # moisture_bio_damage
    (123, 28, 222): 5      # vandalism
}

def rgb_to_index(mask_bgr):
    mask_idx = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
    for bgr_color, class_idx in BGR_COLOR_MAP_1.items():
        matches = np.all(mask_bgr == bgr_color, axis=-1)
        mask_idx[matches] = class_idx
    return mask_idx

def get_rarest_class_in_mask(mask_path):
    mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    mask_idx = rgb_to_index(mask_bgr)
    unique_classes = np.unique(mask_idx)
    
    priority = {5: 50, 3: 40, 4: 30, 2: 20, 1: 10, 0: 0}
    max_prio = -1
    target_class = 0
    for cls in unique_classes:
        if cls in priority and priority[cls] > max_prio:
            max_prio = priority[cls]
            target_class = cls
    return target_class

def process_dataset(raw_dir, masks_dir, img_ext):
    data_list = []
    for mask_file in os.listdir(masks_dir):
        if not mask_file.endswith('.png'):
            continue
            
        base_name = os.path.splitext(mask_file)[0]
        mask_path = masks_dir / mask_file
        img_path = raw_dir / (base_name + img_ext)
        
        if img_path.exists():
            strata_class = get_rarest_class_in_mask(mask_path)
            data_list.append({
                'img_path': img_path,
                'mask_path': mask_path,
                'basename': base_name,
                'strata': strata_class
            })
    return data_list

def main():
    print("Собираем датасет абляции (206 изображений)...")
    all_data = process_dataset(DS1_RAW, DS1_MASKS, img_ext='.png')
    print(f"Найдено полных пар: {len(all_data)}")
    
    y_strata = [item['strata'] for item in all_data]
    
    counts = Counter(y_strata)
    most_common_class = counts.most_common(1)[0][0]
    for i in range(len(y_strata)):
        if counts[y_strata[i]] < 2:
            print(f"Класс {y_strata[i]} имеет меньше 2 представителей. Меняем на {most_common_class}.")
            y_strata[i] = most_common_class
            
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42, stratify=y_strata)
    
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}")
    
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
        
    for split in ['train', 'val']:
        (OUT_IMAGES / split).mkdir(parents=True, exist_ok=True)
        (OUT_MASKS / split).mkdir(parents=True, exist_ok=True)
        
    def copy_files(data_subset, split_name):
        for item in tqdm(data_subset, desc=f"Обработка {split_name}"):
            img = cv2.imread(str(item['img_path']))
            mask_bgr = cv2.imread(str(item['mask_path']), cv2.IMREAD_COLOR)
            mask = rgb_to_index(mask_bgr)
            
            cv2.imwrite(str(OUT_IMAGES / split_name / f"{item['basename']}.png"), img)
            cv2.imwrite(str(OUT_MASKS / split_name / f"{item['basename']}.png"), mask)

    copy_files(train_data, 'train')
    copy_files(val_data, 'val')
    print(f"Готово! Абляционный датасет сохранен в: {OUT_DIR}")

if __name__ == "__main__":
    main()
