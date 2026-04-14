import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Пути к исходным данным
BASE_DIR = Path("/home/neuralist/HSE_AI_master/master-thesis/building-facades-damage-prediction")
DS1_RAW = BASE_DIR / "data/raw/screenshots_206_imgs"
DS1_MASKS = BASE_DIR / "data/raw/facade-damage-seg-206-imgs-v2/SegmentationClass"

DS2_RAW = BASE_DIR / "data/raw/100_100_imgs_raw"
DS2_MASKS = BASE_DIR / "data/raw/facade-damage-seg-100_100_imgs/SegmentationClass"

# Путь для сохранения объединенного датасета
OUT_DIR = BASE_DIR / "data/processed/unified_dataset"
OUT_IMAGES = OUT_DIR / "images"
OUT_MASKS = OUT_DIR / "masks"

TARGET_SIZE = (800, 800)

COLOR_MAP = {
    (0, 0, 0): 0,          # background
    (217, 21, 54): 1,      # coating_deterioration (CVAT is RGB, cv2 reads BGR, wait. Let's use RGB below)
    (183, 50, 250): 2,     # cracks
    (209, 125, 42): 3,     # masonry_degradation
    (98, 221, 38): 4,      # moisture_bio_damage
    (123, 28, 222): 5      # vandalism
}
# Note: cv2 reads in BGR. So we should reverse the CVAT RGB tuples.

# Датасет 1 (206 патчей) BGR:
BGR_COLOR_MAP_1 = {
    (0, 0, 0): 0,          # background
    (217, 21, 54): 1,      # coating_deterioration
    (183, 50, 250): 2,     # cracks
    (209, 125, 42): 3,     # masonry_degradation
    (98, 221, 38): 4,      # moisture_bio_damage
    (123, 28, 222): 5      # vandalism
}

# Датасет 2 (100 патчей) BGR из его labelmap.txt:
BGR_COLOR_MAP_2 = {
    (0, 0, 0): 0,          # background
    (224, 69, 94): 1,      # coating_deterioration
    (152, 89, 219): 2,     # cracks
    (209, 125, 42): 3,     # masonry_degradation
    (160, 244, 184): 4,    # moisture_bio_damage
    (116, 86, 241): 5      # vandalism
}

def rgb_to_index(mask_bgr, color_map):
    # Converts BGR mask to 1D index mask
    mask_idx = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
    for bgr_color, class_idx in color_map.items():
        # Find matches for this color
        matches = np.all(mask_bgr == bgr_color, axis=-1)
        mask_idx[matches] = class_idx
    return mask_idx

def get_rarest_class_in_mask(mask_path, is_ds1):
    mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    color_map = BGR_COLOR_MAP_1 if is_ds1 else BGR_COLOR_MAP_2
    mask_idx = rgb_to_index(mask_bgr, color_map)
    unique_classes = np.unique(mask_idx)
    
    priority = {5: 50, 3: 40, 4: 30, 2: 20, 1: 10, 0: 0}
    max_prio = -1
    target_class = 0
    for cls in unique_classes:
        if cls in priority and priority[cls] > max_prio:
            max_prio = priority[cls]
            target_class = cls
    return target_class

def process_dataset(raw_dir, masks_dir, img_ext, resize_needed=False, is_ds1=True):
    """
    Ищет все маски в masks_dir, сопоставляет с изображениями из raw_dir.
    Возвращает список словарей с путями и меткой стратификации.
    """
    data_list = []
    
    # Перебираем все маски (они всегда .png)
    for mask_file in os.listdir(masks_dir):
        if not mask_file.endswith('.png'):
            continue
            
        base_name = os.path.splitext(mask_file)[0]
        mask_path = masks_dir / mask_file
        
        img_path = None
        
        # Проверяем возможные варианты расширений
        possible_exts = [img_ext, '.jpg', '.JPG', '.png', '.jpeg']
        for ext in possible_exts:
            p = raw_dir / (base_name + ext)
            if p.exists():
                img_path = p
                break
                
        if img_path is not None:

            # Находим доминирующий/редкий класс для стратификации
            strata_class = get_rarest_class_in_mask(mask_path, is_ds1)
            
            data_list.append({
                'img_path': img_path,
                'mask_path': mask_path,
                'basename': base_name,
                'strata': strata_class,
                'resize': resize_needed,
                'is_ds1': is_ds1
            })
        else:
            print(f"Предупреждение: для маски {mask_file} не найдено изображение {img_path.name}")
            
    return data_list

def main():
    print("Собираем информацию о датасетах...")
    # Датасет 1 (206 патчей, img .png, 800x800)
    data1 = process_dataset(DS1_RAW, DS1_MASKS, img_ext='.png', resize_needed=False, is_ds1=True)
    
    # Датасет 2 (100 патчей, img .JPG, 1200x1200)
    data2 = process_dataset(DS2_RAW, DS2_MASKS, img_ext='.JPG', resize_needed=True, is_ds1=False)
    
    all_data = data1 + data2
    print(f"Найдено полных пар (изображение + маска): Датасет 1: {len(data1)}, Датасет 2: {len(data2)}. Всего: {len(all_data)}")
    
    # Подготовка стратификации
    y_strata = [item['strata'] for item in all_data]
    
    # Исправление ошибки с единственным представителем класса (ValueError)
    from collections import Counter
    counts = Counter(y_strata)
    most_common_class = counts.most_common(1)[0][0]
    for i in range(len(y_strata)):
        if counts[y_strata[i]] < 2:
            print(f"Класс {y_strata[i]} имеет меньше 2 представителей. Меняем страту на {most_common_class} для разбиения.")
            y_strata[i] = most_common_class
            
    # Разбиение на train/val (80/20) с учетом страт
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42, stratify=y_strata)
    
    print(f"Обучающая выборка: {len(train_data)} изображений")
    print(f"Валидационная выборка: {len(val_data)} изображений")
    
    # Очистка и создание директорий выходного датасета
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
        
    for split in ['train', 'val']:
        (OUT_IMAGES / split).mkdir(parents=True, exist_ok=True)
        (OUT_MASKS / split).mkdir(parents=True, exist_ok=True)
        
    def copy_and_resize(data_subset, split_name):
        for item in tqdm(data_subset, desc=f"Обработка {split_name}"):
            img_out_path = OUT_IMAGES / split_name / f"{item['basename']}.png"
            mask_out_path = OUT_MASKS / split_name / f"{item['basename']}.png"
            
            # Чтение
            img = cv2.imread(str(item['img_path']))
            mask_bgr = cv2.imread(str(item['mask_path']), cv2.IMREAD_COLOR)
            
            # Конвертация в индексы с нужной палитрой
            color_map = BGR_COLOR_MAP_1 if item['is_ds1'] else BGR_COLOR_MAP_2
            mask = rgb_to_index(mask_bgr, color_map)
            
            # Ресайз, если нужно (метод INTER_AREA лучше для уменьшения)
            if item['resize']:
                img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                # Для масок обязательно использовать INTER_NEAREST, чтобы не смазать классы
                mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
            else:
                # Даже если ресайз не нужен, пересохраним изображение в .png (если оно было .JPG) 
                # для унификации, хотя в Dataset 1 они уже png.
                pass
                
            cv2.imwrite(str(img_out_path), img)
            cv2.imwrite(str(mask_out_path), mask)

    copy_and_resize(train_data, 'train')
    copy_and_resize(val_data, 'val')
    
    print("Готово! Объединенный датасет сохранен в:", OUT_DIR)

if __name__ == "__main__":
    main()
