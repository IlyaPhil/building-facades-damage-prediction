import os
import cv2
import glob
from tqdm import tqdm
from pathlib import Path

# Папка с широкими планами (исходниками для инференса)
INPUT_DIR = Path("data/processed/inference_test")
if not INPUT_DIR.exists():
    print(f"Внимание: папка {INPUT_DIR} не найдена. Создаю пустую.")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

# Папка куда сохраняем нарезанные тайлы
OUTPUT_DIR = Path("data/processed/tiles_800x800")

# Параметры инференса
TILE_SIZE = 800
STRIDE = 700

# Константы масштабирования в зависимости от источника:
DSLR_SCALE = 800 / 1200     # Для фото 5456х3632
YANDEX_SCALE = 1.41          # Для фото 1920х1280 (не меняем)
PHONE_SCALE = 1.0           # TODO: Вставь сюда нужный коэффициент, когда откалибруешь телефон (4096х3072)

def get_scale_factor(w, h):
    # Я.Панорамы имеют ширину около 1920
    if w <= 2000:
        return YANDEX_SCALE
    # Зеркалка имеет ширину более 5000
    elif w >= 5000:
        return DSLR_SCALE
    # Остальное считаем телефоном (около 4096)
    else:
        return PHONE_SCALE

def create_tiles():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    for ext in ["*.jpg", "*.JPG", "*.png", "*.PNG"]:
        image_paths.extend(list(INPUT_DIR.glob(ext)))
        
    print(f"Найдено изображений для нарезки: {len(image_paths)}")
    if len(image_paths) == 0:
        return
        
    total_tiles_created = 0
    
    for img_path in tqdm(image_paths, desc="Нарезка тайлов"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Ошибка чтения: {img_path}")
            continue
            
        h_orig, w_orig, _ = img.shape
        filename = img_path.stem
        
        # --- Блок нормализации масштаба ---
        scale_factor = get_scale_factor(w_orig, h_orig)
        
        if scale_factor != 1.0:
            new_w = int(w_orig * scale_factor)
            new_h = int(h_orig * scale_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        h, w, _ = img.shape
        
        y_starts = list(range(0, h, STRIDE))
        x_starts = list(range(0, w, STRIDE))
        
        for y in y_starts:
            for x in x_starts:
                tile = img[y:y+TILE_SIZE, x:x+TILE_SIZE]
                
                # Дополняем нулями (черный цвет) до нужного размера, если это остаток края
                if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                    tile = cv2.copyMakeBorder(
                        tile, 
                        0, max(0, TILE_SIZE - tile.shape[0]), 
                        0, max(0, TILE_SIZE - tile.shape[1]), 
                        cv2.BORDER_CONSTANT, 
                        value=[0, 0, 0]
                    )
                
                tile_filename = f"{filename}_{y}_{x}.jpg"
                tile_path = OUTPUT_DIR / tile_filename
                
                cv2.imwrite(str(tile_path), tile)
                total_tiles_created += 1

    print(f"\nГотово! Создано всего тайлов: {total_tiles_created} (с шагом {STRIDE})")
    print(f"Тайлы сохранены в: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    create_tiles()
