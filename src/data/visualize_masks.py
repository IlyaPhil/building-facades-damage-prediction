import os
import cv2
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

# Пути к готовому объединенному датасету
BASE_DIR = "/home/neuralist/HSE_AI_master/master-thesis/building-facades-damage-prediction"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "data/processed/unified_dataset/images/train")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "data/processed/unified_dataset/masks/train")

color_map = {
    0: (0, 0, 0),         # background
    1: (217, 21, 54),     # coating_deterioration (RGB)
    2: (250, 50, 183),    # cracks (RGB)
    3: (209, 125, 42),    # masonry_degradation (RGB)
    4: (98, 221, 38),     # moisture_bio_damage (RGB)
    5: (222, 28, 123)     # vandalism (RGB)
}

def label_to_color_image(label_mask):
    color_img = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        color_img[label_mask == class_id] = color
    return color_img

def main():
    images = glob.glob(os.path.join(TRAIN_IMG_DIR, "*.png"))
    # Выбираем случайные 10 изображений
    sample_images = random.sample(images, min(10, len(images)))

    # Собираем фигуру 10x2 (слева фото, справа маска)
    fig, axes = plt.subplots(10, 2, figsize=(10, 40))
    axes[0, 0].set_title("Оригинал")
    axes[0, 1].set_title("Индексная маска (GT)")

    for i, img_path in enumerate(sample_images):
        basename = os.path.basename(img_path)
        mask_path = os.path.join(TRAIN_MASK_DIR, basename)

        orig_img = cv2.imread(img_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_color = label_to_color_image(gt_mask)

        # Выводим уникальные классы на этой маске в консоль для дебага
        unique_classes = np.unique(gt_mask)
        print(f"{basename} содержит классы: {unique_classes}")

        axes[i, 0].imshow(orig_img)
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(gt_color)
        axes[i, 1].axis("off")

    plt.tight_layout()
    output_path = os.path.join(BASE_DIR, "docs/mask_verification.png")
    plt.savefig(output_path, dpi=150)
    print(f"\nВизуализация сохранена в {output_path}. Открой её, чтобы убедиться!")

if __name__ == "__main__":
    main()
