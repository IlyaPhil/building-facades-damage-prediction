import pandas as pd
import glob
import os

def check_consistency(filepath):
    print(f"Checking {filepath}...")
    df = pd.read_csv(filepath)
    pairs = {}
    contradictions = []
    
    for idx, row in df.iterrows():
        img_a = str(row['image_a'])
        img_b = str(row['image_b'])
        winner = int(row['winner'])
        
        # Normalize pair
        if img_a < img_b:
            key = (img_a, img_b)
            val = winner
        else:
            key = (img_b, img_a)
            # if winner is 0 (a wins), then b is second, so new winner is 1 (b wins)
            # if winner is 1 (b wins), then a is second, so new winner is 0 (a wins)
            if winner == 0:
                val = 1
            elif winner == 1:
                val = 0
            else:
                val = 2
                
        if key in pairs:
            if pairs[key] != val:
                contradictions.append({
                    'pair': key,
                    'old_val': pairs[key],
                    'new_val': val,
                    'row_idx': idx
                })
        else:
            pairs[key] = val
            
    if contradictions:
        print(f"Found {len(contradictions)} contradictions in {filepath}:")
        for c in contradictions:
            print(c)
    else:
        print("No contradictions found.")
    return pairs


if __name__ == "__main__":
    # Указываем путь к новой объединенной разметке
    dataset_dir = "/home/neuralist/HSE_AI_master/master-thesis/building-facades-damage-prediction/data/processed/ranking_dataset"
    csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
    
    if not csv_files:
        print(f"CSV файлы не найдены в директории: {dataset_dir}")
        
    # Проверяем внутреннюю непротиворечивость каждого файла
    for f in csv_files:
        print("-" * 40)
        check_consistency(f)

