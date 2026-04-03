from playwright.sync_api import sync_playwright
import geopandas as gpd
from pathlib import Path

def screenshot_with_playwright(geojson_path, width=1920, height=1080):
    """
    Downloads screenshots from Yandex Panorama URLs found in the GeoJSON file.
    Expected GeoJSON properties:
    - id: Unique identifier (e.g. SPB_001)
    - panorama_url: URL to the panorama
    - screenshot_saved: Boolean flag
    """
    # Check if file exists
    if not Path(geojson_path).exists():
        print(f"File not found: {geojson_path}")
        return

    buildings = gpd.read_file(geojson_path)
    
    # Ensure screenshot_saved column exists
    if 'screenshot_saved' not in buildings.columns:
        buildings['screenshot_saved'] = False

    output_dir = Path('screenshots')
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loaded {len(buildings)} buildings from {geojson_path}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        # Set viewport slightly larger to ensure controls are out of the crop area if needed
        # But for cropping we will use the clip argument
        page = browser.new_page(viewport={'width': width, 'height': height + 300}) 
        
        for idx, row in buildings.iterrows():
            if row.get('screenshot_saved'):
                continue
            
            building_id = row.get('id')
            panorama_url = row.get('panorama_url')
            
            if not panorama_url or not building_id:
                print(f"Skipping row {idx}: Missing URL or ID")
                continue
            
            try:
                page.goto(panorama_url)
                # Wait for the panorama to load. 
                # Ideally we would wait for a specific element key to the panorama player
                page.wait_for_timeout(5000)
                
                # Скрыть элементы интерфейса Яндекс.Панорам
                page.evaluate("""
                    const mapElement = document.querySelector('.panorama-mini-map');
                    if (mapElement) mapElement.style.display = 'none';
                    
                    const controls = document.querySelectorAll('.panorama-controls, .panorama-footer, .panorama-overlay');
                    controls.forEach(el => el.style.display = 'none');
                """)
                
                page.wait_for_timeout(500)  # Подождать применения стилей
                
                screenshot_path = output_dir / f'{building_id}.jpg'
                
                # Скриншот с обрезкой до целевого разрешения
                page.screenshot(path=str(screenshot_path), clip={'x': 0, 'y': 0, 'width': width, 'height': height})
                
                buildings.loc[idx, 'screenshot_saved'] = True
                print(f'✓ Saved {building_id}')
                
            except Exception as e:
                print(f'✗ Error processing {building_id}: {e}')
        
        browser.close()
    
    buildings.to_file(geojson_path, driver='GeoJSON')
    print("Updated GeoJSON saved.")

# Установка Playwright
# pip install playwright
# playwright install chromium


# Запуск
if __name__ == '__main__':
    # Update this path to your actual GeoJSON file
    geo_file = 'master-thesis/damage-analysis/data/geodata/spb_buildings_2026-02-21_part2.geojson' 
    
    print(f"Starting downloader for {geo_file}...")
    screenshot_with_playwright(
        geojson_path=geo_file,
        width=1920,
        height=1080
    )

# python master-thesis/damage-analysis/src/utils/download_screenshots.py

# Убрать мини-карту при создании скриншота вручную
# Ввести в консоль в браузере
# document.querySelector('.panorama-minimap-view').style.display = 'none';