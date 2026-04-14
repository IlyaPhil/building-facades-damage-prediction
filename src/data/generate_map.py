import pandas as pd
import requests
import time
import folium
import re
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

def extract_coords_from_yandex(short_url):
    """
    Отправляет запрос по короткой ссылке и достает координаты из полного URL.
    Мы используем заголовки (headers), чтобы Яндекс не принял скрипт за бота.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        response = requests.get(short_url, headers=headers, allow_redirects=True, timeout=10)
        full_url = response.url
        
        # В Яндексе координаты передаются в параметре ll в формате долгота%2Cширота
        match = re.search(r'll=([\d\.]+)%2C([\d\.]+)', full_url)
        if match:
            lon, lat = float(match.group(1)), float(match.group(2))
            return lat, lon
        else:
            print(f"Не удалось найти координаты в URL: {full_url}")
            return None, None
    except Exception as e:
        print(f"Ошибка при обработке {short_url}: {e}")
        return None, None

def geocode_address(address, geolocator):
    """
    Получает координаты по текстовому адресу с задержкой 1.1с, чтобы не забанил сервер Nominatim.
    """
    full_address = f"Санкт-Петербург, {address}"
    try:
        # Устанавливаем разумный таймаут
        location = geolocator.geocode(full_address, timeout=10)
        time.sleep(1.1)  # Обязательная задержка (правила OpenStreetMap)
        if location:
            return location.latitude, location.longitude
        else:
            print(f"Не найден адрес: {full_address}")
            return None, None
    except GeocoderTimedOut:
        print(f"Таймаут геокодера для адреса: {full_address}")
        return None, None
    except Exception as e:
        print(f"Ошибка геокодирования {full_address}: {e}")
        return None, None

def main():
    print("1. Обработка Яндекс Панорам (Выборка для Инференса)...")
    df_yandex = pd.read_csv('data/interim/yandex_panoramas_for_inference.csv')
    
    # Добавляем пустые колонки под координаты
    df_yandex['lat'] = None
    df_yandex['lon'] = None
    df_yandex['split'] = 'inference'
    df_yandex['name'] = 'Building ' + df_yandex['building_id'].astype(str)
    
    for idx, row in df_yandex.iterrows():
        lat, lon = extract_coords_from_yandex(row['panorama_url'])
        df_yandex.at[idx, 'lat'] = lat
        df_yandex.at[idx, 'lon'] = lon
        print(f"Обработана панорама {idx+1}/{len(df_yandex)}")
        
    print("\n2. Обработка адресов с камеры (Обучающая выборка)...")
    df_camera = pd.read_csv('data/interim/Camera_train_addresses.csv')
    df_camera['lat'] = None
    df_camera['lon'] = None
    df_camera['split'] = 'train'
    
    # Инициализация геокодера с понятным названием приложения
    geolocator = Nominatim(user_agent="hse_master_thesis_facades")
    
    for idx, row in df_camera.iterrows():
        lat, lon = geocode_address(row['Адрес'], geolocator)
        df_camera.at[idx, 'lat'] = lat
        df_camera.at[idx, 'lon'] = lon
        print(f"Обработан адрес {idx+1}/{len(df_camera)}")
        
    print("\n3. Подготовка итоговых данных...")
    # Оставляем только успешные строки и нужные колонки
    df_yandex_clean = df_yandex[['name', 'lat', 'lon', 'split']].dropna()
    df_camera_clean = df_camera[['Название фото', 'lat', 'lon', 'split']].rename(columns={'Название фото': 'name'}).dropna()
    
    # Склеиваем таблицы в одну
    df_final = pd.concat([df_yandex_clean, df_camera_clean], ignore_index=True)
    print(f"Итого успешных точек: {len(df_final)} из {len(df_yandex) + len(df_camera)}")
    
    # Сохраняем объединенный датасет с координатами в csv для истории
    df_final.to_csv('data/interim/all_mapped_buildings.csv', index=False)
    
    print("\n4. Создание карты...")
    # Центрируем карту примерно на центре Санкт-Петербурга
    m = folium.Map(location=[59.9386, 30.3141], zoom_start=12, tiles='CartoDB positron')

    
    for idx, row in df_final.iterrows():
        # Задаем цвета: Обучение — синий, Инференс — красный
        color = 'blue' if row['split'] == 'train' else 'red'
        
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"{row['name']} ({row['split']})",
            icon=folium.Icon(color=color)
        ).add_to(m)
        
    # Сохраняем HTML файл
    output_map = 'buildings_map.html'
    m.save(output_map)
    print(f"Отлично! Интерактивная карта успешно сохранена в файл: {output_map}")

if __name__ == '__main__':
    main()
