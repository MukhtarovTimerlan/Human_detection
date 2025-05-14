"""
Скрипт для запуска приложения детекции людей
"""
import os
import sys
import subprocess
import platform
import numpy as np

def check_dependencies():
    """Проверка установленных зависимостей"""
    try:
        import streamlit
        import ultralytics
        import cv2
        import numpy
        print("Все необходимые зависимости установлены")
        return True
    except ImportError as e:
        print(f"Отсутствуют необходимые зависимости: {e}")
        print("Установите зависимости с помощью команды: pip install -r requirements.txt")
        return False

def check_model():
    """Проверка наличия модели"""
    model_path = "runs/detect/custom_yolov8n2/weights/best.onnx"
    
    if not os.path.exists(model_path):
        print(f"Внимание: Модель не найдена по пути: {model_path}")
        print("Убедитесь, что файл модели существует или используйте обученную модель")
        return False
    
    print(f"Модель найдена: {model_path}")
    return True

def run_app():
    print("🚀 Запуск приложения...")
    result = subprocess.run([
        "streamlit", 
        "run", 
        "app.py",
        "--server.fileWatcherType=none",
        "--server.port=8502",
        "--logger.level=error"
    ])
    return result.returncode == 0

def main():
    print("=" * 50)
    print("Приложение для детекции людей на изображениях и видео")
    print("=" * 50)
    
    # Проверка зависимостей
    if not check_dependencies():
        return
    
    # Проверка наличия модели
    check_model()
    
    # Запуск приложения
    run_app()

if __name__ == "__main__":
    main() 