import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER"] = "none"

import cv2
import streamlit as st
import tempfile
import numpy as np
import time
from threading import Lock
from ultralytics import YOLO

# Инициализация блокировки для потокобезопасности
camera_lock = Lock()

# Должен быть ПЕРВЫМ Streamlit-вызовом
st.set_page_config(page_title="People Detector", layout="wide")

# Конфигурация
MODEL_PATH = "runs/detect/custom_yolov8n2/weights/best.onnx"
CLASS_NAMES = ["person"]
CONF_THRESH = 0.5
CAMERA_ID = 0

@st.cache_resource
def load_model():
    """Загрузка ONNX-модели с кэшированием"""
    return YOLO(MODEL_PATH, task='detect')

def process_frame(_model, frame):
    """Обработка одного кадра с замерами времени"""
    start_time = time.perf_counter()
    
    # Инференс модели
    results = _model.predict(frame, conf=CONF_THRESH, verbose=False)
    inference_time = time.perf_counter() - start_time
    
    # Отрисовка результатов
    res_img = results[0].plot()
    postprocess_time = time.perf_counter() - start_time - inference_time
    
    # Добавление метрик времени
    timing_text = f"Inference: {inference_time*1000:.1f}ms | FPS: {1/(inference_time+1e-6):.1f}"
    cv2.putText(res_img, timing_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    return res_img, len(results[0].boxes)

def process_video(_model, video_path):
    """Обработка видео с прогресс-баром"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Создаем временный файл с правильным расширением
    output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, mode='wb')
    output.close()  # Закрываем файл для использования VideoWriter
    
    # Настройка VideoWriter с правильными параметрами
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        output.name,
        fourcc,
        fps/4,  # Частота кадров с учетом пропуска
        (frame_width, frame_height)
    )
    
    try:
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
            
            if frame_count % 4 == 0:
                processed, _ = process_frame(_model, frame)
                writer.write(processed)
                
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            frame_count += 1
            
    except Exception as e:
        st.error(f"Ошибка обработки видео: {str(e)}")
    finally:
        cap.release()
        writer.release()
        progress_bar.empty()
    
    # Перечитываем файл в бинарном режиме
    with open(output.name, 'rb') as f:
        video_bytes = f.read()
    
    os.unlink(output.name)  # Удаляем временный файл
    return video_bytes


def camera_processing():
    """Основной цикл обработки видео с камеры"""
    model = load_model()
    cap = None
    
    try:
        with camera_lock:
            cap = cv2.VideoCapture(CAMERA_ID)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            st_frame = st.empty()
            
            # Инициализация состояния кнопки
            if 'camera_running' not in st.session_state:
                st.session_state.camera_running = True
            
            # Создаем кнопку ОДИН РАЗ с уникальным ключом
            stop_button = st.button(
                "Остановить камеру",
                key="unique_stop_button_key"
            )
            
            while cap.isOpened() and st.session_state.camera_running:
                success, frame = cap.read()
                if not success:
                    st.error("Ошибка чтения камеры!")
                    break
                
                # Обработка кадра
                processed, count = process_frame(model, frame)
                
                # Отображение кадра
                st_frame.image(processed[:, :, ::-1], 
                             caption=f"Людей в кадре: {count}", 
                             use_container_width=True)
                
                # Обновляем состояние через session_state
                if stop_button:
                    st.session_state.camera_running = False
                    break
                
    finally:
        if cap is not None:
            cap.release()
            st.success("Камера успешно отключена")

# Инициализация модели
model = load_model()

# Интерфейс
st.title("Детекция людей в реальном времени 📷")
st.sidebar.header("Настройки")

# Выбор режима работы
app_mode = st.sidebar.selectbox("Режим работы:", 
                               ["Фото/Видео", "Веб-камера"])

if app_mode == "Веб-камера":
    if st.sidebar.button("Запустить камеру"):
        camera_processing()
else:
    # Оригинальный код для обработки фото/видео
    upload_type = st.sidebar.radio("Тип файла:", ["Изображение", "Видео"])
    uploaded_file = st.sidebar.file_uploader("Выберите файл...", 
                                          type=["jpg", "png", "jpeg", "mp4"])
    
    if uploaded_file:
        if upload_type == "Изображение":
            # Обработка изображения
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            start_total = time.perf_counter()
            processed, count = process_frame(model, img)
            total_time = time.perf_counter() - start_total
            
            # Вывод результатов
            st.image(processed[:, :, ::-1], 
                    use_container_width=True,
                    caption=f"Найдено людей: {count} | Время: {total_time*1000:.1f}ms")
            
        else:
            # Обработка видео
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            with st.spinner("Обработка видео..."):
                video_bytes = process_video(model, tfile.name)
                st.video(video_bytes)

# Боковая панель с информацией
st.sidebar.markdown("---")
st.sidebar.markdown("## Информация о системе")
st.sidebar.info(
    f"Модель: YOLOv8n (ONNX)\n"
    f"Порог уверенности: {CONF_THRESH}\n"
    f"Разрешение кадра: 640x480"
)
