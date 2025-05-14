import os
import cv2

# Пути к исходным данным
root_dir = "datasets/WiderPerson"
images_dir = os.path.join(root_dir, "Images")
annotations_dir = os.path.join(root_dir, "Annotations")

# Пути для YOLO-формата
output_dir = "datasets/widerperson_yolo"
os.makedirs(output_dir, exist_ok=True)

def convert_bbox(img_width, img_height, x1, y1, x2, y2):
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return [x_center, y_center, width, height]

# Обработка каждого набора данных
for split in ["train", "val", "test"]:
    list_file = os.path.join(root_dir, f"{split}.txt")
    
    with open(list_file, "r") as f:
        lines = f.read().splitlines()
    
    for line in lines:
        img_name = line.strip()
        img_name = img_name + '.jpg'
        img_path = os.path.join(images_dir, img_name)
        txt_path = os.path.join(annotations_dir, f"{img_name}.txt")
        
        # Пропускаем изображения без аннотаций (для тестового набора)
        if not os.path.exists(txt_path):
            print(f"[WARNING] Аннотация для {img_name} не найдена. Пропускаем.")
            continue
        
        # Чтение изображения для получения размеров
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Не удалось прочитать изображение: {img_path}")
            continue
        h, w, _ = img.shape
        
        # Чтение аннотаций
        with open(txt_path, "r") as f:
            annos = f.read().splitlines()
        
        # Первая строка - количество объектов
        try:
            num_objs = int(annos[0])
        except:
            print(f"[ERROR] Некорректный формат аннотации: {txt_path}")
            continue
        
        yolo_annos = []
        for anno in annos[1:num_objs+1]:
            parts = anno.split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[0])
                x1, y1, x2, y2 = map(int, parts[1:])
            except:
                continue
            
            # Фильтрация классов
            if class_id in [1, 2, 3]:  # Объединяем в класс 0
                bbox = convert_bbox(w, h, x1, y1, x2, y2)
                yolo_annos.append(f"0 {' '.join(map(str, bbox))}")
        
        # Копирование изображения
        output_img_path = os.path.join(output_dir, "images", split, img_name)
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        os.system(f"cp '{img_path}' '{output_img_path}'")  # Кавычки для путей с пробелами
        
        # Сохранение аннотаций (только если есть объекты)
        if yolo_annos:
            output_txt_path = os.path.join(output_dir, "labels", split, f"{os.path.splitext(img_name)[0]}.txt")
            os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
            with open(output_txt_path, "w") as f:
                f.write("\n".join(yolo_annos))
        else:
            print(f"[INFO] Нет объектов для {img_name}")