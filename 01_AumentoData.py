# Este script realiza aumento de datos en imágenes y etiquetas en formato YOLO usando Albumentations.
# El objetivo es incrementar la diversidad del dataset para robustecer el entrenamiento de modelos de visión por computadora.

import os                          # Interacción con el sistema operativo, creación de carpetas y manipulación de archivos
from pathlib import Path           # Manejo de rutas de archivos de forma clara y multiplataforma
import cv2                         # Lectura y guardado de imágenes en disco
import albumentations as A         # Aumentos de datos avanzados en imágenes y bounding boxes, compatible con YOLO

# 1. Definición de rutas de entrada y salida para imágenes y etiquetas de train, val y test.
input_dirs = {
    'train': Path(r'D:\12_Pros_Ocool\80_YoloVx\02_Entrenamiento\80_DataSets\DataSet_Yolo_m\D_15img\train\images'),
    'val': Path(r'D:\12_Pros_Ocool\80_YoloVx\02_Entrenamiento\80_DataSets\DataSet_Yolo_m\D_15img\val\images'),
    'test': Path(r'D:\12_Pros_Ocool\80_YoloVx\02_Entrenamiento\80_DataSets\DataSet_Yolo_m\D_15img\test\images')
}
annotations_dirs = {
    'train': Path(r'D:\12_Pros_Ocool\80_YoloVx\02_Entrenamiento\80_DataSets\DataSet_Yolo_m\D_15img\train\labels'),
    'val': Path(r'D:\12_Pros_Ocool\80_YoloVx\02_Entrenamiento\80_DataSets\DataSet_Yolo_m\D_15img\val\labels'),
    'test': Path(r'D:\12_Pros_Ocool\80_YoloVx\02_Entrenamiento\80_DataSets\DataSet_Yolo_m\D_15img\test\labels')
}
output_dir = Path(r'D:\12_Pros_Ocool\80_YoloVx\02_Entrenamiento\80_DataSets\DataSet_Yolo_m\D_15img_A4')

# 2. Crea subcarpetas de salida para imágenes y etiquetas aumentadas si no existen.
for subset in ['train', 'val', 'test']:
    os.makedirs(output_dir / subset / 'images', exist_ok=True)
    os.makedirs(output_dir / subset / 'labels', exist_ok=True)

# 3. Define la función de aumentos de datos usando transformaciones aleatorias de Albumentations.
def get_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),  # Volteo horizontal aleatorio
        A.Rotate(limit=20, border_mode=0, p=0.5),  # Rotación aleatoria
        A.RandomScale(scale_limit=0.2, p=0.5),  # Escalado aleatorio
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Brillo/contraste aleatorio
        A.Sharpen(alpha=(0.1, 0.3), p=0.5),  # Nitidez aleatoria
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),  # Cambios de color
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),  # Niebla aleatoria
        A.MotionBlur(blur_limit=5, p=0.2)  # Desenfoque de movimiento
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.1))

# 4. Función para cargar etiquetas en formato YOLO desde archivo.
def load_yolo_labels(label_path):
    with open(label_path, "r") as f:
        labels = [line.strip().split() for line in f.readlines()]
        labels = [[int(label[0])] + list(map(float, label[1:])) for label in labels]
    return labels

# 5. Función para guardar etiquetas en formato YOLO en archivo.
def save_yolo_labels(labels, output_path):
    with open(output_path, "w") as f:
        for label in labels:
            f.write(" ".join(map(str, label)) + "\n")

# 6. Aplica los aumentos a cada imagen y sus etiquetas, y guarda los resultados.
def augment_data(image_path, label_path, output_img_dir, output_label_dir, img_suffix="aug"):
    image = cv2.imread(str(image_path))  # Carga la imagen original
    labels = load_yolo_labels(label_path)  # Carga las etiquetas YOLO
    class_labels = [label[0] for label in labels]  # Extrae clases
    bboxes = [label[1:] for label in labels]  # Extrae bboxes

    aug = get_augmentation()  # Obtiene aumentos configurados
    augmented = aug(image=image, bboxes=bboxes, class_labels=class_labels)  # Aplica aumentos
    aug_image = augmented['image']
    aug_bboxes = augmented['bboxes']
    aug_class_labels = augmented['class_labels']

    base_name = image_path.stem  # Nombre base del archivo
    output_img_path = output_img_dir / f"{base_name}_{img_suffix}.jpg"
    output_label_path = output_label_dir / f"{base_name}_{img_suffix}.txt"

    cv2.imwrite(str(output_img_path), aug_image)  # Guarda imagen aumentada
    aug_labels = [[cls] + list(bbox) for cls, bbox in zip(aug_class_labels, aug_bboxes)]  # Une clase y bbox
    save_yolo_labels(aug_labels, output_label_path)  # Guarda etiquetas aumentadas

# 7. Itera sobre cada conjunto (train, val, test), y para cada imagen genera varias versiones aumentadas.
for subset, img_dir in input_dirs.items():
    label_dir = annotations_dirs[subset]
    output_img_dir = output_dir / subset / 'images'
    output_label_dir = output_dir / subset / 'labels'

    for img_file in img_dir.glob("*.jpg"):
        label_file = label_dir / img_file.with_suffix(".txt").name
        if label_file.exists():
            for i in range(3):  # Número de aumentos por imagen
                augment_data(img_file, label_file, output_img_dir, output_label_dir, img_suffix=f"aug{i+1}")

print("Aumento de datos completado.")  # Mensaje final al completar el proceso
