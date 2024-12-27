import os
from pathlib import Path
import cv2
import albumentations as A

# Directorios de entrada y salida
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

# Crea subcarpetas de salida si no existen
for subset in ['train', 'val', 'test']:
    os.makedirs(output_dir / subset / 'images', exist_ok=True)
    os.makedirs(output_dir / subset / 'labels', exist_ok=True)

# Configuración de aumentos de datos
def get_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, border_mode=0, p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Sharpen(alpha=(0.1, 0.3), p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
        A.MotionBlur(blur_limit=5, p=0.2)
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.1))


# Función para cargar etiquetas YOLO
def load_yolo_labels(label_path):
    with open(label_path, "r") as f:
        labels = [line.strip().split() for line in f.readlines()]
        labels = [[int(label[0])] + list(map(float, label[1:])) for label in labels]
    return labels

# Función para guardar etiquetas YOLO
def save_yolo_labels(labels, output_path):
    with open(output_path, "w") as f:
        for label in labels:
            f.write(" ".join(map(str, label)) + "\n")

# Función para aplicar aumentos de datos
def augment_data(image_path, label_path, output_img_dir, output_label_dir, img_suffix="aug"):
    # Carga la imagen y las etiquetas
    image = cv2.imread(str(image_path))
    labels = load_yolo_labels(label_path)
    class_labels = [label[0] for label in labels]
    bboxes = [label[1:] for label in labels]

    # Aplica aumentos de datos
    aug = get_augmentation()
    augmented = aug(image=image, bboxes=bboxes, class_labels=class_labels)
    aug_image = augmented['image']
    aug_bboxes = augmented['bboxes']
    aug_class_labels = augmented['class_labels']

    # Genera nombre de archivo de salida
    base_name = image_path.stem
    output_img_path = output_img_dir / f"{base_name}_{img_suffix}.jpg"
    output_label_path = output_label_dir / f"{base_name}_{img_suffix}.txt"

    # Guarda la imagen aumentada
    cv2.imwrite(str(output_img_path), aug_image)

    # Guarda las etiquetas aumentadas
    aug_labels = [[cls] + list(bbox) for cls, bbox in zip(aug_class_labels, aug_bboxes)]
    save_yolo_labels(aug_labels, output_label_path)

# Itera sobre cada conjunto (train, val, test)
for subset, img_dir in input_dirs.items():
    label_dir = annotations_dirs[subset]
    output_img_dir = output_dir / subset / 'images'
    output_label_dir = output_dir / subset / 'labels'

    # Itera sobre las imágenes en el directorio
    for img_file in img_dir.glob("*.jpg"):
        label_file = label_dir / img_file.with_suffix(".txt").name

        # Verifica que exista el archivo de etiqueta correspondiente
        if label_file.exists():
            # Genera múltiples versiones aumentadas si es necesario
            for i in range(3):  # Cambia el rango para ajustar la cantidad de aumentos
                augment_data(img_file, label_file, output_img_dir, output_label_dir, img_suffix=f"aug{i+1}")

print("Aumento de datos completado.")
