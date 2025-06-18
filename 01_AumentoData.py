# This script performs data augmentation on images and YOLO-format labels using Albumentations.
# The goal is to increase dataset diversity and improve the robustness of computer vision model training[1][2].

import os                          # Interacts with the operating system, creates folders, and manages files[3].
from pathlib import Path           # Handles file paths in a clear, cross-platform way[4].
import cv2                         # Reads and saves images to disk[5].
import albumentations as A         # Advanced image and bounding box augmentations, compatible with YOLO[1].

# 1. Define input and output directories for train, val, and test images and labels.
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

# 2. Create output subfolders for augmented images and labels if they do not exist.
for subset in ['train', 'val', 'test']:
    os.makedirs(output_dir / subset / 'images', exist_ok=True)
    os.makedirs(output_dir / subset / 'labels', exist_ok=True)

# 3. Define the augmentation function using random Albumentations transforms.
def get_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
        A.Rotate(limit=20, border_mode=0, p=0.5),  # Random rotation
        A.RandomScale(scale_limit=0.2, p=0.5),  # Random scaling
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Random brightness/contrast
        A.Sharpen(alpha=(0.1, 0.3), p=0.5),  # Random sharpening
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),  # Color changes
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),  # Random fog
        A.MotionBlur(blur_limit=5, p=0.2)  # Random motion blur
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.1))

# 4. Function to load YOLO-format labels from file.
def load_yolo_labels(label_path):
    with open(label_path, "r") as f:
        labels = [line.strip().split() for line in f.readlines()]
        labels = [[int(label[0])] + list(map(float, label[1:])) for label in labels]
    return labels

# 5. Function to save YOLO-format labels to file.
def save_yolo_labels(labels, output_path):
    with open(output_path, "w") as f:
        for label in labels:
            f.write(" ".join(map(str, label)) + "\n")

# 6. Apply augmentations to each image and its labels, then save the results.
def augment_data(image_path, label_path, output_img_dir, output_label_dir, img_suffix="aug"):
    image = cv2.imread(str(image_path))  # Load the original image
    labels = load_yolo_labels(label_path)  # Load YOLO labels
    class_labels = [label[0] for label in labels]  # Extract class IDs
    bboxes = [label[1:] for label in labels]  # Extract bounding boxes

    aug = get_augmentation()  # Get configured augmentations
    augmented = aug(image=image, bboxes=bboxes, class_labels=class_labels)  # Apply augmentations
    aug_image = augmented['image']
    aug_bboxes = augmented['bboxes']
    aug_class_labels = augmented['class_labels']

    base_name = image_path.stem  # Base file name
    output_img_path = output_img_dir / f"{base_name}_{img_suffix}.jpg"
    output_label_path = output_label_dir / f"{base_name}_{img_suffix}.txt"

    cv2.imwrite(str(output_img_path), aug_image)  # Save augmented image
    aug_labels = [[cls] + list(bbox) for cls, bbox in zip(aug_class_labels, aug_bboxes)]  # Combine class and bbox
    save_yolo_labels(aug_labels, output_label_path)  # Save augmented labels

# 7. Iterate over each dataset split (train, val, test), generating several augmented versions per image.
for subset, img_dir in input_dirs.items():
    label_dir = annotations_dirs[subset]
    output_img_dir = output_dir / subset / 'images'
    output_label_dir = output_dir / subset / 'labels'

    for img_file in img_dir.glob("*.jpg"):
        label_file = label_dir / img_file.with_suffix(".txt").name
        if label_file.exists():
            for i in range(3):  # Number of augmentations per image
                augment_data(img_file, label_file, output_img_dir, output_label_dir, img_suffix=f"aug{i+1}")

print("Data augmentation completed.")  # Final message when the process is done
