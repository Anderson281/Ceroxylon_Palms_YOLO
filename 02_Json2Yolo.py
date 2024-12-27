import json
import os
import shutil
from sklearn.model_selection import train_test_split


def labelme_to_yolo(json_dir, output_dir, img_width, img_height, val_split=0.2, test_split=0.1):
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train_y8x_m')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Obtén todos los archivos JSON
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    # Divide en train_y8x_m, val y test
    train_files, val_test_files = train_test_split(json_files, test_size=(val_split + test_split))
    val_files, test_files = train_test_split(val_test_files, test_size=test_split / (val_split + test_split))

    def convert_and_save(files, target_dir):
        for json_file in files:
            with open(os.path.join(json_dir, json_file), 'r') as f:
                data = json.load(f)
                yolo_data = []
                for shape in data['shapes']:
                    label = shape['label']
                    points = shape['points']
                    x_center = (points[0][0] + points[1][0]) / 2 / img_width
                    y_center = (points[0][1] + points[1][1]) / 2 / img_height
                    width = abs(points[1][0] - points[0][0]) / img_width
                    height = abs(points[1][1] - points[0][1]) / img_height
                    yolo_data.append(f"{label} {x_center} {y_center} {width} {height}\n")

                # Guarda el archivo en formato YOLO en la carpeta destino
                output_file = os.path.join(target_dir, json_file.replace(".json", ".txt"))
                with open(output_file, 'w') as out_f:
                    out_f.writelines(yolo_data)

                # Copia la imagen correspondiente
                img_file = json_file.replace(".json", ".jpg")  # Cambia la extensión si es diferente
                shutil.copy2(os.path.join(json_dir, img_file), target_dir)

    # Convierte y guarda en las carpetas correspondientes
    convert_and_save(train_files, train_dir)
    convert_and_save(val_files, val_dir)
    convert_and_save(test_files, test_dir)


labelme_to_yolo(r'D:\12_Pros_Ocool\80_YoloVx\02_Entrenamiento\80_DataSets\DataSet_Json\10img_parcela', r'D:\12_Pros_Ocool\80_YoloVx\02_Entrenamiento\80_DataSets\DataSet_Yolo_m', img_width=640, img_height=480)
