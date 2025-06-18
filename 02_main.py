# Este script realiza inferencia con un modelo YOLO sobre una imagen georreferenciada (TIFF), 
# convierte las detecciones a coordenadas geoespaciales y exporta los resultados a un archivo Shapefile.
# Es útil para aplicaciones de teledetección y mapeo automático de objetos detectados en imágenes satelitales o aéreas.

import cv2
import json
import geopandas as gpd                        # Manipulación y análisis de datos espaciales vectoriales, como Shapefiles
import pyproj
from shapely.geometry import box               # Creación de cajas (bounding boxes) como polígonos espaciales
from ultralytics import YOLO                   # Carga y ejecución de modelos YOLOv8 para detección de objetos
from rasterio import open as raster_open       # Lectura de archivos raster (TIFF georreferenciados) y extracción de información espacial
from rasterio.transform import from_origin

# 1. Cargar el modelo YOLO previamente entrenado.
model = YOLO("../runs/detect_Final/train_y8m/weights/best.pt")  # Ruta al modelo entrenado

# 2. Abrir la imagen TIFF para extraer información geoespacial (transformación y sistema de referencia de coordenadas).
tif_path = "../70_otm/08_otm.tif"
with raster_open(tif_path) as src:
    transform = src.transform  # Matriz de transformación de píxeles a coordenadas reales
    crs = src.crs              # Sistema de referencia de coordenadas (CRS)
    width = src.width          # Ancho de la imagen en píxeles
    height = src.height        # Alto de la imagen en píxeles

    # 3. Realizar la detección de objetos sobre la imagen TIFF.
    results = model(tif_path)

    # 4. Preparar una lista para almacenar los resultados de las detecciones.
    detection_data = []

    # 5. Iterar sobre los resultados de detección.
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Coordenadas de las cajas (x1, y1, x2, y2) en píxeles
        confidences = result.boxes.conf.cpu().numpy()  # Confianza de cada detección
        class_ids = result.boxes.cls.cpu().numpy()     # ID de la clase detectada

        # 6. Convertir cada caja de píxeles a coordenadas geoespaciales y almacenar la información.
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            confidence = confidences[i]
            class_id = int(class_ids[i])

            # Transformar coordenadas de píxeles a coordenadas reales usando la matriz de transformación.
            minx, miny = transform * (x1, y1)
            maxx, maxy = transform * (x2, y2)

            detection_data.append({
                'class_id': class_id,
                'confidence': float(confidence),
                'geometry': box(minx, miny, maxx, maxy)  # Crear un polígono tipo caja (bounding box)
            })

    # 7. Convertir la lista de detecciones a un GeoDataFrame de GeoPandas, usando el CRS original del TIFF.
    gdf = gpd.GeoDataFrame(detection_data, geometry='geometry', crs=crs)

    # 8. Transformar el CRS del GeoDataFrame a UTM 18S (EPSG:32718), común en Sudamérica.
    utm18s_crs = "EPSG:32718"
    gdf = gdf.to_crs(utm18s_crs)

    # 9. Exportar el GeoDataFrame a un archivo Shapefile para su uso en SIG.
    gdf.to_file('../60_Shapes_Entren/otm/prueba.shp')

print("Exportación a Shapefile completada.")  # Mensaje final de éxito
