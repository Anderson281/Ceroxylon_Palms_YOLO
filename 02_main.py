import cv2
import json
import geopandas as gpd
import pyproj
from shapely.geometry import box
from ultralytics import YOLO
from rasterio import open as raster_open
from rasterio.transform import from_origin

# Cargar el modelo
model = YOLO("../runs/detect_Final/train_y11x/weights/best.pt")                  #aaaaaaaaaaaaaaaaa

# Leer la imagen TIFF para obtener la información geoespacial
tif_path = "../70_otm/37_otma.tif"                                                #aaaaaaaaaaaaaaaaa
with raster_open(tif_path) as src:
    # Obtener la transformación geoespacial y CRS de la imagen TIFF
    transform = src.transform
    crs = src.crs

    # Obtener las dimensiones de la imagen
    width = src.width
    height = src.height

    # Realizar la detección
    results = model(tif_path)

    # Crear una lista para almacenar los datos de detección
    detection_data = []

    # Iterar sobre los resultados
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Coordenadas de las cajas (x1, y1, x2, y2)
        confidences = result.boxes.conf.cpu().numpy()  # Confianza para cada detección
        class_ids = result.boxes.cls.cpu().numpy()  # ID de la clase para cada detección

        # Almacenar datos de detección
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            confidence = confidences[i]
            class_id = int(class_ids[i])

            # Convertir coordenadas de píxeles a coordenadas geoespaciales
            minx, miny = transform * (x1, y1)
            maxx, maxy = transform * (x2, y2)

            # Añadir los datos de detección a la lista
            detection_data.append({
                'class_id': class_id,
                'confidence': float(confidence),
                'geometry': box(minx, miny, maxx, maxy)
            })

    # Convertir los datos de detección a un GeoDataFrame
    gdf = gpd.GeoDataFrame(detection_data,
                           geometry='geometry',
                           crs=crs)  # Utiliza el CRS original del TIFF

    # Transformar el CRS del GeoDataFrame a UTM 18S
    utm18s_crs = "EPSG:32718"  # EPSG para UTM 18S
    gdf = gdf.to_crs(utm18s_crs)

    # Guardar el GeoDataFrame en un archivo Shapefile
    gdf.to_file('../60_Shapes_Entren/37_otm/37_otm_train_y11x.shp')                 #aaaaaaaaaaaaaaaaa

print("Exportación a Shapefile completada.")