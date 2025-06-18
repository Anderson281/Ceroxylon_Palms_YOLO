# This script performs inference with a YOLO model on a georeferenced TIFF image,
# converts detections to geospatial coordinates, and exports the results to a Shapefile.
# It is useful for remote sensing and automatic mapping of objects detected in satellite or aerial images.

import cv2
import json
import geopandas as gpd                        # Manipulation and analysis of vector spatial data, such as Shapefiles[1].
import pyproj
from shapely.geometry import box               # Creation of bounding boxes as spatial polygons[2].
from ultralytics import YOLO                   # Loading and running YOLOv8 models for object detection[3].
from rasterio import open as raster_open       # Reading georeferenced raster files (TIFF) and extracting spatial information[4].
from rasterio.transform import from_origin

# 1. Load the pre-trained YOLO model.
model = YOLO("../runs/detect_Final/train_y8m/weights/best.pt")  # Path to the trained model

# 2. Open the TIFF image to extract geospatial information (transform and coordinate reference system).
tif_path = "../70_otm/08_otm.tif"
with raster_open(tif_path) as src:
    transform = src.transform  # Transformation matrix from pixels to real-world coordinates
    crs = src.crs              # Coordinate Reference System (CRS)
    width = src.width          # Image width in pixels
    height = src.height        # Image height in pixels

    # 3. Perform object detection on the TIFF image.
    results = model(tif_path)

    # 4. Prepare a list to store detection results.
    detection_data = []

    # 5. Iterate over the detection results.
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Box coordinates (x1, y1, x2, y2) in pixels
        confidences = result.boxes.conf.cpu().numpy()  # Confidence for each detection
        class_ids = result.boxes.cls.cpu().numpy()     # Detected class ID

        # 6. Convert each box from pixel to geospatial coordinates and store the information.
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            confidence = confidences[i]
            class_id = int(class_ids[i])

            # Transform pixel coordinates to real-world coordinates using the transformation matrix.
            minx, miny = transform * (x1, y1)
            maxx, maxy = transform * (x2, y2)

            detection_data.append({
                'class_id': class_id,
                'confidence': float(confidence),
                'geometry': box(minx, miny, maxx, maxy)  # Create a bounding box polygon
            })

    # 7. Convert the list of detections to a GeoPandas GeoDataFrame, using the original TIFF CRS.
    gdf = gpd.GeoDataFrame(detection_data, geometry='geometry', crs=crs)

    # 8. Transform the GeoDataFrame CRS to UTM 18S (EPSG:32718), common in South America.
    utm18s_crs = "EPSG:32718"
    gdf = gdf.to_crs(utm18s_crs)

    # 9. Export the GeoDataFrame to a Shapefile for use in GIS.
    gdf.to_file('../60_Shapes_Entren/otm/prueba.shp')

print("Shapefile export completed.")  # Final success message
