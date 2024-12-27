import json
import geopandas as gpd
import rasterio
from shapely.geometry import Point, box
from math import sqrt

# Rutas de archivos
json_path = r"D:\12_Pros_Ocool\80_YoloVx\02_Entrenamiento\70_otm\val_otm\json\37_otm.json"
tif_path = r"D:\12_Pros_Ocool\80_YoloVx\02_Entrenamiento\70_otm\val_otm\json\37_otm.tif"  # Imagen TIFF de referencia
shapefile_path = r"D:\12_Pros_Ocool\80_YoloVx\02_Entrenamiento\50_Shape_val\37_otm_val.shp"

# Función para calcular el radio de un círculo a partir de dos puntos
def calcular_radio(punto1, punto2):
    return sqrt((punto2[0] - punto1[0]) ** 2 + (punto2[1] - punto1[1]) ** 2) / 2

# Leer el sistema de coordenadas y transformación de la imagen TIFF
with rasterio.open(tif_path) as src:
    crs = src.crs  # Sistema de referencia de la imagen
    transform = src.transform  # Transformación de píxeles a coordenadas

# Función para crear un cuadro en coordenadas geoespaciales con escalado al 150%
def crear_cuadro_escalado(centro, radio, escala=1.5):
    # Ajustar el radio según el factor de escala
    radio_escalado = radio * escala
    minx, miny = centro[0] - radio_escalado, centro[1] - radio_escalado
    maxx, maxy = centro[0] + radio_escalado, centro[1] + radio_escalado
    cuadro = box(minx, miny, maxx, maxy)
    return cuadro

# Cargar el archivo JSON y procesar las etiquetas circulares
try:
    with open(json_path, 'r') as f:
        datos = json.load(f)

    if 'shapes' in datos:
        shapes = datos['shapes']
    else:
        raise KeyError("El JSON no contiene la estructura 'shapes' esperada.")

    cuadros = []
    for shape in shapes:
        if 'points' in shape and len(shape['points']) >= 2:
            # Convertir los puntos originales a coordenadas de la imagen TIFF
            centro_pix = shape['points'][0]
            radio_pix = calcular_radio(shape['points'][0], shape['points'][1])

            # Convertir el centro a coordenadas geoespaciales
            centro_geo = transform * centro_pix
            cuadro_geom = crear_cuadro_escalado(centro_geo, radio_pix, escala=0.08)
            cuadros.append({'geometry': cuadro_geom, 'label': shape['label']})
        else:
            print(f"Etiqueta omitida: No se encontraron puntos suficientes en {shape}")

    # Crear el GeoDataFrame y asignarle el CRS de la imagen TIFF
    gdf = gpd.GeoDataFrame(cuadros, crs=crs)

    # Guardar el GeoDataFrame en un archivo Shapefile
    gdf.to_file(shapefile_path, driver='ESRI Shapefile')
    print(f"Shapefile georreferenciado con escala 150% guardado en: {shapefile_path}")

except FileNotFoundError:
    print(f"El archivo JSON no se encontró en la ruta especificada: {json_path}")
except json.JSONDecodeError:
    print("Error al decodificar el archivo JSON. Asegúrate de que el archivo tenga el formato JSON correcto.")
except KeyError as e:
    print(f"Error de clave en JSON: {e}")
