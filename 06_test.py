import geopandas as gpd
import pyogrio
from shapely.geometry import Polygon
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def calculate_iou(pred_poly, gt_poly):
    intersection = pred_poly.intersection(gt_poly).area
    union = pred_poly.union(gt_poly).area
    return intersection / union if union != 0 else 0


def calculate_metrics(pred_shp, gt_shp, iou_threshold=0.5, total_possible_detections=1000):
    gt_gdf = gpd.GeoDataFrame(pyogrio.read_dataframe(gt_shp))
    pred_gdf = gpd.GeoDataFrame(pyogrio.read_dataframe(pred_shp))

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    total_gt_palmeras = len(gt_gdf)
    pred_gdf["matched"] = False
    gt_gdf["matched"] = False

    for i, gt_poly in enumerate(gt_gdf.geometry):
        for j, pred_poly in enumerate(pred_gdf.geometry):
            iou = calculate_iou(pred_poly, gt_poly)
            if iou >= iou_threshold:
                true_positives += 1
                pred_gdf.at[j, "matched"] = True
                gt_gdf.at[i, "matched"] = True
                break

    false_positives = len(pred_gdf[~pred_gdf["matched"]])
    false_negatives = len(gt_gdf[~gt_gdf["matched"]])

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall


def create_pr_curve(pred_shp, gt_shp, output_dir):
    # Calcular puntos para la curva PR
    thresholds = np.linspace(0.1, 0.9, 20)
    precisions = []
    recalls = []

    print("Calculando puntos de la curva PR...")
    for threshold in thresholds:
        try:
            p, r = calculate_metrics(pred_shp, gt_shp, iou_threshold=threshold)
            precisions.append(p)
            recalls.append(r)
            print(f"Threshold: {threshold:.2f}, Precision: {p:.3f}, Recall: {r:.3f}")
        except Exception as e:
            print(f"Error en threshold {threshold}: {str(e)}")
            continue

    # Configurar el estilo de la gráfica
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
    })

    # Plotear la curva
    print("Generando gráfica...")
    try:
        plt.plot(recalls, precisions, '-', color='blue', linewidth=2,
                 label='Ceroxylon')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.02)
        plt.legend()

        # Guardar la gráfica
        output_path = output_dir / 'PR_curve.png'
        plt.savefig(output_path, dpi=900, bbox_inches='tight')
        plt.close()
        print(f"Gráfica guardada en: {output_path}")
    except Exception as e:
        print(f"Error al generar la gráfica: {str(e)}")


def main():
    # Rutas
    pred_shp_path = r'D:\12_Pros_Ocool\80_YoloVx\02_Entrenamiento\60_Shapes_Entren\08_otm\08_otm_train_y8m.shp'
    gt_shp_path = r'D:\12_Pros_Ocool\80_YoloVx\02_Entrenamiento\50_Shape_val\08_otm_val.shp'
    output_dir = Path(r"D:\12_Pros_Ocool\80_YoloVx\02_Entrenamiento\runs\detect_Final\train_y8m\imagenes")

    print("Iniciando generación de curva PR...")
    print(f"Archivo de predicciones: {pred_shp_path}")
    print(f"Archivo de ground truth: {gt_shp_path}")
    print(f"Directorio de salida: {output_dir}")

    # Crear directorio si no existe
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generar curva PR
    create_pr_curve(pred_shp_path, gt_shp_path, output_dir)


if __name__ == "__main__":
    main()