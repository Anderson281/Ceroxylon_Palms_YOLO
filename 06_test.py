# This script generates a Precision-Recall (PR) curve for spatial object detection results.
# It compares predicted and ground truth polygons from Shapefiles, computes precision and recall at various IoU thresholds,
# and saves the PR curve as a high-resolution image. The code is annotated to explain each step and the purpose of each library.

import geopandas as gpd              # For reading, manipulating, and analyzing vector spatial data (e.g., Shapefiles).
import pyogrio                       # Efficient reading/writing of geospatial files, faster than Fiona for large datasets[2].
from shapely.geometry import Polygon # For creating and manipulating geometric objects (polygons, etc.).
import numpy as np                   # Numerical operations, array handling, and generating threshold values.
import matplotlib.pyplot as plt      # For plotting and saving the Precision-Recall curve.
from pathlib import Path             # For clear, cross-platform file and directory path handling.

# Function to compute Intersection over Union (IoU) between two polygons.
def calculate_iou(pred_poly, gt_poly):
    intersection = pred_poly.intersection(gt_poly).area
    union = pred_poly.union(gt_poly).area
    return intersection / union if union != 0 else 0

# Function to compute precision and recall at a given IoU threshold.
def calculate_metrics(pred_shp, gt_shp, iou_threshold=0.5, total_possible_detections=1000):
    gt_gdf = gpd.GeoDataFrame(pyogrio.read_dataframe(gt_shp))   # Load ground truth polygons.
    pred_gdf = gpd.GeoDataFrame(pyogrio.read_dataframe(pred_shp)) # Load predicted polygons.

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    total_gt_palmeras = len(gt_gdf)  # Total ground truth objects
    pred_gdf["matched"] = False      # Track matched predictions
    gt_gdf["matched"] = False        # Track matched ground truths

    # For each ground truth polygon, check if any prediction matches above the IoU threshold.
    for i, gt_poly in enumerate(gt_gdf.geometry):
        for j, pred_poly in enumerate(pred_gdf.geometry):
            iou = calculate_iou(pred_poly, gt_poly)
            if iou >= iou_threshold:
                true_positives += 1
                pred_gdf.at[j, "matched"] = True
                gt_gdf.at[i, "matched"] = True
                break

    false_positives = len(pred_gdf[~pred_gdf["matched"]])  # Predictions with no matching ground truth
    false_negatives = len(gt_gdf[~gt_gdf["matched"]])      # Ground truths with no matching prediction

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall

# Function to generate and save the Precision-Recall curve.
def create_pr_curve(pred_shp, gt_shp, output_dir):
    thresholds = np.linspace(0.1, 0.9, 20)  # IoU thresholds from 0.1 to 0.9
    precisions = []
    recalls = []

    print("Calculating PR curve points...")
    for threshold in thresholds:
        try:
            p, r = calculate_metrics(pred_shp, gt_shp, iou_threshold=threshold)
            precisions.append(p)
            recalls.append(r)
            print(f"Threshold: {threshold:.2f}, Precision: {p:.3f}, Recall: {r:.3f}")
        except Exception as e:
            print(f"Error at threshold {threshold}: {str(e)}")
            continue

    # Configure plot style
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
    })

    # Plot the PR curve
    print("Generating plot...")
    try:
        plt.plot(recalls, precisions, '-', color='blue', linewidth=2, label='Ceroxylon')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.02)
        plt.legend()

        # Save the plot as a high-resolution PNG
        output_path = output_dir / 'PR_curve.png'
        plt.savefig(output_path, dpi=900, bbox_inches='tight')
        plt.close()
        print(f"Plot saved at: {output_path}")
    except Exception as e:
        print(f"Error generating plot: {str(e)}")

# Main function to set file paths and run the PR curve generation.
def main():
    pred_shp_path = r'D:\12_Pros_Ocool\80_YoloVx\02_Entrenamiento\60_Shapes_Entren\08_otm\08_otm_train_y8m.shp'
    gt_shp_path = r'D:\12_Pros_Ocool\80_YoloVx\02_Entrenamiento\50_Shape_val\08_otm_val.shp'
    output_dir = Path(r"D:\12_Pros_Ocool\80_YoloVx\02_Entrenamiento\runs\detect_Final\train_y8m\imagenes")

    print("Starting PR curve generation...")
    print(f"Prediction file: {pred_shp_path}")
    print(f"Ground truth file: {gt_shp_path}")
    print(f"Output directory: {output_dir}")

    # Create output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the PR curve
    create_pr_curve(pred_shp_path, gt_shp_path, output_dir)

if __name__ == "__main__":
    main()
