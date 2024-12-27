import pandas as pd

# Ruta al archivo de resultados
results_path = r"D:\12_Pros_Ocool\80_YoloVx\02_Entrenamiento\runs\detect_Final\train_y8m\results.csv"

# Cargar los resultados en un DataFrame
results = pd.read_csv(results_path)

# Eliminar espacios adicionales en los nombres de las columnas
results.columns = results.columns.str.strip()

# Mostrar los nombres de las columnas después de la limpieza
print("Nombres de columnas limpiados:", results.columns)

# Buscar la mejor época basándose en 'val/box_loss'
best_epoch = results['val/box_loss'].idxmin()
print(f"La mejor época fue la número: {best_epoch}")
