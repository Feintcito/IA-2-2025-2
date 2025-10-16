import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


def find_best_weighted_threshold(loss_normal, loss_anomalies):
    """
    Encuentra el umbral que maximiza un F1-Score ponderado,
    dando más importancia a la clase de anomalía.
    """
    best_weighted_f1 = -1
    best_threshold = 0
    y_true = np.concatenate([np.zeros(len(loss_normal)), np.ones(len(loss_anomalies))])
    all_loss = np.concatenate([loss_normal, loss_anomalies])

    thresholds = np.percentile(all_loss, np.arange(1, 100))

    # Definimos los pesos: la clase Anomalía es más importante.
    weight_normal = 0.3
    weight_anomaly = 0.7

    for threshold in thresholds:
        y_pred = [1 if e > threshold else 0 for e in all_loss]

        f1_normal = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        f1_anomaly = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        # Calcular el F1-score ponderado
        current_weighted_f1 = (weight_normal * f1_normal) + (weight_anomaly * f1_anomaly)

        if current_weighted_f1 > best_weighted_f1:
            best_weighted_f1 = current_weighted_f1
            best_threshold = threshold

    print(
        f"Búsqueda completada: Mejor F1 ponderado ({best_weighted_f1:.4f}) encontrado con umbral = {best_threshold:.4f}")
    return best_threshold


def plot_error_distribution(train_loss, anomaly_loss, threshold):
    plt.figure(figsize=(12, 6))
    sns.histplot(train_loss, bins=50, kde=True, label='Normal', color='blue')
    sns.histplot(anomaly_loss, bins=50, kde=True, label='Anomalía (Ataque)', color='red')
    plt.axvline(threshold, color='orange', linestyle='--', linewidth=2,
                label=f'Umbral Ponderado Óptimo ({threshold:.4f})')
    plt.title('Distribución del Error de Reconstrucción')
    plt.xlabel('Error Absoluto Medio (MAE)')
    plt.ylabel('Densidad')
    plt.legend()
    plt.savefig('error_distribution.png')
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomalía'],
                yticklabels=['Normal', 'Anomalía'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig('confusion_matrix.png')
    plt.show()


# --- Ejecución del Script ---
if __name__ == '__main__':
    try:
        model = tf.keras.models.load_model('lstm_autoencoder.keras')
        X_normal = np.load('X_train_sequences.npy')
        X_anomalies = np.load('X_anomaly_sequences.npy')
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo necesario: {e.filename}.")
        exit()

    print("Modelo y datasets cargados.")

    X_full = np.concatenate([X_normal, X_anomalies])
    y_true = np.concatenate([np.zeros(len(X_normal)), np.ones(len(X_anomalies))])

    print("Calculando el error de reconstrucción...")
    reconstructions = model.predict(X_full)
    mae_loss = np.mean(np.abs(reconstructions - X_full), axis=(1, 2))

    loss_normal = mae_loss[y_true == 0]
    loss_anomalies = mae_loss[y_true == 1]

    # --- 3. Encontrar y Establecer el Umbral Ponderado Óptimo ---
    optimal_threshold = find_best_weighted_threshold(loss_normal, loss_anomalies)

    # --- 4. Generar Predicciones con el umbral final ---
    y_pred = [1 if e > optimal_threshold else 0 for e in mae_loss]

    # --- 5. Calcular y Mostrar Métricas Finales ---
    precision_normal = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_normal = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_normal = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    support_normal = len(X_normal)

    precision_anomaly = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_anomaly = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_anomaly = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    support_anomaly = len(X_anomalies)

    accuracy = accuracy_score(y_true, y_pred)

    print("\n--- Reporte de Clasificación con Umbral Ponderado Óptimo ---")
    print(f"{'':<12}{'precision':<12}{'recall':<12}{'f1-score':<12}{'support':<12}")
    print(f"{'Normal':<12}{precision_normal:<12.2f}{recall_normal:<12.2f}{f1_normal:<12.2f}{support_normal:<12}")
    print(f"{'Anomalía':<12}{precision_anomaly:<12.2f}{recall_anomaly:<12.2f}{f1_anomaly:<12.2f}{support_anomaly:<12}")
    print("\n")
    print(f"{'accuracy':<36}{accuracy:<12.2f}{len(y_true):<12}")

    # --- 6. Visualizar Resultados ---
    print("\nGenerando visualizaciones...")
    plot_error_distribution(loss_normal, loss_anomalies, optimal_threshold)
    plot_confusion_matrix(y_true, y_pred)

    print("\n¡Análisis completado!")