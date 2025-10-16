import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def create_sequences(data, seq_length):
    """
    Crea secuencias de un tamaño fijo a partir de los datos.
    """
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)


def prepare_data_for_model(input_csv_path, seq_length=10):
    """
    Prepara los datos procesados para el entrenamiento del modelo LSTM Autoencoder.

    Args:
        input_csv_path (str): Ruta al archivo CSV con los logs procesados.
        seq_length (int): La longitud de cada secuencia de eventos.

    Returns:
        tuple: Contiene los arrays de secuencias normales y de ataque.
    """
    print("Iniciando la preparación de datos para el modelo...")
    df = pd.read_csv(input_csv_path)

    # --- 1. Feature Engineering: Codificación One-Hot ---
    # Seleccionamos las características categóricas más importantes
    features_to_encode = ['event_type', 'process']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    encoded_features = encoder.fit_transform(df[features_to_encode])

    # Crear un nuevo DataFrame con las características codificadas
    df_encoded = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

    print(
        f"Se han creado {df_encoded.shape[1]} características numéricas a partir de '{', '.join(features_to_encode)}'.")

    # --- 2. Creación de Secuencias ---
    # Convertir el DataFrame codificado a un array de NumPy para crear secuencias
    data_numeric = df_encoded.to_numpy()

    sequences = create_sequences(data_numeric, seq_length)
    print(f"Se han creado {len(sequences)} secuencias de longitud {seq_length}.")

    # --- 3. Separación de Datos Normales y Anómalos ---
    # Para identificar secuencias anómalas, revisamos los 'event_type' originales
    is_anomaly = df['event_type'].isin(['Failed password', 'Invalid user'])

    normal_sequences = []
    anomaly_sequences = []

    for i in range(len(sequences)):
        # Si en la ventana de la secuencia (de i a i+seq_length) hay algún evento anómalo,
        # la secuencia completa se considera una anomalía.
        if is_anomaly[i:i + seq_length].any():
            anomaly_sequences.append(sequences[i])
        else:
            normal_sequences.append(sequences[i])

    # Convertir listas a arrays de NumPy
    X_train_normal = np.array(normal_sequences)
    X_test_anomalies = np.array(anomaly_sequences)

    print(f"Separación completada:")
    print(f" - Secuencias de entrenamiento (normales): {X_train_normal.shape[0]}")
    print(f" - Secuencias de prueba (anómalas): {X_test_anomalies.shape[0]}")

    return X_train_normal, X_test_anomalies


# --- Ejecución del Script ---
if __name__ == '__main__':
    processed_log_file = 'processed_auth_logs.csv'
    SEQUENCE_LENGTH = 15  # Puedes ajustar este valor. 15 es un buen punto de partida.

    # Preparar los datos
    X_train, X_anomalies = prepare_data_for_model(processed_log_file, seq_length=SEQUENCE_LENGTH)

    # --- 4. Guardar los datos listos para el entrenamiento ---
    # Guardamos los arrays en formato .npy para cargarlos fácilmente después
    np.save('X_train_sequences.npy', X_train)
    np.save('X_anomaly_sequences.npy', X_anomalies)

    print(f"\nDatos listos y guardados en 'X_train_sequences.npy' y 'X_anomaly_sequences.npy'.")
    print("\nEl siguiente paso es construir y entrenar el modelo LSTM Autoencoder. 🚀")