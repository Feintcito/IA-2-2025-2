import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed


def build_autoencoder(seq_length, n_features):
    """
    Construye el modelo LSTM Autoencoder.

    Args:
        seq_length (int): La longitud de cada secuencia de entrada.
        n_features (int): El número de características en cada paso de tiempo de la secuencia.

    Returns:
        tensorflow.keras.Model: El modelo Autoencoder compilado.
    """
    model = Sequential()

    # --- Encoder ---
    # La primera capa LSTM comprime la secuencia de entrada.
    model.add(LSTM(128, activation='relu', input_shape=(seq_length, n_features), return_sequences=False))
    # Repite el vector de salida del encoder para que coincida con la longitud de la secuencia original.
    model.add(RepeatVector(seq_length))

    # --- Decoder ---
    # La segunda capa LSTM intenta reconstruir la secuencia original a partir del vector comprimido.
    model.add(LSTM(128, activation='relu', return_sequences=True))
    # La capa TimeDistributed aplica una capa Dense a cada paso de tiempo de la salida del decoder.
    model.add(TimeDistributed(Dense(n_features)))

    # Compilar el modelo
    model.compile(optimizer='adam', loss='mae')  # Usamos Mean Absolute Error como métrica de error

    model.summary()
    return model


# --- Ejecución del Script ---
if __name__ == '__main__':
    # --- 1. Cargar los Datos de Entrenamiento ---
    try:
        X_train = np.load('X_train_sequences.npy')
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'X_train_sequences.npy'.")
        print("Asegúrate de ejecutar el script 'prepare_data.py' primero.")
        exit()

    if X_train.shape[0] == 0:
        print("Error: El archivo de entrenamiento está vacío. No hay secuencias normales para entrenar.")
        exit()

    print(f"Datos de entrenamiento cargados. Forma del dataset: {X_train.shape}")

    # Obtener dimensiones para el modelo
    n_sequences, seq_length, n_features = X_train.shape

    # --- 2. Construir el Modelo ---
    autoencoder = build_autoencoder(seq_length, n_features)

    # --- 3. Entrenar el Modelo ---
    print("\nIniciando el entrenamiento del modelo...")
    history = autoencoder.fit(
        X_train, X_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
        ]
    )
    print("Entrenamiento completado.")

    # --- 4. Guardar el Modelo Entrenado (CORREGIDO) ---
    # Se guarda en un solo archivo con la extensión .keras
    model_filename = 'lstm_autoencoder.keras'
    autoencoder.save(model_filename)
    print(f"\nModelo entrenado y guardado exitosamente como '{model_filename}'")

    print("\nEl siguiente paso es usar este modelo para detectar anomalías y evaluar su rendimiento. ✅")