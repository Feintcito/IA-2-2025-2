"""
Laboratorio: Comparación de Sistemas de Seguridad
Sistema con IA (TensorFlow/Keras) vs Sistema sin IA (Reglas)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt


# ============================================================================
# GENERACIÓN DE DATOS SIMULADOS DE TRÁFICO DE RED (VERSIÓN MEJORADA)
# ============================================================================

def generar_datos_trafico(n_samples=5000):
    """
    Genera datos sintéticos más realistas con superposición entre clases.
    Introduce un "ataque sutil" para desafiar al sistema basado en reglas.
    """
    np.random.seed(42)

    # --- CATEGORÍAS DE TRÁFICO ---
    # 1. Tráfico Normal (65%): Comportamiento estándar.
    # 2. Ataque de Fuerza Bruta (20%): Ruidoso y fácil de detectar.
    # 3. Ataque Sutil (15%): Diseñado para estar en la "zona gris".

    n_normal = int(n_samples * 0.65)
    n_bruto = int(n_samples * 0.20)
    n_sutil = n_samples - n_normal - n_bruto

    # 1. Tráfico normal: Aumentamos la desviación estándar para crear más variabilidad.
    trafico_normal = np.column_stack([
        np.random.normal(60, 25, n_normal),      # Paquetes/seg con más variación
        np.random.normal(1800, 700, n_normal),     # Bytes con más variación
        np.random.randint(1, 8, n_normal),       # Puertos (algunos podrían parecer sospechosos)
        np.random.randint(1, 4, n_normal),       # Intentos
        np.random.normal(0.15, 0.08, n_normal)   # Tiempos de respuesta
    ])
    labels_normal = np.zeros(n_normal)

    # 2. Ataque de Fuerza Bruta: Sigue siendo obvio, pero con valores más cercanos a los umbrales.
    trafico_bruto = np.column_stack([
        np.random.normal(150, 40, n_bruto),      # Paquetes altos
        np.random.normal(4500, 1000, n_bruto),    # Bytes altos
        np.random.randint(15, 50, n_bruto),      # Muchos puertos
        np.random.randint(8, 25, n_bruto),       # Muchos intentos
        np.random.normal(0.4, 0.1, n_bruto)      # Tiempos altos
    ])
    labels_bruto = np.ones(n_bruto)

    # 3. Ataque Sutil (¡LA CLAVE!): Los valores individuales están diseñados
    #    para estar JUSTO DEBAJO de los umbrales del sistema tradicional.
    #    Un sistema de reglas simple fallará, pero la IA puede aprender el patrón combinado.
    trafico_sutil = np.column_stack([
        np.random.normal(95, 10, n_sutil),       # Justo debajo del umbral de 100 paquetes
        np.random.normal(2900, 300, n_sutil),      # Justo debajo del umbral de 3000 bytes
        np.random.randint(8, 11, n_sutil),       # Justo debajo del umbral de 10 puertos
        np.random.randint(4, 6, n_sutil),        # Justo debajo del umbral de 5 intentos
        np.random.normal(0.28, 0.05, n_sutil)    # Justo debajo del umbral de 0.3 de tiempo
    ])
    labels_sutil = np.ones(n_sutil) # Sigue siendo una amenaza (label=1)

    # Combinar los tres tipos de datos
    X = np.vstack([trafico_normal, trafico_bruto, trafico_sutil])
    y = np.concatenate([labels_normal, labels_bruto, labels_sutil])

    # Mezclar datos
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    return X, y


# ============================================================================
# SISTEMA DE SEGURIDAD TRADICIONAL (SIN IA)
# ============================================================================

class SistemaSeguridadTradicional:
    """Sistema basado en reglas fijas predefinidas"""

    def __init__(self):
        # Umbrales definidos manualmente
        self.umbral_paquetes = 100
        self.umbral_bytes = 3000
        self.umbral_puertos = 10
        self.umbral_intentos = 5
        self.umbral_tiempo = 0.3

    def detectar_amenaza(self, trafico):
        """
        Detecta amenazas usando reglas simples
        trafico: [paquetes/seg, bytes, puertos, intentos, tiempo]
        """
        paquetes, bytes_env, puertos, intentos, tiempo = trafico

        # Reglas básicas
        if paquetes > self.umbral_paquetes:
            return 1  # Amenaza
        if bytes_env > self.umbral_bytes:
            return 1
        if puertos > self.umbral_puertos:
            return 1  # Posible escaneo de puertos
        if intentos > self.umbral_intentos:
            return 1
        if tiempo > self.umbral_tiempo:
            return 1

        return 0  # Normal

    def predecir(self, X):
        """Predice para múltiples muestras"""
        return np.array([self.detectar_amenaza(x) for x in X])


# ============================================================================
# SISTEMA DE SEGURIDAD CON IA (TENSORFLOW/KERAS)
# ============================================================================

class SistemaSeguridadIA:
    """Sistema basado en Red Neuronal con Keras"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.entrenado = False

    def crear_modelo(self, input_dim):
        """Crea una red neuronal para clasificación binaria"""
        model = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_dim=input_dim),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def entrenar(self, X_train, y_train, epochs=50, verbose=0):
        """Entrena el modelo con los datos"""
        # Normalizar datos
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Crear y entrenar modelo
        self.model = self.crear_modelo(X_train.shape[1])

        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=verbose
        )

        self.entrenado = True
        return history

    def predecir(self, X):
        """Predice amenazas"""
        if not self.entrenado:
            raise Exception("El modelo no ha sido entrenado")

        X_scaled = self.scaler.transform(X)
        predicciones = self.model.predict(X_scaled, verbose=0)
        return (predicciones > 0.5).astype(int).flatten()


# ============================================================================
# EVALUACIÓN Y COMPARACIÓN
# ============================================================================

def calcular_metricas(y_true, y_pred):
    """Calcula métricas de rendimiento"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'falsos_positivos': fp,
        'falsos_negativos': fn
    }


def mostrar_comparacion(metricas_tradicional, metricas_ia, tiempo_tradicional, tiempo_ia):
    """Muestra comparación detallada"""
    print("=" * 80)
    print("RESULTADOS DEL LABORATORIO: SEGURIDAD CON IA vs SIN IA")
    print("=" * 80)

    print("\n SISTEMA TRADICIONAL (Reglas Fijas)")
    print("-" * 80)
    print(f"  Exactitud (Accuracy):      {metricas_tradicional['accuracy']:.2%}")
    print(f"  Precisión:                 {metricas_tradicional['precision']:.2%}")
    print(f"  Recall (Sensibilidad):     {metricas_tradicional['recall']:.2%}")
    print(f"  F1-Score:                  {metricas_tradicional['f1_score']:.4f}")
    print(f"  Falsos Positivos:          {metricas_tradicional['falsos_positivos']}")
    print(f"  Falsos Negativos:          {metricas_tradicional['falsos_negativos']} (¡Ataques sutiles no detectados!)")
    print(f"  Tiempo de predicción:      {tiempo_tradicional:.4f} segundos")

    print("\n SISTEMA CON IA (TensorFlow/Keras)")
    print("-" * 80)
    print(f"  Exactitud (Accuracy):      {metricas_ia['accuracy']:.2%}")
    print(f"  Precisión:                 {metricas_ia['precision']:.2%}")
    print(f"  Recall (Sensibilidad):     {metricas_ia['recall']:.2%}")
    print(f"  F1-Score:                  {metricas_ia['f1_score']:.4f}")
    print(f"  Falsos Positivos:          {metricas_ia['falsos_positivos']}")
    print(f"  Falsos Negativos:          {metricas_ia['falsos_negativos']}")
    print(f"  Tiempo de predicción:      {tiempo_ia:.4f} segundos")

    print("\n COMPARACIÓN")
    print("-" * 80)
    if metricas_tradicional['accuracy'] > 0:
      mejora_accuracy = (metricas_ia['accuracy'] - metricas_tradicional['accuracy'])
      print(f"  La IA fue un {mejora_accuracy:.2%} más exacta.")

    print(f"  El sistema tradicional tuvo {metricas_tradicional['falsos_negativos']} falsos negativos,")
    print(f"  mientras que la IA solo tuvo {metricas_ia['falsos_negativos']}.")


# ============================================================================
# EJECUCIÓN DEL LABORATORIO
# ============================================================================

def ejecutar_laboratorio():
    """Ejecuta el laboratorio completo"""
    print(" Iniciando Laboratorio de Seguridad con IA...\n")

    # 1. Generar datos
    print(" Generando datos de tráfico de red...")
    X, y = generar_datos_trafico(n_samples=5000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"   ✓ {len(X_train)} muestras de entrenamiento")
    print(f"   ✓ {len(X_test)} muestras de prueba")
    print(f"   ✓ {int(np.sum(y_test))} amenazas en datos de prueba\n")

    # 2. Sistema Tradicional
    print(" Configurando sistema tradicional...")
    sistema_tradicional = SistemaSeguridadTradicional()
    inicio = time.time()
    pred_tradicional = sistema_tradicional.predecir(X_test)
    tiempo_tradicional = time.time() - inicio
    metricas_tradicional = calcular_metricas(y_test, pred_tradicional)
    print("   ✓ Sistema tradicional evaluado\n")

    # 3. Sistema con IA
    print(" Entrenando sistema con IA (esto puede tomar unos segundos)...")
    sistema_ia = SistemaSeguridadIA()
    sistema_ia.entrenar(X_train, y_train, epochs=50, verbose=0)
    print("   ✓ Modelo entrenado\n")

    print("🔍 Evaluando sistema con IA...")
    inicio = time.time()
    pred_ia = sistema_ia.predecir(X_test)
    tiempo_ia = time.time() - inicio
    metricas_ia = calcular_metricas(y_test, pred_ia)
    print("   ✓ Sistema con IA evaluado\n")

    # 4. Mostrar resultados
    mostrar_comparacion(metricas_tradicional, metricas_ia, tiempo_tradicional, tiempo_ia)

    # 5. Demostración práctica
    print("\n DEMOSTRACIÓN PRÁCTICA: El punto ciego del sistema de reglas")
    print("=" * 80)
    print("Probando con tres tipos de tráfico:\n")

    # Tráfico 100% normal
    trafico_normal = np.array([[45, 1200, 2, 1, 0.08]])

    # Tráfico 100% malicioso (Fuerza Bruta)
    trafico_bruto_obvio = np.array([[250, 6000, 30, 15, 0.6]])

    # Tráfico Sutil Engañoso: diseñado para estar JUSTO DEBAJO de cada umbral
    # El sistema de reglas fallará aquí, pero la IA debería detectarlo.
    trafico_sutil_enganoso = np.array([[98, 2950, 9, 4, 0.29]])

    print("1. Tráfico Normal: [45 paq/s, 1200 bytes, 2 puertos, 1 intento, 0.08s]")
    print("   Ambos sistemas deberían identificarlo como normal.")
    print(
        f"  → Sistema Tradicional: {' AMENAZA' if sistema_tradicional.predecir(trafico_normal)[0] == 1 else '✅ NORMAL'}")
    print(f"  → Sistema con IA:      {' AMENAZA' if sistema_ia.predecir(trafico_normal)[0] == 1 else '✅ NORMAL'}")

    print("\n2. Tráfico Sospechoso (Bruto): [250 paq/s, 6000 bytes, 30 puertos, 15 intentos, 0.6s]")
    print("   Ambos sistemas deberían identificarlo como amenaza.")
    print(
        f"  → Sistema Tradicional: {' AMENAZA' if sistema_tradicional.predecir(trafico_bruto_obvio)[0] == 1 else '✅ NORMAL'}")
    print(f"  → Sistema con IA:      {' AMENAZA' if sistema_ia.predecir(trafico_bruto_obvio)[0] == 1 else '✅ NORMAL'}")

    print("\n3. Tráfico Sutil (Engañoso): [98 paq/s, 2950 bytes, 9 puertos, 4 intentos, 0.29s]")
    print("   💥 ¡AQUÍ ES DONDE EL SISTEMA DE REGLAS FALLA! 💥")
    print(
        f"  → Sistema Tradicional: {' AMENAZA' if sistema_tradicional.predecir(trafico_sutil_enganoso)[0] == 1 else '✅ NORMAL'}")
    print(f"  → Sistema con IA:      {' AMENAZA' if sistema_ia.predecir(trafico_sutil_enganoso)[0] == 1 else '✅ NORMAL'}")


    print("\n" + "=" * 80)
    print(" Laboratorio completado exitosamente!")
    print("=" * 80)


# Ejecutar el laboratorio
if __name__ == "__main__":
    ejecutar_laboratorio()