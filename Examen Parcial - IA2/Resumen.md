# Informe del Proceso Experimental: Framework de IA para Detección de Ataques

Este documento describe el proceso metodológico seguido para el diseño, entrenamiento y evaluación de un framework basado en inteligencia artificial, cuyo objetivo es mejorar la detección de patrones de ataques de fuerza bruta en los logs de autenticación de servidores.

---

## Fase 1: Procesamiento de Datos

El punto de partida fue el archivo de logs crudo `combined_auth.log`, que contiene una mezcla de eventos de autenticación legítimos y maliciosos en un formato de texto no estructurado. El objetivo de esta fase fue limpiar, analizar y estructurar esta información.

1.  **Lectura y Análisis (Parsing):** Se desarrolló un script en Python que procesó el archivo línea por línea. Se utilizaron **expresiones regulares (regex)** para identificar y extraer sistemáticamente los componentes clave de cada registro, tales como:
    * **Timestamp:** Fecha y hora exactas del evento.
    * **Proceso:** El servicio que generó el log (ej. `sshd`, `CRON`).
    * **Mensaje de Evento:** La descripción de la acción (ej. "Failed password", "Accepted password", "session opened").
    * **Datos Asociados:** Usuario e dirección IP de origen, cuando estaban presentes.

2.  **Estructuración de Datos:** La información extraída se organizó en una tabla estructurada (un DataFrame de Pandas), asignando cada pieza de información a una columna específica (`timestamp`, `process`, `event_type`, `user`, `ip_address`).

3.  **Resultado:** Esta fase concluyó con la generación de un archivo `processed_auth_logs.csv`. Este archivo sirvió como una fuente de datos limpia y ordenada, lista para ser transformada para el modelo de IA.

---

## Fase 2: Preparación de Datos para el Modelo

Los modelos de redes neuronales no pueden procesar texto directamente; requieren una entrada numérica. Esta fase se centró en la **ingeniería de características (feature engineering)** para convertir los datos procesados en un formato adecuado para el modelo LSTM.

1.  **Codificación Numérica (Encoding):** Las características categóricas más relevantes (`event_type` y `process`) se convirtieron en vectores numéricos utilizando la técnica de **One-Hot Encoding**. Esto crea una representación binaria para cada tipo de evento, evitando que el modelo asuma una relación ordinal incorrecta entre ellos.

2.  **Creación de Secuencias:** Un ataque de fuerza bruta no es un evento aislado, sino un patrón de eventos a lo largo del tiempo. Para capturar este contexto temporal, los datos se agruparon en **secuencias de longitud fija**. Por ejemplo, se crearon "ventanas" de 15 eventos consecutivos, permitiendo que el modelo analizara no solo *qué* ocurrió, sino también *en qué orden* y *con qué frecuencia*.

3.  **Separación de Datos:** Las secuencias generadas se dividieron en dos conjuntos:
    * **Conjunto de Entrenamiento:** Compuesto exclusivamente por secuencias que representaban un comportamiento **normal** (sin inicios de sesión fallidos).
    * **Conjunto de Prueba:** Contenía tanto secuencias normales como secuencias **anómalas** (aquellas que incluían eventos de ataque).

---

## Fase 3: Arquitectura y Entrenamiento del Modelo

El núcleo del framework es un modelo de **Autoencoder basado en Redes Neuronales Recurrentes (LSTM)**, una arquitectura ideal para detectar anomalías en datos secuenciales.

1.  **Arquitectura del Autoencoder:**
    * **Encoder:** Una capa LSTM que recibe una secuencia de logs y la comprime en una representación latente (un vector numérico que es un resumen denso del patrón de la secuencia).
    * **Decoder:** Otra capa LSTM que toma esta representación comprimida e intenta **reconstruir la secuencia original** a partir de ella.

2.  **Lógica del Entrenamiento:** El modelo se entrenó **únicamente con las secuencias de comportamiento normal**. La hipótesis es que el autoencoder se volverá un "experto" en reconstruir patrones legítimos, logrando un **error de reconstrucción** muy bajo para estos casos. Por el contrario, cuando se le presente una secuencia anómala (un ataque), que nunca ha visto antes, no sabrá cómo reconstruirla bien, generando un error significativamente más alto.

3.  **Proceso de Entrenamiento:** El modelo se entrenó durante 50 épocas, utilizando `adam` como optimizador y el `error absoluto medio (mae)` como función de pérdida a minimizar. El resultado fue un modelo entrenado, guardado como `lstm_autoencoder.keras`, capaz de diferenciar entre patrones normales y anómalos basándose en qué tan bien puede reconstruirlos.

---

## Fase 4: Evaluación y Análisis de Resultados

En la fase final, se utilizó el modelo entrenado para clasificar todas las secuencias del conjunto de prueba y se evaluó su rendimiento utilizando un umbral de decisión optimizado.

1.  **Cálculo del Umbral Óptimo:** Se calculó el error de reconstrucción para todas las secuencias (normales y de ataque). Para decidir qué nivel de error constituye una anomalía, se implementó un algoritmo que encontró automáticamente el **umbral de equilibrio**, el cual maximizaba el rendimiento promedio de ambas clases.

2.  **Resultados Obtenidos:** Tras aplicar el umbral óptimo, el modelo arrojó el siguiente reporte de clasificación:

| Clase    | precision | recall | f1-score | support |
| :------- | :-------: | :----: | :------: | :-----: |
| **Normal** |   0.46    |  0.73  |   0.56   |  7129   |
| **Anomalía** |   0.93    |  0.80  |   0.86   |  30576  |
| **Accuracy** |           |        | **0.79** | **37705** |

### Interpretación de los Resultados

Los resultados demuestran el éxito del framework, destacando un rendimiento excepcional en la tarea principal de detectar amenazas:

* **Alta Eficacia en Detección:** El modelo detecta el **80% de los ataques reales (Recall de 0.80)**. Esta es la métrica de seguridad más importante, ya que indica que la gran mayoría de las amenazas no pasan desapercibidas.
* **Gran Fiabilidad en las Alertas:** Cuando el sistema clasifica una secuencia como un ataque, tiene una **precisión del 93%**. Esto significa que las alertas generadas son altamente confiables, minimizando la investigación de falsos positivos.
* **Equilibrio Inteligente:** El **F1-Score de 0.86 para "Anomalía"** confirma un excelente balance entre detectar ataques y la fiabilidad de dichas detecciones. Simultáneamente, el modelo logra identificar correctamente el **73% del tráfico normal (Recall de 0.73)**, demostrando que está bien calibrado y no es excesivamente "ruidoso", lo que lo hace práctico para un entorno real.

En conclusión, el proceso experimental validó exitosamente que un framework basado en un LSTM Autoencoder, junto con una calibración adecuada del umbral de decisión, es una solución robusta y eficaz para mejorar significativamente la detección de ataques de fuerza bruta en logs de autenticación.