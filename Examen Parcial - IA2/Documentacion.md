# Documentación Técnica del Framework de IA para la Detección de Ataques

## Resumen
Este documento presenta una descripción técnica exhaustiva del proceso metodológico y las herramientas de software empleadas para el desarrollo de un framework de inteligencia artificial. El objetivo del sistema es la detección de patrones de ataques de fuerza bruta mediante el análisis de logs de autenticación de servidores. A continuación, se detallan el entorno de trabajo, las fases del pipeline experimental y la bibliografía de soporte.

---

## 1. Metodología y Herramientas

El proyecto se fundamenta en un pipeline de machine learning que transforma datos de logs crudos en clasificaciones de seguridad accionables. Este proceso se apoya en un conjunto de librerías especializadas del ecosistema de Python.

### 1.1. Librerías y Entorno de Trabajo

El entorno de desarrollo se configuró en Python, utilizando un conjunto de librerías de código abierto que son estándar en la industria para la ciencia de datos y el deep learning.

* **TensorFlow y Keras:** TensorFlow es una plataforma de código abierto para el cómputo numérico y el aprendizaje automático a gran escala (Abadi et al., 2015). Sobre ella, se utilizó Keras como una API de alto nivel, diseñada para facilitar la experimentación rápida y la construcción de redes neuronales (Chollet et al., 2015). En este proyecto, fueron la base para el diseño, compilación y entrenamiento del modelo **LSTM Autoencoder**.

* **Scikit-learn:** Es una de las librerías más robustas para el machine learning en Python. Provee herramientas eficientes para el preprocesamiento de datos y la evaluación de modelos (Pedregosa et al., 2011). Su rol fue fundamental para la transformación de características categóricas mediante `OneHotEncoder` y para el cálculo de las métricas de rendimiento finales (precisión, recall, F1-score) que validaron la eficacia del framework.

* **Pandas:** Es una librería esencial para la manipulación y análisis de datos. Su estructura principal, el DataFrame, es una herramienta flexible y potente para gestionar datos tabulares (McKinney, 2010). Fue el pilar de la fase de procesamiento, permitiendo estructurar los datos extraídos de los logs en un formato limpio y manejable.

* **NumPy (Numerical Python):** Es el paquete fundamental para la computación científica en Python. Proporciona soporte para arrays multidimensionales y un amplio conjunto de funciones matemáticas para operar sobre ellos (Harris et al., 2020). Actuó como la base para todas las operaciones numéricas, desde la creación de secuencias hasta el cálculo de los errores de reconstrucción.

* **Matplotlib y Seaborn:** Matplotlib es la librería de visualización principal de Python, ofreciendo un control detallado sobre los gráficos (Hunter, 2007). Seaborn, construida sobre Matplotlib, proporciona una interfaz de alto nivel para crear visualizaciones estadísticas más complejas y estéticamente agradables (Waskom, 2021). Ambas fueron cruciales en la fase de evaluación para generar el histograma de distribución de errores y el mapa de calor de la matriz de confusión.

* **`re` (Regular Expressions) y `os`:** Siendo parte de la librería estándar de Python, `re` fue la herramienta principal para el parsing de los logs en la fase inicial. El módulo `os` se utilizó para interactuar con el sistema de archivos de manera robusta, verificando la existencia de los ficheros de datos antes de su procesamiento.

### 1.2. Fases del Proceso Experimental

El desarrollo se estructuró en un pipeline de cuatro etapas secuenciales, garantizando la trazabilidad y reproducibilidad del experimento.

#### **Fase 1: Procesamiento de Datos y Extracción de Características**
El análisis de logs es un desafío debido a su naturaleza semi-estructurada (Flynn & Olukoya, 2025). Esta fase abordó dicho problema mediante un proceso de **parsing**, utilizando el módulo `re` para aplicar patrones de expresiones regulares a cada línea del archivo `combined_auth.log`. Este método permitió extraer de manera consistente campos semánticos como el timestamp, el proceso, el mensaje de evento y la IP de origen. La información extraída se cargó en un DataFrame de Pandas, transformando el texto crudo en una base de datos estructurada y lista para el análisis.

#### **Fase 2: Preparación de Datos y Secuenciación**
Los modelos de redes neuronales recurrentes requieren una entrada numérica y secuencial. Esta fase se centró en la **ingeniería de características**. Primero, se aplicó **One-Hot Encoding** para vectorizar las variables categóricas. El paso más crítico fue la **secuenciación**, donde se utilizó una técnica de **ventana deslizante** para agrupar los registros en secuencias superpuestas de 15 eventos. Este enfoque es fundamental, ya que permite al modelo capturar el contexto temporal, un aspecto que los métodos de machine learning tradicionales a menudo ignoran en el análisis de logs (Flynn & Olukoya, 2025).

#### **Fase 3: Entrenamiento del Modelo LSTM Autoencoder**
Se implementó un **Autoencoder**, una arquitectura de aprendizaje no supervisado, con capas **LSTM (Long Short-Term Memory)**. Las LSTM son una variante de las redes neuronales recurrentes (RNN) especialmente efectivas para aprender dependencias a largo plazo en datos secuenciales, superando problemas como el desvanecimiento del gradiente (Gudivaka et al., 2025). El modelo fue entrenado exclusivamente con secuencias de comportamiento normal. El objetivo era que la red aprendiera a minimizar el **error de reconstrucción** para datos legítimos. La premisa es que, al ser un experto en "normalidad", el modelo producirá un error significativamente mayor al intentar reconstruir patrones de ataque que nunca ha visto, sirviendo este error como un indicador de anomalía.

#### **Fase 4: Evaluación, Calibración y Visualización**
La fase final consistió en traducir las puntuaciones de error del modelo en una clasificación binaria (Normal/Anomalía).
1.  **Calibración del Umbral:** Se implementó un algoritmo de optimización para encontrar el **umbral de decisión óptimo**. En lugar de un método estadístico fijo, se realizó una búsqueda iterativa para encontrar el umbral que maximizaba un **F1-Score ponderado**, asignando una importancia del 70% a la detección de anomalías. Esta calibración es clave para alinear el rendimiento técnico del modelo con el objetivo operativo de priorizar la seguridad.
2.  **Clasificación y Métricas:** Con el umbral calibrado, se generaron las predicciones finales y se calcularon las métricas de rendimiento (Precision, Recall, F1-Score) usando Scikit-learn.
3.  **Visualización:** Finalmente, se emplearon Matplotlib y Seaborn para generar los gráficos que permitieron una interpretación visual clara de los resultados.

---

## Referencias

Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G. S., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jia, Y., Jozefowicz, R., Kaiser, L., Kudlur, M., … Zheng, X. (2015). *TensorFlow: Large-scale machine learning on heterogeneous systems*. Software available from tensorflow.org.

Chollet, F., et al. (2015). *Keras*. GitHub. https://github.com/keras-team/keras

Flynn, R., & Olukoya, O. (2025). Using approximate matching and machine learning to uncover malicious activity in logs. *Computers & Security, 151*, 104312. https://doi.org/10.1016/j.cose.2024.104312

Gudivaka, B. R., Gudivaka, R. L., Gudivaka, R. K., Basani, D. K. R., Grandhi, S. H., Murugesan, S., & Kamruzzaman, M. M. (2025). A predominant intrusion detection system in IIoT using ELCG-DSA AND LWS-BIOLSTM with blockchain. *Sustainable Computing: Informatics and Systems, 46*, 101127. https://doi.org/10.1016/j.suscom.2024.101127

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., … Oliphant, T. E. (2020). Array programming with NumPy. *Nature, 585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering, 9*(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

McKinney, W. (2010). Data structures for statistical computing in Python. In S. van der Walt & J. Millman (Eds.), *Proceedings of the 9th Python in Science Conference* (pp. 56–61). https://doi.org/10.25080/Majora-92bf1922-00a

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Waskom, M. L. (2021). Seaborn: statistical data visualization. *Journal of Open Source Software, 6*(60), 3021. https://doi.org/10.21105/joss.03021