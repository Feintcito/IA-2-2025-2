# Análisis del Algoritmo de Cifrado SecureMD5

## 1. Introducción

Este informe detalla la implementación y evaluación del algoritmo de cifrado **SecureMD5**. El objetivo de este laboratorio es analizar su rendimiento y seguridad en comparación con algoritmos estándar de la industria como **AES** y **RC4**.

Para lograr esto, se realizaron dos experimentos principales:

1. **Análisis de Rendimiento:** Se midió el tiempo de ejecución, el uso de CPU y el consumo de memoria durante el cifrado y descifrado de un archivo de 10 MB.

2. **Análisis de Entropía:** Se evaluó la capacidad del algoritmo para ofuscar patrones en los datos de un archivo de texto, utilizando el "vocabulario" de palabras binarias como métrica de aleatoriedad.

## 2. Descripción del Código

El script de Python proporcionado se divide en tres componentes principales: la implementación de SecureMD5, los algoritmos de comparación y el framework de pruebas.

### 2.1. Clase SecureMD5

Esta clase encapsula la lógica del algoritmo, que se basa en un cifrador de flujo. Para cada byte del archivo de entrada, genera un byte pseudoaleatorio a partir de una función hash y lo combina con el byte original.

- **`_generar_byte_hash(posicion, frn, clave)`**: Es el núcleo del algoritmo. Esta función privada genera un byte único para cada posición del archivo. Sigue una serie de pasos complejos:

  1. **Construcción del Mensaje (M)**: Concatena la posición del byte, un número aleatorio de archivo (frn) y la clave para formar una larga cadena de dígitos.

  2. **Rotación y Ajuste**: Rota la lista de dígitos y la ajusta a un tamaño fijo de 33 elementos mediante operaciones de XOR y concatenación.

  3. **Conversión a Enteros**: Agrupa los 33 dígitos en 11 enteros de 8 bits (0-255).

  4. **Funciones de Mezcla (F, G, H, I)**: Aplica una serie de operaciones a nivel de bit (AND, OR, NOT, XOR) sobre los 11 enteros para producir un resultado final.

  5. **Resultado**: El resultado de la función I se convierte en el byte hash final.

- **`cifrar(datos_plano, clave)`**: Itera sobre cada byte del archivo original, genera el byte hash correspondiente y lo suma (módulo 256) para obtener el byte cifrado. `cB = pB + f(...)`

- **`descifrar(datos_cifrados, clave, frn)`**: Realiza el proceso inverso. Itera sobre cada byte cifrado, genera el mismo byte hash y lo resta (módulo 256) para recuperar el byte original. `dB = cB - f(...)`

### 2.2. Algoritmos de Comparación

Para tener una base de comparación, el script implementa dos cifradores estándar utilizando la librería cryptography:

- **AES-256-CTR**: Un cifrador de bloque robusto y ampliamente utilizado, operando en modo contador (CTR) que lo convierte en un cifrador de flujo.

- **RC4**: Un cifrador de flujo clásico, conocido por su velocidad pero con vulnerabilidades conocidas. **Nota:** La librería cryptography emite una advertencia de que RC4 está obsoleto y no se considera seguro para aplicaciones modernas.

### 2.3. Framework de Pruebas

- **`probar_rendimiento(...)`**: Mide y reporta las métricas clave de rendimiento:
  - **Tiempo**: Usa la librería time para medir la duración del cifrado y descifrado.
  - **CPU y Memoria**: Usa la librería psutil para monitorear el uso de CPU y el pico de memoria RAM consumida durante el cifrado.

- **`probar_entropia(...)`**: Evalúa la aleatoriedad del archivo de salida. Lo hace dividiendo el contenido del archivo en "palabras" de 2 bytes y contando cuántas palabras únicas existen. Un texto plano tiene pocas palabras únicas (baja entropía), mientras que un archivo bien cifrado debería tener un número cercano al máximo posible (65,536), indicando alta entropía.

## 3. Análisis de Resultados

A continuación se presentan y analizan los resultados obtenidos de la ejecución del script.

### Experimento 1: Análisis de Rendimiento

Se cifró y descifró un archivo aleatorio de 10 MB con cada algoritmo.

| Algoritmo | Tiempo Cifrado (s) | Tiempo Descifrado (s) | CPU (%) | Memoria Pico (MB) |
|-----------|-------------------:|----------------------:|--------:|------------------:|
| **SecureMD5** | 291.3931 | 286.9984 | 99.60 | 10.67 |
| **AES-256-CTR** | 0.0163 | 0.0048 | 104.20 | 11.89 |
| **RC4** | 0.0200 | 0.0190 | 97.70 | 10.09 |

**Interpretación:**

- **Tiempo de Ejecución**: SecureMD5 es **extremadamente lento** en comparación con los algoritmos estándar. Tarda casi 5 minutos en procesar 10 MB, mientras que AES y RC4 lo hacen en milisegundos. Esta diferencia de más de **15,000 veces** se debe a que SecureMD5 está implementado en Python puro, con bucles, conversiones de tipo y manipulaciones de listas para cada byte, mientras que AES y RC4 son algoritmos optimizados implementados a bajo nivel.

- **Uso de CPU**: Todos los algoritmos maximizan el uso de un núcleo de CPU, lo cual es esperado para tareas computacionalmente intensivas.

- **Uso de Memoria**: El consumo de memoria es similar en todos los casos, ya que la mayor parte corresponde a cargar el archivo de 10 MB en la RAM.

**Conclusión del Rendimiento**: Desde una perspectiva de rendimiento, SecureMD5 es completamente inviable para cualquier aplicación práctica.

### Experimento 2: Análisis de Entropía (Seguridad)

Se analizó el "vocabulario" (palabras únicas de 2 bytes) de un archivo de texto plano (quijote.txt) y sus versiones cifradas con SecureMD5 y RC4. El máximo teórico de palabras de 2 bytes es 2^16 = 65,536.

| Archivo | Vocabulario (Palabras únicas de 2 bytes) |
|---------|------------------------------------------:|
| quijote.txt (Original) | 1,669 |
| quijote_smd5.bin | 65,535 |
| quijote_rc4.bin | 65,537 |

*Nota: un valor superior a 65,536, como el de RC4, puede deberse a que el archivo no tiene un tamaño par de bytes, dejando un último byte "huérfano" que se cuenta como una palabra única adicional.*

**Interpretación:**

- El archivo de texto original tiene un vocabulario muy bajo (1,669), lo que refleja la naturaleza predecible y repetitiva del lenguaje humano.

- Tanto SecureMD5 como RC4 transforman el texto plano en un archivo con un vocabulario que se acerca al máximo teórico. Esto indica que ambos algoritmos son **efectivos para eliminar los patrones** del texto original, produciendo una salida que parece estadísticamente aleatoria.

**Conclusión de la Entropía**: Según esta métrica, SecureMD5 logra su objetivo de ofuscar los datos de manera efectiva, alcanzando un nivel de entropía comparable al de un cifrador de flujo estándar.

## 4. Conclusiones Generales

El algoritmo SecureMD5 demuestra un contraste radical entre seguridad teórica y viabilidad práctica.

1. **Seguridad**: El diseño del algoritmo logra producir un texto cifrado con alta entropía, ocultando eficazmente las características estadísticas del texto plano. Desde este punto de vista, cumple con un requisito fundamental de la criptografía.

2. **Rendimiento**: Su rendimiento es **abismalmente pobre**. La complejidad de generar un hash para cada byte individual lo hace miles de veces más lento que los algoritmos estándar, haciéndolo inútil para cualquier aplicación en el mundo real.

En resumen, SecureMD5 es un ejercicio académico interesante que ilustra los principios de los cifradores de flujo y la generación de datos pseudoaleatorios, pero su ineficiencia computacional resalta por qué los algoritmos criptográficos prácticos deben equilibrar cuidadosamente la seguridad con el rendimiento.
