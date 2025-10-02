import os
import time
import psutil
import requests
from collections import Counter
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


# -----------------------------------------------------------------------------
# IMPLEMENTACIÓN DE SECUREMD5
# Basado en la Sección 3.1 y Figura 2 del paper.
# -----------------------------------------------------------------------------

class SecureMD5:
    """
    Implementación del algoritmo de cifrado SecureMD5.
    """

    def _generar_byte_hash(self, posicion, frn, clave):
        """
        Implementa la función hash 'f(p, frn, k)' descrita en la Figura 2.
        """
        # Paso 1: Construir el Mensaje (M) como una lista de dígitos [cite: 156-157]
        # Convertimos todos los parámetros a una cadena de dígitos y luego a una lista de enteros.
        p_str = str(posicion)
        frn_str = str(int.from_bytes(frn, 'big'))
        # La clave (str) se convierte a sus valores de byte y luego a dígitos
        k_str = "".join([str(b) for b in clave.encode('utf-8')])

        m_str = p_str + frn_str + k_str
        M = [int(digit) for digit in m_str]

        # Si M está vacía, evitamos errores.
        if not M:
            return 0

        # Paso 2: Rotar la lista M [cite: 217]
        size_m = len(M)
        rotacion = posicion % size_m
        M = M[rotacion:] + M[:rotacion]

        # Paso 3: Ajustar el tamaño del mensaje a 33 posiciones [cite: 218-220]
        while len(M) != 33:
            if len(M) > 33:
                sublistas = [M[i:i + 33] for i in range(0, len(M), 33)]
                M_result = [0] * 33
                for sub in sublistas:
                    # Rellenamos la sublista si es más corta de 33
                    sub.extend([0] * (33 - len(sub)))
                    for i in range(33):
                        M_result[i] ^= sub[i]  # Operación XOR entre sublistas [cite: 220]
                M = M_result
            else:  # len(M) < 33
                M.extend(M)  # Concatenar M consigo misma [cite: 219]

        # Nos aseguramos de que M tenga exactamente 33 elementos
        M = M[:33]

        # Paso 4: Crear lista de 11 enteros (m₀ a m₁₀) a partir de M [cite: 221]
        m_list = []
        for i in range(0, 33, 3):
            # Tomamos 3 dígitos, los unimos para formar un número y aplicamos módulo 256
            if i + 3 <= 33:
                num_str = "".join(map(str, M[i:i + 3]))
                num = int(num_str)
                m_list.append(num % 256)

        # Paso 5: Calcular funciones F, G, H, I [cite: 224-227]
        # Usamos operadores a nivel de bit de Python: & (AND), | (OR), ~ (NOT), ^ (XOR)
        # La máscara & 0xFF asegura que el resultado de NOT se mantenga en 8 bits (0-255)
        m = m_list
        F = (m[0] & m[1]) | (~m[2] & 0xFF & m[3])
        G = (F & m[4]) | (m[5] & (~m[6] & 0xFF))
        H = G ^ m[7] ^ m[8]
        I = H ^ (m[9] | (~m[10] & 0xFF))

        # Paso 6: El resultado de la función es el byte final [cite: 236]
        return I % 256

    def cifrar(self, datos_plano, clave):
        """
        Cifra un array de bytes usando la Ecuación (1): cB = pB + f(...) [cite: 147]
        """
        frn = os.urandom(4)  # Genera un número aleatorio de 4 bytes para el archivo [cite: 155]
        datos_cifrados = bytearray()

        for i, byte_original in enumerate(datos_plano):
            byte_hash = self._generar_byte_hash(i, frn, clave)
            byte_cifrado = (byte_original + byte_hash) % 256
            datos_cifrados.append(byte_cifrado)

        return frn, bytes(datos_cifrados)

    def descifrar(self, datos_cifrados, clave, frn):
        """
        Descifra un array de bytes usando la Ecuación (2): dB = cB - f(...) [cite: 148]
        """
        datos_descifrados = bytearray()

        for i, byte_cifrado in enumerate(datos_cifrados):
            byte_hash = self._generar_byte_hash(i, frn, clave)
            # Se suma 256 para evitar resultados negativos con el módulo
            byte_original = (byte_cifrado - byte_hash + 256) % 256
            datos_descifrados.append(byte_original)

        return bytes(datos_descifrados)


# -----------------------------------------------------------------------------
# IMPLEMENTACIÓN DE ALGORITMOS DE COMPARACIÓN (AES y RC4)
# -----------------------------------------------------------------------------

def cifrar_aes_ctr(datos, clave):
    clave_bytes = clave.encode('utf-8')
    # AES necesita una clave de 16, 24 o 32 bytes. La ajustamos.
    clave_ajustada = (clave_bytes * (32 // len(clave_bytes) + 1))[:32]
    nonce = os.urandom(16)
    cipher = Cipher(algorithms.AES(clave_ajustada), modes.CTR(nonce), backend=default_backend())
    encryptor = cipher.encryptor()
    ct = encryptor.update(datos) + encryptor.finalize()
    return nonce, ct


def descifrar_aes_ctr(datos_cifrados, clave, nonce):
    clave_bytes = clave.encode('utf-8')
    clave_ajustada = (clave_bytes * (32 // len(clave_bytes) + 1))[:32]
    cipher = Cipher(algorithms.AES(clave_ajustada), modes.CTR(nonce), backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(datos_cifrados) + decryptor.finalize()


def cifrar_rc4(datos, clave):
    clave_bytes = clave.encode('utf-8')
    # RC4 necesita una clave de 5 a 256 bytes.
    clave_ajustada = (clave_bytes * (16 // len(clave_bytes) + 1))[:16]
    nonce = os.urandom(16)  # Aunque RC4 no usa nonce, lo generamos para mantener la interfaz.
    cipher = Cipher(algorithms.ARC4(clave_ajustada), mode=None, backend=default_backend())
    encryptor = cipher.encryptor()
    ct = encryptor.update(datos) + encryptor.finalize()
    return nonce, ct


def descifrar_rc4(datos_cifrados, clave, nonce):
    clave_bytes = clave.encode('utf-8')
    clave_ajustada = (clave_bytes * (16 // len(clave_bytes) + 1))[:16]
    cipher = Cipher(algorithms.ARC4(clave_ajustada), mode=None, backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(datos_cifrados) + decryptor.finalize()


# -----------------------------------------------------------------------------
# FRAMEWORK DE PRUEBAS
# -----------------------------------------------------------------------------

def crear_archivo_de_prueba(nombre, tamano_mb):
    """Crea un archivo con datos aleatorios para las pruebas."""
    print(f"Creando archivo de prueba '{nombre}' de {tamano_mb} MB...")
    with open(nombre, 'wb') as f:
        f.write(os.urandom(tamano_mb * 1024 * 1024))
    print("Archivo creado.")


def probar_rendimiento(nombre_algoritmo, func_cifrar, func_descifrar, datos, clave):
    """Mide el rendimiento de un algoritmo de cifrado/descifrado."""
    print(f"\n--- Probando rendimiento de {nombre_algoritmo} ---")

    proceso = psutil.Process(os.getpid())

    # Prueba de cifrado
    mem_antes = proceso.memory_info().rss
    cpu_antes = proceso.cpu_percent(interval=None)
    tiempo_inicio = time.time()

    nonce_o_frn, datos_cifrados = func_cifrar(datos, clave)

    tiempo_fin = time.time()
    cpu_despues = proceso.cpu_percent(interval=None)
    mem_despues = proceso.memory_info().rss

    tiempo_cifrado = tiempo_fin - tiempo_inicio
    uso_cpu = cpu_despues - cpu_antes
    uso_mem = (mem_despues - mem_antes) / (1024 * 1024)

    print(f"Cifrado:   Tiempo = {tiempo_cifrado:.4f}s | CPU = {uso_cpu:.2f}% | Memoria Pico = {uso_mem:.2f} MB")

    # Prueba de descifrado
    tiempo_inicio = time.time()
    datos_descifrados = func_descifrar(datos_cifrados, clave, nonce_o_frn)
    tiempo_fin = time.time()
    tiempo_descifrado = tiempo_fin - tiempo_inicio

    print(f"Descifrado: Tiempo = {tiempo_descifrado:.4f}s")

    # Verificación de integridad
    assert datos == datos_descifrados
    print("Verificación: El archivo original y el descifrado coinciden.")

    return tiempo_cifrado


def probar_entropia(nombre_archivo):
    """
    Calcula la entropía contando palabras binarias únicas de 2 bytes.
    Este método replica el análisis de vocabulario descrito en[cite: 579, 583].
    """
    print(f"\n--- Analizando entropía de '{nombre_archivo}' ---")
    try:
        with open(nombre_archivo, "rb") as f:
            contenido = f.read()
            # Creamos "palabras" de 2 bytes
            palabras_binarias = [contenido[i:i + 2] for i in range(0, len(contenido), 2)]
            conteo = Counter(palabras_binarias)
            vocabulario = len(conteo)
            print(f"Vocabulario (palabras únicas de 2 bytes): {vocabulario}")
            return vocabulario
    except FileNotFoundError:
        print(f"Error: Archivo '{nombre_archivo}' no encontrado.")
        return 0


# -----------------------------------------------------------------------------
# EJECUCIÓN DEL LABORATORIO
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    CLAVE_SECRETA = "miClaveSuperSegura123"

    # ===== EXPERIMENTO 1: ANÁLISIS DE RENDIMIENTO =====
    print("=" * 50)
    print("EXPERIMENTO 1: ANÁLISIS DE RENDIMIENTO")
    print("=" * 50)

    archivo_prueba = "prueba_10mb.bin"
    tamano_prueba = 10  # en MB
    crear_archivo_de_prueba(archivo_prueba, tamano_prueba)

    with open(archivo_prueba, "rb") as f:
        datos_prueba = f.read()

    # Instancia de SecureMD5
    securemd5_cipher = SecureMD5()

    # Pruebas
    probar_rendimiento("SecureMD5", securemd5_cipher.cifrar, securemd5_cipher.descifrar, datos_prueba, CLAVE_SECRETA)
    probar_rendimiento("AES-256-CTR", cifrar_aes_ctr, descifrar_aes_ctr, datos_prueba, CLAVE_SECRETA)
    probar_rendimiento("RC4", cifrar_rc4, descifrar_rc4, datos_prueba, CLAVE_SECRETA)

    # Limpieza
    os.remove(archivo_prueba)

    # ===== EXPERIMENTO 2: ANÁLISIS DE ENTROPÍA (SEGURIDAD) =====
    print("\n" + "=" * 50)
    print("EXPERIMENTO 2: ANÁLISIS DE ENTROPÍA (SEGURIDAD)")
    print("=" * 50)

    # Descargar un archivo de texto grande (Don Quijote), como en el paper [cite: 580]
    url_quijote = "https://www.gutenberg.org/files/2000/2000-0.txt"
    nombre_quijote = "quijote.txt"
    try:
        print(f"Descargando '{nombre_quijote}' para prueba de entropía...")
        response = requests.get(url_quijote)
        response.raise_for_status()  # Lanza un error si la descarga falla
        with open(nombre_quijote, 'wb') as f:
            f.write(response.content)
        print("Descarga completa.")

        with open(nombre_quijote, "rb") as f:
            datos_texto_plano = f.read()

        # Cifrar el texto con SecureMD5 y RC4
        _, datos_cifrados_smd5 = securemd5_cipher.cifrar(datos_texto_plano, CLAVE_SECRETA)
        with open("quijote_smd5.bin", "wb") as f:
            f.write(datos_cifrados_smd5)

        _, datos_cifrados_rc4 = cifrar_rc4(datos_texto_plano, CLAVE_SECRETA)
        with open("quijote_rc4.bin", "wb") as f:
            f.write(datos_cifrados_rc4)

        # Realizar las pruebas de entropía
        probar_entropia(nombre_quijote)
        probar_entropia("quijote_smd5.bin")
        probar_entropia("quijote_rc4.bin")

        # Limpieza
        os.remove(nombre_quijote)
        os.remove("quijote_smd5.bin")
        os.remove("quijote_rc4.bin")

    except requests.exceptions.RequestException as e:
        print(f"No se pudo descargar el archivo de prueba: {e}")
        print("Saltando la prueba de entropía.")