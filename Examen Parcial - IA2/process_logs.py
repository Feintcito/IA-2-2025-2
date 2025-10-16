import pandas as pd
import re
import os


def parse_log_file(file_path):
    """
    Analiza un archivo de log de autenticación para extraer y estructurar los datos.

    Args:
        file_path (str): La ruta al archivo de log (ej. 'combined_auth.log').

    Returns:
        pandas.DataFrame: Un DataFrame con los datos de log procesados y estructurados.
    """
    # Expresión regular para capturar la estructura principal de cada línea de log
    # Grupo 1: Timestamp (Fecha y Hora)
    # Grupo 2: Hostname
    # Grupo 3: Proceso (ej. sshd, CRON)
    # Grupo 4: PID (ID del Proceso)
    # Grupo 5: Mensaje
    log_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+\d{2}:\d{2})\s+([\w-]+)\s+([\w-]+)(?:\[(\d+)\])?:\s+(.*)')

    parsed_data = []

    print(f"Iniciando el procesamiento del archivo: {file_path}...")

    with open(file_path, 'r') as f:
        for line in f:
            match = log_pattern.match(line)
            if not match:
                continue  # Ignora líneas que no coinciden con el formato esperado

            timestamp, hostname, process, pid, message = match.groups()

            # Extraer detalles específicos del mensaje usando regex adicionales
            user = None
            ip_address = None
            event_type = 'info'  # Valor por defecto

            # Intento de ataque o inicio de sesión fallido
            fail_match = re.search(
                r'Failed password for (invalid user )?(\S+) from (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', message)
            if fail_match:
                user = fail_match.group(2)
                ip_address = fail_match.group(3)
                event_type = 'Failed password'

            # Usuario inválido
            invalid_user_match = re.search(r'Invalid user (\S+) from (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', message)
            if not fail_match and invalid_user_match:
                user = invalid_user_match.group(1)
                ip_address = invalid_user_match.group(2)
                event_type = 'Invalid user'

            # Inicio de sesión exitoso
            accepted_match = re.search(r'Accepted password for (\S+) from (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})',
                                       message)
            if accepted_match:
                user = accepted_match.group(1)
                ip_address = accepted_match.group(2)
                event_type = 'Accepted password'

            # Sesión abierta o cerrada (CRON, su, etc.)
            session_match = re.search(r'session (opened|closed) for user (\S+)', message)
            if session_match:
                user = session_match.group(2)
                event_type = f'Session {session_match.group(1)}'

            parsed_data.append({
                'timestamp': timestamp,
                'hostname': hostname,
                'process': process,
                'pid': pid,
                'event_type': event_type,
                'user': user,
                'ip_address': ip_address,
                'raw_message': message.strip()
            })

    df = pd.DataFrame(parsed_data)

    # --- Limpieza y Formateo ---
    # Convertir la columna de timestamp a un objeto datetime real
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Rellenar valores nulos para mantener la consistencia
    df['pid'] = pd.to_numeric(df['pid'], errors='coerce').fillna(0).astype(int)
    df['user'] = df['user'].fillna('N/A')
    df['ip_address'] = df['ip_address'].fillna('N/A')

    print(f"Procesamiento completado. Se encontraron {len(df)} registros.")

    return df


# --- Ejecución del Script ---
if __name__ == '__main__':
    log_file = 'combined_auth.log'
    output_csv_file = 'processed_auth_logs.csv'

    if not os.path.exists(log_file):
        print(f"Error: El archivo '{log_file}' no se encuentra en el directorio.")
        print("Por favor, asegúrate de que el script y el archivo de log estén en la misma carpeta.")
    else:
        # Procesar el archivo de log
        processed_df = parse_log_file(log_file)

        # Guardar el DataFrame procesado en un archivo CSV
        processed_df.to_csv(output_csv_file, index=False)

        print(f"\nDatos procesados y guardados exitosamente en '{output_csv_file}'")

        # Mostrar las primeras 10 filas del resultado
        print("\n--- Muestra de los datos procesados: ---")
        print(processed_df.head(10))

        # Mostrar un resumen de los tipos de eventos encontrados
        print("\n--- Resumen de tipos de evento: ---")
        print(processed_df['event_type'].value_counts())