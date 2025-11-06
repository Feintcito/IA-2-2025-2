import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.primitives import Sampler

# --- 1. Configuración ---
N_BITS = 20  # Aumentamos los bits para ver mejor el efecto
np.random.seed(41)

# --- 2. Alice prepara sus qubits ---
alice_bits = np.random.randint(2, size=N_BITS)
alice_bases = np.random.randint(2, size=N_BITS)
print(f"Bits de Alice:   {alice_bits}")
print(f"Bases de Alice:  {alice_bases} (0=Z, 1=X)\n")


def encode_qubits(bits, bases):
    qc = QuantumCircuit(N_BITS, N_BITS)
    for i in range(N_BITS):
        if bits[i] == 1: qc.x(i)
        if bases[i] == 1: qc.h(i)
    return qc


alice_circuit = encode_qubits(alice_bits, alice_bases)
alice_circuit.barrier()  # Fin de la sección de Alice

# --- 3. EVE INTERCEPTA ---
# Eve tiene que adivinar sus propias bases
eve_bases = np.random.randint(2, size=N_BITS)
print(f"Bases de EVE:    {eve_bases} (0=Z, 1=X)\n")


def eve_intercept_measure(qc, bases):
    """Eve mide los qubits de Alice"""
    for i in range(N_BITS):
        if bases[i] == 1:
            qc.h(i)  # Eve aplica H si quiere medir en Base X
        qc.measure(i, i)  # Eve mide y colapsa el estado
    return qc


# Eve toma el circuito de Alice y añade sus mediciones
eve_circuit = eve_intercept_measure(alice_circuit, eve_bases)
eve_circuit.barrier()  # Fin de la medición de Eve

# ¡IMPORTANTE! Eve no puede pasar el circuito.
# Tiene que SIMULARLO para obtener sus resultados y RE-ENVIAR.

# --- SIMULACIÓN HASTA EVE ---
# Ejecutamos el circuito hasta Eve para saber qué midió
sampler_sim = Sampler()
job_eve = sampler_sim.run(eve_circuit, shots=1)
result_eve = job_eve.result()
counts_eve = result_eve.quasi_dists[0].binary_probabilities()
eve_results_str = list(counts_eve.keys())[0]
eve_measured_bits = list(reversed([int(bit) for bit in eve_results_str]))

print(f"Eve midió:       {eve_measured_bits}")


# --- EVE RE-ENVÍA ---
def eve_resend(bits, bases):
    """Eve prepara NUEVOS qubits basados en lo que midió"""
    qc = QuantumCircuit(N_BITS, N_BITS)
    for i in range(N_BITS):
        if bits[i] == 1: qc.x(i)  # Prepara el bit que ELLA midió
        if bases[i] == 1: qc.h(i)  # Usando la base que ELLA usó para medir
    return qc


# Este es el nuevo circuito que Bob recibe
eve_resend_circuit = eve_resend(eve_measured_bits, eve_bases)
eve_resend_circuit.barrier()  # Fin de la sección de Eve

# --- 4. Bob mide los qubits de Eve ---
bob_bases = np.random.randint(2, size=N_BITS)
print(f"Bases de Bob:    {bob_bases} (0=Z, 1=X)\n")


def measure_qubits(qc, bases):
    for i in range(N_BITS):
        if bases[i] == 1: qc.h(i)
        qc.measure(i, i)
    return qc


# Bob añade sus mediciones al *nuevo* circuito que Eve le envió
full_circuit_with_eve = measure_qubits(eve_resend_circuit, bob_bases)

# --- 5. Simulación (Bob) ---
job_bob = sampler_sim.run(full_circuit_with_eve, shots=1)
result_bob = job_bob.result()
counts_bob = result_bob.quasi_dists[0].binary_probabilities()
bob_results_str = list(counts_bob.keys())[0]
bob_results = list(reversed([int(bit) for bit in bob_results_str]))

print(f"Resultados Bob (ordenados): {bob_results}\n")

# --- 6. Comparación de Bases (Sondeo) ---
final_key_alice = []
final_key_bob = []
# Registramos las bases para ver el QBER
error_count = 0
key_size = 0

for i in range(N_BITS):
    # Paso 1: Bob y Alice comparan bases (canal público)
    if alice_bases[i] == bob_bases[i]:
        # Coincidieron. Estos bits formarán la clave.
        key_size += 1
        alice_bit = int(alice_bits[i])
        bob_bit = bob_results[i]

        final_key_alice.append(alice_bit)
        final_key_bob.append(bob_bit)

        # Paso 2: Verificamos si los bits coinciden
        if alice_bit != bob_bit:
            error_count += 1

print("--- Sondeo (Sifting) y Detección de Eve ---")
print(f"Bases Alice: {alice_bases}")
print(f"Bases Bob:   {bob_bases}")
print(f"Bases Eve:   {eve_bases}")

print(f"\nClave de Alice (filtrada): {final_key_alice}")
print(f"Clave de Bob (filtrada):   {final_key_bob}")

# --- 7. Verificación ---
if final_key_alice == final_key_bob:
    print("\n✅ Éxito: La clave secreta compartida es:")
    print(f"   {final_key_alice}")
    print("¡Eve tuvo suerte y adivinó todas las bases correctas (muy improbable!)")
else:
    print("\n❌ ¡ALARMA! ¡ESPÍA DETECTADO!")

    qber = (error_count / key_size) * 100
    print(f"   Tamaño de clave filtrada: {key_size} bits")
    print(f"   Errores encontrados:      {error_count}")
    print(f"   Tasa de Error (QBER):     {qber:.2f}%")
    print("\n   REMEDIACIÓN: ¡Descartar esta clave inmediatamente!")