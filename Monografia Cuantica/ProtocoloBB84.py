import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.primitives import Sampler

# --- 1. Configuración ---
N_BITS = 10
np.random.seed(41)

# --- 2. Alice prepara sus qubits ---
alice_bits = np.random.randint(2, size=N_BITS)
alice_bases = np.random.randint(2, size=N_BITS)

print(f"Bits de Alice:   {alice_bits}")
print(f"Bases de Alice:  {alice_bases} (0=Z, 1=X)\n")

def encode_qubits(bits, bases):
    qc = QuantumCircuit(N_BITS, N_BITS)
    for i in range(N_BITS):
        if bits[i] == 1:
            qc.x(i)
        if bases[i] == 1:
            qc.h(i)
    return qc

alice_circuit = encode_qubits(alice_bits, alice_bases)
alice_circuit.barrier()

# --- 3. Bob mide los qubits ---
bob_bases = np.random.randint(2, size=N_BITS)
print(f"Bases de Bob:    {bob_bases} (0=Z, 1=X)\n")

def measure_qubits(qc, bases):
    for i in range(N_BITS):
        if bases[i] == 1:
            qc.h(i)
        qc.measure(i, i)
    return qc

full_circuit = measure_qubits(alice_circuit, bob_bases)

# --- 4. Simulación (¡CORREGIDO!) ---
print("Iniciando el simulador AerSampler...")
sampler_sim = Sampler()

job = sampler_sim.run(full_circuit, shots=1)
result = job.result()

counts = result.quasi_dists[0].binary_probabilities()
bob_results_str = list(counts.keys())[0]

# Convertimos 'c9...c0' a [c9, ..., c0]
bob_results_list_reversed = [int(bit) for bit in bob_results_str]

# Revertimos la lista para que sea [c0, c1, ..., c9]
bob_results = list(reversed(bob_results_list_reversed))

print(f"Resultados Bob (ordenados): {bob_results}\n")


# --- 5. Comparación de Bases (Ahora funciona) ---
final_key_alice = []
final_key_bob = []

for i in range(N_BITS):
    if alice_bases[i] == bob_bases[i]:
        final_key_alice.append(alice_bits[i])
        # Ahora bob_results[i] es c[i], que es correcto
        final_key_bob.append(bob_results[i])

print("--- Sondeo (Sifting) ---")
print(f"Coincidencias (A=B): {alice_bases == bob_bases}")
print(f"Clave de Alice (filtrada): {final_key_alice}")
print(f"Clave de Bob (filtrada):   {final_key_bob}")

# --- 6. Verificación ---
# (Convertimos los np.int32 a int nativos para la comparación)
final_key_alice_py = [int(bit) for bit in final_key_alice]

if final_key_alice_py == final_key_bob:
    print("\n✅ Éxito: La clave secreta compartida es:")
    print(f"   {final_key_alice_py}")
else:
    print("\n❌ Fallo: Las claves no coinciden (¡Revisar la lógica!)")