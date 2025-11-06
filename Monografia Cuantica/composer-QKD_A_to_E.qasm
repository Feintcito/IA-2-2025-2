// GENERADO POR QISKIT (BASADO EN SEED 42, N=10)
// CASO 2 - PARTE 1: ALICE -> EVE

OPENQASM 2.0;
include "qelib1.inc";

qreg q[10];
creg c[10];

// --- 2. Codificación de Alice ---
// (Es idéntica a la del Caso 1)
x q[1];
h q[4];
x q[5];
h q[6];
h q[7];
h q[8];
x q[9];
barrier q;

// --- 3. Medición de Eve ---
// (Eve usa sus bases 'eve_bases')
h q[0];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];

// --- 4. Medición Final (de Eve) ---
measure q -> c;