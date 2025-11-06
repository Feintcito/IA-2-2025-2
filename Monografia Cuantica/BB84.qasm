// GENERADO POR QISKIT (BASADO EN SEED 42, N=10)
// CASO 1: ALICE -> BOB

OPENQASM 2.0;
include "qelib1.inc";

qreg q[10];
creg c[10];

// --- 2. Codificación de Alice ---
// q[0]: bit=0, base=0 -> (nada)
// q[1]: bit=1, base=0 -> x
x q[1];
// q[2]: bit=0, base=0 -> (nada)
// q[3]: bit=0, base=0 -> (nada)
// q[4]: bit=0, base=1 -> h
h q[4];
// q[5]: bit=1, base=0 -> x
x q[5];
// q[6]: bit=0, base=1 -> h
h q[6];
// q[7]: bit=0, base=1 -> h
h q[7];
// q[8]: bit=0, base=1 -> h
h q[8];
// q[9]: bit=1, base=0 -> x
x q[9];
barrier q;

// --- 3. Medición de Bob ---
// q[0]: base=1 -> h
h q[0];
// q[1]: base=0 -> (nada)
// q[2]: base=1 -> h
h q[2];
// q[3]: base=1 -> h
h q[3];
// q[4]: base=1 -> h
h q[4];
// q[5]: base=1 -> h
h q[5];
// q[6]: base=1 -> h
h q[6];
// q[7]: base=1 -> h
h q[7];
// q[8]: base=1 -> h
h q[8];
// q[9]: base=1 -> h
h q[9];

// --- 4. Medición Final ---
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];