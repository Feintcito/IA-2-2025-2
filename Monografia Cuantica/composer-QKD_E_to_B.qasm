// GENERADO POR QISKIT (BASADO EN SEED 42, N=10)
// CASO 2 - PARTE 2: EVE -> BOB

OPENQASM 2.0;
include "qelib1.inc";

qreg q[10];
creg c[10];

// --- 3. Re-envío de Eve ---
// Eve usa sus bits medidos [0,0,0,0,0,0,0,0,1,1]
// Eve usa sus bases [1,0,1,1,1,1,1,1,1,1]

// q[0]: bit=0, base=1 -> h
h q[0];
// q[1]: bit=0, base=0 -> (nada)
// q[2]: bit=0, base=1 -> h
h q[2];
// q[3]: bit=0, base=1 -> h
h q[3];
// q[4]: bit=0, base=1 -> h
h q[4];
// q[5]: bit=0, base=1 -> h
h q[5];
// q[6]: bit=0, base=1 -> h
h q[6];
// q[7]: bit=0, base=1 -> h
h q[7];
// q[8]: bit=1, base=1 -> x, h
x q[8];
h q[8];
// q[9]: bit=1, base=1 -> x, h
x q[9];
h q[9];
barrier q;

// --- 4. Medición de Bob ---
// Bob usa sus bases [1,1,0,0,1,0,0,1,0,1]
// q[0]: base=1 -> h
h q[0];
// q[1]: base=1 -> h
h q[1];
// q[2]: base=0 -> (nada)
// q[3]: base=0 -> (nada)
// q[4]: base=1 -> h
h q[4];
// q[5]: base=0 -> (nada)
// q[6]: base=0 -> (nada)
// q[7]: base=1 -> h
h q[7];
// q[8]: base=0 -> (nada)
// q[9]: base=1 -> h
h q[9];

// --- 5. Medición Final (de Bob) ---
measure q -> c;