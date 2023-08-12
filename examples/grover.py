import sys
sys.path.append('./..')

import numpy as np
from quantum_circuit import QuantumCircuit
from quantum_gate import HadamardGate
from quantum_state import QuantumState
        

class PhaseOracle(QuantumCircuit):
    def __init__(self, n_qubits, state_label, invert=True):
        super().__init__(n_qubits)
        self.state_label = state_label
        self.invert = invert
    
    def act_on(self, state):
        new_state = state.copy()
        if self.invert: new_state *= (-1)
        new_state[self.state_label] *= (-1)
        return new_state
    

class Diffuser(QuantumCircuit):
    def __init__(self, n_qubits, state_label):
       super().__init__(n_qubits)
       self.state_label = state_label

    def act_on(self, state):
        new_state = state.copy()
        new_state = PhaseOracle(self.n_qubits, self.state_label, False).act_on(new_state)
        for i in range(self.n_qubits):
            new_state = HadamardGate(self.n_qubits, i).act_on(new_state)
        new_state = PhaseOracle(self.n_qubits, 0, True).act_on(new_state)
        for i in range(self.n_qubits):
            new_state = HadamardGate(self.n_qubits, i).act_on(new_state)
        return new_state
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n_qubits = 4
    tag = 15
    N = int(np.pi/(4*np.arcsin(1/np.sqrt(2**n_qubits)))-1/2)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.add_gate(HadamardGate(n_qubits, i))
    for _ in range(N):
        qc.add_gate(Diffuser(n_qubits, state_label=tag))
    results = qc.run(register=2**n_qubits, shots=100)
    plt.hist(results, bins=np.arange(16))
    plt.show()


