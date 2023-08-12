import numpy as np
from quantum_gate import *
from quantum_state import QuantumState


class QuantumCircuit(QuantumGate):
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.n_dimensions = 2**n_qubits
        self.gates = []
        self.psi0 = QuantumState(state=[1]+(self.n_dimensions-1)*[0])

    def add_gate(self, gate):
        self.gates.append(gate)

    def act_on(self, state=None):
        new_state = self.psi0.copy() if state is None else state
        for gate in self.gates:
            new_state = gate.act_on(new_state)
        return new_state
    
    def execute(self, register=None, state=None):
        if register is None: register=self.n_dimensions
        new_state = self.act_on(state)
        probs = np.array([np.abs(coefficient)**2 for coefficient in new_state])
        probs_cumulated = np.cumsum(probs)
        p = np.random.random()
        for i, prob in enumerate(probs_cumulated):
            if p < prob:
                return i % register
        return i % register
    
    def run(self, register=None, shots=100, state=None):
        if register is None: register=self.n_dimensions
        results = []
        for _ in range(shots):
            results.append(self.execute(register, state))
        return np.array(results)

    
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     quantum_circuit = QuantumCircuit(10)
#     for i in range(10):
#         quantum_circuit.add_gate(HadamardGate(10, i))
#     out = quantum_circuit.run(shots=1000)
#     plt.hist(out)
#     plt.show()
    
