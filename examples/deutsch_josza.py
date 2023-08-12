import sys
sys.path.append('./..')

import numpy as np
from quantum_circuit import QuantumCircuit
from quantum_gate import NotGate, HadamardGate
        

class PhaseOracle(QuantumCircuit):
    def __init__(self, n_qubits, action='random'):
        super().__init__(n_qubits)
        if action == 'random':
            self.action = 'constant' if np.random.random() < 0.5 else 'balanced'
        else:
            self.action = action
    
    def act_on(self, state):
        new_state = state.copy()
        if self.action == 'balanced':
            pass
            for i in range(self.n_qubits-1):
               new_state = NotGate(self.n_qubits, self.n_qubits-1, i).act_on(new_state)
        elif self.action == 'constant':
            pass
        return new_state


def main():
    import matplotlib.pyplot as plt
    n_qubits = 4
    quantum_circuit = QuantumCircuit(n_qubits+1)
    quantum_circuit.add_gate(NotGate(n_qubits+1, n_qubits))
    for i in range(n_qubits+1):
        quantum_circuit.add_gate(HadamardGate(n_qubits+1, i))
    quantum_circuit.add_gate(PhaseOracle(n_qubits+1, action='random'))
    for i in range(n_qubits+1):
        quantum_circuit.add_gate(HadamardGate(n_qubits+1, i))

    quantum_circuit.execute()
    results = quantum_circuit.run(register=n_qubits, shots=1000)
    plt.hist(results)
    plt.show()


if __name__ == '__main__':
    main()
    print('Done')