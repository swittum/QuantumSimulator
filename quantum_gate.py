from abc import ABC, abstractmethod
import numpy as np
from numpy.linalg import eigvals
from utils import get_nth_bit, invert_nth_bit
from quantum_state import QuantumState


class QuantumGate(ABC):
    def __init__(self, n_qubits, n_act=None, n_control=None):
        self.n_qubits = n_qubits
        self.n_dimensions = 2**n_qubits
        self.n_act = n_act
        self.n_control = n_control

    def __repr__(self):
        return str(self.get_matrix())

    @abstractmethod
    def act_on(self, state):
        """ Override this in child classes """

    def get_matrix(self):
        matrix = np.zeros((self.n_dimensions, self.n_dimensions), dtype=complex)
        for i in range(self.n_dimensions):
            state = QuantumState(n_qubits=self.n_qubits)
            state[i] = 1
            matrix[:, i] = self.act_on(state)
        return matrix


class NotGate(QuantumGate):
    def __init__(self, n_qubits, n_act, n_control=None):
        super().__init__(n_qubits, n_act, n_control)

    def act_on(self, state):
        # new_state = np.zeros(len(state), dtype=complex)
        new_state = QuantumState(n_qubits=state.n_qubits)
        for state_label in range(self.n_dimensions):
            state_control = get_nth_bit(state_label, self.n_control) if self.n_control is not None else 1
            if state_control == 1:
                state_inv_label = invert_nth_bit(state_label, self.n_act)
                new_state[state_label] = state[state_inv_label]
                new_state[state_inv_label] = state[state_label]
            else:
                new_state[state_label] = state[state_label]
        return new_state
    

class HadamardGate(QuantumGate):
    def __init__(self, n_qubits, n_act, n_control=None):
        super().__init__(n_qubits, n_act, n_control)

    def act_on(self, state):
        new_state = np.zeros(self.n_dimensions, dtype=complex)
        for state_label in range(self.n_dimensions):
            state_control = get_nth_bit(state_label, self.n_control) if self.n_control is not None else 1
            if state_control == 1:
                state_act = get_nth_bit(state_label, self.n_act)
                new_state[state_label] += (-1)**state_act*state[state_label]/np.sqrt(2)
                state_inv_label = invert_nth_bit(state_label, self.n_act)
                new_state[state_inv_label] += state[state_label]/np.sqrt(2)
            else:
                new_state[state_label] = state[state_label]
        return new_state
    

class PauliXGate(QuantumGate):
    def __init__(self, n_qubits, n_act, n_control=None):
        super().__init__(n_qubits, n_act, n_control)
    
    def act_on(self, state):
        new_state = np.zeros(self.n_dimensions, dtype=complex)
        for state_label in range(self.n_dimensions):
            state_control = get_nth_bit(state_label, self.n_control) if self.n_control is not None else 1
            if state_control == 1:
                new_state[invert_nth_bit(state_label, self.n_act)] = state[state_label]
            else:
                new_state[state_label] = state[state_label]
        return new_state
    

class PauliYGate(QuantumGate):
    def __init__(self, n_qubits, n_act, n_control=None):
        super().__init__(n_qubits, n_act, n_control)

    def act_on(self, state):
        new_state = np.zeros(self.n_dimensions, dtype=complex)
        for state_label in range(self.n_dimensions):
            state_control = get_nth_bit(state_label, self.n_control) if self.n_control is not None else 1
            if state_control == 1:
                state_act = get_nth_bit(state_label, self.n_act)
                new_state[invert_nth_bit(state_label, self.n_act)] = (-1)**state_act*1j*state[state_label]
            else:
                new_state[state_label] = state[state_label]
        return new_state
    

class PauliZGate(QuantumGate):
    def __init__(self, n_qubits, n_act, n_control=None):
        super().__init__(n_qubits, n_act, n_control)

    def act_on(self, state):
        new_state = np.zeros(self.n_dimensions, dtype=complex)
        for state_label in range(self.n_dimensions):
            state_control = get_nth_bit(state_label, self.n_control) if self.n_control is not None else 1
            if state_control == 1:
                state_act = get_nth_bit(state_label, self.n_act)
                new_state[state_label] = (-1)**state_act*state[state_label]
            else:
                new_state[state_label] = state[state_label]
        return new_state
    

class XRotationGate(QuantumGate):
    def __init__(self, n_qubits, n_act, phi, n_control=None):
        super().__init__(n_qubits, n_act, n_control)
        self.phi = phi

    def act_on(self, state):
        new_state = np.zeros(self.n_dimensions, dtype=complex)
        for state_label in range(self.n_dimensions):
            state_control = get_nth_bit(state_label, self.n_control) if self.n_control is not None else 1
            if state_control == 1:
                new_state[state_label] += np.cos(self.phi/2)*state[state_label]
                new_state[invert_nth_bit(state_label, self.n_act)] += -1j*np.sin(self.phi/2)*state[state_label]
            else:
                new_state[state_label] = state[state_label]
        return new_state
    

class YRotationGate(QuantumGate):
    def __init__(self, n_qubits, n_act, phi, n_control=None):
        super().__init__(n_qubits, n_act, n_control)
        self.phi = phi

    def act_on(self, state):
        new_state = np.zeros(self.n_dimensions, dtype=complex)
        for state_label in range(self.n_dimensions):
            state_control = get_nth_bit(state_label, self.n_control) if self.n_control is not None else 1
            if state_control == 1:
                state_act = get_nth_bit(state_label, self.n_act)
                new_state[state_label] += np.cos(self.phi/2)*state[state_label]
                new_state[invert_nth_bit(state_label, self.n_act)] += (-1)**state_act*np.sin(self.phi/2)*state[state_label]
            else:
                new_state[state_label] = state[state_label]
        return new_state
    

class ZRotationGate(QuantumGate):
    def __init__(self, n_qubits, n_act, phi, n_control=None):
        super().__init__(n_qubits, n_act, n_control)
        self.phi = phi

    def act_on(self, state):
        new_state = np.zeros(self.n_dimensions, dtype=complex)
        for state_label in range(self.n_dimensions):
            state_control = get_nth_bit(state_label, self.n_control) if self.n_control is not None else 1
            if state_control == 1:
                state_act = get_nth_bit(state_label, self.n_act)
                arg = (-1)**(state_act+1)*self.phi/2
                new_state[state_label] = np.exp(1j*arg)*state[state_label]
            else:
                new_state[state_label] = state[state_label]
        return new_state


class RotationGate(QuantumGate):
    def __init__(self, n_qubits, n_act, phi, n_control=None):
        super().__init__(n_qubits, n_act, n_control)
        self.phi = phi

    def act_on(self, state):
        new_state = np.zeros(self.n_dimensions, dtype=complex)
        w = np.exp(1j*self.phi)
        for state_label in range(self.n_dimensions):
            state_control = get_nth_bit(state_label, self.n_control) if self.n_control is not None else 1
            if state_control == 1:
                state_act = get_nth_bit(state_label, self.n_act)
                new_state[state_label] = w**state_act*state[state_label]
            else:
                new_state[state_label] = state[state_label]
        return new_state
    

class QFTGate(QuantumGate):
    def __init__(self, n_qubits):
        super().__init__(n_qubits)

    def act_on(self, state):
        new_state = state.copy()
        for qubit in range(self.n_qubits):
            new_state = HadamardGate(self.n_qubits, qubit).act_on(new_state)
            for i in range(1, self.n_qubits-qubit):
                phi = 2*np.pi/2**(1+i)
                new_state = RotationGate(self.n_qubits, qubit, phi, i+qubit).act_on(new_state)
        return new_state
    

class InverseQFTGate(QuantumGate):
    def __init__(self, n_qubits):
        super().__init__(n_qubits)

    def act_on(self, state):
        new_state = state.copy()
        for qubit in range(self.n_qubits-1, -1, -1):
            for i in range(1, self.n_qubits-qubit):
                phi = -2*np.pi/2**(1+i)
                new_state = RotationGate(self.n_qubits, qubit, phi, i+qubit).act_on(new_state)
            new_state = HadamardGate(self.n_qubits, qubit).act_on(new_state)
        return new_state
    

if __name__ == '__main__':
    pass