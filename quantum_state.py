import warnings
import numpy as np
from exceptions import NormZeroError


class QuantumState:
    def __init__(self, *, state=None, n_qubits=None, configuration=None):
        if state is not None:
            self._state = state
            self.n_dimensions = len(state)
            self.n_qubits = round(np.log2(self.n_dimensions))
            self.normalize()
        elif n_qubits is not None:
            self.n_qubits = n_qubits
            self.n_dimensions = 2**n_qubits
            if configuration is not None: 
                self._state = np.zeros(self.n_dimensions, dtype=complex)
                for key, item in configuration.items():
                    self._state[int(key, 2)] = item
                self.normalize()
            else:
                self._state = np.zeros(self.n_dimensions, dtype=complex)

    def normalize(self):
        norm2 = 0
        for coefficient in self._state:
            norm2 += np.abs(coefficient)**2
        if norm2 < 1e-15:
            raise NormZeroError
        if norm2 != 1.:
            warnings.warn('Attention: State is not normalized. Performing normalization automatically.')
            self._state /= np.sqrt(norm2)

    def copy(self):
        return QuantumState(state=self._state.copy())

    def __getitem__(self, index):
        return self._state[index]
    
    def __setitem__(self, index, value):
        self._state[index] = value

    def __repr__(self):
        repr = ''
        for index, el in enumerate(self._state):
            if np.abs(el) > 1e-15:
                label = bin(index)[2:]
                label = (self.n_qubits-len(label))*'0'+label
                repr += f'{label}: {el}\n'
        return repr
    
    def __len__(self):
        return len(self._state)
    
    def __imul__(self, factor):
        self._state *= factor
        return self
    

