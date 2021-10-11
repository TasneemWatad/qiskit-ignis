# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Gate set tomography preparation and measurement basis
"""

# Needed for functions
import functools
from typing import Tuple, Callable, Union, Optional, Dict
import numpy as np
import itertools
# Import QISKit classes
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import U2Gate, XGate, YGate, HGate, CXGate
from qiskit.quantum_info import PTM
from qiskit.quantum_info import Pauli


class GateSetBasis:

    """
    This class contains the gateset data needed to perform gateset tomgography.
    The gateset tomography data consists of two sets, G and F
    G = (G1,...,Gn) is the set of the gates we wish to characterize
    F = (F1,..,Fm) is a set of SPAM (state preparation and measurement)
    circuits. The SPAM circuits are constructed from the elements of G
    and all the SPAM combinations are appended before and after elements of G
    when performing the tomography measurements
    (i.e. we measure all circuits of the form Fi * Gk * Fj)

    The gateset data is comprised of four elements:
    1) The labels (strings) of the gates
    2) A function f(circ, qubit, op)
        which adds to circ at qubit the gate labeled by op
    3) The labels of the SPAM circuits for the gate set tomography
    4) For SPAM label, tuple of gate labels for the gates in this SPAM circuit
    """
    def __init__(self,
                 name: str,
                 gates: Dict[str, Union[Callable, Gate]],
                 spam: Dict[str, Tuple[str]],
                 num_qubits: int
                 ):
        """
        Initialize the gate set basis data

        Args:
            name: Name of the basis.
            gates: The gate data (name -> gate/gate function)
            spam: The spam data (name -> sequence of gate names)
        """
        self.num_qubits = num_qubits
        self.name = name
        self.gate_labels = list(gates.keys())
        self.gates = gates
        self.gate_matrices = {name: np.real(gate_matrix(self.num_qubits,gate))
                              for (name, gate) in gates.items()}
        self.spam_labels = tuple(sorted(spam.keys()))
        self.spam_spec = spam

        if(gram_matrix_rank(self.num_qubits,list(self.spam_spec.values()),self.gates)[0]<(2**self.num_qubits)**2):
            raise QiskitError("Gram matrix is rank deficient")

    def add_gate(self, gate: Union[Callable, Gate], name: Optional[str] = None):
        """Adds a new gate to the gateset
            Args:
                gate: Either a qiskit gate object or a function taking
                (QuantumCircuit, QuantumRegister)
                and adding the gate to the circuit
                name: the name of the new gate
            Raises:
                RuntimeError: If the gate is given as a function but without
                a name.
        """
        if name is None:
            if isinstance(gate, Gate):
                name = gate.name
            else:
                raise RuntimeError("Gate name is missing")
        self.gate_labels.append(name)
        self.gates[name] = gate
        self.gate_matrices[name] = gate_matrix(self.num_qubits,gate)

    def add_gate_to_circuit(self, circ: QuantumCircuit,
                            qubits: QuantumRegister,
                            op: str
                            ):

        """
        Adds the gate op to circ at qubits
        Args:
            circ: the circuit to apply op on
            qubits: qubit to be operated on
            op: gate name
        Raises:
            RuntimeError: if `op` does not describe a gate
        """
        if op not in self.gates:
            raise RuntimeError("{} is not a SPAM circuit".format(op))
        gate = self.gates[op]
        if callable(gate):
            gate(circ, *qubits)
        if isinstance(gate, Gate):
            circ.append(gate, qubits, [])

    def add_spam_to_circuit(self,
                            circ: QuantumCircuit,
                            qubits: QuantumRegister,
                            op: str
                            ):
        """
        Adds the SPAM circuit op to circ at qubits

        Args:
            circ: the circuit to apply op on
            qubits: qubits to be operated on
            op: SPAM circuit name

        Raises:
            RuntimeError: if `op` does not describe a SPAM circuit
        """
        if op not in self.spam_spec:
            raise RuntimeError("{} is not a SPAM circuit".format(op))
        op_gates = self.spam_spec[op]
        for gate_name in op_gates:
            self.add_gate_to_circuit(circ, qubits, gate_name)

    def spam_matrix(self, label: str) -> np.array:
        """
        Returns the matrix corresponding to a spam label
        Every spam is a sequence of gates, and so the result matrix
        is the product of the matrices corresponding to those gates
        Params:
            label: Spam label
        Returns:
            The corresponding matrix
        """
        spec = self.spam_spec[label]
        f_matrices = [self.gate_matrices[gate_label] for gate_label in spec]
        result = functools.reduce(lambda a, b: a @ b, f_matrices)
        return result


def default_gateset_basis(num_qubits):
    """Returns a default tomographically-complete gateset basis

       Args:
       num_qubits: The number of qubits. This takes one of the two values: 1 for the case of performing gateset
       tomography on one single qubit and 2 for the two-qubit case.


       Return value: The gateset given as example 3.4.1 in arXiv:1509.02921

       Raises:
          QiskitError: if num_qubits is larger than 2.
    """

    if num_qubits > 2:
        raise QiskitError("No default basis for three qubits or more")


    default_gates_single = {
        'Id': lambda circ, qubit: None,
        'X_Rot_90': lambda circ, qubit: circ.append(U2Gate(-np.pi / 2, np.pi / 2), [qubit]),
        'Y_Rot_90': lambda circ, qubit: circ.append(U2Gate(0, 0), [qubit])
    }

    default_spam_single = {
        'F0': ('Id',),
        'F1': ('X_Rot_90',),
        'F2': ('Y_Rot_90',),
        'F3': ('X_Rot_90', 'X_Rot_90')
    }

    ####################################

    # Two-Qubit
    # AB stands for acting with A gate on qubit2 and with gate B on qubit1.
    # Ex. XI- Apply X on qubit 1, and do nothing for qubit2.

    default_gates_two = {

        'IdId': lambda circ, qubit1, qubit2: None,
        'X I': lambda circ, qubit1, qubit2: circ.append(XGate(), [qubit2]),
        'I X': lambda circ, qubit1, qubit2: circ.append(XGate(), [qubit1]),
        'H I': lambda circ, qubit1, qubit2: circ.append(HGate(), [qubit2]),
        'I H': lambda circ, qubit1, qubit2: circ.append(HGate(), [qubit1]),
        'X_Rot_90 I': lambda circ, qubit1, qubit2: circ.append(U2Gate(-np.pi / 2, np.pi / 2), [qubit2]),
        'I X_Rot_90': lambda circ, qubit1, qubit2: circ.append(U2Gate(-np.pi / 2, np.pi / 2), [qubit1]),
        # qubit1 is the ctrl qubit, qubit2 is the target
    }

    default_spam_two = {
        'F0': ('IdId',),
        'F1': ('X I',),
        'F2': ('I X',),
        'F3': ('H I',),
        'F4': ('I H',),
        'F5': ('I X_Rot_90',),
        'F6': ('X_Rot_90 I',),
        'F7': ('X I', 'I X'),
        'F8': ('X I', 'I H'),
        'F9': ('X I', 'I X_Rot_90'),
        'F10': ('I X', 'H I'),
        'F11': ('I X', 'X_Rot_90 I'),
        'F12': ('H I', 'I H'),
        'F13': ('H I', 'I X_Rot_90'),
        'F14': ('I H', 'X_Rot_90 I'),
        'F15': ('I X_Rot_90', 'X_Rot_90 I')
    }

    return GateSetBasis('Default GST', default_gates_single, default_spam_single,
                        1) if num_qubits == 1 else GateSetBasis('Default GST', default_gates_two, default_spam_two, 2)


def gate_matrix(num_qubits, gate):
    """Gets a PTM representation of the gate"""
    if isinstance(gate, Gate):
        return PTM(gate).data
    if callable(gate):
        c = QuantumCircuit(num_qubits)
        qubits = []
        for i in range(num_qubits):
            qubits.append(c.qubits[i])
        gate(c, *qubits)
        return PTM(c).data
    return None


def spam_matrix(spec, basis_gates, num_qubits) -> np.array:
    """
    Returns the matrix corresponding to a spam label
    Every spam is a sequence of gates, and so the result matrix
    is the product of the matrices corresponding to those gates
    Params:
        label: Spam label
    Returns:
        The corresponding matrix
    """
    f_matrices = [gate_matrix(num_qubits, basis_gates[gate_label]) for gate_label in spec]
    result = functools.reduce(lambda a, b: a @ b, f_matrices)
    return result


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def find_spam_gates_from_gatesetbasis(basis_gates, rho, num_qubits):
    """Given a basis_gates, the function returns a set of spam labels that form an informationally complete
    set of independent states that yield a full rank gram matrix"""
    basis_gates_labels = list(basis_gates.keys())
    allstrings = [[(','.join(p)).split(',') for p in itertools.product(basis_gates_labels, repeat=i)] for i in range(4)]
    # basis_gates_labels[0] should always be the identity
    Selected = [(basis_gates_labels[0],), ]
    matrix = []
    for j in range(len(Selected)):
        matrix.append(spam_matrix(Selected[j], basis_gates, num_qubits) @ rho)
    for k in range(1, len(allstrings)):
        for i in range(len(allstrings[k])):
            spam_temp = spam_matrix(allstrings[k][i], basis_gates, num_qubits)
            matrix2 = np.copy(matrix)
            matrix2 = np.vstack([matrix2, spam_temp @ rho])
            # print(np.linalg.matrix_rank(matrix2))
            if (np.linalg.matrix_rank(matrix2) > np.linalg.matrix_rank(matrix)):
                if (check_symmetric(spam_temp == True)):
                    # print(1,i,'yes',np.linalg.matrix_rank(matrix2))
                    matrix.append(spam_temp @ rho)
                    Selected.append(allstrings[k][i])
                    if (len(Selected) == (2 ** num_qubits) ** 2):
                        return Selected
    raise QiskitError(
        f'No informationally complete fidicuals set with a full rank corresponding gram matrix can be '
        f'found for the gateset basis"{repr(basis_gates)}"')


def Pauli_strings(num_qubits):
    """Returns the matrix representation of Pauli strings of size=num_qubits. e.g., for num_qubits=2, it returns
    the matrix representations of ['II','IX','IY','IZ,'XI','YI',...]"""
    pauli_labels = ['I', 'X', 'Y', 'Z']
    Pauli_strings_matrices = [Pauli(''.join(p)).to_matrix() for p in
                              itertools.product(pauli_labels, repeat=num_qubits)]
    return Pauli_strings_matrices


def default_init_state(num_qubits):
    """Returns the PTM representation of the usual ground state |00...>"""
    d = np.power(2, num_qubits)

    # matrix representation of #rho in regular Hilbert space
    matrix_init_0 = np.zeros((d, d), dtype=complex)
    matrix_init_0[0, 0] = 1

    # decompoition into Pauli strings basis (PTM representation)
    matrix_init_pauli = [np.trace(np.dot(matrix_init_0, Pauli_strings(num_qubits)[i])) for i in
                         range(np.power(d, 2))]
    return np.reshape(matrix_init_pauli, (np.power(d, 2), 1))


def gram_matrix_rank(num_qubits, spam_gates_labels, basis_gates):
    """Returns the rank of the gram matrix and its singular values"""

    rho = default_init_state(num_qubits)
    ds = (2 ** num_qubits) ** 2
    g = np.zeros((ds, ds), dtype=complex)
    for i in range(ds):
        for j in range(ds):
            E = rho.reshape((1, ds))
            g[i][j] = (E @ (
                    spam_matrix(spam_gates_labels[i], basis_gates, num_qubits) @ spam_matrix(spam_gates_labels[j],
                                                                                             basis_gates,
                                                                                             num_qubits) @ rho))[0]
    # rank
    rank = np.linalg.matrix_rank(g)
    # singular values
    _, s, _ = np.linalg.svd(g, full_matrices=True)

    return rank, s


def gatesetbasis_constrction_from_basis_gates(basis_gates: Dict, num_qubits: int):
    """Returns a default tomographically-complete gateset basis and the gram matrix singular values spectrum.
       Given a Dict of basis gates, it constructs a list of informationally complete spam gates labels built
       from the given set of  gates. The spam gates found, correspond to a full rank ideal gram matrix.
       This is found by searching for a complete set for which:
       1. PTM is symmetric, and since it is always real, it is hermitian as well.
       2. Acting on #rho and #E(the native measurement and preparation states, which are assumed to be
       the same) gives an informationally complete set of independent states.


       Args:
       basis_gates_labels: A list of labels corresponding to the gates Gi from which the spam gates are
       constructed. For example:

       basis_gates = {
         'IdId': lambda circ, qubit1, qubit2: None,
         'X I': lambda circ, qubit1, qubit2: circ.append(XGate(), [qubit2]),
         'Y I': lambda circ, qubit1, qubit2: circ.append(YGate(), [qubit2]),
         'I X': lambda circ, qubit1, qubit2: circ.append(XGate(), [qubit1]),
         'I Y': lambda circ, qubit1, qubit2: circ.append(YGate(), [qubit1]),
         'H I': lambda circ, qubit1, qubit2: circ.append(HGate(), [qubit2]),
         'I H': lambda circ, qubit1, qubit2: circ.append(HGate(), [qubit1]),
         'I X_Rot_90': lambda circ, qubit1, qubit2: circ.append(U2Gate(-np.pi / 2, np.pi / 2), [qubit1]),
         'X_Rot_90 I': lambda circ, qubit1, qubit2: circ.append(U2Gate(-np.pi / 2, np.pi / 2), [qubit2]),
          # qubit1 is the ctrl qubit, qubit2 is the target
        }

       num_qubits: number of qubits on which GST is applied

       Return value: The full gatesetbasis, gram matrix full singular values spectrum

       Raises:
          QiskitError: if no informationally complete fidicuals set with a full rank corresponding
          gram matrix could be found
    """
    ds = (2 ** num_qubits) ** 2
    rho = default_init_state(num_qubits).reshape(ds)

    spam_gates_labels = find_spam_gates_from_gatesetbasis(basis_gates, rho, num_qubits)
    spam_gates = {}
    keys = ['F' + str(i) for i in range(ds)]
    for i in range(len(keys)):
        spam_gates[keys[i]] = tuple(spam_gates_labels[i])

    gram_rank, gram_matrix_singular_values = gram_matrix_rank(num_qubits, spam_gates_labels, basis_gates)
    if gram_rank == ds:
        return GateSetBasis('GST basis', basis_gates, spam_gates, num_qubits), gram_matrix_singular_values
    else:
        raise QiskitError(
            f'No informationally complete fidicuals set with a full rank corresponding gram matrix'
            f' could be found for the gateset basis"{repr(basis_gates_labels)}"')
