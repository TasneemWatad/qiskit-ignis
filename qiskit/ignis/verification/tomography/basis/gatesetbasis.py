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

# Import QISKit classes
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import U2Gate
from qiskit.quantum_info import PTM
from .tomographybasis import TomographyBasis

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
                 num_qubits:int
                 ):
        """
        Initialize the gate set basis data

        Args:
            name: Name of the basis.
            gates: The gate data (name -> gate/gate function)
            spam: The spam data (name -> sequence of gate names)
        """
        self.name = name
        self.gate_labels = list(gates.keys())
        self.gates = gates
        self.gate_matrices = {name: np.real(self._gate_matrix(gate))
                              for (name, gate) in gates.items()}
        self.spam_labels = tuple(sorted(spam.keys()))
        self.spam_spec = spam
        self.num_qubits= num_qubits

    def _gate_matrix(self, gate):
        """Gets a PTM representation of the gate"""
        if isinstance(gate, Gate):
            return PTM(gate).data
        if callable(gate):
            c = QuantumCircuit(self.num_qubits)
            qubits=[]
            for i in range(self.num_qubits):
                qubits.append(c.qubits[i])
            gate(c, *qubits)
            return PTM(c).data
        return None

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
        self.gate_matrices[name] = self._gate_matrix(gate)

    def add_gate_to_circuit(self,circ: QuantumCircuit,
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
        #print(self.gates,'gates')
        if op not in self.gates:
            raise RuntimeError("{} is not a SPAM circuit".format(op))
        gate = self.gates[op]
        if callable(gate):
            gate(circ, *qubits)
        if isinstance(gate, Gate):
            circ.append(gate,qubits,[])

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


       Return value: The default gateset (for 1 qubit, it is the gateset as in example 3.4.1 in arXiv:1509.02921)

       Raises:
          QiskitError: if num_qubits is larger than 2.
    """

    if num_qubits > 2:
        raise QiskitError("Three qubits or more are not supported")

    # In lambda func., the argument qubit here is a single qubit: i.e., Qubit(QuantumRegister(),i)
    default_gates_single = {
        'Id': lambda circ, qubit: None,
        'X_Rot_90': lambda circ, qubit: circ.append(U2Gate(-np.pi / 2, np.pi / 2), [qubit]),
        'Y_Rot_90': lambda circ, qubit: circ.append(U2Gate(0, 0), [qubit])  # changesshould be this way
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

    """
    default_gates_two = {

    'IdId': lambda circ, qubit1,qubit2: None,
    'XI': lambda circ, qubit1,qubit2: circ.x(qubit2),
    'YI': lambda circ, qubit1, qubit2: circ.x(qubit2),  #Q.      #should we do it this way or .append?
    'IX': lambda circ, qubit1,qubit2: circ.x(qubit1),
    'IY': lambda circ, qubit1, qubit2: circ.x(qubit1),
    'CX': lambda circ, qubit1, qubit2: circ.cx(qubit1,qubit2) #qubit1 is the ctrl qubit, qubit2 is the target
    }

    default_spam_two = {
        'F0': ('IdId',),
        'F1': ('XI',),
        'F2': ('YI',),
        'F3': ('IX',),
        'F4': ('IY',),
        'F5': ('CX',),
        'F6': ('XI','IX',),
        'F7': ('XI','IY',),
        'F8': ('XI','YI',),
        'F9': ('IX','IY',),
        'F10': ('YI','IX',),
        'F11': ('YI', 'IY',),
        'F12': ('CX','YI',),
        'F13': ('YI','CX',),
        'F14': ('XI','YI','IY',),
        'F15': ('XI', 'YI','IX',)
    }
    """
    default_gates_two = {

        'IdId': lambda circ, qubit1, qubit2: None,
        'X_Rot_90 I': lambda circ, qubit1, qubit2: circ.append(U2Gate(-np.pi / 2, np.pi / 2), [qubit2]),
        'Y_Rot_90 I': lambda circ, qubit1, qubit2: circ.append(U2Gate(0, 0), [qubit2]),
        # Q.      #should we do it this way or .append?
        'I X_Rot_90': lambda circ, qubit1, qubit2: circ.append(U2Gate(-np.pi / 2, np.pi / 2), [qubit1]),
        'I Y_Rot_90': lambda circ, qubit1, qubit2: circ.append(U2Gate(0, 0), [qubit2]),
        'CX': lambda circ, qubit1, qubit2: circ.cx(qubit1, qubit2)  # qubit1 is the ctrl qubit, qubit2 is the target
    }

    default_spam_two = {
        'F0': ('IdId',),
        'F1': ('X_Rot_90 I',),
        'F2': ('Y_Rot_90 I',),
        'F3': ('I X_Rot_90',),
        'F4': ('I Y_Rot_90',),
        'F5': ('CX',),
        'F6': ('X_Rot_90 I', 'I X_Rot_90',),
        'F7': ('X_Rot_90 I', 'I Y_Rot_90',),
        'F8': ('X_Rot_90 I', 'Y_Rot_90 I',),
        'F9': ('I X_Rot_90', 'I Y_Rot_90',),
        'F10': ('Y_Rot_90 I', 'I X_Rot_90',),
        'F11': ('Y_Rot_90 I', 'I Y_Rot_90',),
        'F12': ('CX', 'Y_Rot_90 I',),
        'F13': ('Y_Rot_90 I', 'CX',),
        'F14': ('X_Rot_90 I', 'Y_Rot_90 I', 'I Y_Rot_90',),
        'F15': ('X_Rot_90 I', 'Y_Rot_90 I', 'I X_Rot_90',)
    }

    return GateSetBasis('Default GST', default_gates_single, default_spam_single,
                        1) if num_qubits == 1 else GateSetBasis('Default GST', default_gates_two, default_spam_two, 2)
