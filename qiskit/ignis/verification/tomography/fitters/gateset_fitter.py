# -*- coding: utf-8 -*-
#
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

# pylint: disable=missing-docstring,invalid-name

"""
Quantum gate set tomography fitter
"""

import itertools
from typing import Union, List, Dict, Tuple, Optional
import numpy as np
from scipy.linalg import schur
import scipy.optimize as opt
from qiskit.result import Result
from qiskit.quantum_info import Choi, PTM, Operator, DensityMatrix
from ..basis.gatesetbasis import default_gateset_basis, GateSetBasis
from .base_fitter import TomographyFitter
from qiskit.quantum_info import Pauli
from functools import reduce
import scipy.linalg as la


class GatesetTomographyFitter:
    def __init__(self,
                 result: Result,
                 circuits: List,
                 gateset_basis: Union[GateSetBasis, str] = 'default'
                 ):
        """Initialize gateset tomography fitter with experimental data.

        Args:
            result: a Qiskit Result object obtained from executing
                            tomography circuits.
            circuits: a list of circuits or circuit names to extract
                            count information from the result object.
            gateset_basis: (default: 'default') Representation of
            the gates and SPAM circuits of the gateset

        Additional information:
            The fitter attempts to output a GST result from the collected
            experimental data. The output will be a dictionary of the computed
            operators for the gates, as well as the measurment operator and
            initial state of the system.

            The input for the fitter consists of the experimental data
            collected by the backend, the circuits on which it operated
            and the gateset basis used when collecting the data.

        Example::

            from qiskit.circuits.library.standard import *
            from qiskit.ignis.verification.basis import default_gateset_basis
            from qiskit.ignis.verification import gateset_tomography_circuits
            from qiskit.ignis.verification import GateSetTomographyFitter
            num_qubits=1
            gate = HGate()
            basis = default_gateset_basis(num_qubits)
            basis.add_gate(gate)
            backend = ...
            circuits = gateset_tomography_circuits(num_qubits,[*range(num_qubits)],gateset_basis=basis)
            qobj = assemble(circuits, shots=10000)
            result = backend.run(qobj).result()
            fitter = GatesetTomographyFitter(result, circuits, basis)
            result_gates = fitter.fit()
            result_gate = result_gates[gate.name]
        """
        self.num_qubits = gateset_basis.num_qubits
        self.gateset_basis = gateset_basis
        if gateset_basis == 'default':
            self.gateset_basis = default_gateset_basis(self.num_qubits)
        self.num_qubits = self.gateset_basis.num_qubits
        data = TomographyFitter(result, circuits).data
        self.probs = {}
        for key, vals in data.items():
            # We assume the POVM is always |E>=|0000..><0000..|
            self.probs[key] = vals.get('0' * self.num_qubits, 0) / sum(vals.values())

    def linear_inversion(self) -> Dict[str, PTM]:
        """
        Reconstruct a gate set from measurement data using linear inversion.

        Returns:
            For each gate in the gateset: its approximation found
            using the linear inversion process.

        Additional Information:
            Given a gate set (G1,...,Gm)
            and SPAM circuits (F1,...,Fn) constructed from those gates
            the data should contain the probabilities of the following types:
            p_ijk = E*F_i*G_k*F_j*rho
            p_ij = E*F_i*F_j*rho

            We have p_ijk = self.probs[(Fj, Gk, Fi)] since in self.probs
            (Fj, Gk, Fi) indicates first applying Fj, then Gk, then Fi.

            One constructs the Gram matrix g = (p_ij)_ij
            which can be described as a product g=AB
            where A = sum (i> <E F_i) and B=sum (F_j rho><j)
            For each gate Gk one can also construct the matrix Mk=(pijk)_ij
            which can be described as Mk=A*Gk*B
            Inverting g we obtain g^-1 = B^-1A^-1 and so
            g^1 * Mk = B^-1 * Gk * B
            This gives us a matrix similiar to Gk's representing matrix.
            However, it will not be the same as Gk,
            since the observable results cannot distinguish
            between (G1,...,Gm) and (B^-1*G1*B,...,B^-1*Gm*B)
            a further step of *Gauge optimization* is required on the results
            of the linear inversion stage.
            One can also use the linear inversion results as a starting point
            for a MLE optimization for finding a physical gateset, since
            unless the probabilities are accurate, the resulting gateset
            need not be physical.
        """
        n = len(self.gateset_basis.spam_labels)
        m = len(self.gateset_basis.gate_labels)
        gram_matrix = np.zeros((n, n))
        E = np.zeros((1, n))
        rho = np.zeros((n, 1))
        gate_matrices = []
        for i in range(m):
            gate_matrices.append(np.zeros((n, n)))

        for i in range(n):  # row
            F_i = self.gateset_basis.spam_labels[i]
            E[0][i] = self.probs[(F_i,)]
            rho[i][0] = self.probs[(F_i,)]
            for j in range(n):  # column
                F_j = self.gateset_basis.spam_labels[j]
                gram_matrix[i][j] = self.probs[(F_j, F_i)]

                for k in range(m):  # gate
                    G_k = self.gateset_basis.gate_labels[k]
                    gate_matrices[k][i][j] = self.probs[(F_j, G_k, F_i)]

        gram_inverse = np.linalg.inv(gram_matrix)

        gates = [PTM(gram_inverse @ gate_matrix) for gate_matrix in gate_matrices]
        result = dict(zip(self.gateset_basis.gate_labels, gates))
        result['E'] = E
        result['rho'] = gram_inverse @ rho
        return result

    @staticmethod
    def _default_init_state(num_qubits):
        """Returns the PTM representation of the usual ground state |00...>"""
        d = np.power(2, num_qubits)

        # matrix representation of #rho in regular Hilbert space
        matrix_init_0 = np.zeros((d, d), dtype=complex)
        matrix_init_0[0, 0] = 1

        # decompoition into Pauli strings basis (PTM representation)
        matrix_init_pauli = [np.trace(np.dot(matrix_init_0, Pauli_strings(num_qubits)[i])) for i in
                             range(np.power(d, 2))]
        return np.reshape(matrix_init_pauli, (np.power(d, 2), 1))

    @staticmethod
    def _default_measurement_op(num_qubits):
        """The PTM representation of the usual Z-basis measurement"""
        d = np.power(2, num_qubits)

        # matrix representation of #E=|00..><00...| in regular Hilbert space
        matrix_meas_0 = np.zeros((d, d), dtype=complex)
        matrix_meas_0[0, 0] = 1

        # decompoition into Pauli strings basis (PTM representation)
        matrix_meas_pauli = [np.trace(np.dot(matrix_meas_0, Pauli_strings(num_qubits)[i])) for i in
                             range(np.power(d, 2))]
        return matrix_meas_pauli

    def _ideal_gateset(self, num_qubits):
        ideal_gateset = {label: PTM(self.gateset_basis.gate_matrices[label])
                         for label in self.gateset_basis.gate_labels}
        ideal_gateset['E'] = self._default_measurement_op(num_qubits)
        ideal_gateset['rho'] = self._default_init_state(num_qubits)
        return ideal_gateset

    def fit(self) -> Dict:
        """
        Reconstruct a gate set from measurement data using optimization.

        Returns:
           For each gate in the gateset: its approximation found using the
           optimization process.

        Additional Information:
            The gateset optimization process con/.sists of three phases:
            1) Use linear inversion to obtain an initial approximation.
            2) Use gauge optimization to ensure the linear inversion results
            are close enough to the expected optimization outcome to serve
            as a suitable starting point
            3) Use MLE optimization to obtain the final outcome
        """
        linear_inversion_results = self.linear_inversion()
        # n = len(self.gateset_basis.spam_labels)
        gauge_opt = GaugeOptimize(self._ideal_gateset(self.gateset_basis.num_qubits),
                                  linear_inversion_results,
                                  self.gateset_basis)
        past_gauge_gateset = gauge_opt.optimize()

        optimizer = GST_Optimize(self.gateset_basis.gate_labels,
                                 self.gateset_basis.spam_labels,
                                 self.gateset_basis.spam_spec,
                                 self.probs, self.num_qubits)

        optimizer.set_initial_value(past_gauge_gateset)

        optimization_results = optimizer.optimize()
        return optimization_results


class GaugeOptimize():
    def __init__(self,
                 ideal_gateset: Dict[str, PTM],
                 initial_gateset: Dict[str, PTM],
                 gateset_basis: GateSetBasis,
                 ):
        """Initialize gauge optimizer fitter with the ideal and expected
            outcomes.
        Args:
            ideal_gateset: The ideal expected gate matrices
            initial_gateset: The experimentally-obtained gate approximations.
            gateset_basis: The gateset data

        Additional information:
            Gauge optimization aims to find a basis in which the tomography
            results are as close as possible to the ideal (noiseless) results

            Given a gateset specification (E, rho, G1,...,Gn) and any
            invertible matrix B, the gateset specification
            (E*B^-1, B*rho, B*G1*B^-1,...,B*Gn*B^-1)
            is indistinguishable from it by the tomography results.

            B is called the gauge matrix and the goal of gauge optimization
            is finding the B for which the resulting gateset description
            is optimal in some sense; we choose to minimize the norm
            difference between the gates found by experiment
            and the "expected" gates in the ideal (noiseless) case.
        """
        self.gateset_basis = gateset_basis
        self.ideal_gateset = ideal_gateset
        self.initial_gateset = initial_gateset
        self.Fs = [self.gateset_basis.spam_matrix(label)
                   for label in self.gateset_basis.spam_labels]
        self.d = np.shape(ideal_gateset['rho'])[0]
        self.n = len(gateset_basis.gate_labels)
        self.rho = ideal_gateset['rho']

    def _x_to_gateset(self, x: np.array) -> Dict[str, PTM]:
        """Converts the gauge to the gateset defined by it
        Args:
            x: An array representation of the B matrix

        Returns:
            The gateset obtained from B

        Additional information:
            Given a vector representation of B, this functions
            produces the list [B*G1*B^-1,...,B*Gn*B^-1]
            of gates correpsonding to the gauge B

        """
        B = np.array(x).reshape((self.d, self.d))
        try:
            BB = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            #BB = np.linalg.pinv(B)
            return None    #I will let it return the Pseudo inverse matrix eaither way
        gateset = {label: PTM(B @ self.initial_gateset[label].data @ BB)
                   for label in self.gateset_basis.gate_labels}
        gateset['E'] = self.initial_gateset['E'] @ BB
        gateset['rho'] = B @ self.initial_gateset['rho']
        return gateset

    def _obj_fn(self, x: np.array) -> float:
        """The norm-based score function for the gauge optimizer
        Args:
            x: An array representation of the B matrix

        Returns:
            The sum of norm differences between the ideal gateset
            and the one corresponding to B
        """
        gateset = self._x_to_gateset(x)
        result = sum([np.linalg.norm(gateset[label].data -
                                     self.ideal_gateset[label].data)
                      for label in self.gateset_basis.gate_labels])
        result = result + np.linalg.norm(gateset['E'] -
                                         self.ideal_gateset['E'])
        result = result + np.linalg.norm(gateset['rho'] -
                                         self.ideal_gateset['rho'])
        return result

    def optimize(self) -> List[np.array]:
        """The main optimization method
        Returns:
            The optimal gateset found by the gauge optimization
        """
        initial_value = np.array([(F @ self.rho).T[0] for F in self.Fs]).T
        result = opt.minimize(self._obj_fn, initial_value)
        return self._x_to_gateset(result.x)


def Pauli_strings(num_qubits):
    """Returns the normalized matrix representation of Pauli strings basis of size=num_qubits. e.g., for num_qubits=2,
     it returns the matrix representations of 0.5*['II','IX','IY','IZ,'XI','YI',...]"""
    pauli_labels = ['I', 'X', 'Y', 'Z']
    pauli_strings_matrices = [Pauli(''.join(p)).to_matrix() for p in itertools.product(pauli_labels, repeat=num_qubits)]
    # normalization
    pauli_strings_matrices_orthonormal = [(1/np.sqrt(2**num_qubits))*pauli_strings_matrices[i] for i in range(len(pauli_strings_matrices))]
    return pauli_strings_matrices_orthonormal


def make_positive_semidefinite(mat: np.array,
                               epsilon: Optional[float] = 0
                               ) -> np.array:
    """
    Rescale a Hermitian matrix to nearest postive semidefinite matrix.
    Args:
        mat: a hermitian matrix.
        epsilon: (default: 0) the threshold for setting
            eigenvalues to zero. If epsilon > 0 positive eigenvalues
            below epsilon will also be set to zero.
    Raises:
        ValueError: If epsilon is negative
    Returns:
        The input matrix rescaled to have non-negative eigenvalues.
    References:
        [1] J Smolin, JM Gambetta, G Smith, Phys. Rev. Lett. 108, 070502
            (2012). Open access: arXiv:1106.5458 [quant-ph].
    """

    if epsilon < 0:
        raise ValueError('epsilon must be non-negative.')

    # Get the eigenvalues and eigenvectors of rho
    # eigenvalues are sorted in increasing order
    # v[i] <= v[i+1]

    dim = len(mat)
    v, w = la.eigh(mat)
    for j in range(dim):
        if v[j] < epsilon:
            tmp = v[j]
            v[j] = 0.
            # Rescale remaining eigenvalues
            x = 0.
            for k in range(j + 1, dim):
                x += tmp / (dim - (j + 1))
                v[k] = v[k] + tmp / (dim - (j + 1))

    # Build positive matrix from the rescaled eigenvalues
    # and the original eigenvectors

    mat_psd = np.zeros([dim, dim], dtype=complex)
    for j in range(dim):
        mat_psd += v[j] * np.outer(w[:, j], np.conj(w[:, j]))

    return mat_psd


class GST_Optimize:
    def __init__(self,
                 Gs: List[str],
                 Fs_names: Tuple[str],
                 Fs: Dict[str, Tuple[str]],
                 probs: Dict[Tuple[str], float],
                 qubits: int
                 ):
        """Initializes the data for the MLE optimizer
        Args:
            Gs: The names of the gates in the gateset
            Fs_names: The names of the SPAM circuits
            Fs: The SPAM specification (SPAM name -> gate names)
            probs: The probabilities obtained experimentally
            qubits: number of qubits
        """
        self.probs = probs
        self.Gs = Gs
        self.Fs_names = Fs_names
        self.Fs = Fs
        self.qubits = qubits
        self.obj_fn_data = self._compute_objective_function_data()
        self.initial_value = None

    # auxiliary functions
    @staticmethod
    def _split_list(input_list: List, sizes: List) -> List[List]:
        """Splits a list to several lists of given size
        Args:
            input_list: A list
            sizes: The sizes of the splitted lists
        Returns:
            list: The splitted lists
        Example:
            >> split_list([1,2,3,4,5,6,7], [1,4,2])
            [[1],[2,3,4,5],[6,7]]

        Raises:
            RuntimeError: if length of l does not equal sum of sizes
        """
        if sum(sizes) != len(input_list):
            msg = "Length of list ({}) " \
                  "differs from sum of split sizes ({})".format(len(input_list), sizes)
            raise RuntimeError(msg)
        result = []
        i = 0
        for s in sizes:
            result.append(input_list[i:i + s])
            i = i + s
        return result

    @staticmethod
    def _vec_to_complex_matrix(vec: np.array) -> np.array:
        n = int(np.sqrt(vec.size / 2))
        if 2 * n * n != vec.size:
            raise RuntimeError("Vector of length {} cannot be reshaped"
                               " to square matrix".format(vec.size))
        size = n * n
        return np.reshape(vec[0:size] + 1j * vec[size: 2 * size], (n, n))

    @staticmethod
    def _complex_matrix_to_vec(M):
        mvec = M.reshape(M.size)
        return list(np.concatenate([mvec.real, mvec.imag]))

    def _compute_objective_function_data(self) -> List:
        """Computes auxiliary data needed for efficient computation
        of the objective function.

        Returns:
             The objective function data list
        Additional information:
            The objective function is
            sum_{ijk}(<|E*R_Fi*G_k*R_Fj*Rho|>-m_{ijk})^2
            We expand R_Fi*G_k*R_Fj to a sequence of G-gates and store
            indices. We also obtain the m_{ijk} value from the probs list
            all that remains when computing the function is thus
            performing the matrix multiplications and remaining algebra.
        """
        m = len(self.Fs)
        n = len(self.Gs)
        obj_fn_data = []
        for (i, j) in itertools.product(range(m), repeat=2):
            for k in range(n):
                Fi = self.Fs_names[i]
                Fj = self.Fs_names[j]
                m_ijk = (self.probs[(Fj, self.Gs[k], Fi)])
                Fi_matrices = [self.Gs.index(gate) for gate in self.Fs[Fi]]
                Fj_matrices = [self.Gs.index(gate) for gate in self.Fs[Fj]]
                matrices = Fj_matrices + [k] + Fi_matrices
                obj_fn_data.append((matrices, m_ijk))
        return obj_fn_data

    def _split_input_vector(self, x: np.array, fullness: str) -> Tuple:
        """Reconstruct the GST data from its vector representation
        Args:
            x: The vector representation of the GST data
            fullness: Takes two values, 'full': in case x has no missing elements and '0': in case the
            last elements of each corresponding lower triangle matrix of the Cholesky decomposition of
            the gates are missing.

        Returns:
            The GST data (E, rho, Gs) (see additional info)

        Additional information:
            The gate set tomography data is a tuple (E, rho, Gs) consisting of
            1) A POVM measurement operator E
            2) An initial quantum state rho
            3) A list Gs = (G1, G2, ..., Gk) of gates, represented as matrices

            This function reconstructs (E, rho, Gs) from the vector x
            Since the MLE optimization procedure has PSD constraints on
            E, rho and the Choi represetnation of the PTM of the Gs,
            we rely on the following property: M is PSD iff there exists
            T such that M = T @ T^{dagger}.
            Hence, x stores those T matrices for E, rho and the Gs

            if x has missing values, it replaces them with zeros and returns the corresponding GST data.
        """
        n = len(self.Gs)
        d = (2 ** self.qubits)
        ds = d ** 2  # d squared - the dimension of the density operator

        d_t = 2 * d ** 2
        ds_t = 2 * ds ** 2

        if fullness == 'full':
            T_vars = self._split_list(x, [d_t, d_t] + [ds_t] * n)
        else:  # =='0'
            T_vars = self._split_list(x, [d_t, d_t] + [ds_t - 1] * n)
            # add zeros instead of last real element and it will be changed in the obj func
            for i in range(n):
                T_vars[2 + i] = np.insert(T_vars[2 + i], ds ** 2 - 1, 0)

        E_T = self._vec_to_complex_matrix(T_vars[0])
        rho_T = self._vec_to_complex_matrix(T_vars[1])

        Gs_T = [self._vec_to_complex_matrix(T_vars[2 + i]) for i in range(n)]

        E = np.reshape(E_T @ np.conj(E_T.T), (1, ds))
        rho = np.reshape(rho_T @ np.conj(rho_T.T), (ds, 1))
        Gs = [PTM(Choi(G_T @ np.conj(G_T.T))).data for G_T in Gs_T]

        return (E, rho, Gs)

    def _join_input_vector(self,
                           E: np.array,
                           rho: np.array,
                           Gs: List[np.array]
                           ) -> np.array:
        """Converts the GST data into a vector representation
        Args:
            E: The POVM measurement operator
            rho: The initial state
            Gs: The gates list

        Returns:
            The vector representation of (E, rho, Gs)

        Additional information:
            The GST data is encoded in terms of Cholesky decomposition to make sure: E, rho and every
             G in the gateset are all PSD.
            This function performs the inverse operation to
            split_input_vector; the notations are the same.
        """
        d = (2 ** self.qubits)
        #np.linalg returns the lower diagonal matrix T of the Cholesky decomposition T_dagger*T, but it
        #works only with positive definite matrix. To solve this, we simply add a small matrix to shift
        # the zero eigenvalues.

        E_T = np.linalg.cholesky(make_positive_semidefinite(E.reshape((d, d))) + 1e-14 * np.eye(d))
        rho_T = np.linalg.cholesky(make_positive_semidefinite(rho.reshape((d, d))) + 1e-14 * np.eye(d))

        Gs_Choi = [Choi(PTM(G)).data for G in Gs]
        Gs_Choi_mod = [make_positive_semidefinite(G) + 1e-14 * np.eye(d ** 2) for G in Gs_Choi]
        Gs_T = [np.linalg.cholesky(G) for G in Gs_Choi_mod]
        E_vec = self._complex_matrix_to_vec(E_T)
        rho_vec = self._complex_matrix_to_vec(rho_T)
        result = E_vec + rho_vec
        for G_T in Gs_T:
            result += self._complex_matrix_to_vec(G_T)
        return np.array(result)

    def _obj_fn(self, x: np.array) -> float:
        """The MLE objective function
        Args:
            x: The vector representation of the GST data (E, rho, Gs) with the last elements of the
            Cholesky matrices corresponding to all the gates in the gateset are missing

        Returns:
            The MLE cost function (see additional information)

        Additional information:
            The MLE objective function is obtained by approximating
            the MLE estimator using the central limit theorem.

            It is computed as the sum of all terms of the form
            (m_{ijk} - p_{ijk})^2
            Where m_{ijk} are the experimental results, and
            p_{ijk} are the predicted results for the given GST data:
            p_{ijk} = E*F_i*G_k*F_j*rho.

            For additional info, see section 3.5 in arXiv:1509.02921
            and about the filling of the x vector, see the information in _complete_x method below.
        """
        E, rho, G_matrices = self._split_input_vector(x, '0')
        n = len(G_matrices)
        x1 = np.copy(x)
        x2 = np.copy(x)
        d = (2 ** self.qubits)  # rho is dxd and starts at variable d^2
        ds = d ** 2
        l = 0
        for i in range(n):
            m = 0
            for j in range(2 * ds ** 2 - 1):
                m += (abs(x[4 * ds + 2 * i * ds ** 2 - l + j])) ** 2
            l += 1
            x1 = np.insert(x1, 4 * ds + (2 * i + 1) * ds ** 2, np.sqrt(abs(d - m)))
            x2 = np.insert(x2, 4 * ds + (2 * i + 1) * ds ** 2, -np.sqrt(abs(d - m)))

        # E, rho, G_matrices = self._split_input_vector(x)
        E, rho, G_matrices = self._split_input_vector(x1, 'full')
        val1 = 0
        for term in self.obj_fn_data:
            term_val = rho
            for G_index in term[0]:
                term_val = G_matrices[G_index] @ term_val
            term_val = E @ term_val
            term_val = np.real(term_val[0][0])
            term_val = term_val - term[1]  # m_{ijk}
            term_val = term_val ** 2
            val1 = val1 + term_val

        E, rho, G_matrices = self._split_input_vector(x2, 'full')
        val2 = 0
        for term in self.obj_fn_data:
            term_val = rho
            for G_index in term[0]:
                term_val = G_matrices[G_index] @ term_val
            term_val = E @ term_val
            term_val = np.real(term_val[0][0])
            term_val = term_val - term[1]  # m_{ijk}
            term_val = term_val ** 2
            val2 = val2 + term_val
        return np.min([val1, val2])

    def _complete_x(self, x: np.array) -> float:
        """ Completes the x vector by adding the suitable elements to fill the deleted elements; last
        element of each Cholesky decomposition matrices _T of the gates. The suitable elements are
        those that give trace=2**num_qubits for the Choi matrix corresponding to each gate in the gateset.
        As the trace does not depend on the signs of these elements, their sign is picked as the one that
        returns lower estimation of the objection function.

        Args:
            x: The vector representation of the GST data (E, rho, Gs) with the last elements of the
            Cholesky matrices corresponding to all the gates in the gateset are missing

        Returns:
            The full x vector representation of the GST data
        """
        E, rho, G_matrices = self._split_input_vector(x, '0')
        n = len(G_matrices)
        x1 = np.copy(x)
        x2 = np.copy(x)
        d = (2 ** self.qubits)  # rho is dxd and starts at variable d^2
        ds = d ** 2

        l = 0
        for i in range(n):
            m = 0

            for j in range(2 * ds ** 2 - 1):
                m += (abs(x[4 * ds + 2 * i * ds ** 2 - l + j])) ** 2
            l += 1
            x1 = np.insert(x1, 4 * ds + (2 * i + 1) * ds ** 2, np.sqrt(abs(d - m)))
            x2 = np.insert(x2, 4 * ds + (2 * i + 1) * ds ** 2, -np.sqrt(abs(d - m)))

        # E, rho, G_matrices = self._split_input_vector(x)
        E, rho, G_matrices = self._split_input_vector(x1, 'full')
        val1 = 0
        for term in self.obj_fn_data:
            term_val = rho
            for G_index in term[0]:
                term_val = G_matrices[G_index] @ term_val
            term_val = E @ term_val
            term_val = np.real(term_val[0][0])
            term_val = term_val - term[1]  # m_{ijk}
            term_val = term_val ** 2
            val1 = val1 + term_val

        E, rho, G_matrices = self._split_input_vector(x2, 'full')
        val2 = 0
        for term in self.obj_fn_data:
            term_val = rho
            for G_index in term[0]:
                term_val = G_matrices[G_index] @ term_val
            term_val = E @ term_val
            term_val = np.real(term_val[0][0])
            term_val = term_val - term[1]  # m_{ijk}
            term_val = term_val ** 2
            val2 = val2 + term_val

        if (np.min([val1, val2]) == val1):
            x = x1
        else:
            x = x2
        return x

    def _ptm_matrix_values(self, x: np.array) -> List[np.array]:
        """Returns a vectorization of the gates matrices
        Args:
            x: The vector representation of the GST data

        Returns:
            A vectorization of all the PTM matrices for the gates
            in the GST data

        Additional information:
            This function is not trivial since the returned vector
            is not a subset of x, since for each gate G, what x
            stores in practice is a matrix T, such that the
            Choi matrix of G is T@T^{dagger}. This needs to be
            converted into the PTM representation of G.
        """
        xfull = self._complete_x(x)
        _, _, G_matrices = self._split_input_vector(xfull, 'full')
        result = []
        for G in G_matrices:
            result = result + self._complex_matrix_to_vec(G)
        return result

    def _rho_trace(self, x: np.array) -> Tuple[float]:
        """Returns the trace of the GST initial state
        Args:
            x: The vector representation of the GST data
        Returns:
            The trace of rho - the initial state of the GST. The real
            and imaginary part are returned separately.
        """
        xfull = self._complete_x(x)
        _, rho, _ = self._split_input_vector(xfull, 'full')
        d = (2 ** self.qubits)  # rho is dxd and starts at variable d^2
        rho = self._convert_from_ptm(rho.reshape((d, d)))
        trace = sum([rho[i][i] for i in range(d)])
        return (np.real(trace), np.imag(trace))

    def _bounds_eq_constraint(self, x: np.array) -> List[float]:
        """Equality MLE constraints on the GST data

        Args:
            x: The vector representation of the GST data

        Returns:
            The list of computed constraint values (should equal 0)

        Additional information:
            We have the following constraints on the GST data, due to
            the PTM representation we are using:
            1) G_{0,0} is 1 for every gate G
            2) The rest of the first row of each G is 0.
            3) G only has real values, so imaginary part is 0.

            For additional info, see section 3.5.2 in arXiv:1509.02921
        """
        ptm_matrix = self._ptm_matrix_values(x)
        bounds_eq = []
        n = len(self.Gs)
        d = (2 ** self.qubits)  # rho is dxd and starts at variable d^2
        ds = d ** 2

        i = 0
        for _ in range(n):  # iterate over all Gs
            bounds_eq.append(ptm_matrix[i] - 1)  # G^k_{0,0} is 1
            i += 1
            for _ in range(ds - 1):
                bounds_eq.append(ptm_matrix[i] - 0)  # G^k_{0,i} is 0
                i += 1
            for _ in range((ds - 1) * ds):  # rest of G^k
                i += 1
            for _ in range(ds ** 2):  # the complex part of G^k
                bounds_eq.append(ptm_matrix[i] - 0)  # G^k_{0,i} is 0
                i += 1
        return bounds_eq

    def _bounds_ineq_constraint(self, x: np.array) -> List[float]:
        """Inequality MLE constraints on the GST data

        Args:
            x: The vector representation of the GST data

        Returns:
            The list of computed constraint values (should be >= 0)

        Additional information:
            We have the following constraints on the GST data, due to
            the PTM representation we are using:
            1) Every row of G except the first has entries in [-1,1]

            We implement this as two inequalities per entry.

            For additional info, see section 3.5.2 in arXiv:1509.02921
        """
        ptm_matrix = self._ptm_matrix_values(x)
        bounds_ineq = []
        n = len(self.Gs)
        d = (2 ** self.qubits)  # rho is dxd and starts at variable d^2
        ds = d ** 2

        i = 0
        for _ in range(n):  # iterate over all Gs
            i += 1
            for _ in range(ds - 1):
                i += 1
            for _ in range((ds - 1) * ds):  # rest of G^k
                bounds_ineq.append(ptm_matrix[i] + 1)  # G_k[i] >= -1
                bounds_ineq.append(-ptm_matrix[i] + 1)  # G_k[i] <= 1
                i += 1
            for _ in range(ds ** 2):  # the complex part of G^k
                i += 1
        return bounds_ineq

    def _rho_trace_constraint(self, x: np.array) -> List[float]:
        """The constraint Tr(rho) = 1
        Args:
            x: The vector representation of the GST data

        Return:
            The list of computed constraint values (should be equal 0)

        Additional information:
            We demand real(Tr(rho)) == 1 and imag(Tr(rho)) == 0
        """
        trace = self._rho_trace(x)
        return [trace[0] - 1, trace[1]]

    def _constraints(self) -> List[Dict]:
        """Generates the constraints for the MLE optimization

        Returns:
            A list of constraints.

        Additional information:
            Each constraint is a dictionary containing
            type ('eq' for equality == 0, 'ineq' for inequality >= 0)
            and a function generating from the input x the values
            that are being constrained.
        """
        cons = []
        cons.append({'type': 'eq', 'fun': self._rho_trace_constraint})
        cons.append({'type': 'eq', 'fun': self._bounds_eq_constraint})
        cons.append({'type': 'ineq', 'fun': self._bounds_ineq_constraint})
        return cons

    def _convert_from_ptm(self, vector):
        """Converts a vector back from PTM representation"""
        num_qubits = self.qubits
        pauli_strings_matrices = Pauli_strings(num_qubits)
        v = vector.reshape(np.size(vector))
        n = [a * b for a, b in zip(v, pauli_strings_matrices)]
        return reduce(lambda x, y: np.add(x, y), n)

    def _process_result(self, x: np.array) -> Dict:
        """Completes and transforms the optimization result to a friendly format
        Args:
            x: the optimization result vector

        Returns:
            The final GST data, as dictionary.
        """

        xfinal = self._complete_x(x)

        E, rho, G_matrices = self._split_input_vector(xfinal, 'full')

        result = {}
        result['E'] = Operator(self._convert_from_ptm(E))
        result['rho'] = DensityMatrix(self._convert_from_ptm(rho))
        for i in range(len(self.Gs)):
            result[self.Gs[i]] = PTM(G_matrices[i])
        return result

    def set_initial_value(self, initial_value: Dict[str, PTM]):
        """Sets the initial value for the MLE optimization
        Args:
            initial_value: The dictionary of the initial gateset
        """
        E = initial_value['E']
        rho = initial_value['rho']
        Gs = [initial_value[label] for label in self.Gs]

        initial_value_temp = self._join_input_vector(E, rho, Gs)
        n = len(Gs)
        d = (2 ** self.qubits)  # rho is dxd and starts at variable d^2
        ds = d ** 2
        m = 0
        for i in range(n):
            # Deleting the final elements of the cholesky decomposition matrices _T of the gates from the
            # initial_value_temp vector (containing the vectorization of GST data).
            initial_value_temp = np.delete(initial_value_temp, 4 * ds + (2 * i + 1) * ds ** 2 - 1 - m)
            m += 1
        self.initial_value = initial_value_temp


    def optimize(self, initial_value: Optional[np.array] = None) -> Dict:
        """Performs the MLE optimization for gate set tomography
        Args:
            initial_value: Vector representation of the initial value data
        Returns:
            The formatted results of the MLE optimization.
        """
        if initial_value is not None:
            self.initial_value = initial_value
        result = opt.minimize(self._obj_fn, self.initial_value,
                              method='SLSQP',
                              constraints=self._constraints())
        formatted_result = self._process_result(result.x)
        return formatted_result
