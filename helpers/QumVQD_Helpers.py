# QumVQD: variational quantum deflation on qumode-based processors.
#
# This file contains code derived from the qumode_est_paper repository
# by Rishab Dutta (https://github.com/rishabdchem/qumode_est_paper),
# licensed under the MIT License.
# Copyright (c) 2025 Rishab Dutta.
# See /third_party/LICENSE_qumode_est_paper.txt for the full upstream license.
#
# Modifications and new code by the QumVQD authors (2026):
#   - Refactored notebook cells into an installable module
#   - Generalized two-mode SNAP-displacement-beamsplitter ansatz to
#     arbitrary N modes
#   - Added VQD loss and excited-state driver routines
#   - Added particle-number-preserving slicing and bosonic Hamiltonian
#     construction utilities
#


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from openfermion.chem import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner, bravyi_kitaev
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.ops.representations import get_tensors_from_integrals
from openfermionpyscf import run_pyscf

from pyscf import gto, scf, fci

import os

import qutip as qt

import json


def destroy_tf(N, dtype=tf.complex128):
    """Annihilation operator"""
    a = tf.linalg.diag(tf.sqrt(tf.range(1, N, dtype=tf.float64)), k=1)
    return tf.cast(a, dtype=dtype)

def create_tf(N, dtype=tf.complex128):
    """Creation operator"""
    return tf.linalg.adjoint(destroy_tf(N, dtype))

def displace_tf(N, alpha, dtype=tf.complex128):
    """Displacement operator D(alpha)"""
    a = destroy_tf(N, dtype)
    a_dag = create_tf(N, dtype)
    gen = tf.cast(alpha, dtype) * a_dag - tf.cast(tf.math.conj(alpha), dtype) * a
    return tf.linalg.expm(gen)

def snap_op_tf(theta_vec, nfock, dtype=tf.complex128):
    """SNAP operator S(θ)"""
    phases = tf.exp(1j * tf.cast(theta_vec, dtype))
    return tf.linalg.diag(phases)

def beam_splitter_tf(theta, phi, n1, n2, dtype=tf.complex128):
    """Beam splitter BS(θ,φ) between two modes"""
    a1 = destroy_tf(n1, dtype)
    a2 = destroy_tf(n2, dtype)
    a1_dag = create_tf(n1, dtype)
    a2_dag = create_tf(n2, dtype)

    I1 = tf.eye(n1, dtype=dtype)
    I2 = tf.eye(n2, dtype=dtype)

    a1_dag_a2 = tf.experimental.numpy.kron(a1_dag, a2)
    a1_a2_dag = tf.experimental.numpy.kron(a1, a2_dag)

    # Generator
    gen = tf.cast(theta/2, dtype) * (
        tf.exp(1j * tf.cast(phi, dtype)) * a1_dag_a2 +
        tf.exp(-1j * tf.cast(phi, dtype)) * a1_a2_dag
    )

    return tf.linalg.expm(1j * gen)


def tensor_product_list(operators):
    """Compute tensor product of a list of operators"""
    result = operators[0]
    for op in operators[1:]:
        result = tf.experimental.numpy.kron(result, op)
    return result

def embed_single_mode_op(op, mode_idx, nfocks, dtype=tf.complex128):
    """Embed single-mode operator into multi-mode space"""
    ops = []
    for i, nfock in enumerate(nfocks):
        if i == mode_idx:
            ops.append(op)
        else:
            ops.append(tf.eye(nfock, dtype=dtype))
    return tensor_product_list(ops)

def embed_two_mode_op(op, mode_i, mode_j, nfocks, dtype=tf.complex128):
    """Embed two-mode operator into multi-mode space"""
    identities = [tf.eye(n, dtype=dtype) for n in nfocks]

    # The beam splitter acts on modes i and j
    if mode_i > mode_j:
        mode_i, mode_j = mode_j, mode_i

    # Reshape beam splitter to act on correct modes
    ops = []
    bs_applied = False
    j_idx = 0

    for k in range(len(nfocks)):
        if k == mode_i:
            continue
        elif k == mode_j:
            ops.append(op)
            bs_applied = True
        else:
            ops.append(identities[k])

    return tensor_product_list(ops)

def unpack_params_multi_qumode(X, nfocks, ndepth):
    """Unpack parameters for multi-qumode ansatz"""
    n_modes = len(nfocks)
    n_bs_pairs = n_modes * (n_modes - 1) // 2  # Number of beam splitter pairs

    idx = 0

    n_disp = ndepth * n_modes
    alphas = tf.reshape(X[idx:idx+n_disp], (ndepth, n_modes))
    idx += n_disp

    n_snap = ndepth * sum(nfocks)
    thetas = []
    for i, nfock in enumerate(nfocks):
        theta_mode = tf.reshape(X[idx:idx+ndepth*nfock], (ndepth, nfock))
        thetas.append(theta_mode)
        idx += ndepth * nfock

    n_bs = ndepth * n_bs_pairs * 2
    bs_params = tf.reshape(X[idx:idx+n_bs], (ndepth, n_bs_pairs, 2))

    return alphas, thetas, bs_params

def multi_qumode_ansatz_layer(alpha_vec, theta_vecs, bs_params, nfocks, dtype=tf.complex128):
    """Single layer of multi-qumode ansatz"""
    n_modes = len(nfocks)
    total_dim = np.prod(nfocks)

    U = tf.eye(total_dim, dtype=dtype)

    for i in range(n_modes):
        S = snap_op_tf(theta_vecs[i], nfocks[i], dtype)
        D = displace_tf(nfocks[i], alpha_vec[i], dtype)
        SD = tf.matmul(S, D)
        SD_full = embed_single_mode_op(SD, i, nfocks, dtype)
        U = tf.matmul(SD_full, U)

    # Apply beam splitters between all pairs
    bs_idx = 0
    for i in range(n_modes):
        for j in range(i+1, n_modes):
            theta_bs = bs_params[bs_idx, 0]
            phi_bs = bs_params[bs_idx, 1]

            BS = beam_splitter_tf(theta_bs, phi_bs, nfocks[i], nfocks[j], dtype)

            BS_full = embed_two_mode_op(BS, i, j, nfocks, dtype)

            U = tf.matmul(BS_full, U)
            bs_idx += 1

    return U

@tf.function
def multi_qumode_ansatz_tf(Xvec, nfocks, ndepth, dtype=tf.complex128):
    """Full multi-qumode ansatz with depth D"""
    alphas, thetas, bs_params = unpack_params_multi_qumode(Xvec, nfocks, ndepth)

    total_dim = tf.reduce_prod(nfocks)

    U = tf.eye(total_dim, dtype=dtype)

    for d in range(ndepth):
        alpha_d = alphas[d]
        theta_d = [thetas[i][d] for i in range(len(nfocks))]
        bs_d = bs_params[d]

        U_layer = multi_qumode_ansatz_layer(alpha_d, theta_d, bs_d, nfocks, dtype)
        U = tf.matmul(U_layer, U)

    return U

def multi_qumode_vacuum(nfocks, dtype=tf.complex128):
    """Create vacuum state |0,0,...,0⟩"""
    total_dim = np.prod(nfocks)
    vac = tf.zeros([total_dim, 1], dtype=dtype)
    # Vacuum is the first basis state
    vac = tf.tensor_scatter_nd_update(vac, [[0, 0]], tf.cast([1.0], dtype))
    return vac

@tf.function
def state_from_ansatz_multi_qumode(Xvec, nfocks, ndepth, dtype=tf.complex128):
    """Prepare state |ψ⟩ = U|0,0,...,0⟩"""
    vac = multi_qumode_vacuum(nfocks, dtype)
    U = multi_qumode_ansatz_tf(Xvec, nfocks, ndepth, dtype)
    return tf.matmul(U, vac)


@tf.function
def energy_multi_qumode(psi, H_ops, coeffs):
    """Compute ⟨ψ|H|ψ⟩ for multi-qumode system"""
    energy = tf.constant(0.0, dtype=tf.float64)

    for i in tf.range(tf.shape(coeffs)[0]):
        H_psi = tf.matmul(H_ops[i], psi)
        expectation = tf.matmul(
            tf.transpose(tf.math.conj(psi)),
            H_psi
        )[0, 0]
        energy += tf.cast(coeffs[i], tf.float64) * tf.math.real(expectation)

    return energy

@tf.function
def energy_full_H(psi, H):
    """Compute ⟨ψ|H|ψ⟩ for full Hamiltonian"""
    H_psi = tf.matmul(H, psi)
    expectation = tf.matmul(
        tf.transpose(tf.math.conj(psi)),
        H_psi
    )[0, 0]

    return tf.math.real(expectation)

@tf.function
def compute_overlap(psi1, psi2):
    """Compute |⟨ψ1|ψ2⟩|²"""
    overlap = tf.matmul(
        tf.transpose(tf.math.conj(psi1)),
        psi2
    )[0, 0]
    return tf.abs(overlap) ** 2



def vqd_loss_multi_qumode(Xvec, H_ops, coeffs, prev_states_params, betas,
                          nfocks, ndepth, full_bosonic_H = None):
    """VQD loss with overlap penalties for multi-qumode system"""

    psi = state_from_ansatz_multi_qumode(Xvec, nfocks, ndepth)

    # Energy
    if full_bosonic_H is not None:
      energy = energy_full_H(psi, full_bosonic_H)
    else:
      energy = energy_multi_qumode(psi, H_ops, coeffs)


    # Overlap penalties
    overlap_penalty = tf.constant(0.0, dtype=tf.float64)

    for i, prev_params in enumerate(prev_states_params):
        psi_prev = state_from_ansatz_multi_qumode(prev_params, nfocks, ndepth)
        overlap = compute_overlap(psi, psi_prev)
        overlap_penalty += tf.cast(betas[i], tf.float64) * tf.cast(overlap, tf.float64)

    return energy + overlap_penalty


def multi_qumode_vqd(H_ops=None, coeffs=None, nfocks=[16], ndepth=20,
                     prev_states_params=None, betas=None,
                     niter=1000, threshold=1e-8, Xvec=None,
                     parallel_iterations=1, full_bosonic_H = None):
    """
    VQD for multi-qumode systems

    Arguments:
    H_ops -- List of Hamiltonian operators in full multi-qumode space
    coeffs -- coefficients for each operator in H_ops
    nfocks -- List of Fock cutoffs for each mode
    ndepth -- Ansatz depth
    betas -- List of overlap penalty weights for previously found states (length should match prev_states_params)
    niter -- Maximum number of iterations for classical optimizer
    threshold -- Convergence threshold for optimizer
    parallel_iterations -- Number of parallel iterations for optimizer
    Xvec -- Initial parameters for ansatz 
    full_bosonic_H -- Optional full Hamiltonian matrix for direct energy evaluation. Tends to be more efficient than using H_ops.
    prev_state_params -- List of parameter vectors for previously found states (for overlap penalties)
    """
    if prev_states_params is None:
        prev_states_params = []
    if betas is None:
        betas = []

    H_ops_tf = None
    coeffs_tf = None
    if H_ops is not None and coeffs is not None:
      H_ops_tf = tf.stack([tf.constant(H, dtype=tf.complex128) for H in H_ops])
      coeffs_tf = tf.constant(coeffs, dtype=tf.float64)

    n_modes = len(nfocks)
    n_bs_pairs = n_modes * (n_modes - 1) // 2
    dim_params = ndepth * (n_modes + sum(nfocks) + 2 * n_bs_pairs)

    # Initialize parameters
    if Xvec is None:
        # Random initialization
        disp_init = tf.random.uniform([ndepth * n_modes], -0.5, 0.5, dtype=tf.float64)
        snap_init = tf.random.uniform([ndepth * sum(nfocks)], 0, 2*np.pi, dtype=tf.float64)
        bs_init = tf.random.uniform([ndepth * n_bs_pairs * 2], 0, np.pi, dtype=tf.float64)
        Xvec = tf.Variable(tf.concat([disp_init, snap_init, bs_init], axis=0))
    else:
        Xvec = tf.Variable(Xvec, dtype=tf.float64)

    # Loss function
    @tf.function
    def loss_fn(X):
        return vqd_loss_multi_qumode(X, H_ops_tf, coeffs_tf, prev_states_params,
                                     betas, nfocks, ndepth, full_bosonic_H = full_bosonic_H)

    # Gradient function
    @tf.function
    def value_and_gradients(X):
        with tf.GradientTape() as tape:
            tape.watch(X)
            loss = loss_fn(X)
        grads = tape.gradient(loss, X)
        return loss, grads

    # Optimize
    result = tfp.optimizer.lbfgs_minimize(
        value_and_gradients,
        initial_position=Xvec,
        tolerance=threshold,
        max_iterations=niter,
        parallel_iterations=parallel_iterations
    )

    final_state = state_from_ansatz_multi_qumode(result.position, nfocks, ndepth)
    if full_bosonic_H is not None:
      final_energy = energy_full_H(final_state, full_bosonic_H)
    else:
      final_energy = energy_multi_qumode(final_state, H_ops_tf, coeffs_tf)

    return final_energy, result.position, result, final_state


def qt2tf(qt_object, dtype=tf.complex128):
    """Convert QuTip object to TensorFlow tensor"""
    if hasattr(qt_object, 'full'):
        # QuTip object
        return tf.constant(qt_object.full(), dtype=dtype)
    elif isinstance(qt_object, np.ndarray):
        # Numpy array
        return tf.constant(qt_object, dtype=dtype)
    else:
        # Already a tensor or other type
        return tf.cast(qt_object, dtype=dtype)

def get_multi_qumode_ops(qubit_ham, op_dict, n_qubits, quditvec, verb=1):
    """
    Map qubit Hamiltonian to multi-qumode operators

    Arguments:
    qubit_ham -- OpenFermion QubitOperator
    op_dict -- Pre-computed SNAP parameters dictionary
    n_qubits -- Total number of qubits
    quditvec -- List of qubits per mode, e.g., [4, 4] for H4

    """
    import qutip as qt

    n_modes = len(quditvec)
    nfock = 16  

    ops_list, coeffs = qubit_string_from_openfermion(qubit_ham, n_qubits, quditvec)

    if verb:
      print(f"len(coeffs) = {len(coeffs)}")

    # Initialize lists for each mode
    mode_op_lists = [[] for _ in range(n_modes)]

    # Build operators for each mode separately
    for i in range(len(coeffs)):
      if verb and i%len(coeffs)%5 == 0:
        print(f"building op {i}")
      for mode_idx in range(n_modes):
        if verb and i%len(coeffs)%5 == 0:
          print(f"building ops for qumode {mode_idx}")

        pauli_str = ops_list[i][mode_idx]

        if pauli_str == 'I' * quditvec[mode_idx]:
            # Identity on this mode
            mode_op_lists[mode_idx].append(qt.qeye(nfock))
        else:
            # Get SNAP unitary from pre-computed dictionary
            U = snap_uni_four_pauli(pauli_str, op_dict)
            mode_op_lists[mode_idx].append(U)

    mode_op_lists_tf = []
    for mode_ops in mode_op_lists:
        mode_ops_tf = [qt2tf(op) for op in mode_ops]
        mode_op_lists_tf.append(mode_ops_tf)

    return mode_op_lists_tf, coeffs

def partition_qubits(n_qubits, qubits_per_mode=4):
    """
    Partition qubits into groups for multi-qumode mapping
    Following paper's approach: each group has ≤ 4 qubits
    """
    n_modes = (n_qubits + qubits_per_mode - 1) // qubits_per_mode
    partitions = []

    for i in range(n_modes):
        start = i * qubits_per_mode
        end = min(start + qubits_per_mode, n_qubits)
        partitions.append(list(range(start, end)))

    return partitions


def find_excited_states_multi_qumode(H_ops=None, coeffs=None, nfocks=[16], ndepth=20, num_states=1,
                                     beta_factor=3.0, niter=2000, threshold=1e-10,
                                     parallel_iterations=1, Xvec = None, full_bosonic_H = None,
                                     prev_state_params = [], save_files = False, save_dir = None, hhdis = None):
    """
    Find ground and excited states using multi-qumode VQD
    
    Arguments:
    H_ops -- List of Hamiltonian operators in full multi-qumode space
    coeffs -- coefficients for each operator in H_ops
    nfocks -- List of Fock cutoffs for each mode
    ndepth -- Ansatz depth
    num_states -- Total number of states to find (including ground state)
    beta_factor -- Overlap penalty weight for excited states
    niter -- Maximum number of iterations for classical optimizer
    threshold -- Convergence threshold for optimizer
    parallel_iterations -- Number of parallel iterations for optimizer
    Xvec -- Initial parameters for ansatz 
    full_bosonic_H -- Optional full Hamiltonian matrix for direct energy evaluation. Tends to be more efficient than using H_ops.
    prev_state_params -- List of parameter vectors for previously found states (for overlap penalties)
    save_files -- Whether to save results to files after each state is found
    save_dir -- Directory to save results if save_files is True
    hhdis -- radial distances used (only used when saving data to files)    
    """

    energies = []
    state_params = prev_state_params.copy()
    results = []
    states = []
    overlaps_list = []

    num_prev_states = len(prev_state_params)
    print(f"num_prev_states = {num_prev_states}")

    for k in range(num_states):
        k+=num_prev_states

        print(f"\nFinding state {k}...")

        if k == 0:
            # Ground state
            energy, params, result, state = multi_qumode_vqd(
                H_ops, coeffs, nfocks, ndepth,
                niter=niter, threshold=threshold,
                parallel_iterations=parallel_iterations, Xvec = Xvec,
                full_bosonic_H = full_bosonic_H
            )
        else:
            # Excited states with overlap penalties
            betas = [beta_factor] * k

            energy, params, result, state = multi_qumode_vqd(
                H_ops, coeffs, nfocks, ndepth,
                prev_states_params=state_params,
                betas=betas,
                niter=niter, threshold=threshold,
                parallel_iterations=parallel_iterations, Xvec = Xvec,
                full_bosonic_H = full_bosonic_H
            )

        # Check orthogonality
        state_overlaps = []
        if k > 0:
            for i in range(k - num_prev_states):
                overlap = compute_overlap(state, states[i]).numpy()
                print(f"Overlap with state {i+num_prev_states}: {overlap:.6e}")
                state_overlaps.append({'with_state': i + num_prev_states, 'overlap': float(overlap)})


        overlaps_list.append(state_overlaps)
        energies.append(energy.numpy())
        state_params.append(params.numpy())
        results.append(result)
        states.append(state)
        if save_files:
          save_data_to_files(energies, state_params, results, psis=states, overlaps_list=overlaps_list,
                              save_dir=save_dir, ndepth=ndepth, nfocks=nfocks, num_states=num_states, beta_factor=beta_factor, 
                              niter=niter, threshold=threshold, hhdis=hhdis)

        print(f"State {k} energy: {energy.numpy():.6f}")
        print(f"Converged: {result.converged.numpy()}")


    return energies, state_params, results, states

def unpack_params_snap_disp_tf(X, nfock):
    """
    TensorFlow-compatible parameter unpacking
    """
    ndepth = tf.shape(X)[0] // (nfock + 1)

    alpha = X[:ndepth]
    d1 = ndepth * nfock
    theta = tf.reshape(X[ndepth:ndepth+d1], (ndepth, nfock))

    return alpha, theta

def snap_disp_op_tf(alpha, theta_vec, nfock, dtype=tf.complex128):
    """
    SNAP-displacement operator S(θ)D(α)
    """
    S = snap_op_tf(theta_vec, nfock, dtype)
    D = displace_tf(nfock, alpha, dtype)
    return tf.matmul(S, D)

def ham_linear_sym_h4(R, f2q='JWT'):
    """
    (0) -- (R) -- (2R) -- (3R)

    Arguments:
    R -- bond distance in Angstrom
    f2q -- Fermion to qubit mapping
    """
    # Define parameters
    basis = "sto-3g"
    multiplicity = 3
    charge = 0

    # XYZ coordinates

    geometry = [("H", (0, 0, 0)), \
                ("H", (0, 0, R)), \
                ("H", (0, 0, 2*R)), \
                ("H", (0, 0, 3*R))]

    # Define molecule
    molecule = MolecularData(geometry, basis, multiplicity, charge)

    # Hartree-Fock
    hf_molecule = run_pyscf(molecule, run_scf=1, run_fci=0, verbose = 0) 

    ham_fermi = get_fermion_operator(hf_molecule.get_molecular_hamiltonian())


    # Qubit mapping
    if f2q == 'JWT':
        ham = jordan_wigner(ham_fermi)
    elif f2q == 'BKT':
        ham = bravyi_kitaev(ham_fermi)

    return ham


def ham_linear_sym_h2(R, f2q='JWT'):
    """
    (0) -- (R)

    Arguments:
    R -- bond distance in Angstrom
    f2q -- Fermion to qubit mapping
    """
    # Define parameters
    basis = "sto-3g"
    multiplicity = 1
    charge = 0

    # XYZ coordinates

    geometry = [("H", (0, 0, 0)), \
                ("H", (0, 0, R))]

    # Define molecule
    molecule = MolecularData(geometry, basis, multiplicity, charge)

    # Hartree-Fock
    hf_molecule = run_pyscf(molecule, run_scf=1, verbose = 1)

    #print(hf_molecule.get_molecular_hamiltonian()) # for testing

    # Fermionic Hamiltonian
    ham_fermi = get_fermion_operator(hf_molecule.get_molecular_hamiltonian())

    # Qubit mapping
    if f2q == 'JWT':
        ham = jordan_wigner(ham_fermi)
    elif f2q == 'BKT':
        ham = bravyi_kitaev(ham_fermi)

    return ham

def snap_uni_four_pauli(pword, data_dict):
    """
    Get SNAP-displacement unitary with Fock cutoff = 16.

    Arguments:
    pword -- four qubit Pauli word
    data_dict -- parameter dictionary
    """
    nfock = 16
    Xvec = np.array( data_dict[pword][2] )
    U = snap_disp_ansatz_tf(Xvec, nfock)

    return U

def qubit_partition(qubitop, nqubit, quditvec):
    """
    Transform one qubit string to a set of a qubit strings based on
    the partition chosen.

    Arguments:
    qubitop  -- A single Pauli operator string as a tuple of tuples
    nqubit   -- Number of qubit operators including identity
    quditvec -- Qudit vector for reorganizing qubit Hilbert space
    """
    if sum(quditvec) != nqubit:
        raise ValueError("Wrong qudit vector elements")

    qubitstr = append_ids(qubitop, nqubit)

    newstrings = []
    for j in range(len(quditvec)):
        templist = []
        for p in range(quditvec[j]):
            temptuple = (p, qubitstr[sum(quditvec[:j]) + p][1])
            templist.append(temptuple)
        newstrings.append(templist)

    return newstrings

def snap_disp_ansatz_tf(Xvec, nfock, dtype=tf.complex128):
    """
    SNAP-displacement ansatz implemented entirely in TensorFlow
    """
    alphavec, thetamat = unpack_params_snap_disp_tf(Xvec, nfock)

    ndepth = tf.shape(alphavec)[0]

    # print(f"ndepth = {ndepth}") # for testing

    # Initialize with identity
    uni = tf.eye(nfock, dtype=dtype)

    # Apply layers
    def apply_layer(i, current_uni):
        # Get parameters for this layer
        alpha_i = alphavec[i]
        theta_i = thetamat[i, :]

        # Create SNAP-disp operator for this layer
        layer_op = snap_disp_op_tf(alpha_i, theta_i, nfock, dtype)

        # Apply to current unitary
        return tf.matmul(layer_op, current_uni)

    # apply all layers
    for i in tf.range(ndepth):
        uni = apply_layer(i, uni)

    return uni



def qubit_string_from_openfermion(qubitop, nqubit, quditvec):
    """
    Transform a QubitOperator to a set of a qubit strings and coefficients based on
    the partition chosen.

    Arguments:
    qubitop  -- QubitOperator
    nqubit   -- Number of qubit operators including identity
    quditvec -- Qudit vector for reorganizing qubit Hilbert space
    """
    
    if sum(quditvec) != nqubit:
        raise ValueError("Wrong qudit vector elements")


    qubit_strings = list(qubitop.terms.keys())
    coeffs = np.array(list( qubitop.terms.values() ))


    ops_list = []
    for qubit_op in qubit_strings:
        ops_list.append( simple_qubit_string(qubit_op, nqubit, quditvec) )

    return ops_list, coeffs

def append_ids(qubitstr, nqubits):
    """
    Modify a qubit string so that identities are explicity included.

    Arguments:
    qubitstr -- Input qubit string as a tuple of tuples
    nqubits  -- Number of qubits
    """
    newstr = [None] * nqubits

    for p in range(len(qubitstr)):
        newstr[qubitstr[p][0]] = qubitstr[p]

    for p in range(nqubits):
        if newstr[p] == None:
            newstr[p] = (p, 'I')

    return newstr

def qubit_partition(qubitop, nqubit, quditvec):
    """
    Transform one qubit string to a set of a qubit strings based on
    the partition chosen.

    Arguments:
    qubitop  -- A single Pauli operator string as a tuple of tuples
    nqubit   -- Number of qubit operators including identity
    quditvec -- Qudit vector for reorganizing qubit Hilbert space
    """
    # Check
    if sum(quditvec) != nqubit:
        raise ValueError("Wrong qudit vector elements")

    qubitstr = append_ids(qubitop, nqubit)

    newstrings = []
    for j in range(len(quditvec)):
        templist = []
        for p in range(quditvec[j]):
            temptuple = (p, qubitstr[sum(quditvec[:j]) + p][1])
            templist.append(temptuple)
        newstrings.append(templist)

    return newstrings


def simple_qubit_string(qubitop, nqubit, quditvec):
    """
    Transform one qubit string to a set of a qubit strings based on
    the partition chosen.

    Arguments:
    qubitop  -- A single Pauli operator string as a tuple of tuples
    nqubit   -- Number of qubit operators including identity
    quditvec -- Qudit vector for reorganizing qubit Hilbert space
    """
    
    if sum(quditvec) != nqubit:
        raise ValueError("Wrong qudit vector elements")

    qubitstr = qubit_partition(qubitop, nqubit, quditvec)

    string_list = []
    for j in range(len(quditvec)):
        string_list.append(''.join(op for _, op in qubitstr[j]))

    return string_list

def prepare_ops_tf_no_hadamard(U_list, dtype=tf.complex128):
    """
    Convert list of QuTip operators to TensorFlow tensors without applying Hadamard transformation.

    Arguments:
    U_list -- list of four-qubit unitaries
    """
    T_list = [[] for _ in range(len(U_list))]

    for qumode_index in range(len(U_list)):
      for op_index in range(len(U_list[qumode_index])):
        U_op = qt.Qobj(U_list[qumode_index][op_index])
        T_list[qumode_index].append(qt2tf(U_op, dtype=dtype))

    return T_list

def construct_bosonic_hamiltonian(H_ops, coeffs):
    """Construct the full bosonic Hamiltonian from the list of operators and coefficients.
    """
    nqumodes = len(H_ops)
    depth = len(H_ops[0])

    dim = 1
    for qumode_idx in range(nqumodes):
        dim *= len(H_ops[qumode_idx][0])

    H_bosonic = tf.zeros((dim, dim), dtype=H_ops[0][0].dtype)

    for gate_idx in range(depth):
        multimode_op = H_ops[0][gate_idx]
        for qumode_idx in range(1, nqumodes):
            multimode_op = tf.experimental.numpy.kron(
                multimode_op,
                H_ops[qumode_idx][gate_idx]
            )
        H_bosonic = tf.add(H_bosonic, coeffs[gate_idx] * multimode_op)

    return H_bosonic

def slice_bosonic_hamiltonian(H_bosonic, n_electrons):
  """Slice the full bosonic Hamiltonian to the subspace with a fixed number of electrons."""
  indices_to_keep = [i for i in range(H_bosonic.shape[0]) if bin(i).count('1') == n_electrons]
  H_bosonic_sliced = tf.gather(tf.gather(H_bosonic, indices_to_keep, axis=0), indices_to_keep, axis=1)
  return H_bosonic_sliced

def compute_fci_states_h4(R, nroots):
    """
    Compute FCI states for H4 using PySCF and PySCF's FCI solver.
    """
    mol = gto.M(atom=[['H', (i*R,0,0)] for i in range(4)],
                basis='sto-3g')
    mf = scf.RHF(mol).run(verbose=0)
    cis = fci.FCI(mol, mf.mo_coeff)
    e, _ = cis.kernel(nroots=nroots)
    return np.sort(e)

def compute_fci_states_h2(R, nroots):
    """
    Compute FCI states for H2 using PySCF and PySCF's FCI solver.
    """
    mol = gto.M(atom=[['H', (i*R,0,0)] for i in range(2)],
                basis='sto-3g')
    mf = scf.RHF(mol).run(verbose=0)
    cis = fci.FCI(mol, mf.mo_coeff)
    e, _ = cis.kernel(nroots=nroots)
    return np.sort(e)


def save_data_to_files(energies=[], state_params=[], results=[], psis=[], overlaps_list = [], save_dir = None, 
                       ndepth=None, nfocks=None, num_states=None, beta_factor=None, niter=None, threshold=None,
                         hhdis=None, coeffs=None):

    """
    Save data to JSON and NumPy files.
    """
    SAVE_DIR = save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    with open(f'{SAVE_DIR}/overlaps.json', 'w') as f:
        json.dump(overlaps_list, f, indent=2)

    with open(f'{SAVE_DIR}/energies.json', 'w') as f:
        energies_list = [e.item() if hasattr(e, 'item') else e for e in energies]
        json.dump(energies_list, f, indent=2)


    state_params_np = [p for p in state_params]
    np.save(f'{SAVE_DIR}/state_params.npy', state_params_np)

    results_summary = [{
        'num_iterations': int(r.num_iterations.numpy()),
        'converged': bool(r.converged.numpy()),
        'failed': bool(r.failed.numpy()),
        'objective_value': float(r.objective_value.numpy())
    } for r in results]
    with open(f'{SAVE_DIR}/results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    psis_np = [p.numpy() for p in psis]
    np.save(f'{SAVE_DIR}/psis.npy', psis_np)
    scalar_values = {}
    if (hhdis):
        scalar_values['hhdis'] = float(hhdis)
    if (ndepth):
        scalar_values['ndepth'] = int(ndepth)
    if (num_states):
        scalar_values['num_states'] = int(num_states)
    if (beta_factor):
        scalar_values['beta_factor'] = float(beta_factor)
    if (niter):
        scalar_values['niter'] = int(niter)
    if (threshold):
        scalar_values['threshold'] = float(threshold)
        
    if hhdis is not None:
        scalar_values['hhdis'] = float(hhdis)

    if coeffs is not None:
        scalar_values['coeffs'] = [float(c) for c in coeffs]
    with open(f'{SAVE_DIR}/run_parameters.json', 'w') as f:
        json.dump(scalar_values, f, indent=2)

    print(f"  [saved checkpoint to {SAVE_DIR} after state {len(energies)-1}]")