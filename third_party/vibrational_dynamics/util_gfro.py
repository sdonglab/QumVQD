# Original source: https://github.com/shreyasvm3/Vibrational_Dynamics_on_Bosonic_QCs
# Copyright (c) 2026 Shreyas Malpathak and Sangeeth Das Kallullathil
# Licensed under the MIT License
# Reference: Malpathak, Das Kallullathil & Izmaylov, JPCL 2025, arXiv:2411.17950
# Modifications by Marlon Jost: edited import statements  of util_hamil.py

import numpy as np
import json

from .util_hamil import extract_coeffs_and_ops # [Marlon Jost] edited import statement of util_hamil.py, which is now in the same folder as util_gfro.py, so we can use relative imports

from openfermion.ops import BosonOperator
from openfermion.transforms import normal_ordered

from scipy.optimize import minimize
from scipy import linalg


# find norm of the Hamiltonian but only for cubic and quartic terms
def norm_m(H):

    H_vector, ops = extract_coeffs_and_ops(H)

    indices_cq = [i for i, op in enumerate(ops) if len(op) == 3 or len(op) == 4]

    norm_cq = []

    for i in indices_cq:
        value = H_vector[i] ** 2
        norm_cq.append(value)

    norm_cqs = np.real(sum(norm_cq))

    return norm_cqs


# creates vectors that contains the coefficients with matching indices of two bosonic operators
def vectors(H, H_f):

    H_coeffs, H_ops = extract_coeffs_and_ops(H)
    H_f_coeffs, H_f_ops = extract_coeffs_and_ops(H_f)

    H_dict = {op: coeff for op, coeff in zip(H_ops, H_coeffs)}
    H_f_dict = {op: coeff for op, coeff in zip(H_f_ops, H_f_coeffs)}

    # Ensure both Hamiltonians have the same operators
    all_ops = set(H_ops).union(set(H_f_ops))

    # Sort the operators
    sorted_ops = sorted(all_ops, key=lambda x: (len(x), x))

    # Create vectors with matching indices
    H_vector = [H_dict.get(op, 0) for op in sorted_ops]
    H_f_vector = [H_f_dict.get(op, 0) for op in sorted_ops]

    return H_vector, H_f_vector, sorted_ops


# reconstruct bosonic operator from coefficients and operators
def reconstruct_boson_operator(coeffs, ops):
    boson_operator = BosonOperator()
    for coeff, op in zip(coeffs, ops):
        boson_operator += BosonOperator(op, coeff)
    return boson_operator


# subtract elements in two lists
def subtract_lists(list1, list2):
    return [a - b for a, b in zip(list1, list2)]


# Subtracting one Bosonic Hamitonian from another one


def subtract(H, H_f):
    # Extract coefficients and operators
    H_coeffs, H_ops = extract_coeffs_and_ops(H)
    H_f_coeffs, H_f_ops = extract_coeffs_and_ops(H_f)

    H_dict = {op: coeff for op, coeff in zip(H_ops, H_coeffs)}
    H_f_dict = {op: coeff for op, coeff in zip(H_f_ops, H_f_coeffs)}

    # Ensure both Hamiltonians have the same operators
    all_ops = set(H_ops).union(set(H_f_ops))

    # Sort the operators
    sorted_ops = sorted(all_ops, key=lambda x: (len(x), x))

    # Create vectors with matching indices
    H_vector = [H_dict.get(op, 0) for op in sorted_ops]
    H_f_vector = [H_f_dict.get(op, 0) for op in sorted_ops]

    H_new = subtract_lists(H_vector, H_f_vector)

    H = reconstruct_boson_operator(H_new, sorted_ops)

    return H


# Define the initial parameters in the exponent A=[[P,Q],[Q,P]], L and G
def initial(n):
    p = np.random.uniform(-2, 2, (n, n))
    P = p.T + p  # P satifies P=P^T

    q = np.random.uniform(-2, 2, (n, n))
    Q = q.T + q  # Q satifies Q=Q^T

    # Random gamma vector
    G = np.random.uniform(-1, 1, n)

    # Random lambda matrix
    L = np.random.uniform(0, 1, (n, n))

    return P, Q, G, L


# create super matrix A=[[P,Q],[Q,P]] and exponentiate it(find [[U,-V],[-V,U]]= e^A)
def super_matrix_exp(P, Q):

    # Create the super matrix A
    A = np.block([[P, Q], [Q, P]])

    expA = linalg.expm(A)

    return expA


# exatract U and V matrices from expA [[U,-V],[-V,U]]= e^A)
def extract_sub_matrices(expA, n):

    # Extract sub-matrices
    U = expA[:n, :n]
    V = -expA[:n, n:]

    return U, V


# create a vector contain upper trangular elements of a symmetric matrix
def triu_to_vector(matrix):

    upper_tri_ind = np.triu_indices_from(matrix)
    upper_tri_vector = matrix[upper_tri_ind]

    return upper_tri_vector


# recreate symmetric matrix from triangular vector of size n(n+1)/2
def vector_to_triu(vector, n):

    matrix = np.zeros((n, n))

    # upper triangular indices
    upper_tri_ind = np.triu_indices(n)

    # Adding the values from the vector to matrix
    matrix[upper_tri_ind] = vector

    # copy the upper triangular to the lower triangular
    i, j = upper_tri_ind
    matrix[j, i] = matrix[i, j]

    return matrix


# extract n(n-1)/2 off diagonal upper terms from P
def skew_sym_to_vec(P, n):

    vec = []

    for i in range(n):
        for j in range(i + 1, n):
            vec.append(P[i, j])

    return vec


# Create skew_symmetric P
def vec_to_skew_sym(vec, n):

    A = np.zeros((n, n))

    # Fill the upper triangular part
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = vec[k]
            A[j, i] = -vec[k]
            k += 1

    return A


# Calculate quartic fragments from calculated L, C_b, C_bd
def compute_fragment(L, C_b, C_bd, n):

    result = BosonOperator()

    for i in range(n):
        for j in range(n):

            # Calculate the term L[i, j] * C_bd[i]^T * C_b[i] * C_bd[j]^T * C_b[j]
            term = L[i, j] * C_bd[i] * C_b[i] * C_bd[j] * C_b[j]

            result += term

    result = normal_ordered(result)

    return result


# combine necessary parmaters in P,Q,G,L in to single vector X
def flatten_matrices(P, Q, G, L, n):
    # Flatten the matrices and the vector
    P_flat = skew_sym_to_vec(P, n)
    Q_flat = triu_to_vector(Q)
    L_flat = L.flatten()

    # Concatenate all flattened arrays into a single vector
    X = np.concatenate([P_flat, Q_flat, G, L_flat])

    return X


# recreate P,Q,G,L from vector X
def recreate_matrices(X, n):
    # Calculate the sizes of the arrays
    sp = n * (n - 1) // 2
    sq = n * (n + 1) // 2
    sg = n

    # Extract the arrays from X
    P_flat = X[:sp]
    Q_flat = X[sp : sp + sq]
    G = X[sp + sq : sp + sq + sg]
    L_flat = X[sp + sq + sg :]

    # Recreate matrices
    P = vec_to_skew_sym(P_flat, n)
    Q = vector_to_triu(Q_flat, n)
    L = L_flat.reshape((n, n))

    return P, Q, G, L


# Remove the fragments that have norm less than 10^-3
def filter_params(params, n):
    threshold = 1e-3
    filtered_params = [
        param for param in params if norm_m(fragments(param, n)) >= threshold
    ]
    return filtered_params


# Print the value of the cost in a file
def printx(x):
    cost_value = cost_function(x)
    with open("co2.txt", "a") as f:
        f.write(f"cost: {cost_value}\n")
    print(f"cost: {cost_value}")


# Create a vecctor for calculations
def create_bosonic_vector(n):
    return [1.0] * n


# Calculate list of bosonic terms for U b^dagger
def bosonic_BD(U):

    n = np.shape(U)[1]

    B = create_bosonic_vector(n)

    # bosonic operators
    bosonic_operators_listU = []

    for row in U:
        # Initialize an empty BosonOperator for the current row
        bosonic_operator = BosonOperator()

        # Iterate over the elements of B and the current row of U
        for coeff, index in zip(B, range(len(row))):
            # Multiply the coefficient with corresponding number in the row and create a term
            term = BosonOperator(((index, 1),), coeff * row[index])

            # Add the term to the BosonOperator for the current row
            bosonic_operator += term

        # Append the resulting BosonOperator to the list
        bosonic_operators_listU.append(bosonic_operator)

    return bosonic_operators_listU


# Calculate list of bosonic terms for U b
def bosonic_B(U):

    n = np.shape(U)[1]

    B = create_bosonic_vector(n)

    # bosonic operators
    bosonic_operators_listU = []

    for row in U:
        # Initialize an empty BosonOperator for the current row
        bosonic_operator = BosonOperator()

        # Iterate over the elements of B and the current row of U
        for coeff, index in zip(B, range(len(row))):
            # Multiply the coefficient with the corresponding number in the row and create a term
            term = BosonOperator(((index, 0),), coeff * row[index])

            # Add the term to the BosonOperator for the current row
            bosonic_operator += term

        # Append the resulting BosonOperator to the list
        bosonic_operators_listU.append(bosonic_operator)

    return bosonic_operators_listU


# Calculate list of bosonic terms corresponding to the constant term


def bosonic_G(G):

    # resulting terms
    iden_list = []

    for coeff in G.flat:
        # Create a term by multiplying the coefficient with the bosonic identity operator
        term = BosonOperator("", coeff)

        # Append the resulting term to the list
        iden_list.append(term)

    return iden_list


# Calculate transformed operators
def tranformed_operators(U, V, G):

    Ub = bosonic_B(U)  # Calculate list of bosonic terms for U b
    Ubd = bosonic_BD(U)  # Calculate list of bosonic terms for U b^dagger
    Vb = bosonic_B(V)  # Calculate list of bosonic terms for V b
    Vbd = bosonic_BD(V)  # Calculate list of bosonic terms for V b^dagger
    Gi = bosonic_G(G)  # Calculate list of bosonic terms for I G

    # tranformed operators
    C_b = []
    C_bd = []

    # Iterate over corresponding elements of the input lists
    for inUb, inUbd, inVb, inVbd, inGi in zip(Ub, Ubd, Vb, Vbd, Gi):
        # Add corresponding elements and append it to the result list
        C_b.append(inUb - inVbd + inGi)
        C_bd.append(inUbd - inVb + inGi)

    return C_b, C_bd


# remove the cubic and quartic terms from the Hamiltonian


def H_quadatic_op(H):

    Hv, ops = extract_coeffs_and_ops(H)

    indices_cq = [i for i, op in enumerate(ops) if len(op) < 3]

    Hvs = list()
    opss = list()

    for i in indices_cq:

        Hvs.append(Hv[i])
        opss.append(ops[i])

    H_qd = reconstruct_boson_operator(Hvs, opss)

    return H_qd


# Obtain fragments from previously calculated parameters in non-rotated coordinates
def non_rotated_fragments(name, H, n):
    # Read the JSON file
    file_path = f"Calculated_fragments/params_{name}.json"
    with open(file_path, "r") as file:
        data = json.load(file)

    # Convert the data into a list of NumPy arrays
    params = [np.array(entry) for entry in data]

    norms = list()

    norms.append(norm_m(H))  # norm of the initial H (cubic and quartic)

    # List to store each H2
    H_frags = []

    for i in range(len(params)):
        H2 = fragments(params[i], n)
        H_frags.append(H2)  # Save each H2 to the list
        H = subtract(H, H2)
        norms.append(norm_m(H))

    H_frags.insert(0, H_quadatic_op(H))
    # Edited by SVM 09/10/24. Original appended the upto quadratic term.
    # Now it is inserted in the 1st position.

    return H_frags, norms


# calculating the cost function(sum of square of the coeffiecients of H-H_f operator)


def cost_function(X, H, n):

    P, Q, G, L = recreate_matrices(X, n)  # Calculate P,Q,G,L from paramter vector X

    expA = super_matrix_exp(P, Q)  # Calculate exp([[P,Q],[Q,P]])

    U, V = extract_sub_matrices(
        expA, n
    )  # Extract U and V from exp([[P,Q],[Q,P]]) = [[U,-V],[-V,U]]

    C_b, C_bd = tranformed_operators(U, V, G)  # Calculate transformed operators

    H_f = compute_fragment(
        L, C_b, C_bd, n
    )  # Calculate quartic fragments from calculated L, C_b, C_bd

    H_vector, H_f_vector, sorted_ops = vectors(
        H, H_f
    )  # extract coeffiecients and corresponding operators from H and H_f

    # Finding indices corresponds to cubic and quartic terms in the Hamiltonian and the Fragment
    indices_cq = [i for i, op in enumerate(sorted_ops) if len(op) == 3 or len(op) == 4]

    cost_fn = []

    for i in indices_cq:
        value = H_vector[i] - H_f_vector[i]
        value_sq = value**2
        cost_fn.append(value_sq)

    cost = np.real(sum(cost_fn))  # sum of the squares of the coefficients of H-H_f

    return cost


# Compute quartic fragments from X
def fragments(X, n):

    P, Q, G, L = recreate_matrices(X, n)

    expA = super_matrix_exp(P, Q)

    U, V = extract_sub_matrices(expA, n)

    C_b, C_bd = tranformed_operators(U, V, G)

    H_f = compute_fragment(L, C_b, C_bd, n)

    return H_f


# Obtain the fragments
def obtain_fragment(H, n, options):

    # Cost function

    def cost(X):
        return cost_function(X, H, n)

    P, Q, G, L = initial(n)
    X = 1e-3 * flatten_matrices(
        P, Q, G, L, n
    )  # 1e-3 is to make sure that exponentiation does not blowup

    # Minimize the cost function
    result = minimize(
        cost, X, method="BFGS", tol=None, options=options
    )  # , callback=printx)

    return result
