# Original source: https://github.com/shreyasvm3/Vibrational_Dynamics_on_Bosonic_QCs
# Copyright (c) 2026 Shreyas Malpathak and Sangeeth Das Kallullathil
# Licensed under the MIT License
# Reference: Malpathak, Das Kallullathil & Izmaylov, JPCL 2025, arXiv:2411.17950
# Modifications by Marlon Jost: edited import statements of util_hamil.py and util_gfro.py


import numpy as np

from openfermion.ops import BosonOperator
from openfermion.transforms import normal_ordered
from openfermion import get_sparse_operator

from scipy.linalg import eig
from scipy.sparse.linalg import eigsh

from .util_hamil import extract_coeffs_and_ops 
from .util_gfro import tranformed_operators,reconstruct_boson_operator 



# reorder so that dagger operators always comes first
def reorder_brackets(ops):
    reordered_terms = []
    for term in ops:
        first, second = term
        if first[1] < second[1]:
            reordered_terms.append((second, first))
        else:
            reordered_terms.append((first, second))
    return reordered_terms

# Create A and B matrices as in eq 2.1 Intro to Quantum stat mech Bogolubov
def create_AB(H_q,n):
    
    H_c,ops=extract_coeffs_and_ops(H_q)
    
    ops = reorder_brackets(ops) # reorder so that dagger operators always comes first

    A = np.zeros((n, n))
    B = np.zeros((n, n))

    for i, term in enumerate(ops):

        first, second = term
        
        if first[1] == 1 and second[1] == 0: # b_p' b_q                   
                A[first[0], second[0]] = H_c[i]
                A[second[0], first[0]] = H_c[i]
        elif first[1] == 1 and second[1] == 1: # b_p' b_q' 
            if first[0] == second[0]:
                B[first[0], second[0]] =  2*H_c[i]
            else:
                B[first[0], second[0]] = H_c[i]
                B[second[0], first[0]] = H_c[i]
               
    return A,B


def solve_uv_lambda(A, B, n):
    #I = np.eye(n)
    # Construct the block matrix for the eigenvalue problem
    top_block = np.hstack((A, B))
    bottom_block = np.hstack((-B, -A))
    full_matrix = np.vstack((top_block, bottom_block))

    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = eig(full_matrix)

    # Extract the solutions for lambda, u, and v
    lambdas = []
    us = []
    vs = []

    for i in range(len(eigenvalues)):
        lambda_k = eigenvalues[i]
        
        # Split the eigenvector into u and v components
        u = eigenvectors[:n, i]
        v = eigenvectors[n:, i]
        
        lambdas.append(lambda_k)
        us.append(u)
        vs.append(v)       

    return  np.array(us).T, np.array(vs).T,np.array(lambdas)


def choose_uv(U,V,L,n):
    
    ind = []
    
    for i in range(2*n):
        
        if U[:,i].T @ U[:,i] > V[:,i].T @ V[:,i]:
        
            ind.append(i)
            
    ind = np.array(ind)
    
    U = U[:,ind]
    V = V[:,ind]
    L = L[ind]
    
    return U,V,L.reshape(-1, 1)


def scale_matrices(U, V,n):

    U_scaled = np.zeros_like(U)
    V_scaled = np.zeros_like(V)
    
    for j in range(n):
        u_j = np.dot(U[:, j], U[:, j])
        v_j = np.dot(V[:, j], V[:, j])
        
        alpha_j = 1 / np.sqrt(u_j - v_j)
        
        U_scaled[:, j] = alpha_j * U[:, j]
        V_scaled[:, j] = alpha_j * V[:, j]
    
    return U_scaled, V_scaled

def create_C(H_l,n):
    
    coeff,ops = extract_coeffs_and_ops(H_l)

    C = np.zeros((n, 1))

    for i, term in enumerate(ops):
        if term[0][1] == 1 : # b_p'                  
            C[term[0][0]] = coeff[i]
            
    C = np.array(C).reshape(-1, 1)
                
    return np.array(C)

def create_D(C,U,V): 
    return np.array(U.T@C  + V.T@C)

def create_K(L,V):
    
    sum_V = np.sum(V**2, axis=0)
    
    return -(L.T@sum_V)

def create_D_over_L(D,L): 
    
    D = np.array(D)
    
    L = np.array(L)
    
    return D/L

def create_new_K(K,L,D):
    
    D2 = D**2
    
    K_new = K - np.sum(D2/L)
    
    return K_new

# remove the cubic and quartic terms from the Hamiltonian

def H_quadatic(H):
    
    Hv,ops = extract_coeffs_and_ops(H)    

    indices_cq = [i for i, op in enumerate(ops) if len(op) < 3]

    Hvs = list()
    opss = list()

    for i in indices_cq:
        
        Hvs.append(Hv[i])
        opss.append(ops[i])
        
    H_qd = reconstruct_boson_operator(Hvs, opss)
    
    return H_qd


# separte just quadratic terms (b'b , bb, and b'b')

def H_quadatic_only(H):
    
    Hv,ops = extract_coeffs_and_ops(H)
    

    indices_cq = [i for i, op in enumerate(ops) if len(op) == 2]

    Hvs = list()
    opss = list()

    for i in indices_cq:
        
        Hvs.append(Hv[i])
        opss.append(ops[i])
        
    H_qd = reconstruct_boson_operator(Hvs, opss)
    
    return H_qd

# separte just linear terms (b' and b)

def H_linear_only(H):
    
    Hv,ops = extract_coeffs_and_ops(H)
    
    indices_cq = [i for i, op in enumerate(ops) if len(op) == 1]

    Hvs = list()
    opss = list()

    for i in indices_cq:
        
        Hvs.append(Hv[i])
        opss.append(ops[i])
        
    H_l = reconstruct_boson_operator(Hvs, opss)
    
    return H_l

# separte just constant term ('')

def H_constant_only(H):
    
    Hv,ops = extract_coeffs_and_ops(H)
    
    indices_cq = [i for i, op in enumerate(ops) if len(op) == 0]
    
    Hvs = list()
    opss = list()

    for i in indices_cq:
        
        Hvs.append(Hv[i])
        opss.append(ops[i])
        
    
    return Hvs

#Performing B-Transform on quadratic terms
def b_transform(H,n):
    
    H_q = H_quadatic_only(H)
    
    A,B = create_AB(H_q,n)
    
    U,V,L = solve_uv_lambda(A, B, n)
    
    U,V,L = choose_uv(U,V,L,n)
    
    U,V = scale_matrices(U, V,n)
    
    H_l = H_linear_only(H)
    
    C = create_C(H_l,n)
    
    D = create_D(C,U,V)
    
    K = create_K(L,V)
    
    K_n = create_new_K(K,L,D)
    
    G = create_D_over_L(D,L)
    
    K_c = H_constant_only(H)
    
    K = K_n + K_c
        
    return U,V,L,G,K

# Recreate the quadratic Hamiotnian from the diagonalized form
def recreate_quadH(U,V,L,G,K,n):
    
    C_b,C_bd = tranformed_operators(U.T,V.T,G)
        
    # Initialize zero BosonOperator
    result = BosonOperator()
    
    I = BosonOperator('')
    
    for i in range(n):
        # Calculate the term L[i] * C_bd[i]^T * C_b[i] + K[i] * I
        term = L.flat[i] * C_bd[i] * C_b[i] 
        # Add the term to the result
        result += term
           
    result = result + K.item()* I    
    result = normal_ordered(result)
        
    return result 

# Calculate eigenspectrum for a bosonic operator
def boson_eigenspectrum_sparse(operator, truncation, k):
    
    sparse_op = get_sparse_operator(operator, trunc=truncation)
    
    return eigsh(sparse_op, k=k, which='SA', return_eigenvectors=False)