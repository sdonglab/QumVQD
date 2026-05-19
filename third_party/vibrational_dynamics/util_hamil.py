# Original source: https://github.com/shreyasvm3/Vibrational_Dynamics_on_Bosonic_QCs
# Copyright (c) 2026 Shreyas Malpathak and Sangeeth Das Kallullathil
# Licensed under the MIT License
# Reference: Malpathak, Das Kallullathil & Izmaylov, JPCL 2025, arXiv:2411.17950
# Modifications by Marlon Jost: Modified file path for Hamiltonian files.

from openfermion.ops import BosonOperator
from openfermion.ops import QuadOperator
from openfermion.transforms import normal_ordered
from openfermion.transforms import get_boson_operator

import numpy as np
import os


# Creating bosonic operators from file_name, divisor(coefficients of Taylors series), include_p_terms for quadratic
def process_file(file_name, divisor, include_p_terms=False):
    with open(file_name, "r") as file:
        lines = file.readlines()

    operators = []
    for line in lines:
        parts = line.strip().split()
        indices = parts[:-1]
        value = float(parts[-1]) / divisor
        indices_str_q = " ".join(f"q{int(index)-1}" for index in indices)
        operator_q = QuadOperator(indices_str_q, value)
        operators.append(operator_q)

        if include_p_terms:
            indices_str_p = " ".join(f"p{int(index)-1}" for index in indices)
            operator_p = QuadOperator(indices_str_p, value)
            operators.append(operator_p)

    return operators


# Making sure that the coefficients are all real
def H_real(H):
    Hv, ops = extract_coeffs_and_ops(H)
    Hv = [np.real(element) for element in Hv]
    H = reconstruct_boson_operator(Hv, ops)

    return H


# Take molecule name and extract files from the folder named molecule and output bosonic Hamiltonian
def operator_quad(molecule):
    base_dir = "molecules"
    files = [f"f2{molecule}.dat", f"f3{molecule}.dat", f"f4{molecule}.dat"]
    divisors = {"f2": 2.0, "f3": 6.0, "f4": 24.0}
    # depends on whether the divisors are included or not

    H = QuadOperator()

    for file in files:
        file_path = os.path.join(base_dir, file)
        file_prefix = file[:2]
        if file_prefix in divisors:
            divisor = divisors[file_prefix]
            include_p_terms = file_prefix == "f2"
            operators = process_file(file_path, divisor, include_p_terms)
            for operator in operators:
                H += operator

    H_b = get_boson_operator(H)
    H = normal_ordered(H_b)

    return H_real(H)


# Function to extract coefficients and operators
def extract_coeffs_and_ops(boson_operator):
    coeffs = []
    ops = []
    for term, coeff in boson_operator.terms.items():
        ops.append(term)
        coeffs.append(coeff)
    return coeffs, ops


# Reconstruct bosonic operator from coefficients and operators
def reconstruct_boson_operator(coeffs, ops):
    boson_operator = BosonOperator()
    for coeff, op in zip(coeffs, ops):
        boson_operator += BosonOperator(op, coeff)
    return boson_operator


# for the Hamiltonians provided from Xanadu
## The author of the following part is Ignacio Loaiza Ganem

import h5py  # used for all sections in this notebook, everything else is self-contained
from .bosonic_form import harmonic_oscillators, taylor_to_bosonic # [Marlon Jost] edited import statement to reflect new location of bosonic_form.py, which is now in the same folder as util_hamil.py, so we can use relative imports

from pathlib import Path # [Marlon Jost] added import statement

def hamil(name):
    data_path = Path(__file__).parent.parent.parent / "data" / f"{name}_V2.hdf5" # [Marlon Jost] edited to use pathlib for constructing the file path
    with h5py.File(data_path, "r") as f:
        # with h5py.File(f'data/{name}.hdf5', 'r') as f:
        taylor_1D = f["taylor_1D"][()]
        taylor_2D = f["taylor_2D"][()]
        # taylor_3D = f["taylor_3D"][()]
        freqs = f["freqs"][()]

    Hharm = harmonic_oscillators(freqs)
    Hanh = taylor_to_bosonic([taylor_1D, taylor_2D])  # , taylor_3D])

    bosonic_hamiltonian = Hanh + Hharm

    au_to_cm = 219474.63068
    return au_to_cm * normal_ordered(bosonic_hamiltonian)
