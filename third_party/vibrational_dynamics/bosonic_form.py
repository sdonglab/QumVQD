# Original source: https://github.com/shreyasvm3/Vibrational_Dynamics_on_Bosonic_QCs
# Copyright (c) 2026 Shreyas Malpathak and Sangeeth Das Kallullathil
# Licensed under the MIT License
# Reference: Malpathak, Das Kallullathil & Izmaylov, JPCL 2025, arXiv:2411.17950
# Modifications by Marlon Jost: edited import statement to reflect new location of qff.py


import openfermion as of
import numpy as np

from .qff import find_2d_degs, find_3d_degs # [Marlon Jost] edited import statement to reflect new location of qff.py, which is now in the same folder as bosonic_form.py, so we can use relative imports

def position_to_boson(index):
	"""
	Given an index of a product of position operators,
	expand them using HO ladder operators and simplify. 
	"""

	# form bosonic operator
	# creation operator
	factors_c = tuple([ (int(entry),1) for entry in index ])
	# annihilation operator
	factors_a = tuple([ (int(entry),0) for entry in index ])
	expr = of.BosonOperator.identity()
	for ii in range(len(index)):
		expr *= (of.BosonOperator(factors_c[ii], 1.) + of.BosonOperator(factors_a[ii], 1.)) / np.sqrt(2)

	return of.normal_ordered(expr)

def kinetic_term(freqs):
	nmodes = len(freqs)
	expr = of.BosonOperator.zero()
	for ii in range(nmodes):
		pi = of.BosonOperator((ii,1), 1. ) - of.BosonOperator((ii,0), 1.)
		expr -= 0.25 * freqs[ii] * pi*pi

	return of.normal_ordered(expr)

def harmonic_oscillators(freqs):
	nmodes = len(freqs)
	kin = kinetic_term(freqs)

	pot = of.BosonOperator.zero()
	for ii in range(nmodes):
		pot += position_to_boson([ii,ii]) * freqs[ii] * 0.5

	return kin + pot

def taylor_to_bosonic(f_arr, verbose=False):
	num_coups = len(f_arr)

	nmodes, deg = np.shape(f_arr[0])
	deg += 2

	degs_2d = find_2d_degs(deg)
	degs_3d = find_3d_degs(deg)

	b_op = of.BosonOperator.zero()
	for nc in range(num_coups):
		f_eff = f_arr[nc]

		if nc == 0:
			for ii in range(nmodes):
				for i_deg, f_val in enumerate(f_eff[ii,:]):
					idx = (i_deg+3) * [ii]
					if verbose:
						print(f"Adding element with indices {idx} with associated coefficient {f_val}")
					b_op += f_val * position_to_boson(idx)

		if nc == 1:
			for i1 in range(nmodes):
				for i2 in range(i1):
					for deg_idx, Qs in enumerate(degs_2d):
						idx = Qs[0]*[i1] + Qs[1]*[i2]
						if verbose:
							print(f"Adding element with indices {idx} with associated coefficient {f_eff[i1,i2,deg_idx]}")
						b_op += f_eff[i1,i2,deg_idx] * position_to_boson(idx)

		if nc == 2:
			for i1 in range(nmodes):
				for i2 in range(i1):
					for i3 in range(i2):
						for deg_idx, Qs in enumerate(degs_3d):
							idx = Qs[0]*[i1] + Qs[1]*[i2] + Qs[2]*[i3]
							if verbose:
								print(f"Adding element with indices {idx} with associated coefficient {f_eff[i1,i2,i3,deg_idx]}")
							b_op += f_eff[i1,i2,i3,deg_idx] * position_to_boson(idx)

		if nc > 2:
			print("Warning, entered array for more than 3-mode couplings, not implemented!")
			print("Returning up to three-mode couplings")

	return of.normal_ordered(b_op)
