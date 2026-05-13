# Original source: https://github.com/shreyasvm3/Vibrational_Dynamics_on_Bosonic_QCs
# Copyright (c) 2026 Shreyas Malpathak and Sangeeth Das Kallullathil
# Licensed under the MIT License
# Reference: Malpathak, Das Kallullathil & Izmaylov, JPCL 2025, arXiv:2411.17950
# Modifications by Marlon Jost: None

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import itertools


def obtain_r2(ytrue, yfit):
	ymean = np.sum(ytrue) / len(ytrue)
	ssres = np.sum((ytrue - yfit)**2) 
	sstot = np.sum((ytrue - ymean)**2)

	return 1-ssres/sstot

def remove_harmonic(freqs, pes_onebody):
	nmodes, quad_order = np.shape(pes_onebody)
	gauss_grid, gauss_weights = np.polynomial.hermite.hermgauss(quad_order)

	harmonic_pes = np.zeros((nmodes, quad_order))
	anh_pes = np.zeros((nmodes, quad_order))

	for ii in range(nmodes):
		ho_const = freqs[ii] / 2
		harmonic_pes[ii,:] = ho_const * (gauss_grid**2)
		anh_pes[ii,:] = pes_onebody[ii,:] - harmonic_pes[ii,:]

	return nmodes, quad_order, anh_pes, harmonic_pes

def c1_fit(anh_pes, deg, verbose=False):
	if deg < 3:
		raise Exception(f"Taylor expansion degree is {deg}<3, minimal degree is 3!")

	nmodes, quad_order = np.shape(anh_pes)
	gauss_grid, _ = np.polynomial.hermite.hermgauss(quad_order)

	fs = np.zeros((nmodes, deg - 2))
	
	predicted_1D = np.zeros_like(anh_pes)

	for i1 in range(nmodes):
		poly1D = PolynomialFeatures(degree=(3,deg), include_bias=False)
		poly1D_features = poly1D.fit_transform(gauss_grid.reshape(-1, 1))
		poly1D_reg_model = LinearRegression()
		poly1D_reg_model.fit(poly1D_features, anh_pes[i1,:])
		fs[i1,:] = poly1D_reg_model.coef_
		predicted_1D[i1,:] = poly1D_reg_model.predict(poly1D_features)

	if verbose:
		for i1 in range(nmodes):
			for my_deg in range(deg-2):
				print(f"Fit coefficient for q{i1+1}^{my_deg+3} = {fs[i1,my_deg]}")

	return fs, predicted_1D

def find_2d_degs(deg):
	fit_degs = []
	deg_idx = 0
	for feat_deg in range(3,deg+1):
		max_deg = feat_deg - 1
		for deg_dist in range(1,max_deg+1):
			q1deg = max_deg-deg_dist+1
			q2deg = deg_dist
			fit_degs.append((q1deg,q2deg))
			deg_idx += 1

	return fit_degs

def c2_fit(pes_twobody, deg, verbose=False):
	nmodes,_,quad_order,_ = np.shape(pes_twobody)
	gauss_grid, _ = np.polynomial.hermite.hermgauss(quad_order)

	if deg < 3:
		raise Exception(f"Taylor expansion degree is {deg}<3, minimal degree is 3!")
	
	num_fs = int((deg-1)*deg/2-1)
	fs = np.zeros((nmodes,nmodes,num_fs))

	predicted_2D = np.zeros_like(pes_twobody)

	grid_2D = np.array(np.meshgrid(gauss_grid,gauss_grid))
	q1 = grid_2D[0,::].flatten()
	q2 = grid_2D[1,::].flatten()
	idx_2D = np.array(np.meshgrid(range(quad_order),range(quad_order)))
	idx1 = idx_2D[0,::].flatten()
	idx2 = idx_2D[1,::].flatten()
	num_2D = len(q1)

	features = np.zeros((num_2D, num_fs))
	fit_degs = find_2d_degs(deg)
	for deg_idx,Qs in enumerate(fit_degs):
		q1deg,q2deg = Qs
		features[:,deg_idx] = q1**(q1deg) * q2**(q2deg)
		
	for i1 in range(nmodes):
		for i2 in range(i1):
			poly2D = PolynomialFeatures(degree=(3,deg), include_bias=False, interaction_only=True)
			Y = []
			for idx in range(num_2D):
				idx_q1 = idx1[idx]
				idx_q2 = idx2[idx]
				Y.append(pes_twobody[i1,i2,idx_q1,idx_q2])
			poly2D_reg_model = LinearRegression()
			poly2D_reg_model.fit(features, Y)
			fs[i1,i2,:] = poly2D_reg_model.coef_
			predicted = poly2D_reg_model.predict(features)
			for idx in range(num_2D):
				idx_q1 = idx1[idx]
				idx_q2 = idx2[idx]
				predicted_2D[i1,i2,idx_q1,idx_q2] = predicted[idx]

	if verbose:
		for i1 in range(nmodes):
			for i2 in range(i1):
				for fit_num in range(num_fs):
					qi_deg, qj_deg = fit_degs[fit_num]
					print(f"Fit coefficient for q{i1+1}^{qi_deg} q{i2+1}^{qj_deg} = {fs[i1,i2,fit_num]}")

	return fs, predicted_2D

def generate_bin_occupations(max_occ, nbins):
    # Generate all combinations placing max_occ balls in nbins
    combinations = list(itertools.product(range(max_occ+1), repeat=nbins))

    # Filter valid combinations
    valid_combinations = [combo for combo in combinations if sum(combo) == max_occ]

    return valid_combinations

def find_3d_degs(deg):
	fit_degs = []
	deg_idx = 0
	for feat_deg in range(3,deg+1):
		max_deg = feat_deg - 3
		possible_occupations = generate_bin_occupations(max_deg, 3)
		for occ in possible_occupations:
			q1deg = 1 + occ[0]
			q2deg = 1 + occ[1]
			q3deg = 1 + occ[2]
			fit_degs.append((q1deg,q2deg,q3deg))
			deg_idx += 1

	return fit_degs

def c3_fit(pes_threebody, deg, verbose=False):
	nmodes,_,_,quad_order,_,_ = np.shape(pes_threebody)
	gauss_grid, _ = np.polynomial.hermite.hermgauss(quad_order)

	if deg < 3:
		raise Exception(f"Taylor expansion degree is {deg}<3, minimal degree is 3!") 
	
	predicted_3D = np.zeros_like(pes_threebody)

	grid_3D = np.array(np.meshgrid(gauss_grid,gauss_grid,gauss_grid))
	q1 = grid_3D[0,::].flatten()
	q2 = grid_3D[1,::].flatten()
	q3 = grid_3D[2,::].flatten()
	idx_3D = np.array(np.meshgrid(range(quad_order),range(quad_order),range(quad_order)))
	idx1 = idx_3D[0,::].flatten()
	idx2 = idx_3D[1,::].flatten()
	idx3 = idx_3D[2,::].flatten()
	num_3D = len(q1)

	num_fs = int((deg-2)*(deg-1)*deg/6)
	print(f"num_fs = {num_fs}")
	fs = np.zeros((nmodes,nmodes,nmodes,num_fs))


	features = np.zeros((num_3D, num_fs))
	fit_degs = find_3d_degs(deg)
	for deg_idx,Qs in enumerate(fit_degs):
		q1deg,q2deg,q3deg = Qs
		features[:,deg_idx] = q1**(q1deg) * q2**(q2deg) * q3**(q3deg)
	
	for i1 in range(nmodes):
		for i2 in range(i1):
			for i3 in range(i2):
				poly3D = PolynomialFeatures(degree=(3,deg), include_bias=False, interaction_only=True)
				Y = []
				for idx in range(num_3D):
					idx_q1 = idx1[idx]
					idx_q2 = idx2[idx]
					idx_q3 = idx3[idx]
					Y.append(pes_threebody[i1,i2,i3,idx_q1,idx_q2,idx_q3])

				poly3D_reg_model = LinearRegression()
				poly3D_reg_model.fit(features, Y)
				fs[i1,i2,i3,:] = poly3D_reg_model.coef_
				predicted = poly3D_reg_model.predict(features)
				for idx in range(num_3D):
					idx_q1 = idx1[idx]
					idx_q2 = idx2[idx]
					idx_q3 = idx3[idx]
					predicted_3D[i1,i2,i3,idx_q1,idx_q2,idx_q3] = predicted[idx]

	if verbose:
		for i1 in range(nmodes):
			for i2 in range(i1):
				for i3 in range(i2):
					for fit_num in range(num_fs):
						q1deg, q2deg, q3deg = fit_degs[fit_num]
						print(f"Fit coefficient for q{i1+1}^{q1deg} q{i2+1}^{q2deg} q{i3+1}^{q3deg} = {fs[i1,i2,i3,fit_num]}")

	return fs, predicted_3D

def f_1D_to_tensor(f_1D):
	nmodes, deg = np.shape(f_1D)
	deg += 2
	if deg != 4:
		raise Exception(f"Trying to convert Taylor fit coefficients into tensor for degree {deg}!=4, not implemented!")

	phi_3 = np.zeros((nmodes,nmodes,nmodes))
	phi_4 = np.zeros((nmodes,nmodes,nmodes,nmodes))

	for ii in range(nmodes):
		phi_3[ii,ii,ii] = 6*f_1D[ii,0]
		phi_4[ii,ii,ii,ii] = 24*f_1D[ii,1]

	return phi_3, phi_4

def f_2D_to_tensor(f_2D, deg):
	nmodes, _, num_f2 = np.shape(f_2D)
	assert num_f2 == deg*(deg-1)/2 - 1
	if deg != 4:
		raise Exception(f"Trying to convert Taylor fit coefficients into tensor for degree {deg}!=4, not implemented!")

	phi_3 = np.zeros((nmodes,nmodes,nmodes))
	phi_4 = np.zeros((nmodes,nmodes,nmodes,nmodes))

	fit_degs = find_2d_degs(deg)

	for i1 in range(nmodes):
		for i2 in range(i1):
			for deg_idx,Qs in enumerate(fit_degs):
				q1deg,q2deg = Qs
				if q1deg+q2deg == 3:
					if q1deg == 2:
						phi_3[i1,i1,i2] = 2*f_2D[i1,i2,deg_idx]
						phi_3[i1,i2,i1] = 2*f_2D[i1,i2,deg_idx]
						phi_3[i2,i1,i1] = 2*f_2D[i1,i2,deg_idx]
					elif q2deg == 2:
						phi_3[i1,i2,i2] = 2*f_2D[i1,i2,deg_idx]
						phi_3[i2,i1,i2] = 2*f_2D[i1,i2,deg_idx]
						phi_3[i2,i2,i1] = 2*f_2D[i1,i2,deg_idx]
				elif q1deg+q2deg == 4:
					if q1deg == 3:
						phi_4[i1,i1,i1,i2] = 6*f_2D[i1,i2,deg_idx]
						phi_4[i1,i1,i2,i1] = 6*f_2D[i1,i2,deg_idx]
						phi_4[i1,i2,i1,i1] = 6*f_2D[i1,i2,deg_idx]
						phi_4[i2,i1,i1,i1] = 6*f_2D[i1,i2,deg_idx]
					elif q1deg == 2:
						phi_4[i1,i1,i2,i2] = 4*f_2D[i1,i2,deg_idx]
						phi_4[i1,i2,i1,i2] = 4*f_2D[i1,i2,deg_idx]
						phi_4[i1,i2,i2,i1] = 4*f_2D[i1,i2,deg_idx]
						phi_4[i2,i1,i1,i2] = 4*f_2D[i1,i2,deg_idx]
						phi_4[i2,i1,i2,i1] = 4*f_2D[i1,i2,deg_idx]
						phi_4[i2,i2,i1,i1] = 4*f_2D[i1,i2,deg_idx]
					elif q1deg == 1:
						phi_4[i2,i2,i2,i1] = 6*f_2D[i1,i2,deg_idx]
						phi_4[i2,i2,i1,i2] = 6*f_2D[i1,i2,deg_idx]
						phi_4[i2,i1,i2,i2] = 6*f_2D[i1,i2,deg_idx]
						phi_4[i1,i2,i2,i2] = 6*f_2D[i1,i2,deg_idx]
				else:
					print(f"Error, sum of polynomial degrees is {q1deg+q2deg} != 3,4. Returning only 3 and 4 degree tensors...")

	return phi_3, phi_4

def f_3D_to_tensor(f_3D, deg):
	nmodes, _, _, num_f3 = np.shape(f_3D)
	assert num_f3 == deg*(deg-1)*(deg-2)/6
	if deg != 4:
		raise Exception(f"Trying to convert Taylor fit coefficients into tensor for degree {deg}!=4, not implemented!")

	phi_3 = np.zeros((nmodes,nmodes,nmodes))
	phi_4 = np.zeros((nmodes,nmodes,nmodes,nmodes))

	fit_degs = find_3d_degs(deg)

	for i1 in range(nmodes):
		for i2 in range(i1):
			for i3 in range(i2):
				for deg_idx,Qs in enumerate(fit_degs):
					q1deg,q2deg,q3deg = Qs
					if q1deg+q2deg+q3deg == 3:
						phi_3[i1,i2,i3] = f_3D[i1,i2,i3,deg_idx]
						phi_3[i1,i3,i2] = f_3D[i1,i2,i3,deg_idx]
						phi_3[i2,i1,i3] = f_3D[i1,i2,i3,deg_idx]
						phi_3[i2,i3,i1] = f_3D[i1,i2,i3,deg_idx]
						phi_3[i3,i1,i2] = f_3D[i1,i2,i3,deg_idx]
						phi_3[i3,i2,i1] = f_3D[i1,i2,i3,deg_idx]
					elif q1deg+q2deg+q3deg == 4:
						if q1deg == 2:
							phi_4[i1,i1,i2,i3] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i1,i1,i3,i2] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i1,i2,i1,i3] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i1,i2,i3,i1] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i1,i3,i1,i2] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i1,i3,i2,i1] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i2,i1,i1,i3] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i2,i1,i3,i1] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i2,i3,i1,i1] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i3,i1,i1,i2] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i3,i1,i2,i1] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i3,i2,i1,i1] = 2*f_3D[i1,i2,i3,deg_idx]
						elif q2deg == 2:
							phi_4[i2,i2,i1,i3] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i2,i2,i3,i1] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i2,i1,i2,i3] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i2,i1,i3,i2] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i2,i3,i2,i1] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i2,i3,i1,i2] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i1,i2,i2,i3] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i1,i2,i3,i2] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i1,i3,i2,i2] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i3,i2,i2,i1] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i3,i2,i1,i2] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i3,i1,i2,i2] = 2*f_3D[i1,i2,i3,deg_idx]
						elif q3deg == 2:
							phi_4[i3,i3,i2,i1] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i3,i3,i1,i2] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i3,i2,i3,i1] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i3,i2,i1,i3] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i3,i1,i3,i2] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i3,i1,i2,i3] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i2,i3,i3,i1] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i2,i3,i1,i3] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i2,i1,i3,i3] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i1,i3,i3,i2] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i1,i3,i2,i3] = 2*f_3D[i1,i2,i3,deg_idx]
							phi_4[i1,i2,i3,i3] = 2*f_3D[i1,i2,i3,deg_idx]
					else:
						print(f"Error, sum of polynomial degrees is {q1deg+q2deg+q3deg} != 3,4. Returning only 3 and 4 degree tensors...")

	return phi_3, phi_4

def f_arr_to_tensor(f_arr, deg):
	if deg != 4:
		raise Exception(f"Trying to convert Taylor fit coefficients into tensor for degree {deg}!=4, not implemented!")

	phi_3, phi_4 = f_1D_to_tensor(f_arr[0])

	if len(f_arr) > 1:
		phi_3_2D, phi_4_2D = f_2D_to_tensor(f_arr[1], deg)
		phi_3 += phi_3_2D
		phi_4 += phi_4_2D

	if len(f_arr) > 2:
		phi_3_3D, phi_4_3D = f_3D_to_tensor(f_arr[2], deg)
		phi_3 += phi_3_3D
		phi_4 += phi_4_3D

	if len(f_arr) > 3:
		print(f"{len(f_arr)}>3-mode couplings have been added to fit, not implemented! Returning values for only 3-mode couplings")

	return phi_3, phi_4

def taylor_fit(freqs, pes_arr, deg=4, diagnostics=True, plot_dir='./', verbose=True, do_plots=True, return_tensors=True):
	num_couplings = len(pes_arr)
	print(f"Performing Taylor fitting, obtaining T{deg}C{num_couplings} fit from PES array input...")

	pes_onebody = pes_arr[0]
	print("Starting one-mode fitting...")
	nmodes, quad_order, anh_pes, harmonic_pes = remove_harmonic(freqs, pes_onebody)
	f_1D,predicted_1D = c1_fit(anh_pes, deg, verbose=verbose)
	predicted_1D += harmonic_pes
	f_arr = [f_1D]
	predicted_arr = [predicted_1D]

	if num_couplings > 1:
		pes_twobody = pes_arr[1]
		print("Starting two-mode fitting...")
		f_2D,predicted_2D = c2_fit(pes_twobody, deg, verbose=verbose)
		f_arr.append(f_2D)
		predicted_arr.append(predicted_2D)

	if num_couplings > 2:
		pes_threebody = pes_arr[2]
		print("Starting three-mode fitting...")
		f_3D,predicted_3D = c3_fit(pes_threebody, deg, verbose=verbose)
		f_arr.append(f_3D)
		predicted_arr.append(predicted_3D)

	if num_couplings > 3:
		print(f'Detected number of couplings is {num_couplings}>3, not implemented! Returning results for 3-mode couplings...')

	if diagnostics:
		taylor_diagnostics(freqs, pes_arr, predicted_arr, deg, plot_dir, do_plots)

	if return_tensors:
		return f_arr_to_tensor(f_arr, deg)
	else:
		return f_arr

def taylor_diagnostics(freqs, pes_arr, predicted_arr, deg, plot_dir = './', do_plots=True):
	num_couplings = len(pes_arr)
	tname = f'T{deg}C{num_couplings}'
	print(f"Performing Taylor diagnostics for {tname} fits")

	pes_onebody = pes_arr[0]
	nmodes, quad_order, anh_pes, harmonic_pes = remove_harmonic(freqs, pes_onebody)
	gauss_grid, _ = np.polynomial.hermite.hermgauss(quad_order)

	for i1 in range(nmodes):
		r2 = obtain_r2(pes_onebody[i1,:],predicted_arr[0][i1,:])
		r2_harm = obtain_r2(pes_onebody[i1,:],harmonic_pes[i1,:])
		print(f"Calculated R^2 for mode {i1} = {r2:.4f}, improved from harmonic {r2_harm:.4f}")

		if do_plots:
			fig, ax = plt.subplots()
			ax.plot(gauss_grid, pes_onebody[i1,:], label='PES', marker='o')
			ax.plot(gauss_grid, harmonic_pes[i1, :], label='Harmonic')
			ax.plot(gauss_grid, predicted_arr[0][i1,:], label='Taylor fit')
			ax.legend()
			ax.set_title(f'{tname} mode [{i1}] fit, R^2 correlation is {r2:.4f}')
			fig.savefig(plot_dir + tname + f"_onebody_{i1}.png", format="png")
			plt.close()

	if num_couplings > 1:
		pes_twobody = pes_arr[1]
		for i1 in range(nmodes):
			for i2 in range(i1):
				r2 = obtain_r2(pes_twobody[i1,i2,::].flatten(),predicted_arr[1][i1,i2,::].flatten())
				print(f"Calculated R^2 for modes [{i1},{i2}] = {r2}")

				if do_plots:
					fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
					ax.plot_wireframe(gauss_grid, gauss_grid, pes_twobody[i1,i2,:,:], label='PES')
					ax.plot_wireframe(gauss_grid, gauss_grid, predicted_arr[1][i1,i2,:,:], label='Taylor fit', color='orange')
					ax.legend()
					ax.set_title(f'{tname} modes [{i1},{i2}] fit, R^2 correlation is {r2:.4f}')
					fig.savefig(plot_dir + tname + f"_twobody_{i1}_{i2}.png", format="png")
					plt.close()

	if num_couplings > 2:
		pes_threebody = pes_arr[2]
		for i1 in range(nmodes):
			for i2 in range(i1):
				for i3 in range(i2):
					r2 = obtain_r2(pes_threebody[i1,i2,i3,::].flatten(),predicted_arr[2][i1,i2,i3,::].flatten())
					print(f"Calculated R^2 for modes [{i1},{i2},{i3}] = {r2:.4f}")
		