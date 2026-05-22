# Third-party code

This package contains code derived from the `qumode_est_paper` repository
by Rishab Dutta et al. (https://github.com/rishabdchem/qumode_est_paper) and the 
`Vibrational_Dynamics_on_Bosonic_QCs` repository by Shreyas Malpathak et al. 
(https://github.com/shreyasvm3/Vibrational_Dynamics_on_Bosonic_QCs), both 
released under the MIT License.

The original license text is reproduced in `third_party/LICENSE_qumode_est_paper.txt` 
and `third_party/LICENSE_Vibrational_Dynamics_on_Bosonic_QCs.txt`.

Functions in `helpers/QumVQD_Helpers.py` that are derived from the 
`qumode_est_paper` repository (with modifications) include:

- Single-mode primitives: `destroy_tf`, `create_tf`, `displace_tf`,
  `snap_op_tf`, `snap_disp_op_tf`, `snap_disp_op_qt`,
  `snap_disp_ansatz_qt`, `snap_disp_ansatz_tf`, `unpack_params_snap_disp_tf`
- Qubit-to-qumode partitioning: `qubit_string_from_openfermion`,
  `qubit_partition`, `simple_qubit_string`, `append_ids`,
  `snap_uni_four_pauli`
- Molecule construction: `ham_linear_sym_h2`, `ham_linear_sym_h4`
- Utility: `qt2tf`

The following modifications were applied:

- Consolidated standalone notebook cells into a single Python module
  installable via `pip install -e .`.
- Reimplemented the SNAP operator using `tf.linalg.diag` in place of
  the projector-sum form for efficiency.
- Generalized the two-mode SNAP-displacement-with-beamsplitter ansatz
  to an arbitrary-`N`-mode ansatz with pairwise beamsplitter coupling
  discluding the hadamard test portion from the original code.
- Replaced the upstream VQE loss/optimizer routine with a variational
  quantum deflation (VQD) implementation supporting overlap-penalty
  excited-state search.

Code and data derived from `Vibrational_Dynamics_on_Bosonic_QCs`:

- third_party/vibrational_dynamics/ contains copies of the upstream Python modules (util_hamil.py, util_gfro.py, util_qudratic.py, qff.py, bosonic_form.py) used to construct vibrational Hamiltonians and perform Gaussian fragment optimization (GFRO). See headers in these files for citations and modifications made.
- data/H2S.hdf5 and data/CO2.hdf5 are vibrational Hamiltonian data files sent to us by Shreyas Malpathak and are consumed via the upstream hamil() loader.
- In notebooks/Vibrational_VQD.ipynb, the cells under the "Vibrational Hamiltonian / prereqs" and "Vibrational Hamiltonian / Run Fragmentation" sections are adapted from main_gfro.py code in the upstream repository, with modifications to parameterize molecule, mode count, fragment count, and output paths via the notebook's "Run Options" cell and to store the fragment parameters and the residual quadratic Hamiltonian in pickle files.
