# Third-party code

This package contains code derived from the `qumode_est_paper` repository
by Rishab Dutta (https://github.com/rishabdchem/qumode_est_paper),
released under the MIT License.

The original license text is reproduced in `third_party/LICENSE_qumode_est_paper.txt`.

Functions in `helpers/QumVQD_Helpers.py` that are derived from the upstream
repository (with modifications) include:

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
  installable via `pip install -e .`
- Reimplemented the SNAP operator using `tf.linalg.diag` in place of
  the projector-sum form for efficiency
- Generalized the two-mode SNAP-displacement-with-beamsplitter ansatz
  to an arbitrary-`N`-mode ansatz with pairwise beamsplitter coupling
- Replaced the upstream VQE loss/optimizer routine with a variational
  quantum deflation (VQD) implementation supporting overlap-penalty
  excited-state search