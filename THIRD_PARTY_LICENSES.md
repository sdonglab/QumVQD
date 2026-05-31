# Third-Party Licenses

This package incorporates code from the following open-source projects.
The full license text for each is preserved in the `third_party/` directory.

## 1. qumode_est_paper (Dutta)

Source: https://github.com/rishabdchem/qumode_est_paper
Reference: Dutta et al., JCTC 2026
License: MIT
Copyright (c) 2025 Rishab Dutta
Full text: third_party/LICENSE_qumode_est_paper.txt

Derived files: helpers/QumVQD_Helpers.py, notebooks/Electronic_QumVQD.ipynb (selected functions; 
see file header for details).
Data: data/snap-four-qubit-dict_nd16.npy

## 2. Vibrational_Dynamics_on_Bosonic_QCs (Malpathak & Das Kallullathil)

Source: https://github.com/shreyasvm3/Vibrational_Dynamics_on_Bosonic_QCs
Reference: Malpathak, Das Kallullathil & Izmaylov, JPCL 2025
  (arXiv:2411.17950)
License: MIT
Copyright (c) 2026 Shreyas Malpathak and Sangeeth Das Kallullathil
Full text: third_party/LICENSE_Vibrational_Dynamics_on_Bosonic_QCs.txt

Derived files: bosonic_form.py, qff.py, util_gfro.py, util.hamil.py, util_qudratic.py
Modifications: Modified imports to work with this project structure where necessary, modified file paths