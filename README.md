# QumVQD

Variational Quantum Deflation (VQD) on bosonic / qumode hardware models, applied
to electronic and vibrational excited-state problems. QumVQD partitions qubit
Hamiltonians (e.g. linear-symmetric H2 / H4 under Jordan–Wigner) onto one or
more qumodes and optimizes a SNAP + displacement + beamsplitter ansatz in
TensorFlow to recover the ground state and a tower of excited states via an
overlap-penalty deflation loss.


## Repository layout

```
helpers/QumVQD_Helpers.py    Core ansatz, VQD loss, optimizer, Hamiltonians
third_party/vibrational_dynamics/   Vibrational Hamiltonian builders (QFF, GFRO)
notebooks/
  Electronic_QumVQD.ipynb    Electronic excited states for H2 / H4
  Vibrational_VQD.ipynb      Vibrational excited states
  Manuscript_Plotting.ipynb  Figures for the manuscript
data/                         Data required to run notebooks
pyproject.toml                Package metadata (installable as `QumVQD`)
NOTICE.md                     Third-party attribution
```

## Installation

Requires Python ≥ 3.9.

Warning: Windows users must use Windows Subsystem for Linux due to dependency on [PySCF](https://pyscf.org/user/install.html).

```bash
git clone https://github.com/sdonglab/QumVQD.git
cd QumVQD
pip install -e .
```

This installs the `helpers` and `third_party` packages along with the
dependencies declared in `pyproject.toml` (QuTiP, TensorFlow, TensorFlow
Probability, OpenFermion, `openfermionpyscf`, scikit-learn, Jupyter).

## Quick start

Open one of the notebooks under `notebooks/` and run top-to-bottom. Key run
parameters at the top of `Electronic_QumVQD.ipynb`:

| Parameter      | Meaning                                                          |
| -------------- | ---------------------------------------------------------------- |
| `molecule`     | `"H2"` or `"H4"`                                                 |
| `hhdis`        | H–H bond length in Å                                             |
| `num_states`   | Number of states to compute (ground + excited)                   |
| `ndepth`       | Number of ansatz layers                                          |
| `nfocks`       | List of Fock-space dimensions per qumode, e.g. `[70]` or `[16,16]` |
| `beta_factor`  | Weight of the orthogonality penalty in the VQD cost              |
| `niter`        | Maximum optimizer iterations per state                           |
| `threshold`    | Convergence threshold on the loss                                |


## Acknowledgments

This repository includes code adapted from:

- [Dutta et al., *JCTC* 2026](https://github.com/rishabdchem/qumode_est_paper)
- [Malpathak et al., *JPCL* 2025](https://arxiv.org/abs/2411.17950)

Both are used under the MIT License. See [NOTICE.md](NOTICE.md), the
`third_party/` directory, and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
for full attribution and license text.

## License

MIT — see [LICENSE](LICENSE).
