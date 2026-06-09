# QumVQD

Qumode-based Variational Quantum Deflation (QumVQD): a framework for computing
electronic and vibrational excited-state energies on bosonic / qumode quantum
hardware models. QumVQD maps molecular Hamiltonians onto one or more qumodes and
optimizes a SNAP + displacement + beamsplitter ansatz in TensorFlow, recovering
ground and excited states through an overlap-penalty based cost function.


## Repository layout

```
helpers/QumVQD_Helpers.py    Core ansatz, VQD loss, optimizer, Hamiltonians
third_party/vibrational_dynamics/   Vibrational Hamiltonian fragmentation code
notebooks/
  Electronic_QumVQD.ipynb    Electronic excited states for H2 / H4
  Vibrational_VQD.ipynb      Vibrational excited states
  Manuscript_Plotting.ipynb  Figures for the manuscript
  Plots/                     Stores plots produced by 
data/                         Data required to run notebooks
pyproject.toml                Package metadata (installable as `QumVQD`)
NOTICE.md                     Third-party attribution
.gitattributes                Handles Git LFS configuration
.gitignore                    Designates which files for Git to ignore
LICENSE.md                    MIT License
THIRD_PARTY_LICENSES.md       Attribution for third-party code   
```

## Installation

Requires Python ≥ 3.9.

Warning: Windows users must use Windows Subsystem for Linux due to dependency on [PySCF](https://pyscf.org/user/install.html).
Moreover, this repository uses Git LFS for the .hdf5 data files. Install it first (git lfs install), then clone.

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


## Citation

Marlon F. Jost, Sijia S. Dong*. [Excited-State Quantum Chemistry on Qumode-Based Processors via Variational Quantum Deflation](https://arxiv.org/abs/2604.13457). arXiv. 2026. arXiv:2604.13457 
```
@misc{jost2026excitedstatequantumchemistryqumodebased,
      title={Excited-State Quantum Chemistry on Qumode-Based Processors via Variational Quantum Deflation}, 
      author={Marlon F. Jost and Sijia S. Dong},
      year={2026},
      eprint={2604.13457},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2604.13457}, 
}
```
