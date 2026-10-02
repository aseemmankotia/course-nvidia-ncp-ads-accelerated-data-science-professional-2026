## Chapter 2 Cheat Sheet: RAPIDS Setup & GPU-vs-CPU Decisions

### Install Paths
| Method | Command | Best For |
|---|---|---|
| Conda | `conda create -n rapids -c rapidsai -c conda-forge -c nvidia rapids=<ver> python=<ver> cuda-version=<ver>` | Local iteration, no container runtime |
| NGC Container | `docker pull nvcr.io/nvidia/rapidsai/base:<rapids_ver>-cuda<cuda_ver>-py<py_ver>` | CI/CD, reproducibility, multi-host |

**Channel order matters**: list `-c rapidsai -c conda-forge -c nvidia` in that priority order so GPU-correct builds win over generic conda-forge duplicates.

### Versioning
- RAPIDS uses **calendar versioning**: `YY.MM` (e.g. 24.02, 24.10) — quarterly cadence, NOT semantic versioning.
- NGC tags encode: `<rapids_version>-cuda<cuda_version>-py<python_version>`. Always pin exact tags — never use `latest` in production/CI.

### CUDA/Driver Matching
- RAPIDS binaries are compiled against a specific CUDA toolkit version.
- **Rule**: installed NVIDIA driver's max-supported CUDA version must be ≥ the RAPIDS build's CUDA toolkit version.
- Mismatch symptom: import error or kernel-launch failure at runtime — not a silent fallback.
- Fix: upgrade driver, or select a RAPIDS build matching the current driver's supported CUDA version.

### GPU vs CPU Decision Checklist
| Factor | Favors GPU | Favors CPU |
|---|---|---|
| Dataset size | Multi-GB+, fits in GPU memory | Sub-GB, fits comfortably in CPU cache/RAM |
| Iteration count | Many repeated passes (ETL loops, HPO, retraining) | Single one-off run |
| Workflow shape | Columnar, vectorizable (joins, groupby, linear algebra) | Row-wise, branchy, I/O-bound |
| Job frequency | Recurring pipeline | Ad hoc exploration |

**Overhead cost to remember**: GPU context init + host↔device transfer has a fixed cost — must be amortized over dataset size × iteration count to win over CPU.

### Multi-Tenant GPU (MIG)
- MIG = hardware-level partitioning of a single GPU into isolated compute+memory instances.
- Supported on A100/H100-class GPUs.
- A100 supports up to **7 MIG instances**.
- Use MIG when multiple users/tenants need strict isolation on one physical GPU; use time-slicing/streams only for lighter, non-isolated sharing.

### Jupyter + RAPIDS
1. Install `rapids` + `jupyterlab` + `ipykernel` in the **same** conda env.
2. Register kernel: `python -m ipykernel install --user --name rapids`.
3. Select that kernel in JupyterLab — cross-env imports of cudf/cuml will fail.

### Common Version Mismatch Traps
- RAPIDS version installed but driver too old for its CUDA requirement → import failure.
- Mixing pip-installed CUDA libs with conda RAPIDS env → dependency conflicts.
- Using `latest` NGC tag in CI → non-reproducible builds across runs.
