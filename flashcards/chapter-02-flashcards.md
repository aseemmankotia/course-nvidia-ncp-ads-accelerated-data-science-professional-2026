## Chapter 2: RAPIDS Environment Setup and GPU-vs-CPU Decision Making — Flashcards

| # | Front (Question) | Back (Answer) |
|---|-----------------|---------------|
| 1 | Conda command to install RAPIDS from the correct channels? | conda create -n rapids -c rapidsai -c conda-forge -c nvidia rapids=<ver> python=<ver> cuda-version=<ver>. Exam tip: channel order matters — rapidsai/nvidia listed before conda-forge for correct GPU builds. |
| 2 | NGC RAPIDS container pull pattern? | docker pull nvcr.io/nvidia/rapidsai/base:<rapids_version>-cuda<cuda_version>-py<python_version>. Always pin the tag exactly — 'latest' is not reproducible. |
| 3 | RAPIDS versioning scheme format? | Calendar versioning YY.MM (e.g. 24.02, 24.10) tied to a quarterly release cadence — NOT semantic versioning. |
| 4 | What happens if installed RAPIDS/CUDA toolkit version exceeds the host driver's supported CUDA version? | Import/kernel-launch failure. Fix: upgrade the NVIDIA driver, or install RAPIDS built against a CUDA toolkit version supported by the current driver. |
| 5 | conda vs NGC container: when to choose which for RAPIDS? | conda = fast local iteration, lighter weight, no container runtime needed. NGC container = reproducible CI/CD, guaranteed CUDA/driver/library alignment, easy multi-host deployment. |
| 6 | MIG (Multi-Instance GPU) — what problem does it solve? | Hardware-level partitioning of a single GPU (A100/H100-class) into isolated compute+memory instances for multi-tenant workloads. A100 supports up to 7 MIG instances. |
| 7 | GPU vs CPU decision checklist — dataset size factor? | If data comfortably fits in GPU memory and exceeds roughly the CPU-cache-friendly range (multi-GB+), GPU acceleration pays off. Sub-GB datasets rarely benefit due to transfer/launch overhead. |
| 8 | GPU vs CPU decision checklist — iteration count factor? | High iteration count (repeated transforms, hyperparameter search, ETL loops) amortizes GPU context/transfer overhead. Single one-off run rarely justifies GPU. |
| 9 | GPU vs CPU decision checklist — workflow shape factor? | Workflows dominated by vectorizable, columnar, parallel-friendly ops (joins, groupbys, matrix math) favor GPU/cuDF. Row-wise, branchy, I/O-bound logic favors CPU. |
| 10 | How do you make a RAPIDS conda env usable inside JupyterLab? | Install jupyterlab + ipykernel in the SAME rapids conda env, then run `python -m ipykernel install --user --name rapids` to register it as a selectable kernel. |
| 11 | Most common RAPIDS setup error on the exam? | CUDA/driver version mismatch — RAPIDS build compiled against a newer CUDA toolkit than the installed NVIDIA driver supports, causing import failures. |
| 12 | Why must RAPIDS + JupyterLab be installed in the same conda environment (not separate envs)? | cuDF/cuML are compiled binaries tied to that environment's Python path and shared libraries (CUDA runtime, UCX). A separate Jupyter env cannot import them across environment boundaries. |
