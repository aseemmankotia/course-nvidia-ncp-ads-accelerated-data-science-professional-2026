## Chapter 2: RAPIDS Environment Setup and GPU-vs-CPU Decision Making — Practice Questions

**Q1.** A data scientist has a 500MB CSV file and needs to run a single one-off exploratory analysis with groupby and merge operations. Which approach BEST fits this workload?

A) Load into cuDF on a single GPU for maximum throughput
B) Spin up a multi-GPU Dask-cuDF cluster
C) Load into pandas on CPU
D) Use an NGC RAPIDS container with MIG partitioning

<details>
<summary>Answer</summary>

**Correct: C**

Dataset fits easily in CPU RAM, the job runs once, and the fixed cost of GPU context init + host-to-device transfer exceeds any compute savings for a single-pass workload this small.

**Why the others are wrong:**
- cuDF overhead (context creation, PCIe transfer) is not amortized over a single run of a sub-1GB dataset — CPU is faster wall-clock time.
- A multi-GPU cluster adds orchestration overhead with zero benefit at this scale — massive overkill.
- MIG is for isolating multiple tenants on one physical GPU, irrelevant to a single small one-off job.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** A company wants a reproducible RAPIDS environment for CI/CD pipelines that must produce identical results across every build agent, with no dependency on the host having conda pre-installed. Which install method BEST meets this requirement?

A) Pull a pinned NGC RAPIDS container tag (e.g. nvcr.io/nvidia/rapidsai/base:tag)
B) conda install with the rapidsai channel on each build agent
C) pip install rapids packages directly from PyPI
D) Build RAPIDS from source on each agent

<details>
<summary>Answer</summary>

**Correct: A**

A pinned NGC container tag bakes in the exact CUDA toolkit, driver-compatible libraries, and RAPIDS version, giving byte-identical environments across build agents with no host conda dependency.

**Why the others are wrong:**
- conda still requires conda pre-installed and resolves against upstream channels at build time, risking drift unless every package is pinned exactly.
- RAPIDS is not officially distributed as installable pip wheels for general production use the way conda/NGC are — this is not the supported path.
- Building from source is unnecessary complexity and non-reproducible unless every build flag and dependency is also pinned.

</details>

---

**Q3.** A platform team runs a shared A100 GPU and needs to give 7 data scientists strict hardware-level compute and memory isolation so one user's job cannot starve another's. Which feature should they configure?

A) CUDA streams to schedule each user's kernels separately
B) A larger conda environment shared by all users
C) Time-slicing via the default NVIDIA driver scheduler
D) MIG (Multi-Instance GPU) to partition the A100 into isolated instances

<details>
<summary>Answer</summary>

**Correct: D**

MIG partitions a supported GPU (A100/H100 class) into hardware-isolated instances with dedicated compute and memory slices, giving true isolation for multi-tenant environments — the A100 supports up to 7 MIG instances.

**Why the others are wrong:**
- CUDA streams provide concurrency within a single process/context, not hardware isolation between tenants.
- A shared conda environment does nothing to isolate compute or memory at the hardware level.
- Default time-slicing shares the whole GPU sequentially with no memory isolation — one job can still exhaust GPU memory for others.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q4.** A team's admin wants the lightest-weight way to add RAPIDS to an existing local Python workflow without adopting containers. Which install method should they use?

A) docker pull the RAPIDS NGC base image
B) kubectl apply a RAPIDS Helm chart
C) conda create/install using the rapidsai, conda-forge, and nvidia channels
D) Manually compile cuDF and cuML from GitHub source

<details>
<summary>Answer</summary>

**Correct: C**

conda install/create with -c rapidsai -c conda-forge -c nvidia is the officially supported lightweight path for adding RAPIDS to a local environment without container overhead.

**Why the others are wrong:**
- Docker/NGC requires adopting containers, which the admin explicitly wants to avoid.
- Helm/kubectl implies a Kubernetes cluster, unnecessary overhead for a local workflow.
- Source compilation is the heaviest, least-supported option and unnecessary when conda packages exist.

</details>

---

**Q5.** Which setup correctly enables RAPIDS (cuDF/cuML) inside JupyterLab for interactive exploration?

A) Install JupyterLab only in the base conda environment and import cudf from a separate rapids environment
B) Create a dedicated conda env with rapids + jupyterlab installed together, then register it as an IPython kernel
C) Install RAPIDS via pip inside a Jupyter notebook cell at runtime
D) Run Jupyter from the base environment and set CUDA_VISIBLE_DEVICES in the notebook to enable GPU imports

<details>
<summary>Answer</summary>

**Correct: B**

RAPIDS and jupyterlab must be installed in the same conda environment so all shared library dependencies (CUDA runtime, UCX, etc.) resolve consistently, then that environment is registered as a selectable Jupyter kernel via ipykernel install.

**Why the others are wrong:**
- Importing across environments fails — cudf's compiled binaries are only visible inside their own environment's Python path.
- RAPIDS is not reliably pip-installable at runtime inside a notebook cell; this bypasses conda's binary dependency resolution.
- Setting CUDA_VISIBLE_DEVICES only controls which GPU is visible — it does nothing to make RAPIDS packages importable if they aren't installed in the active kernel's environment.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q6.** When running `conda create -n rapids -c rapidsai -c conda-forge -c nvidia rapids=<ver> python=<ver> cuda-version=<ver>`, why does channel order matter?

A) Channel order has no effect; conda always checks conda-forge first regardless of order
B) The order determines the install directory path only
C) Conda gives priority to channels listed earlier, so rapidsai and nvidia builds are preferred over generic conda-forge builds of overlapping packages
D) Channel order only matters for pip, not conda

<details>
<summary>Answer</summary>

**Correct: C**

Conda's channel priority is order-dependent: packages from earlier-listed channels are preferred when the same package exists in multiple channels, which is why rapidsai/nvidia are listed before conda-forge to ensure GPU-correct builds are selected.

**Why the others are wrong:**
- Channel order absolutely affects package resolution priority in conda's default (strict) channel priority mode.
- Channel order affects package selection/resolution, not filesystem install paths.
- This is a conda-specific mechanism; the command shown uses conda, not pip, and pip has its own separate (unrelated) index priority rules.

</details>

---

**Q7.** A user installs a RAPIDS conda environment requiring CUDA 12.x runtime, but the machine's NVIDIA driver only supports up to CUDA 11.x. What happens and what is the correct fix?

A) cuDF import fails or errors at kernel launch; fix by upgrading the NVIDIA driver to a version supporting CUDA 12.x, or installing an environment pinned to a CUDA 11.x-compatible RAPIDS build
B) RAPIDS auto-downgrades itself silently; no action needed
C) The GPU is bypassed automatically and RAPIDS falls back to CPU pandas transparently
D) The environment installs fine and runs correctly since RAPIDS is driver-version agnostic

<details>
<summary>Answer</summary>

**Correct: A**

RAPIDS binaries built against a given CUDA toolkit require a driver whose supported CUDA version is >= that toolkit version; a driver too old causes import/runtime failures, and the fix is to upgrade the driver or select a RAPIDS/CUDA build matching the installed driver.

**Why the others are wrong:**
- There is no silent auto-downgrade mechanism in conda/RAPIDS for CUDA version mismatches.
- RAPIDS has no automatic CPU fallback — a driver mismatch produces an error, not a transparent CPU execution path.
- RAPIDS is driver-version sensitive; this is one of the most common environment setup failures on the exam.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q8.** RAPIDS releases use a calendar-based version scheme such as 24.02 or 24.10. What does this numbering represent?

A) A semantic version where the first number is major API breaking changes
B) The minimum CUDA version required, expressed as a decimal
C) Year and month of the release, reflecting RAPIDS' quarterly release cadence
D) An internal build number with no relation to release date

<details>
<summary>Answer</summary>

**Correct: C**

RAPIDS uses calendar versioning (YY.MM) tied to its quarterly release cadence, so the version number directly encodes the release year and month rather than semantic API versioning.

**Why the others are wrong:**
- RAPIDS does not use traditional semantic versioning where major/minor denote breaking API changes.
- The version number is unrelated to the CUDA toolkit version — CUDA compatibility is specified separately via a cuda-version parameter.
- The numbers are meaningfully tied to release date, not an arbitrary internal build counter.

</details>

---

