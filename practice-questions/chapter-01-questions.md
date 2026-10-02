## Chapter 1: GPU vs CPU Architecture and the CUDA-X Data Science Stack — Practice Questions

**Q1.** A team runs pandas filter/groupby/join operations on a 500GB dataset, taking 45 minutes on CPU. Which change BEST reduces runtime while requiring minimal code rewrite?

A) Rewrite the pipeline using cuGraph
B) Add more CPU cores to the existing server
C) Use cuSpatial to accelerate the joins
D) Move the DataFrame operations to cuDF running on GPU

<details>
<summary>Answer</summary>

**Correct: D**

cuDF is the RAPIDS GPU DataFrame library with a pandas-like API, purpose-built to accelerate filter, groupby, and join operations using the GPU's SIMT throughput model with minimal code changes.

**Why the others are wrong:**
- cuGraph is for graph algorithms (PageRank, BFS), not general tabular ETL.
- Adding CPU cores helps but CPUs remain latency-optimized (MIMD); data-parallel ETL scales far better on GPU throughput cores.
- cuSpatial is purpose-built for geospatial operations, not generic joins.

</details>

---

**Q2.** Profiling shows 90% of a GPU pipeline's runtime is spent copying a 20GB dataset from host RAM to GPU VRAM over PCIe Gen4 x16, while GPU computation itself takes only 2 seconds. What is the BEST first step to improve performance?

A) Buy a GPU with more CUDA cores
B) Increase the batch size to amortize the transfer
C) Switch cuDF operations back to CPU pandas
D) Keep the data resident on GPU across pipeline stages and use NVLink/GPUDirect Storage to reduce host-to-device transfers

<details>
<summary>Answer</summary>

**Correct: D**

The bottleneck is host-to-device (H2D) transfer bandwidth (PCIe), not GPU compute. Persisting data on GPU across stages or using NVLink/GPUDirect Storage eliminates redundant transfers and directly attacks the bottleneck.

**Why the others are wrong:**
- Compute is already fast (2s); more CUDA cores does nothing for a transfer-bound pipeline.
- Total bytes transferred stay the same regardless of batch size; it does not address the fixed 20GB transfer cost.
- Reverting to CPU pandas removes GPU acceleration entirely and doesn't solve the transfer problem.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** A workload has a sequential parsing step consuming 20% of total runtime and a parallel matrix computation consuming 80%. If the parallel portion is accelerated 10x by GPU, what is the maximum overall speedup per Amdahl's Law?

A) 10x
B) 3.57x
C) 5x
D) 8x

<details>
<summary>Answer</summary>

**Correct: B**

Speedup = 1 / ((1-P) + P/S) = 1 / (0.2 + 0.8/10) = 1 / 0.28 ≈ 3.57x. The sequential 20% caps the achievable speedup.

**Why the others are wrong:**
- 10x is the speedup of only the parallel portion, not the whole application.
- 5x ignores the correct weighting of the sequential fraction in Amdahl's formula.
- 8x incorrectly assumes the entire workload benefits from the 10x speedup.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q4.** A team needs to train a Random Forest classifier on GPU using an API nearly identical to scikit-learn. Which CUDA-X library should they use?

A) cuDF
B) cuGraph
C) cuML
D) cuSpatial

<details>
<summary>Answer</summary>

**Correct: C**

cuML provides GPU-accelerated machine learning algorithms, including Random Forest, with a scikit-learn-compatible API.

**Why the others are wrong:**
- cuDF is for DataFrame/ETL operations, not model training.
- cuGraph is for graph analytics algorithms like PageRank and BFS.
- cuSpatial is for geospatial operations, not general ML.

</details>

---

**Q5.** A company needs to run point-in-polygon queries and spatial joins on millions of GPS trajectory points at scale. Which CUDA-X library is purpose-built for this?

A) cuGraph
B) cuML
C) cuSpatial
D) cuDF

<details>
<summary>Answer</summary>

**Correct: C**

cuSpatial provides GPU-accelerated geospatial primitives including point-in-polygon tests and spatial joins, purpose-built for trajectory/location data.

**Why the others are wrong:**
- cuGraph handles graph/network analytics, not spatial geometry.
- cuML is for machine learning model training, not spatial queries.
- cuDF handles generic tabular operations but lacks native spatial join primitives.

</details>

---

**Q6.** Under the SIMT execution model on current NVIDIA GPU architectures, how many threads execute in lockstep as a single warp?

A) 32
B) 16
C) 64
D) 128

<details>
<summary>Answer</summary>

**Correct: A**

A warp consists of 32 threads that execute the same instruction simultaneously under SIMT (Single Instruction, Multiple Threads); this has remained constant across current CUDA architectures including Hopper.

**Why the others are wrong:**
- 16 is not the warp size on any current CUDA GPU.
- 64 is sometimes confused with AMD's wavefront size, not NVIDIA's warp size.
- 128 is closer to the number of CUDA cores per SM on some architectures, not the warp size.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** Two H100 GPUs need to exchange data directly for a multi-GPU training job. Which statement correctly compares NVLink 4 and PCIe Gen5 x16 for this GPU-to-GPU transfer?

A) PCIe Gen5 x16 provides higher bandwidth because it is the newer standard
B) Both interconnects provide identical bandwidth since both are point-to-point links
C) NVLink 4 provides significantly higher bandwidth (~900 GB/s per GPU) than PCIe Gen5 x16 (~128 GB/s bidirectional) because it is a dedicated GPU-to-GPU interconnect
D) PCIe always outperforms NVLink for peer-to-peer GPU transfers

<details>
<summary>Answer</summary>

**Correct: C**

NVLink 4 on Hopper provides roughly 900 GB/s of bidirectional bandwidth per GPU, dramatically higher than PCIe Gen5 x16's roughly 128 GB/s, because NVLink is a dedicated high-speed interconnect for GPU-to-GPU communication.

**Why the others are wrong:**
- PCIe generation newness doesn't overcome NVLink's dedicated, wider interconnect design.
- The two interconnects have very different bandwidth profiles; they are not equivalent.
- This is backwards — NVLink consistently outperforms PCIe for peer-to-peer GPU transfers.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q8.** Which statement correctly differentiates CPU and GPU core design philosophy?

A) CPU cores are designed for throughput via massive parallelism; GPU cores are designed for low-latency single-thread execution
B) Both architectures use an identical SIMT execution model
C) CPU cores are optimized for low latency on few complex threads using MIMD and out-of-order execution; GPU cores are optimized for throughput across thousands of simple threads executing in SIMT lockstep
D) GPUs have fewer, more complex cores than CPUs

<details>
<summary>Answer</summary>

**Correct: C**

CPUs use few complex, latency-optimized cores with out-of-order execution under a MIMD model, while GPUs use thousands of simple, throughput-optimized cores executing in SIMT lockstep (warps).

**Why the others are wrong:**
- This reverses the actual design goals of CPU and GPU cores.
- CPUs use MIMD, not SIMT; the execution models are fundamentally different.
- GPUs have far more, simpler cores than CPUs, not fewer and more complex.

</details>

---

