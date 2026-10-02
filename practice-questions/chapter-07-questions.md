## Chapter 7: Out-of-Core Workflows and Efficient Parquet/ORC I/O — Practice Questions

**Q1.** A pipeline must process a 500GB Parquet dataset on a single A100 GPU with 40GB VRAM. Loading the full dataset into a cudf.DataFrame raises a CUDA out-of-memory error. Which approach BEST resolves this while keeping the workflow GPU-accelerated?

A) Load the dataset into pandas first, then convert the whole DataFrame to cudf.DataFrame at once
B) Use dask_cudf to read the data in partitioned chunks and enable RMM spilling to host memory
C) Increase CUDF_SPILL_DEVICE_LIMIT to 1.0 so the GPU uses all available memory before spilling
D) Convert the dataset to CSV and read it with pandas to avoid GPU memory limits entirely

<details>
<summary>Answer</summary>

**Correct: B**

dask_cudf partitions the dataset into GPU-sized chunks and, combined with RMM spilling, moves buffers to host memory when device memory pressure is high, enabling out-of-core processing without abandoning GPU acceleration.

**Why the others are wrong:**
- Converting the entire dataset to cudf.DataFrame at once still requires it to fit in VRAM at some point, causing the same OOM
- Setting the spill limit to 1.0 removes headroom, so spilling triggers too late (often after OOM already occurs)
- Dropping to pandas/CSV abandons GPU acceleration entirely and is far slower for this workload

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** A multi-stage RAPIDS pipeline recomputes the same expensive dask_cudf groupby every time a downstream stage calls .compute() on a dependent graph. What is the BEST way to avoid this recomputation?

A) Call .compute() on the groupby result once and never reference it again
B) Write the intermediate result to CSV before each downstream stage
C) Call .persist() on the groupby result to materialize and cache it in distributed memory
D) Increase the RMM pool size so recomputation runs faster

<details>
<summary>Answer</summary>

**Correct: C**

persist() triggers computation and keeps the result resident in cluster memory (GPU/host), so subsequent operations reuse it instead of re-walking the task graph.

**Why the others are wrong:**
- .compute() converts to a concrete result on the client but doesn't keep it distributed/cached for reuse in the task graph
- Writing to CSV each time adds slow row-based I/O overhead and still requires a full write per stage
- A larger RMM pool speeds allocation, not recomputation avoidance

</details>

---

**Q3.** A dataset is partitioned by date on disk as Hive-style directories (date=YYYY-MM-DD). A query only needs 3 columns and rows where date > '2025-06-01'. Which read strategy BEST minimizes I/O and GPU decode time?

A) Read all partitions and columns, then filter with a cudf boolean mask after loading
B) Convert the dataset to a single unpartitioned Parquet file and rely on row-group stats only
C) Read the dataset as ORC instead of Parquet to get better predicate pushdown
D) Use dask_cudf.read_parquet with columns=[...] and filters=[('date','>','2025-06-01')] to prune partitions and columns before decode

<details>
<summary>Answer</summary>

**Correct: D**

Specifying columns and filters lets the reader skip both entire date partitions and unneeded columns before any GPU decode happens, minimizing I/O and compute.

**Why the others are wrong:**
- Loading everything then filtering wastes I/O and GPU decode on data that gets discarded
- Removing partitioning loses the cheap directory-level pruning, forcing reliance on slower row-group stat checks across a huge file
- Switching to ORC doesn't inherently improve pushdown and typically has higher per-stripe decode overhead on GPU

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q4.** For a GPU-accelerated cuDF pipeline that needs the fastest read throughput and effective row-group-level predicate pushdown, which columnar file format should be preferred by default?

A) ORC
B) Parquet
C) Avro
D) CSV

<details>
<summary>Answer</summary>

**Correct: B**

libcudf's Parquet decoder is more highly optimized than its ORC decoder, and Parquet's row-group min/max statistics support effective predicate pushdown on GPU reads.

**Why the others are wrong:**
- ORC is supported but generally has higher per-stripe decode overhead on GPU than Parquet
- Avro is row-based and not GPU-decode optimized, with no columnar pushdown
- CSV is row-based, requires full scans, and has no predicate pushdown

</details>

---

**Q5.** Which observed symptom is the STRONGEST indicator that a workflow needs out-of-core handling (e.g., dask_cudf with spilling) instead of single-GPU cudf?

A) The workflow runs slower than an equivalent pandas implementation on CPU
B) The dataset is stored in multiple small files instead of one large file
C) The pipeline uses more than one transformation step
D) Repeated CUDA out-of-memory errors occur during joins or groupbys as dataset size approaches or exceeds GPU VRAM

<details>
<summary>Answer</summary>

**Correct: D**

CUDA OOM errors during memory-intensive operations (joins, groupbys, sorts) when data size nears or exceeds VRAM is the direct signal that single-GPU in-memory processing is no longer viable.

**Why the others are wrong:**
- Running slower than pandas may indicate a different bottleneck (e.g., small data, launch overhead), not memory exhaustion
- Many small files is a partitioning/I/O concern, not necessarily a sign of exceeding VRAM
- Multiple transformation steps is normal pipeline structure and unrelated to memory capacity

</details>

---

**Q6.** Parquet files written for a GPU pipeline have row groups averaging 10MB each, resulting in thousands of row groups per file and slow reads. What configuration change should be made when writing these files?

A) Decrease row_group_size_bytes further to reduce memory per row group
B) Switch to writing one row group per file regardless of size
C) Disable compression entirely to speed up writes
D) Increase row_group_size_bytes to target roughly 128–256MB per row group

<details>
<summary>Answer</summary>

**Correct: D**

Very small row groups create excessive per-row-group reader overhead; targeting roughly 128–256MB row groups balances GPU decode efficiency against memory pressure.

**Why the others are wrong:**
- Decreasing size further worsens the overhead problem
- A single giant row group per file removes parallelism opportunities and can create excessive memory pressure during decode
- Disabling compression addresses a different concern and does not fix the row-group-count/overhead problem

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** Which environment variable configuration correctly enables cuDF's automatic spilling of GPU buffers to host memory?

A) RMM_POOL_SIZE=on
B) DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=1
C) CUDF_SPILL=on together with CUDF_SPILL_DEVICE_LIMIT set to a fraction of device memory
D) CUDA_VISIBLE_DEVICES=spill

<details>
<summary>Answer</summary>

**Correct: C**

CUDF_SPILL=on activates cuDF's spilling feature, and CUDF_SPILL_DEVICE_LIMIT sets the device-memory fraction threshold that triggers moving buffers to host memory.

**Why the others are wrong:**
- RMM_POOL_SIZE controls pool allocator size, not spilling behavior
- That distributed worker setting governs Dask's CPU-worker memory management, not cuDF GPU buffer spilling
- CUDA_VISIBLE_DEVICES selects which GPUs are visible to the process; it has no spilling function

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q8.** How does Parquet predicate pushdown actually reduce GPU decode work, at what granularity does the filtering occur?

A) It filters individual rows before they are read off disk, byte by byte
B) It rewrites the file on disk to remove non-matching rows before the GPU reads it
C) It only works when the file contains a single row group
D) It uses per-row-group min/max statistics to skip entire row groups that cannot match the filter, avoiding decode of skipped groups

<details>
<summary>Answer</summary>

**Correct: D**

Parquet stores min/max statistics per row group; pushdown filters compare the predicate against these stats to skip whole row groups, avoiding unnecessary I/O and GPU decode — not per-row filtering at read time.

**Why the others are wrong:**
- There is no per-row filtering before decode; filtering granularity is row-group level, not byte level
- Files are never rewritten as part of a read-time filter operation
- Predicate pushdown works across any number of row groups; a single row group is not a requirement

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

