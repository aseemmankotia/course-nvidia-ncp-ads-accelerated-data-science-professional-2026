## Chapter 6: ETL Design and Multi-GPU Scaling with Dask-cuDF — Practice Questions

**Q1.** A team has a 60GB tabular dataset that must be joined and aggregated repeatedly during an ETL pipeline. They have a single workstation with four 24GB GPUs connected via NVLink. What is the BEST cluster topology?

A) LocalCUDACluster with one worker per GPU on the single node
B) Single-GPU cudf processing with chunked reads to stay under 24GB
C) A multi-node Dask cluster with UCX over InfiniBand
D) CPU-based Dask cluster with cudf fallback per partition

<details>
<summary>Answer</summary>

**Correct: A**

60GB exceeds any single 24GB GPU but fits comfortably across four GPUs (96GB aggregate) on one node. LocalCUDACluster with one worker per GPU uses NVLink for fast inter-GPU shuffle without the added complexity and network cost of a multi-node cluster.

**Why the others are wrong:**
- Single-GPU chunking would work but forces manual spilling/re-reads and defeats the purpose of parallel GPU compute — not the best fit when 4 GPUs are already available.
- Multi-node adds InfiniBand/network overhead that is unnecessary when the dataset fits within one node's aggregate GPU memory.
- A CPU-based cluster with cudf fallback abandons GPU acceleration entirely, which is not what the scenario calls for.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** A dask_cudf DataFrame has 20 partitions after multiple filter operations, resulting in wildly uneven partition sizes — some near 0 rows, others with millions. Before a groupby-aggregation, what should you do?

A) Call df.compute() immediately to force materialization
B) Increase the number of Dask workers to compensate
C) Call df.repartition(npartitions=...) or df.repartition(partition_size=...) to rebalance
D) Switch from dask_cudf to plain cudf for the aggregation

<details>
<summary>Answer</summary>

**Correct: C**

Repartitioning rebalances skewed partitions before an expensive shuffle-heavy operation like groupby, preventing stragglers where one worker holds a disproportionate share of the data.

**Why the others are wrong:**
- Calling compute() materializes results but does nothing to fix imbalance — it just returns unbalanced output faster.
- Adding workers doesn't fix skew; if one partition is huge, only one worker processes it regardless of total worker count.
- Plain cudf can't hold a dataset that requires distributed GPUs in the first place, and doesn't address partition skew.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** During a distributed dask_cudf merge across 8 GPUs, GPU utilization graphs show long idle periods on most GPUs while network/NVLink activity spikes. What operation is most likely responsible?

A) Lazy graph construction delaying kernel launches
B) RMM pool allocator fragmentation
C) CUDA context switching overhead between workers
D) The shuffle phase of the merge redistributing rows across GPUs by join key

<details>
<summary>Answer</summary>

**Correct: D**

Merges (and groupbys) on non-partition-aligned keys require a shuffle: rows must be exchanged across workers so matching keys land on the same GPU. This communication-bound phase causes visible idle compute time and spikes in interconnect traffic.

**Why the others are wrong:**
- Lazy graph construction happens at .compute()/.persist() time, not during active execution, and doesn't cause runtime idling on the GPU utilization graph.
- RMM fragmentation can cause slowdowns or OOM errors but doesn't produce the specific pattern of idle compute paired with high interconnect traffic.
- CUDA context switching overhead is negligible compared to shuffle communication cost and isn't the typical culprit for this pattern.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q4.** Which statement about dask_cuda.LocalCUDACluster's default worker topology is correct?

A) It creates a single worker process that round-robins across all visible GPUs
B) It creates one worker per GPU, pinning each worker to a distinct device via CUDA_VISIBLE_DEVICES
C) It creates one worker per CPU core, with GPUs shared across workers
D) It requires manually specifying n_workers equal to the number of GPUs, or it defaults to 1 worker total

<details>
<summary>Answer</summary>

**Correct: B**

LocalCUDACluster's default behavior is to spawn one Dask worker per visible GPU, each pinned to its own device by setting CUDA_VISIBLE_DEVICES for that worker process — this is the standard worker-per-GPU topology tested on the exam.

**Why the others are wrong:**
- A single round-robin worker would serialize GPU access and defeats the purpose of multi-GPU parallelism — not how LocalCUDACluster behaves.
- Workers are tied to GPUs, not CPU cores; sharing GPUs across workers would cause contention and is not the default.
- You don't need to manually pass n_workers — the cluster auto-detects visible GPUs and creates one worker per GPU by default.

</details>

---

**Q5.** A pipeline only needs to process a 15GB CSV file that comfortably fits within a single 40GB GPU's memory, with no other concurrent workloads. Which approach is BEST?

A) LocalCUDACluster with 4 workers for parallel throughput
B) A multi-node cluster to future-proof for scale
C) Single-GPU cudf, no Dask cluster needed
D) dask_cudf with a single partition spread across all available GPUs

<details>
<summary>Answer</summary>

**Correct: C**

When data fits comfortably in a single GPU's memory with room to spare for intermediate operations, plain cudf on one GPU avoids the overhead of cluster setup, scheduler communication, and task graph management — the exam's decision criteria favor the simplest sufficient solution.

**Why the others are wrong:**
- Spinning up a 4-worker cluster for data that fits on one GPU adds unnecessary distributed overhead with no throughput benefit.
- Multi-node is a large overcorrection for a dataset that fits on a single GPU — 'future-proofing' is not a valid exam justification when current requirements don't need it.
- A single dask_cudf partition spread across GPUs provides no parallelism benefit since there's only one partition to schedule.

</details>

---

**Q6.** When choosing a partition size for a dask_cudf DataFrame on GPUs with 32GB memory each, which sizing guideline reflects standard RAPIDS practice?

A) Keep each partition small enough (a fraction of per-GPU memory) to leave headroom for shuffle buffers and intermediate results during operations like joins
B) Maximize partition size to use nearly all 32GB per partition for efficiency
C) Partition size is irrelevant since Dask automatically spills to CPU RAM as needed
D) Always use exactly one partition per GPU regardless of dataset size

<details>
<summary>Answer</summary>

**Correct: A**

Operations like merges, groupbys, and sorts can require several times a partition's memory footprint during shuffle and intermediate materialization. Sizing partitions as a fraction of GPU memory leaves headroom and avoids OOM errors — this is a heavily tested exam concept.

**Why the others are wrong:**
- Filling a GPU to near-capacity with one partition leaves no room for the memory overhead of joins/groupbys, risking OOM during shuffle.
- While spilling to host memory is available (via RMM/UCX-managed memory or Dask's spill-to-disk), relying on it as a substitute for correct partition sizing causes severe performance degradation and is not the recommended practice.
- Partition count should scale with dataset size and available GPUs — forcing exactly one partition per GPU ignores memory headroom needs and load balancing.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** In dask_cudf, calling df.groupby('key').mean() immediately followed by inspecting the returned object shows no computed values yet. What explains this?

A) The groupby failed silently and returned an empty object
B) dask_cudf operations build a lazy task graph; execution only happens when .compute() or .persist() is called
C) GPU memory allocation failed, so no result was produced
D) groupby().mean() is not supported in dask_cudf and requires cudf directly

<details>
<summary>Answer</summary>

**Correct: B**

dask_cudf follows Dask's lazy evaluation model: operations construct a task graph of deferred computations. Nothing is actually computed on the GPU until an explicit trigger like .compute() (returns a concrete cudf object) or .persist() (materializes and keeps as distributed futures) is called.

**Why the others are wrong:**
- A silent failure would typically raise an exception or produce an error in the Dask dashboard, not just an uncomputed lazy object.
- Memory allocation failures raise explicit OOM errors from RMM, not a lazy unmaterialized result.
- groupby().mean() is a fully supported dask_cudf operation and does not require dropping to plain cudf.

</details>

---

**Q8.** An ETL job's total dataset size is 500GB, and the largest available single node has 8 GPUs with 640GB aggregate GPU memory. Intermediate shuffle operations during joins are expected to temporarily require up to 2x the base dataset size in working memory. What cluster topology should be selected?

A) Multi-node cluster spanning at least 2 nodes with UCX/InfiniBand
B) Single-GPU processing with manual chunking
C) Single-node LocalCUDACluster using all 8 GPUs, since 640GB aggregate exceeds the 500GB base size
D) A single-node cluster with only 2 of the 8 GPUs to reduce shuffle traffic

<details>
<summary>Answer</summary>

**Correct: A**

Base data (500GB) plus expected 2x working memory during shuffle means peak demand approaches ~1TB, which exceeds the single node's 640GB aggregate GPU memory. A multi-node cluster with UCX over InfiniBand is required to provide enough aggregate memory and fast inter-node communication for the shuffle-heavy joins.

**Why the others are wrong:**
- Single-GPU chunking cannot handle a 500GB dataset with any reasonable chunk strategy for GPU-accelerated joins at this scale.
- 640GB aggregate on one node looks sufficient for the base data alone, but the 2x working-memory requirement during shuffle pushes peak demand beyond available capacity — this is the key gotcha the exam tests.
- Reducing to 2 GPUs shrinks total memory further, making the memory shortfall worse, not better.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

