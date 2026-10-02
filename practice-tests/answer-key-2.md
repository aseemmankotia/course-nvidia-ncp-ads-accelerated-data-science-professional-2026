# Answer Key — Practice Test 2

**Q1. Correct: D**

Small, frequent host-to-device transfers are latency-bound because each transfer pays a fixed setup cost regardless of size. Consolidating into fewer, larger transfers amortizes that overhead, and pinned memory enables faster DMA-based copies instead of pageable-memory staging, directly targeting the measured bottleneck.

---

**Q2. Correct: B**

NGC container images are pre-built, tested, and versioned by NVIDIA against specific CUDA toolkit and driver combinations, giving guaranteed compatibility and reproducibility across heterogeneous host OS versions without per-host build or dependency resolution work.

---

**Q3. Correct: C**

RMM's pool allocator supports specifying both an initial_pool_size and a larger maximum_pool_size, letting the pool start small and expand on demand as the workload requires more memory, while still keeping allocations pooled to reduce fragmentation.

---

**Q4. Correct: B**

CUDA Unified Memory allows oversubscription: memory can be logically allocated beyond physical GPU capacity, and accessing non-resident pages triggers on-demand page migration between host and device, which works correctly but introduces page-fault latency that can significantly slow the workload compared to fully GPU-resident data.

---

**Q5. Correct: D**

For small datasets, fixed costs — host-to-device transfer, CUDA context/kernel launch overhead — dominate over the already-fast CPU execution time, so GPU acceleration typically does not pay off until data volume and operation complexity are large enough to amortize those costs. The correct guidance is to stay on CPU (pandas) until scale justifies the switch.

---

**Q6. Correct: A**

Dask-CUDA supports spilling GPU data to host memory (and optionally disk) when a configured device_memory_limit is reached, using mechanisms like JIT-unspill. This lets a working set larger than aggregate GPU memory complete successfully by offloading less-active data rather than erroring out, without shrinking the input dataset.

---

**Q7. Correct: C**

For small, one-time jobs, fixed overheads (PCIe host-to-device transfer, CUDA context/driver initialization, kernel launch latency) are on the order of hundreds of milliseconds to seconds — often larger than the 2-second pandas runtime itself. GPU acceleration pays off when compute time on large, repeated, or iterative workloads dominates over these fixed costs, not for tiny one-off jobs.

---

**Q8. Correct: A**

Explicitly pinning cuda-version=11.8 installs RAPIDS packages built against the CUDA 11.8 runtime, which matches what the installed driver can support, avoiding a CUDA runtime/driver mismatch at import or kernel-launch time.

---

**Q9. Correct: A**

cudf.pandas wraps every pandas object in a proxy; when an operation isn't implemented in libcudf/cuDF, the proxy automatically falls back to the real CPU pandas execution path, moving the underlying data as needed, transparently to the user.

---

**Q10. Correct: D**

cuDF stores nullability via an Arrow-style validity bitmask independent of the value buffer, so integer columns keep their int64 dtype even with missing values. Pandas, without opting into nullable extension dtypes, upcasts int64 to float64 and uses NaN as the sentinel for missing data.

---

**Q11. Correct: C**

cuDF's str.split supports the `n` and `expand` parameters exactly like pandas, performing the ragged split entirely on GPU via libcudf strings kernels and padding shorter rows with nulls in the extra columns — no host round-trip required.

---

**Q12. Correct: B**

Category dtype benefits low-cardinality columns because it stores a small dictionary of unique values plus compact integer codes. With only 15 distinct 'country' values across 50M rows, this drastically cuts memory versus storing repeated strings; groupby on the coded column is also faster.

---

**Q13. Correct: D**

cuDF does not implement several pandas extension dtypes, including Period (and Interval, Sparse). Calling .dt.to_period() on a native cuDF Series raises a NotImplementedError; the standard GPU-friendly workaround is to bucket via .dt.year/.dt.month or by flooring the timestamp to the month, then group on those integer/timestamp columns.

---

**Q14. Correct: B**

cudf.datasets.timeseries() (mirroring dask's timeseries generator) produces a synthetic time-indexed DataFrame directly on the GPU with configurable frequency and id/x/y columns, making it a standard tool for benchmarking cuDF groupby/join/rolling operations without any I/O overhead.

---

**Q15. Correct: A**

cudf.pandas transparently proxies pandas objects and dispatches operations to GPU cuDF when supported. Unsupported operations (like arbitrary Python `.apply()` logic) automatically and silently fall back to CPU pandas, incurring device-to-host and host-to-device transfer overhead each time. The correct fix is rewriting the transformation as a vectorized cuDF operation (e.g., using `.str` accessor methods or `cudf.core.column` ops) so it stays GPU-accelerated.

---

**Q16. Correct: A**

Like pandas, cuDF's `category` dtype stores integer codes referencing a per-column categories array; join operations on categorical columns compare codes for performance. If the two columns were built independently with different category orderings, identical strings can have different codes, so equality checks fail even though the displayed values look identical. Casting to `str` (or aligning categories with `.cat.set_categories()`) forces value-based comparison and fixes the join.

---

**Q17. Correct: D**

cuDF's `groupby()` matches pandas' default of `dropna=True` — rows with null group keys are excluded from the result unless `dropna=False` is explicitly passed. So the plain `gdf.groupby('key').sum()` call already reproduces the pandas default behavior with no extra parameters needed.

---

**Q18. Correct: C**

cuDF provides a built-in spilling feature (backed by RMM) that, once enabled via `cudf.set_option("spill", True)` or the `CUDF_SPILL=on` environment variable, automatically moves device buffers to host memory when GPU memory pressure is detected and brings them back on demand — all within a single-process, single-GPU script, with no manual chunking or cluster setup required.

---

**Q19. Correct: A**

1.5TB comfortably fits within 640GB aggregate GPU memory when combined with spilling, and NVLink-connected GPUs on a single DGX node give the fastest intra-node shuffle bandwidth — no need to add nodes.

---

**Q20. Correct: C**

device_memory_limit sets the GPU memory threshold that triggers spilling, and jit_unspill enables the just-in-time unspilling mechanism that can move data to host memory and further to disk, allowing the workload to exceed aggregate GPU memory safely.

---

**Q21. Correct: D**

Row groups in the 128-256MB range balance GPU memory usage against per-partition scheduling overhead, and split_row_groups lets dask_cudf control how row groups are combined into partitions sized appropriately for 32GB GPUs.

---

**Q22. Correct: B**

50TB with heavy shuffles benefits from scaling across all available GPUs, and UCX-enabled dask-cuda automatically selects the fastest available transport (NVLink within a node, InfiniBand between nodes), maximizing shuffle throughput across the full cluster.

---

**Q23. Correct: B**

Shuffle operations like sorts and joins require temporary buffers on top of the base partition data; sizing partitions to a fraction of GPU memory leaves headroom for these intermediates, preventing OOM while still keeping partition counts reasonable for scheduling efficiency.

---

**Q24. Correct: A**

Projecting columns at read time via the columns parameter avoids decoding and transferring the other 47 columns from disk into GPU memory, and applying date filtering on the reduced projection minimizes both I/O and GPU memory footprint before heavier processing.

---

**Q25. Correct: A**

20TB vastly exceeds the aggregate 1.28TB of GPU memory across all 32 GPUs, so both multi-node scale-out and out-of-core spilling are required. A UCX-enabled multi-node dask-cuda cluster uses NVLink for intra-node and InfiniBand for inter-node communication, while JIT-unspill lets intermediate shuffle/merge data overflow to host RAM/disk instead of causing OOM, giving both the aggregate GPU compute needed for throughput and the memory headroom needed for volume.

---

**Q26. Correct: A**

device_memory_limit sets the GPU memory threshold at which a worker begins spilling data to host memory (and disk if host memory is also exhausted), while jit_unspill=True ensures spilled objects are only brought back into GPU memory precisely when needed for computation, minimizing unnecessary transfers. This is the standard dask-cuda pattern for handling merges/shuffles that exceed GPU memory.

---

**Q27. Correct: D**

Passing both columns and filters to read_parquet lets the reader use Parquet's stored row-group min/max statistics to skip entire row groups that cannot match the date/amount predicates (predicate pushdown), and column projection avoids reading the other unused columns entirely — minimizing disk I/O and the GPU memory footprint of the resulting DataFrame.

---

**Q28. Correct: D**

Louvain is a community detection algorithm that partitions the graph to maximize modularity, directly surfacing tightly-connected clusters such as fraud rings.

---

**Q29. Correct: D**

Betweenness centrality explicitly measures how often a node lies on the shortest path between other node pairs, making it the correct choice for identifying bridge/bottleneck nodes between communities.

---

**Q30. Correct: D**

At 1-minute sampling, 24 hours equals 1440 periods, and center=False produces a trailing (causal) window that only uses past and current values, avoiding look-ahead bias in real-time anomaly detection.

---

**Q31. Correct: C**

cuSpatial provides GPU-accelerated geospatial primitives including point-in-polygon testing, spatial joins, distance calculations, and trajectory analysis, which is exactly what determining polygon membership for GPS pings requires.

---

**Q32. Correct: C**

Computing the target-mean encoding out-of-fold (excluding each row's own fold) prevents each row's own target value from leaking into its own feature, which is the root cause of inflated validation accuracy.

---

**Q33. Correct: A**

cugraph.pagerank accepts a personalization parameter that biases the random-walk restart distribution toward specified seed nodes, producing personalized (topic-sensitive) PageRank scores relevant to the seed set.

---

**Q34. Correct: C**

Louvain is a modularity-based community detection algorithm implemented in cuGraph that partitions a graph into clusters of nodes with dense internal connectivity and sparse external connectivity — exactly the structural signature of a fraud ring. It returns a partition assignment and modularity score, letting analysts flag clusters for review.

---

**Q35. Correct: D**

cuDF supports rolling window operations (rolling(), including grouped rolling via groupby().rolling()) with GPU-accelerated aggregations like mean() and std(), which is exactly how you compute a rolling z-score per sensor without leaving the GPU dataframe, keeping the pipeline fully accelerated end-to-end into cuML.

---

**Q36. Correct: C**

dask_cuda plus the Dask XGBoost API shards the data across all 8 GPUs and coordinates distributed histogram construction via NCCL/UCX, so the aggregate GPU memory (not any single GPU) needs to hold the data.

---

**Q37. Correct: B**

cuML's DBSCAN implementation includes a max_mbytes_per_batch parameter that limits the memory used per batch when computing the neighborhood/pairwise distance matrix, allowing it to scale to datasets that would otherwise exceed GPU memory.

---

**Q38. Correct: A**

DALI's decoders.image operator accepts device='mixed' to take CPU-resident encoded bytes as input and produce GPU-decoded output using the hardware nvJPEG decoder, which is the documented way to offload decoding to the GPU.

---

**Q39. Correct: A**

A LocalCUDACluster with one Dask worker pinned per GPU allows Dask-ML to dispatch independent candidate trainings concurrently across all 4 GPUs, maximizing throughput for an embarrassingly parallel search.

---

**Q40. Correct: B**

XGBoost unified its tree construction algorithms so tree_method='hist' is used for both CPU and GPU, with the target device selected via the separate device='cuda' parameter, replacing the older gpu_hist value.

---

**Q41. Correct: C**

The bottleneck is transfer-count overhead, not bandwidth or memory pool sizing: thousands of tiny H2D copies each pay fixed per-transfer latency. Batching the reads into one (or a few) large operations via cudf.read_csv globbing or dask_cudf collapses thousands of small transfers into a handful of large ones, which is the standard RAPIDS best practice for minimizing host-device transfer overhead.

---

**Q42. Correct: B**

The profile shows synchronous, per-iteration small transfers blocking the pipeline. Batching into one contiguous pinned buffer and issuing a single async cudaMemcpyAsync on a stream eliminates the repeated per-loop synchronous copy overhead and lets transfer overlap with subsequent kernel launches — the textbook fix for this exact Nsight Systems signature.

---

**Q43. Correct: B**

DLProf (Deep Learning Profiler) is specifically designed to analyze deep learning training runs, reporting per-op Tensor Core eligibility/utilization and precision (FP32 vs FP16) breakdown, and it exports this to a TensorBoard plugin for interactive visualization — exactly the reporting style described.

---

**Q44. Correct: B**

cuML's RandomForest estimators build trees using multiple internal CUDA streams in parallel by default (n_streams, default > 1), which introduces run-to-run non-determinism in split selection even with a fixed random_state. Setting n_streams=1 forces single-stream execution, which combined with random_state yields reproducible results — a documented cuML reproducibility gotcha.

---

**Q45. Correct: C**

TF32 is a compute mode (not a storage format) enabled by default in cuBLAS/cuDNN on Ampere+ GPUs. It reinterprets FP32 inputs internally with a reduced mantissa for Tensor Core matmuls/convolutions, requiring zero code changes, while tensors remain stored as standard 32-bit FP32 in memory.

---

