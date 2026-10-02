# Answer Key — Practice Test 1

**Q1. Correct: C**

For tiny, one-off datasets, the fixed cost of host-to-device transfer and CUDA context initialization exceeds any compute savings from the GPU; pandas on CPU finishes faster and with less overhead.

---

**Q2. Correct: D**

cuGraph is the RAPIDS library dedicated to GPU-accelerated graph analytics, implementing algorithms like PageRank, BFS, connected components, and Louvain community detection.

---

**Q3. Correct: A**

GPUs contain thousands of simpler cores designed for massively parallel, throughput-oriented execution (SIMT model), whereas CPUs have fewer, more complex cores optimized for fast sequential and branch-heavy execution — this is the core reason GPUs excel at data-parallel workloads like DataFrame operations and matrix math.

---

**Q4. Correct: C**

RAPIDS is officially distributed via conda using the rapidsai, conda-forge, and nvidia channels together, with explicit rapids, python, and cuda-version pins to ensure compatible builds — this is the documented installation pattern from the RAPIDS release selector.

---

**Q5. Correct: A**

Dask-cuDF partitions the dataset into chunks distributed across multiple GPUs (and even multiple nodes), allowing a dataset larger than any single GPU's memory to be processed in parallel using cuDF operations under the hood — this is the standard RAPIDS scale-out pattern for out-of-core/multi-GPU workloads.

---

**Q6. Correct: C**

NVIDIA NGC hosts pre-built, validated RAPIDS container images with matched CUDA drivers, libraries, and dependencies baked in, providing a consistent, reproducible environment that runs identically across on-prem and cloud GPU infrastructure — this is the recommended deployment pattern for reproducibility at scale.

---

**Q7. Correct: C**

For very small datasets (tens of thousands of rows), the fixed costs of PCIe host-to-device data transfer and CUDA kernel launch overhead typically exceed the compute time saved, so cuDF often shows no benefit or is slower than pandas. RAPIDS pays off as data volume and computational intensity grow, not for trivially small jobs.

---

**Q8. Correct: B**

The official RAPIDS conda install requires the rapidsai, conda-forge, and nvidia channels together, with the rapids meta-package, matching Python version, and cuda-version pinned to the installed CUDA toolkit — this is the documented pattern from the RAPIDS release selector.

---

**Q9. Correct: D**

cudf.pandas is the zero-code-change GPU accelerator for pandas. Launching a script with `python -m cudf.pandas script.py` (or `%load_ext cudf.pandas` in a notebook) proxies pandas calls to cuDF on GPU-supported operations and falls back to pandas/CPU automatically for unsupported ones, requiring no source modification.

---

**Q10. Correct: D**

cuDF's GPU string/regex engine (built on libcudf, not Python's `re`) supports a subset of regex syntax and does not support lookbehind/lookahead assertions among other advanced constructs. Patterns using these features that work in pandas will fail or behave differently in cuDF.

---

**Q11. Correct: C**

cuDF follows the Apache Arrow columnar format, which tracks null values with a separate validity bitmask rather than relying on a sentinel value like NaN embedded in the data itself. This lets cuDF represent nulls consistently across all dtypes (including integers, which pandas' legacy NaN approach cannot do without upcasting to float), and it displays as <NA>.

---

**Q12. Correct: B**

Casting a low-cardinality string column to 'category' dtype stores the data as compact integer codes backed by a small dictionary of unique values, drastically reducing memory usage and making groupby/join comparisons operate on integers instead of variable-length strings, which is significantly faster on GPU.

---

**Q13. Correct: C**

cudf.datasets.timeseries() is RAPIDS' built-in helper that generates a configurable synthetic time-indexed DataFrame directly in GPU memory, commonly used for quick benchmarking and testing of cuDF pipelines without needing external data or custom generation logic.

---

**Q14. Correct: A**

cuDF (and cudf.pandas) accelerates groupby operations that map to GPU-native aggregations or Numba-JIT-compilable numeric UDFs. Arbitrary Python-object manipulation (e.g., string formatting, non-numeric logic) cannot be compiled for GPU execution, so cudf.pandas transparently falls back to CPU pandas for that specific call while still accelerating other GPU-compatible operations elsewhere in the script.

---

**Q15. Correct: B**

cuDF's Series.fillna() and .median() are both natively GPU-accelerated, so chaining them keeps computation entirely on the GPU while imputing missing values with the median as required.

---

**Q16. Correct: D**

cudf.pandas is RAPIDS' zero-code-change GPU accelerator mode: it intercepts pandas API calls, executes them on cuDF when possible, and transparently falls back to CPU pandas for unsupported operations, requiring no source code rewrite.

---

**Q17. Correct: B**

Casting the join key to 'category' dtype dictionary-encodes the strings into integer codes, letting cuDF perform the join comparison on integers instead of variable-length strings, which significantly speeds up GPU joins on high-cardinality string keys.

---

**Q18. Correct: A**

libcudf implements string columns using an Arrow-style offsets + character-buffer layout resident entirely on the GPU, enabling many .str accessor methods to execute as GPU kernels without moving data to the host.

---

**Q19. Correct: C**

dask-cuDF splits the dataset into partitions sized to fit in GPU memory and can spill excess partitions to host memory, enabling correct out-of-core processing of joins and groupbys far larger than a single GPU's memory.

---

**Q20. Correct: A**

LocalCUDACluster spins up one Dask worker per GPU on a single node, letting dask-cuDF distribute DataFrame operations across all 4 A100s in parallel.

---

**Q21. Correct: B**

RAPIDS guidance recommends partitions in the low hundreds of MB to about 1GB, since operations like joins and groupbys can temporarily require 2-3x the partition size in GPU memory, and this range balances task scheduling overhead against memory safety.

---

**Q22. Correct: A**

dask_cudf.read_parquet supports column projection via the columns argument and predicate pushdown via filters, using row-group and partition metadata to skip unnecessary data before it is even read from disk, minimizing I/O and GPU memory footprint.

---

**Q23. Correct: A**

The 50GB dataset fits comfortably within the 48GB GPU's memory alongside working overhead, so direct cuDF usage delivers full GPU acceleration without the scheduling and communication overhead that Dask introduces unnecessarily.

---

**Q24. Correct: C**

.persist() triggers computation of the lazy graph and keeps the result materialized as a distributed collection across worker GPU memory, so subsequent references reuse the cached partitions instead of recomputing the graph each time.

---

**Q25. Correct: C**

dask-cuDF partitions the dataset across GPUs and, when memory pressure occurs, spills partitions from GPU to host memory (and disk if configured), enabling true out-of-core processing of datasets far larger than aggregate GPU memory while keeping computation GPU-accelerated.

---

**Q26. Correct: B**

RAPIDS/dask-cuDF guidance recommends larger partitions than typical CPU Dask (roughly 1-3GB) because GPU kernel launch and per-operation overhead is amortized better over larger chunks, while still leaving enough partitions per GPU for parallelism and memory headroom.

---

**Q27. Correct: D**

The single node is already near its utilization and memory ceiling with no room for the projected 8TB growth. A multi-node Dask-CUDA cluster with UCX over InfiniBand adds aggregate GPU memory and compute, provides headroom for growth, and is the standard NCP-ADS-tested pattern for scaling beyond single-node capacity when throughput and future data volume both increase.

---

**Q28. Correct: B**

Circular, colluding money-laundering rings are a relationship pattern, not a per-account statistical or geographic anomaly. Modeling accounts/transactions as a graph and applying cuGraph community detection (Louvain) surfaces tightly-connected clusters indicative of collusion.

---

**Q29. Correct: C**

PageRank ranks nodes by importance based on the quantity and quality of incoming links, recursively weighting links from important nodes more heavily — exactly the described requirement.

---

**Q30. Correct: A**

The resolution parameter in cugraph.louvain controls community granularity — increasing resolution favors splitting the graph into more, smaller communities, while decreasing it merges nodes into fewer, larger communities.

---

**Q31. Correct: B**

cuspatial.point_in_polygon is the GPU-accelerated function designed to test whether each point falls within a given set of polygons, matching the delivery-zone lookup requirement.

---

**Q32. Correct: B**

Betweenness centrality measures how often a node lies on the shortest paths between other pairs of nodes, which directly identifies bridge nodes connecting separate clusters.

---

**Q33. Correct: B**

Out-of-fold target encoding — computing groupby means per fold with dask_cudf/cudf and applying them only to held-out rows — prevents a row's own target from leaking into its encoded feature, and it stays fully GPU-accelerated across a distributed dataset.

---

**Q34. Correct: A**

Louvain community detection partitions a graph into densely-connected subgroups (communities) by optimizing modularity, which is exactly what's needed to surface tightly-knit clusters of accounts that transact heavily with each other, a classic fraud-ring detection pattern in cuGraph.

---

**Q35. Correct: D**

cuDF supports groupby-rolling operations that compute per-group (per-device) rolling window statistics like mean and standard deviation entirely on the GPU, avoiding host-device transfers and giving the fastest end-to-end path for large-scale rolling z-score anomaly detection.

---

**Q36. Correct: D**

cuML DBSCAN is a GPU-accelerated density-based clustering algorithm that discovers arbitrarily shaped clusters and natively flags noise points as outliers, without requiring a predefined cluster count.

---

**Q37. Correct: D**

XGBoost 2.0+ deprecated the legacy 'gpu_hist' tree method in favor of tree_method='hist' combined with the device='cuda' parameter to explicitly select GPU-accelerated training.

---

**Q38. Correct: D**

Pairing Optuna with a Dask-CUDA cluster allows multiple hyperparameter trials to be dispatched and executed concurrently across the GPUs in the cluster, minimizing wall-clock tuning time.

---

**Q39. Correct: D**

cuML is designed as a near drop-in replacement for scikit-learn, exposing a familiar fit/predict API while running the underlying computations on the GPU, yielding large speedups on big tabular datasets.

---

**Q40. Correct: B**

NVIDIA DALI offloads JPEG decoding and augmentation operations onto the GPU, removing the CPU preprocessing bottleneck and keeping the GPU continuously fed with data during training.

---

**Q41. Correct: D**

Keeping intermediate results as cuDF/cuML GPU objects avoids round trips to host memory, which is the root cause of the memcpy overhead shown in the profile.

---

**Q42. Correct: C**

An RMM pool allocator with a tuned device_memory_limit reduces fragmentation and gives predictable control over when host spilling occurs, directly addressing the GPU memory pressure described without abandoning GPU acceleration.

---

**Q43. Correct: A**

Nsight Compute provides per-kernel metrics such as achieved occupancy, memory throughput, and roofline analysis needed to classify a kernel as compute-bound or memory-bound.

---

**Q44. Correct: A**

Pinning environment and version details (RAPIDS, CUDA, driver, and GPU architecture) is the strongest guard against reproducibility failures caused by library version drift, CUDA compatibility mismatches, or hardware-related numerical differences.

---

**Q45. Correct: A**

Mixed precision uses lower-precision formats to leverage Tensor Cores, boosting throughput and reducing memory usage, while techniques like loss scaling help preserve training stability despite reduced numerical precision.

---

