## Chapter 7: Out-of-Core Workflows and Efficient Parquet/ORC I/O — Flashcards

| # | Front (Question) | Back (Answer) |
|---|-----------------|---------------|
| 1 | Recommended Parquet row group size for GPU reads? | Roughly 128–256MB per row group. Exam tip: too small = per-row-group reader overhead; too large = memory pressure during GPU decode. |
| 2 | Environment variable to enable cuDF spilling to host memory? | CUDF_SPILL=on, paired with CUDF_SPILL_DEVICE_LIMIT to set the device-memory fraction that triggers spilling. |
| 3 | dask_cudf command to read Parquet in memory-safe partitioned chunks? | dask_cudf.read_parquet(path, split_row_groups=True, blocksize='256MB'). Exam tip: blocksize sizes partitions, not individual row groups. |
| 4 | How to apply predicate pushdown with dask_cudf.read_parquet? | Pass filters=[('col','>',value)] and columns=[...]. Filters skip whole row groups using min/max stats — not per-row filtering. |
| 5 | Parquet vs ORC read throughput on GPU (cuDF)? | Parquet is generally faster: libcudf's Parquet decoder is more optimized than its ORC decoder, which has higher per-stripe decode overhead. Exam tip: default to Parquet unless legacy Hive/ORC constraints require otherwise. |
| 6 | How to avoid recomputation across a multi-stage dask_cudf pipeline? | Call .persist() to materialize and cache results in distributed memory, or checkpoint intermediates to Parquet on disk. .compute() alone does not cache for reuse in the graph. |
| 7 | Strongest signal a workflow needs out-of-core handling? | Repeated CUDA out-of-memory errors during joins/groupbys/sorts as dataset size nears or exceeds GPU VRAM. Exam tip: switch from single-GPU cudf.DataFrame to dask_cudf with spilling enabled. |
| 8 | Default device_memory_limit behavior in dask-cuda's LocalCUDACluster? | If unset, workers can use the full GPU memory, leaving no headroom for proactive spilling. Exam tip: set device_memory_limit to roughly 80% of GPU memory so spilling triggers before an OOM. |
| 9 | What is JIT-Unspill in dask-cuda? | A feature that defers moving spilled data back to device memory until it's actually needed, reducing redundant PCIe transfers. Enabled via jit_unspill=True on LocalCUDACluster. |
| 10 | Columnar vs row-based formats for GPU predicate pushdown? | Columnar formats (Parquet/ORC) support column pruning and row-group stat filtering. Row-based formats (CSV/JSON) require full-row scans with no pushdown. Exam favors Parquet for selective GPU queries. |
| 11 | Best partitioning scheme for large Parquet datasets filtered by date on GPU? | Hive-style directory partitioning (e.g., date=2026-01-01/) lets the reader skip entire partitions before opening files — cheaper than relying on row-group stats alone. |
| 12 | What actually triggers RMM/cuDF spilling to host memory? | GPU memory usage crossing the CUDF_SPILL_DEVICE_LIMIT threshold; buffers move host-side and page back to device on access. Exam tip: minimize spilling via correct partition/row-group sizing rather than disabling it. |
