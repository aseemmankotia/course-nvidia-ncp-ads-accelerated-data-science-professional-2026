## Chapter 7 Cheat Sheet: Out-of-Core Workflows & Parquet/ORC I/O

### Detecting Out-of-Core Need
| Symptom | Action |
|---|---|
| CUDA OOM during join/groupby/sort | Switch to dask_cudf + spilling |
| Dataset size approaches/exceeds GPU VRAM | Partition with dask_cudf, avoid single cudf.DataFrame load |
| Single-GPU pipeline works on samples but fails at scale | Re-architect as chunked/distributed read |

### Chunked Reading
```python
import dask_cudf
ddf = dask_cudf.read_parquet(
    "s3://bucket/data/",
    columns=["a", "b", "c"],
    filters=[("date", ">", "2025-06-01")],
    split_row_groups=True,
    blocksize="256MB",
)
```
- `columns=` prunes unneeded columns before decode.
- `filters=` prunes row groups/partitions via stats — NOT per-row filtering.
- `blocksize` controls partition size, independent of on-disk row group size.

### Predicate Pushdown Granularity
| Format | Pushdown level |
|---|---|
| Parquet | Row-group min/max stats |
| ORC | Stripe-level stats |
| CSV/JSON | None — full scan required |

### Format Choice (GPU)
| Format | GPU Decode Speed | Predicate Pushdown | Notes |
|---|---|---|---|
| Parquet | Fastest (optimized libcudf decoder) | Row-group stats | Default choice |
| ORC | Slower (higher per-stripe overhead) | Stripe stats | Use for legacy Hive pipelines |
| CSV | Slow, row-based | None | Avoid for large GPU pipelines |

### Row Group Sizing
- Target: **128–256MB** per row group for GPU reads.
- Too small → excessive reader/decode overhead (many small kernel launches).
- Too large → memory pressure spikes during decode.
- Set via `row_group_size_bytes` at write time.

### Spilling (RMM / cuDF)
| Setting | Purpose |
|---|---|
| `CUDF_SPILL=on` | Enables automatic spill-to-host for GPU buffers |
| `CUDF_SPILL_DEVICE_LIMIT` | Fraction of device memory before spill triggers |
| `device_memory_limit` (dask-cuda) | Per-worker VRAM threshold; set ~80% of GPU memory to leave headroom |
| `jit_unspill=True` (dask-cuda) | Defers unspilling until data is actually needed, reducing PCIe churn |

### Caching Intermediate Results
```python
result = ddf.groupby("key").agg({"val": "sum"}).persist()
```
- `.persist()` → materializes and caches across the cluster; reused by downstream ops.
- `.compute()` → pulls to a single concrete result, does NOT keep it cached in the graph.
- Alternative: write intermediates to Parquet as checkpoints for expensive, reusable stages.

### Partitioning Scheme
- Prefer **Hive-style directory partitioning** (`date=YYYY-MM-DD/`) for coarse, cheap pruning before file I/O.
- Combine with row-group filters for fine-grained pruning within remaining files.

### Common Exam Traps
- Assuming predicate pushdown filters individual rows (it filters row groups/stripes/partitions).
- Setting spill device limit to 1.0 (removes headroom, spill triggers too late).
- Choosing ORC by default over Parquet without a legacy-format requirement.
- Forgetting `.persist()` and recomputing expensive stages repeatedly.
