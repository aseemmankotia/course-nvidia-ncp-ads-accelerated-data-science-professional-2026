## ETL Design & Multi-GPU Scaling with Dask-cuDF

### Cluster Topology Decision Table
| Scenario | Best Topology |
|---|---|
| Data fits in one GPU's memory w/ headroom | Single-GPU `cudf`, no cluster |
| Data exceeds one GPU but fits in aggregate memory of GPUs on one node (incl. shuffle overhead) | `LocalCUDACluster` (one node, multi-GPU) |
| Data + shuffle working memory exceeds single-node aggregate GPU memory | Multi-node Dask cluster with UCX over InfiniBand |

**Key gotcha:** Always account for shuffle working-memory overhead (often ~2x base data size) when comparing dataset size to aggregate GPU memory — don't just compare base size to capacity.

### LocalCUDACluster Basics
```python
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

cluster = LocalCUDACluster()  # 1 worker per visible GPU by default
client = Client(cluster)
```
- Default: one worker per GPU, pinned via `CUDA_VISIBLE_DEVICES`.
- Supports `rmm_pool_size`, `device_memory_limit`, `protocol="ucx"` for NVLink/IB.

### dask_cudf Partitioning
- `dask_cudf.DataFrame` = collection of `cudf.DataFrame` partitions across workers.
- Partition size should be a **fraction of per-GPU memory**, not maxed out — leaves headroom for shuffle/join intermediate memory spikes.
- Skewed partitions (from filters) → rebalance with:
```python
ddf = ddf.repartition(npartitions=N)
# or
ddf = ddf.repartition(partition_size="256MB")
```

### Lazy Evaluation / Task Graph
- Operations (`groupby`, `merge`, `map_partitions`, etc.) build a **lazy task graph** — no GPU compute happens yet.
- Trigger execution with:
  - `.compute()` → returns concrete `cudf` object (single GPU/host memory).
  - `.persist()` → materializes across cluster, keeps as distributed futures for reuse.

### Shuffle Cost (groupby / merge / sort)
- Non-partition-aligned keys require **shuffle**: rows redistributed across workers so matching keys co-locate.
- Symptom: GPU idle time + spikes in NVLink/network traffic during merge/groupby.
- Mitigate with `protocol="ucx"` for NVLink (intra-node) / InfiniBand (inter-node) communication.

### Decision Checklist for Exam Scenarios
1. Does data fit in **one** GPU with headroom? → Single-GPU.
2. Does data + shuffle overhead fit in **one node's** aggregate GPU memory? → `LocalCUDACluster`.
3. Does it exceed one node's capacity? → Multi-node + UCX/InfiniBand.
4. Uneven partitions before groupby/merge? → `repartition()` first.
5. Nothing computed yet after an operation? → Normal lazy behavior; call `.compute()`/`.persist()`.
