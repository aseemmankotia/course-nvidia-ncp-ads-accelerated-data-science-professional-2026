## Profiling Tool Selection

| Tool | Scope | Use When | Output |
|---|---|---|---|
| Nsight Systems | Whole pipeline/system timeline | CPU-GPU sync stalls, H2D/D2H transfer frequency, multi-GPU timeline gaps | .nsys-rep / .qdrep |
| Nsight Compute | Single kernel deep-dive | Low occupancy, register pressure, memory throughput inside one kernel | .ncu-rep |
| DLProf | DL training aggregated metrics | Iteration time, Tensor Core %, GPU util % across a training run | .qdrep + CSV/JSON summary (viewable in Nsight Systems) |

**Decision rule:**
- Bottleneck spans multiple pipeline stages → **Nsight Systems**
- Bottleneck inside one kernel → **Nsight Compute**
- Need DL-run-level aggregated stats (Tensor Core %, iteration time) → **DLProf**

## Key Commands
```bash
nsys profile -o report python train.py          # Nsight Systems timeline
ncu --set full -o kreport ./app                 # Nsight Compute kernel deep-dive
dlprof --mode=pytorch python train.py           # DLProf DL training profiling
```

## RMM Pool Allocation
- Default `cudaMalloc`/`cudaFree` per allocation = high overhead for many small/frequent allocations (common in cuDF/RAPIDS ETL).
- RMM pool allocator pre-allocates a large block and sub-allocates from it → fewer driver calls.
- Enable: `rmm.reinitialize(pool_allocator=True, initial_pool_size=...)`

## Mixed Precision Comparison

| Precision | Exponent/Mantissa bits | Memory vs FP32 | Needs loss scaling? |
|---|---|---|---|
| FP32 | 8/23 | Baseline | No |
| TF32 (Ampere+) | 8/10 | Same storage as FP32, faster Tensor Core compute | No |
| FP16 | 5/10 | ~50% reduction | Yes |
| BF16 | 8/7 | ~50% reduction | No (FP32-range exponent) |

## Batching Strategy
- Low GPU util + many short kernel-launch gaps on Nsight Systems timeline → **increase batch size**.
- Larger batches amortize launch overhead and improve Tensor Core utilization.
- Tradeoff: larger batch size increases device memory requirement — balance against available GPU memory.

## Reducing Host-to-Device Transfers
- Batch transfers (fewer, larger copies) instead of per-sample copies.
- Use pinned (page-locked) host memory for async transfers.
- Overlap transfer with compute using CUDA streams/prefetching.
- Detect via Nsight Systems timeline: many small HtoD memcpy blocks = red flag.

## Reproducibility Checklist
- Pin container image tag (specific NGC release), never `latest`.
- Pin library versions (RAPIDS, cuDF, PyTorch, CUDA toolkit).
- Fix random seeds; set deterministic algorithm flags where supported.
- Document GPU driver/CUDA version used in the pipeline.
