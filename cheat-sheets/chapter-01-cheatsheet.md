# Chapter 1 Cheat Sheet: GPU vs CPU Architecture & CUDA-X Stack

## CPU vs GPU Design
| | CPU | GPU |
|---|---|---|
| Optimization goal | Latency | Throughput |
| Execution model | MIMD | SIMT |
| Core count | Few (4–64+) | Thousands (CUDA cores) |
| Core complexity | Complex, out-of-order | Simple, in-order |
| Best for | Sequential/branchy logic | Data-parallel workloads |

## GPU Hardware Hierarchy
- **CUDA Core** → executes one thread's instruction per cycle
- **Warp** → 32 threads executing in lockstep (SIMT) — memorize this number
- **SM (Streaming Multiprocessor)** → schedules warps, holds many CUDA cores + shared memory
- **GPU** → array of SMs (e.g., H100 ≈ 132 SMs)

Exam trap: Branch divergence within a warp forces serialized execution paths — a common performance gotcha.

## CUDA-X / RAPIDS Stack (bottom → top)
| Layer | Purpose |
|---|---|
| CUDA | Low-level parallel compute platform/driver |
| cuDF | GPU DataFrames (pandas-like) |
| cuML | GPU ML algorithms (scikit-learn-like) |
| cuGraph | GPU graph analytics |
| cuSpatial | GPU geospatial analytics |

Exam trap: cuGraph (networks/graphs) vs cuSpatial (geo/location) are often confused — both handle 'connected' data but different domains.

## Interconnect Bandwidth (approximate — memorize order of magnitude)
| Link | Approx Bandwidth |
|---|---|
| PCIe Gen4 x16 | ~64 GB/s bidirectional (~32 GB/s/direction) |
| PCIe Gen5 x16 | ~128 GB/s bidirectional (~64 GB/s/direction) |
| NVLink 3 (A100) | ~600 GB/s per GPU |
| NVLink 4 (H100) | ~900 GB/s per GPU |

Exam trap: Host-to-device (H2D) transfer runs over PCIe (or NVLink-C2C on Grace Hopper) — NVLink alone accelerates GPU-to-GPU, not host-to-GPU, in standard x86+GPU systems.

## Memory Model
- Host RAM and GPU VRAM are **separate address spaces**.
- Data must be explicitly copied H2D before compute and D2H after — unless using Unified Memory or GPUDirect Storage/RDMA.
- Many small transfers cost more than one large transfer due to fixed per-transfer latency overhead.

## Amdahl's Law
**Speedup = 1 / ((1-P) + P/S)**
- P = parallelizable fraction of runtime
- S = speedup achieved on that parallel fraction
- The sequential/host-bound portion caps total achievable speedup, even with an infinitely fast GPU.

## Identifying GPU-Suitable Workloads
- **Good fit:** same operation across many independent rows/elements (filters, aggregations, matrix ops, tree ensembles).
- **Poor fit:** heavy branching, small data, tight sequential dependencies, frequent host round-trips.

**Rule of thumb:** If compute time << transfer time, the bottleneck is data movement, not GPU throughput — fix by batching data onto the GPU once, persisting data across pipeline stages, or using NVLink/GPUDirect Storage.
