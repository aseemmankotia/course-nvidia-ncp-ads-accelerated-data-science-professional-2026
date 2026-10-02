## Chapter 1: GPU vs CPU Architecture and the CUDA-X Data Science Stack — Flashcards

| # | Front (Question) | Back (Answer) |
|---|-----------------|---------------|
| 1 | CUDA warp size (threads executing in lockstep)? | 32 threads. Exam tip: Under SIMT, all 32 threads in a warp execute the same instruction each cycle; branch divergence within a warp degrades performance. |
| 2 | NVLink 4 (Hopper H100) approximate per-GPU bidirectional bandwidth? | ~900 GB/s. Exam tip: Compare to PCIe Gen5 x16 (~128 GB/s) — NVLink is roughly 7x faster for GPU-to-GPU transfers. |
| 3 | PCIe Gen4 x16 approximate bidirectional bandwidth? | ~64 GB/s total (~32 GB/s per direction). Exam tip: Host-to-device transfer over PCIe is often the real bottleneck, not GPU compute. |
| 4 | PCIe Gen5 x16 approximate bidirectional bandwidth? | ~128 GB/s total (~64 GB/s per direction). Exam tip: Even Gen5 PCIe is far slower than NVLink for GPU-to-GPU data movement. |
| 5 | cuDF — purpose in CUDA-X stack? | GPU-accelerated DataFrame library with a pandas-like API. Exam tip: Use for drop-in acceleration of filter/groupby/join ETL operations. |
| 6 | cuML — purpose in CUDA-X stack? | GPU-accelerated machine learning library with a scikit-learn-like API (e.g., Random Forest, KMeans). Exam tip: Choose when replacing sklearn model training. |
| 7 | cuGraph — purpose in CUDA-X stack? | GPU-accelerated graph analytics (PageRank, BFS, connected components). Exam tip: Choose for network/graph problems, not tabular or geospatial data. |
| 8 | cuSpatial — purpose in CUDA-X stack? | GPU-accelerated geospatial operations (point-in-polygon, spatial joins, trajectory analysis). Exam tip: Choose for GPS/location-based scenarios, not generic graph problems. |
| 9 | CPU vs GPU core design philosophy? | CPU = latency-optimized, few complex cores, out-of-order execution, MIMD. GPU = throughput-optimized, thousands of simple cores, SIMT lockstep execution in warps of 32. |
| 10 | What is a Streaming Multiprocessor (SM)? | The GPU's core compute unit containing multiple CUDA cores, warp schedulers, and shared memory. Exam tip: Work is scheduled onto SMs as blocks of warps, not individual threads. |
| 11 | Amdahl's Law formula and exam application? | Speedup = 1 / ((1-P) + P/S), where P = parallel fraction, S = speedup of that fraction. Exam tip: The sequential (non-parallelizable) portion caps maximum speedup regardless of GPU power. |
| 12 | Host RAM vs GPU VRAM — key architectural fact? | They are separate memory spaces requiring explicit H2D/D2H data movement over PCIe or NVLink (unless using Unified Memory or Grace Hopper NVLink-C2C). Exam tip: Minimizing transfers is critical since PCIe bandwidth is far lower than GPU compute throughput. |
