## Chapter 5 Cheat Sheet: cuDF Interop, Synthetic Data, Pipeline Efficiency

### Conversion Cost Comparison
| Operation | Direction | Cost | Justified When |
|---|---|---|---|
| `to_pandas()` | Device → Host | Full copy, scales with data size | Small output/inspection, or unsupported op on a small slice |
| `from_pandas()` | Host → Device | Full copy, scales with data size | Compute savings on GPU outweigh one-time transfer |
| `.values` / `.to_cupy()` | Stays on GPU | Zero-copy (via `__cuda_array_interface__`) | Always preferred for numeric GPU math (cupy interop) |
| `to_arrow()` | Device → Host (Arrow buffer) | Copy to host Arrow format | Interop with Arrow-consuming tools outside RAPIDS |

### Zero-Copy Interop Rules
- cuDF, cuML, cuGraph share an **Arrow-compatible GPU columnar layout** — data moves between them without duplication.
- cuDF ↔ cupy: zero-copy via `__cuda_array_interface__` (`.values`, `.to_cupy()`).
- `to_pandas()` / `from_pandas()` are the **exception** — they always force a host round trip because pandas is CPU-only.

### Synthetic Data Generation
- `cudf.datasets.timeseries(start, end, freq, dtypes, seed)` — GPU-resident synthetic DataFrame, schema-configurable, ideal for scale benchmarking.
- Prefer GPU-native generation over pandas-generate-then-`from_pandas()` — avoids skewing benchmark timing with transfer overhead.
- Use synthetic data to test pipeline behavior at multiple scales before production data exists.

### Pipeline Efficiency Rules
| Rule | Why |
|---|---|
| Prune columns at **read time** (`usecols=`) | Avoids allocating/processing unused columns downstream |
| **Chain** operations in one expression | Intermediate results garbage-collected immediately, lower peak memory |
| Avoid naming every intermediate step | Named refs keep GPU memory alive longer |
| Scope pandas fallback to the smallest slice | Minimizes host transfer size and time |
| Avoid `to_pandas()` inside loops | Each call re-copies full data; use `.head()` for inspection instead |

### Decision Rule: Pandas Fallback
Acceptable ONLY if:
1. cuDF has **no equivalent API**, AND
2. The data needing that operation is a **small slice** (single column/subset), not the whole DataFrame.

Convert only that slice → apply pandas op → reassign back into cuDF. Never convert the entire large DataFrame for a single unsupported method.

### Common Exam Traps
- Assuming `to_pandas()`/`from_pandas()` is "free" or cached — it is not; it's a real copy every call.
- Confusing `to_arrow()` (host Arrow buffer) with true zero-copy GPU interop (cupy `__cuda_array_interface__`).
- Dropping unused columns *after* heavy transforms instead of pruning at read time.
- Generating synthetic test data on CPU and transferring it, instead of generating it natively on GPU.
