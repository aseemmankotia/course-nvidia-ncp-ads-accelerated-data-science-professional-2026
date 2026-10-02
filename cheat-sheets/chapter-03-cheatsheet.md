## cuDF Fundamentals Cheat Sheet

### Core API Parity Table
| Operation | pandas | cuDF | Note |
|---|---|---|---|
| Read CSV | pandas.read_csv | cudf.read_csv | cuDF parses directly on GPU |
| Read Parquet | pandas.read_parquet | cudf.read_parquet | Best GPU I/O, supports pushdown |
| Groupby+agg | df.groupby(k).agg({...}) | same syntax | Fully vectorized on GPU |
| Merge/join | df.merge(df2, on=k) | same syntax | Row order NOT guaranteed unless sort=True |
| Row apply | df.apply(fn, axis=1) | same syntax but Numba JIT | Major perf penalty, avoid |

### Construction
- `cudf.DataFrame({...})` — dict/list constructors mirror pandas
- `cudf.from_pandas(df)` — CPU-to-GPU conversion (data copy)
- `df.to_pandas()` — GPU-to-CPU conversion for unsupported ops

### Groupby/Agg Rules
- Prefer `groupby(key).agg({'col': ['mean','std','count']})` — vectorized
- AVOID `groupby(key).apply(lambda g: ...)` — forces per-group Python execution
- Multiple aggregation functions per column: fully supported, GPU-accelerated

### Merge/Join Semantics
- Default `sort=False`: cuDF does NOT guarantee pandas-matching row order (parallel hash join)
- Set `sort=True` for deterministic, pandas-like ordering
- Join types (inner/left/right/outer) match pandas semantics

### Performance Patterns
| Pattern | Speed | When to use |
|---|---|---|
| Vectorized column ops (`df['a']+df['b']`) | Fastest | Always prefer |
| `.apply(fn, axis=1)` w/ Numba JIT | Slow | Only if truly unavoidable, simple numeric fn |
| Arbitrary Python/regex UDF | Fails/falls back | Use pandas instead |
| Method chaining (`.query().groupby().agg()`) | Faster | Reduces intermediate materialization |

### Unsupported / Fallback-Prone Operations
- Arbitrary string regex manipulation
- Complex object-dtype Python UDFs
- Non-Numba-compatible apply logic
- Some nested JSON parsing edge cases

### Choosing cuDF vs pandas
- Large numeric/columnar data + vectorizable ops → **cuDF**
- Small data (<~1GB) or complex arbitrary Python logic → **pandas** (GPU transfer overhead not worth it)
- Heavy string/regex parsing not vectorizable → **pandas**

### Exam Gotchas
1. Assuming merge() output order always matches pandas (needs `sort=True`)
2. Using `.apply(axis=1)` expecting full GPU speedup — it doesn't vectorize
3. Assuming all pandas string/object ops are supported in cuDF without fallback
4. Forgetting `cudf.read_parquet` outperforms `cudf.read_csv` for large columnar data
