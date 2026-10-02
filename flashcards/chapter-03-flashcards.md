## Chapter 3: cuDF Fundamentals: DataFrame Operations at GPU Speed — Flashcards

| # | Front (Question) | Back (Answer) |
|---|-----------------|---------------|
| 1 | cudf.read_csv vs pandas.read_csv — key exam difference? | cudf.read_csv parses directly into GPU memory (GPU-accelerated), avoiding CPU parse + transfer step. Exam tip: prefer cudf.read_csv/read_parquet for large files needing fast GPU pipeline ingestion. |
| 2 | Best file format for GPU-accelerated columnar I/O in cuDF? | Parquet, via cudf.read_parquet — supports column/predicate pushdown and native GPU decoding. Exam tip: Parquet beats CSV/JSON for large columnar datasets. |
| 3 | cuDF merge() default row order vs pandas? | Not guaranteed to match pandas order by default. Use sort=True for deterministic, pandas-like ordering. Exam gotcha: assuming identical output order without sort=True. |
| 4 | df.apply(fn, axis=1) on cuDF — performance impact? | Row-wise apply uses Numba JIT per row, losing columnar vectorization — significantly slower than vectorized column ops. Exam tip: always prefer vectorized expressions (df['a']+df['b']) over apply. |
| 5 | Which operations commonly fall back or fail on cuDF vs pandas? | Arbitrary Python UDFs with string/object manipulation, complex regex, and non-Numba-compatible logic. Exam tip: these are the 'unsupported ops' the exam heavily tests. |
| 6 | cuDF groupby().agg() with dict of {col: [funcs]} — supported? | Yes, fully vectorized and GPU-accelerated, same API pattern as pandas. Exam tip: prefer this over groupby().apply(lambda...) for multi-aggregation. |
| 7 | Why chain cuDF methods instead of storing each step in a variable? | Reduces intermediate DataFrame materialization and memory allocations, lowering kernel launch overhead. Exam tip: method chaining is a recommended performance pattern for GPU pipelines. |
| 8 | When should you choose pandas over cuDF for a transformation? | When the transformation relies on arbitrary complex Python logic, unsupported string ops, or very small datasets where GPU transfer overhead outweighs benefit. Exam tip: exam tests 'choosing cuDF vs pandas' scenario questions. |
| 9 | cudf.DataFrame construction — key parity note with pandas.DataFrame? | cuDF supports similar constructors (dict, list, from_pandas) but underlying data lives in GPU memory (device memory) via Arrow-based columnar format. Exam tip: cudf.from_pandas() converts CPU DataFrame to GPU. |
| 10 | cuDF groupby().apply(lambda ...) — performance implication? | Forces per-group Python execution similar to row-wise apply, losing GPU vectorization; much slower than groupby().agg(). Exam tip: apply after groupby is a common fallback trap. |
| 11 | cuDF read_csv reading gzip-compressed CSV — performance note? | Requires CPU-bound decompression before GPU parsing, reducing the GPU I/O speed advantage. Exam tip: uncompressed CSV or Parquet is preferred for max GPU read performance. |
| 12 | cuDF vs pandas API parity — what is heavily tested? | read_csv, groupby, and merge are explicitly called out as heavily tested for API parity between cuDF and pandas. Exam tip: know exact parameter differences (e.g., sort in merge). |
