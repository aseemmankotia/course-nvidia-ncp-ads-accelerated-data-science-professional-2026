## Chapter 3: cuDF Fundamentals: DataFrame Operations at GPU Speed — Practice Questions

**Q1.** A data engineer needs to load a 50GB CSV file and perform groupby aggregations as fast as possible on a single GPU node. Which approach BEST meets this requirement?

A) Use pandas.read_csv with chunksize and concatenate results
B) Use cudf.read_csv to load the file directly into GPU memory, then use cuDF groupby/agg
C) Use pandas.read_csv then convert the resulting DataFrame to cuDF with cudf.from_pandas
D) Use cudf.read_csv but perform the groupby with a Python for-loop over unique keys

<details>
<summary>Answer</summary>

**Correct: B**

cudf.read_csv parses directly on the GPU, avoiding CPU-to-GPU transfer overhead, and cuDF groupby/agg is vectorized and GPU-accelerated end to end.

**Why the others are wrong:**
- Chunked pandas reading stays entirely on CPU and does not leverage GPU acceleration at all.
- Reading with pandas first still incurs the full CPU parsing cost before transferring to GPU, wasting time.
- A Python for-loop over groups forces row-wise Python execution, defeating GPU vectorization and causing severe fallback penalties.

</details>

---

**Q2.** A candidate writes: df['result'] = df.apply(lambda row: row['a'] + row['b'], axis=1) on a cuDF DataFrame with 10 million rows. What is the MOST likely performance outcome?

A) It executes but performs significantly slower than a vectorized equivalent because row-wise apply invokes per-row Python/Numba overhead
B) It runs at full GPU speed because cuDF apply is JIT-compiled automatically for all Python functions
C) It raises a NotImplementedError and the script crashes
D) cuDF silently converts the entire DataFrame to pandas and the apply runs on CPU only

<details>
<summary>Answer</summary>

**Correct: A**

cuDF supports row-wise apply via Numba JIT, but it compiles per-row logic and loses the columnar vectorization advantage, making it much slower than df['a'] + df['b'].

**Why the others are wrong:**
- Not all Python functions JIT-compile cleanly; even when they do, per-row invocation overhead remains much slower than vectorized ops.
- cuDF does support axis=1 apply for many simple functions via Numba, so it does not simply crash.
- cuDF does not silently fall back the whole DataFrame to pandas for apply; it attempts GPU execution via Numba, just inefficiently.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** Which cuDF operation is MOST likely to trigger an unsupported-operation error or automatic fallback behavior compared to its pandas equivalent?

A) df.groupby('key').sum()
B) df.apply(some_arbitrary_python_function, axis=1) using arbitrary object dtype columns and string manipulation
C) df.merge(df2, on='key', how='inner')
D) df.sort_values('col')

<details>
<summary>Answer</summary>

**Correct: B**

Arbitrary Python UDFs involving object/string manipulation frequently fail to JIT-compile with Numba on the GPU, forcing errors or requiring conversion to pandas.

**Why the others are wrong:**
- groupby().sum() is a core vectorized cuDF operation with full GPU support.
- merge/join is natively GPU-accelerated in cuDF with pandas-like semantics.
- sort_values is a standard vectorized cuDF operation supported on GPU.

</details>

---

**Q4.** By default, how does cudf.DataFrame.merge() handle row order in the output compared to pandas.DataFrame.merge()?

A) cuDF merge does not guarantee row order matching pandas by default; sort=True must be set for deterministic ordering similar to pandas behavior
B) cuDF guarantees the exact same row order as pandas for all join types
C) cuDF merge always sorts output by the join key regardless of the sort parameter
D) cuDF merge randomizes row order every execution and cannot be made deterministic

<details>
<summary>Answer</summary>

**Correct: A**

Due to parallel GPU hash-join execution, cuDF merge output order is not guaranteed to match pandas unless sort=True is explicitly passed, which is a well-known parity gotcha.

**Why the others are wrong:**
- GPU parallel joins process partitions concurrently, so exact pandas row order is not guaranteed without sort=True.
- sort=False (the default) does not force sorted output; sort must be explicitly enabled.
- Output is deterministic given identical input and hardware/algorithm, it is just not necessarily pandas-ordered.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q5.** A pipeline chains several cuDF operations: df.query(...).groupby(...).agg(...).reset_index(). Why is this method-chaining pattern preferred for GPU performance over storing each intermediate result in a separate variable?

A) Method chaining forces cuDF to run entirely on CPU which is faster for small data
B) It has no performance effect on GPU; it only affects code readability
C) It reduces GPU memory allocations and intermediate materialization, allowing the query planner-like optimizations and reduced kernel launch overhead
D) Chaining automatically converts the pipeline to use Dask for multi-GPU scaling

<details>
<summary>Answer</summary>

**Correct: C**

Chaining avoids unnecessary intermediate DataFrame materializations and extra Python object overhead, reducing memory churn and kernel launch overhead on the GPU.

**Why the others are wrong:**
- Method chaining has nothing to do with forcing CPU execution.
- While readability improves, there is a measurable performance benefit from reduced intermediate allocations.
- Chaining alone does not invoke Dask; multi-GPU scaling requires dask_cudf explicitly.

</details>

---

**Q6.** Which file format and read method combination provides the BEST GPU-accelerated I/O performance in cuDF for large columnar datasets?

A) cudf.read_parquet on a Parquet file with columnar predicate pushdown
B) cudf.read_csv on a gzip-compressed CSV
C) pandas.read_excel then cudf.from_pandas
D) cudf.read_json on a deeply nested JSON file

<details>
<summary>Answer</summary>

**Correct: A**

Parquet is columnar and cudf.read_parquet leverages GPU-accelerated decoding plus column/predicate pushdown, minimizing I/O and maximizing throughput.

**Why the others are wrong:**
- Gzip-compressed CSV requires CPU-bound decompression before GPU parsing can fully benefit, adding overhead.
- Excel reading is CPU-only in pandas and then requires a costly CPU-to-GPU transfer.
- Nested JSON parsing has more overhead and less GPU optimization than columnar Parquet reads.

</details>

---

**Q7.** A dataset has 200 million rows and requires a groupby with multiple aggregation functions (mean, std, count) on several columns. Which cuDF pattern is MOST performant?

A) df.groupby('key').apply(lambda g: pd.Series({'mean': g['x'].mean(), 'std': g['x'].std()}))
B) Loop over unique keys with df[df['key']==k] and compute aggregates manually
C) df.groupby('key').agg({'x': ['mean', 'std'], 'y': 'count'})
D) Convert to pandas first because cuDF groupby cannot support multiple aggregation functions

<details>
<summary>Answer</summary>

**Correct: C**

cuDF's groupby().agg() with a dict of column-to-function mappings is fully vectorized and GPU-accelerated, matching pandas API parity for multi-aggregation.

**Why the others are wrong:**
- groupby().apply() with a custom lambda forces row/group-wise Python execution, losing GPU vectorization benefits.
- Manual filtering per key is extremely slow due to repeated full-column scans and Python loop overhead.
- cuDF groupby fully supports multiple aggregation functions natively; conversion to pandas is unnecessary.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q8.** A team is deciding whether to use cuDF or pandas for a transformation involving heavy string regex parsing with complex arbitrary Python logic on a 2GB dataset. Which choice is BEST?

A) Use cuDF because GPU acceleration is always faster regardless of operation type
B) Use cuDF with axis=1 apply and Numba JIT for the regex logic to guarantee speedup
C) Use pandas, since arbitrary complex regex/Python logic often lacks full GPU-native support and may fall back or underperform on cuDF
D) Split the data into 2GB chunks and load them all as separate cuDF DataFrames simultaneously

<details>
<summary>Answer</summary>

**Correct: C**

Complex arbitrary Python/regex logic often isn't vectorizable or Numba-JIT-compatible on GPU, so pandas (or cuDF's limited string API subset) can be more practical and reliable for such transformations.

**Why the others are wrong:**
- GPU acceleration only helps for vectorizable, supported operations; arbitrary regex/Python logic frequently is not accelerated.
- Numba JIT does not support arbitrary regex operations, so this would likely fail or fall back.
- Splitting into chunks does not solve the fundamental unsupported-operation issue and adds unnecessary complexity.

</details>

---

