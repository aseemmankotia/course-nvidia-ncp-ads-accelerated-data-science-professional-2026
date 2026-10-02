## Chapter 5: cuDF Interoperability, Synthetic Data, and Pipeline Efficiency — Practice Questions

**Q1.** A cuDF pipeline needs one string-cleaning method that exists only in pandas, not in cuDF. The rest of the pipeline runs entirely on GPU over 20M rows. What is the BEST approach?

A) Call df.to_pandas() on the entire DataFrame, run the string method, then call cudf.from_pandas() to continue
B) Rewrite the whole pipeline in pandas since one unsupported operation exists
C) Skip the string-cleaning step entirely to avoid the CPU round trip
D) Extract only the needed column with to_pandas(), apply the string method, then reassign it back into the cuDF DataFrame

<details>
<summary>Answer</summary>

**Correct: D**

Converting only the single column minimizes the size of the device-to-host and host-to-device transfer, isolating the pandas fallback to the smallest possible data slice while keeping the rest of the pipeline GPU-accelerated.

**Why the others are wrong:**
- Converting the entire 20M-row DataFrame forces an unnecessary full copy of all columns, not just the one that needs the fallback
- Rewriting the entire pipeline in pandas discards GPU acceleration for every other step, which is disproportionate to one missing API
- Skipping the required transformation changes pipeline correctness/output, which is not an acceptable trade-off

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** A team wants to benchmark a cuDF groupby-aggregation pipeline at multiple data volumes before production data is available. Which approach BEST generates the test data?

A) Use cudf.datasets.timeseries() to create GPU-resident synthetic data matching the pipeline's schema and scale
B) Generate the data with pandas, then convert with cudf.from_pandas() before each benchmark run
C) Manually type sample rows into a Python list and build a small DataFrame
D) Request a sample export of real customer data from production

<details>
<summary>Answer</summary>

**Correct: A**

cudf.datasets.timeseries() generates synthetic, schema-configurable data directly on the GPU, avoiding CPU generation and host-to-device transfer overhead, and can scale to realistic row counts for benchmarking.

**Why the others are wrong:**
- Generating in pandas first and converting adds unnecessary host-to-device transfer cost that isn't needed for a GPU-native workflow
- A handful of manually typed rows cannot represent realistic scale or distribution for benchmarking
- Real production data may be unavailable, sensitive, or not yet collected at the volumes needed for scale testing

</details>

---

**Q3.** A cuDF Series of floats needs to be passed into a cupy.linalg routine for custom GPU math. Which method avoids any transfer off the GPU?

A) series.to_pandas().values
B) series.to_arrow().to_numpy()
C) np.array(series.to_pandas())
D) series.values (or series.to_cupy())

<details>
<summary>Answer</summary>

**Correct: D**

cudf and cupy both support the __cuda_array_interface__ protocol, so .values / .to_cupy() returns a cupy array pointing at the same GPU memory without any host copy.

**Why the others are wrong:**
- to_pandas().values forces a full device-to-host copy before producing a NumPy array
- to_arrow().to_numpy() converts to a host Arrow buffer and then to NumPy, also leaving the GPU
- np.array(series.to_pandas()) requires the same costly to_pandas() host transfer as the first option

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q4.** A cuDF pipeline reads a 50-column CSV but only 4 columns are needed for a downstream groupby aggregation. Which change reduces memory and copy overhead the MOST?

A) Read all 50 columns, then call df[keep_cols] right before the groupby
B) Pass usecols=[...] (or equivalent column selection) to read_csv() so only the 4 needed columns are ever materialized
C) Read all 50 columns, run all transforms, then drop() unused columns after the aggregation
D) Convert to pandas first, select columns with pandas, then convert back to cuDF

<details>
<summary>Answer</summary>

**Correct: B**

Pruning columns at read time means the unused 46 columns are never allocated in GPU memory at all, minimizing both memory footprint and the volume of data moved through every subsequent pipeline stage.

**Why the others are wrong:**
- Selecting right before the groupby still allocates and briefly holds all 50 columns in GPU memory after the read
- Dropping after all transforms means every intermediate step processes and materializes unused columns unnecessarily
- Converting to pandas first adds an unneeded device-to-host and host-to-device round trip just to select columns

</details>

---

**Q5.** What does 'zero-copy interoperability via Apache Arrow' mean in the context of RAPIDS libraries such as cuDF, cuML, and cuGraph?

A) Data is compressed using Arrow before being passed between libraries to save GPU memory
B) Arrow automatically converts all GPU data to CPU format for compatibility with pandas
C) Arrow encrypts data in transit between RAPIDS libraries for security
D) Libraries share the same Arrow-compatible GPU memory layout, so data can be passed between them without duplicating or serializing it

<details>
<summary>Answer</summary>

**Correct: D**

Because multiple RAPIDS libraries adopt an Arrow-compatible columnar memory format on the GPU, a buffer produced by one library can be consumed by another directly, avoiding the copy and serialization cost that a CPU round trip like to_pandas() would incur.

**Why the others are wrong:**
- Arrow's role here is memory layout compatibility, not compression
- Zero-copy interop specifically avoids CPU conversion, the opposite of what this option describes
- Encryption is unrelated to Arrow's role in GPU dataframe interoperability

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q6.** An analyst is iterating on a large cuDF pipeline and calls df.to_pandas() inside every loop iteration just to visually inspect results. What is the exam-relevant efficiency concern?

A) Each call triggers a full device-to-host copy of the entire DataFrame, so it should be limited to small samples like df.head().to_pandas() rather than the full result each iteration
B) There is no concern; to_pandas() results are cached automatically after the first call
C) to_pandas() should be called once at the very start of the pipeline instead, converting everything to pandas up front
D) to_pandas() only copies metadata, not actual data, so repeated calls are free

<details>
<summary>Answer</summary>

**Correct: A**

to_pandas() performs a real device-to-host transfer and copy of all requested data every time it is called; repeating this on a large DataFrame inside a loop is expensive and unnecessary when only a quick visual check is needed.

**Why the others are wrong:**
- cuDF does not cache to_pandas() output automatically between calls
- Converting everything to pandas up front eliminates GPU acceleration for the entire pipeline, defeating its purpose
- to_pandas() copies actual column data, not just metadata, so it is not free

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** Which pattern best minimizes intermediate materialization and peak GPU memory usage in a cuDF pipeline?

A) Assign the result of every transformation step to a separate named variable so it can be inspected later
B) Call df.copy() before each transformation to preserve the original DataFrame
C) Convert the DataFrame to pandas between each step to simplify debugging
D) Chain operations in a single expression (e.g., gdf.query(...).groupby(...).agg(...)) so intermediate results are not kept alive as separate references

<details>
<summary>Answer</summary>

**Correct: D**

Chaining lets each intermediate DataFrame become eligible for garbage collection as soon as it's consumed by the next operation, reducing peak GPU memory pressure compared to holding multiple named intermediates alive simultaneously.

**Why the others are wrong:**
- Naming every intermediate step keeps references alive longer than necessary, increasing peak memory use, especially on large datasets
- Calling copy() before each step deliberately duplicates memory and is the opposite of an efficient pattern
- Converting to pandas between steps adds repeated host transfer overhead and removes GPU acceleration mid-pipeline

</details>

---

**Q8.** In which scenario is falling back to pandas inside an otherwise GPU-resident cuDF pipeline generally considered acceptable?

A) When cuDF has no equivalent API for the required operation and the data slice needing that operation is small
B) Whenever the developer is more comfortable writing pandas syntax than cuDF syntax
C) Whenever the DataFrame has more than 1 million rows, to reduce GPU load
D) Only during unit testing, never in a production pipeline regardless of data size

<details>
<summary>Answer</summary>

**Correct: A**

A pandas fallback is justified specifically when cuDF lacks the required functionality and the round-trip transfer cost is small because only a limited slice of data needs the operation — this keeps the fallback's overhead proportional to its benefit.

**Why the others are wrong:**
- Developer preference is not a technical justification and ignores the performance cost of the round trip
- Row count alone doesn't determine GPU load appropriateness; larger data generally makes a pandas fallback more costly, not more justified
- Production pipelines can legitimately use scoped pandas fallbacks for unsupported operations; it is not exclusive to testing

</details>

---

