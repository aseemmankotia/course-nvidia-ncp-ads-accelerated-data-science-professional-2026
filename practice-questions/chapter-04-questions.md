## Chapter 4: Advanced cuDF: Strings, Datetimes, Nulls, and Categoricals — Practice Questions

**Q1.** A cuDF DataFrame has a 'country' column with 500 million rows but only 60 unique values. You need to reduce GPU memory footprint before a groupby-join operation. Which approach BEST achieves this?

A) Convert the column to categorical dtype
B) Keep the column as string dtype and rely on cuDF's string pool
C) Hash the strings into a new int64 column and drop the original
D) Convert to pyarrow dictionary type on the CPU before loading, then never touch it on GPU

<details>
<summary>Answer</summary>

**Correct: A**

categorical dtype stores each unique value once in a small dictionary and replaces the column with integer codes (int8/int16 depending on cardinality), which is exactly the low-cardinality-high-row-count case cuDF is optimized for, and groupby/join operate directly on the codes.

**Why the others are wrong:**
- String columns store variable-length data with offset+char buffers, which is much larger for repeated values and slower to hash/compare in joins.
- Manual hashing loses the ability to recover original labels and adds an extra full-column pass; categorical encoding already does this internally and safely.
- Doing dictionary encoding only in pyarrow before GPU load does not give cuDF a native categorical column with GPU-resident codes; you'd need to convert on the GPU side anyway.

</details>

---

**Q2.** You create `cudf.Series([1.0, np.nan, 3.0])` using default constructor arguments. What is the state of the second element?

A) It stays as a valid floating-point NaN value, and isnull() returns False for it, matching pandas behavior
B) It is automatically converted to a null (missing) entry in the underlying validity bitmask because nan_as_null defaults to True
C) cuDF raises a TypeError because NaN is unsupported in GPU float columns
D) It is silently dropped, shortening the Series to length 2

<details>
<summary>Answer</summary>

**Correct: B**

cudf.Series and cudf.DataFrame constructors have nan_as_null=True by default, so any NaN encountered during construction is converted into an actual null entry in the Arrow-style validity bitmask rather than being kept as a distinct floating value.

**Why the others are wrong:**
- This describes pandas semantics, not cuDF's default; pandas keeps NaN as a real float value while still reporting isna()==True for it.
- NaN is fully supported as a float value in cuDF; the constructor just converts it to null by default rather than rejecting it.
- The Series length is unchanged; only the validity bitmask entry changes, not the row count.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** Which cuDF string method should you use to pull out regex capture groups (e.g., area code and number) from a phone-number string column into separate output columns?

A) Series.str.findall()
B) Series.str.match()
C) Series.str.extract()
D) Series.str.split(expand=True)

<details>
<summary>Answer</summary>

**Correct: C**

.str.extract() applies a regex with capture groups and returns a DataFrame with one column per group, mirroring pandas' extract API and running on GPU via libcudf's regex kernels.

**Why the others are wrong:**
- findall() returns all non-overlapping matches as a list per row, not separated capture-group columns.
- match() only tests whether the pattern matches at the start of the string and returns a boolean-like result, not extracted groups.
- split(expand=True) splits on a delimiter/pattern into columns but does not use capture-group semantics.

</details>

---

**Q4.** A pipeline needs to clean a 2GB string column using a regex with a variable-length lookbehind assertion. Which is the BEST approach?

A) Call Series.str.replace(pattern, repl, regex=True) directly on the cuDF Series
B) Convert the column to pandas with .to_pandas(), apply the regex with Python's re module, then convert the result back with cudf.from_pandas()
C) Write a custom Numba CUDA kernel that reimplements the lookbehind logic
D) Chain multiple simpler cudf .str.replace() calls to approximate the lookbehind behavior

<details>
<summary>Answer</summary>

**Correct: B**

cuDF's regex engine is a GPU-optimized subset of regex and does not support variable-length lookbehind assertions; for unsupported regex features, the documented and practical approach is to fall back to CPU pandas/re for that specific operation.

**Why the others are wrong:**
- cuDF's regex engine will either error or silently mis-match on variable-length lookbehind since it's not part of the supported feature set.
- Writing a raw CUDA regex kernel is disproportionate engineering effort compared to a simple CPU fallback for one operation on manageable data size.
- Chaining simpler patterns cannot faithfully reproduce lookbehind semantics and risks silent correctness bugs.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q5.** You convert a cudf.Series with 300 distinct string values to CategoricalDtype. What integer type does cuDF use to store the category codes?

A) int8, since categorical codes always default to the smallest integer type
B) int16, because 300 categories exceed int8's range (max 127) but fit within int16
C) int32, because cuDF never uses types smaller than int32 for codes
D) uint64, to guarantee compatibility with joins on other tables

<details>
<summary>Answer</summary>

**Correct: B**

cuDF (following Arrow dictionary encoding rules) chooses the smallest signed integer type that can represent all category codes: int8 up to 127 categories, int16 up to 32767, int32 beyond that. 300 categories require int16.

**Why the others are wrong:**
- int8 can only represent up to 127 distinct signed values, which is insufficient for 300 categories.
- int32 would be used only once cardinality exceeds int16's range (32767); using it here wastes memory.
- uint64 is never the default choice for categorical codes; it would defeat the memory-savings purpose of categoricals.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q6.** You have a cudf DataFrame with a datetime64 index sampled irregularly throughout each day, and you want the daily mean of a numeric column computed on GPU. Which call is correct?

A) df.resample('D').mean()
B) df.groupby(df.index.date).mean()
C) df.rolling('D').mean()
D) df.resample('D').apply(lambda x: x.mean())

<details>
<summary>Answer</summary>

**Correct: A**

cuDF supports pandas-style resample() on a DatetimeIndex with standard offset aliases like 'D' for daily buckets, and .mean() is a supported GPU-accelerated aggregation on the resampled object.

**Why the others are wrong:**
- df.index.date is not a cuDF-native accessor pattern for grouping and adds unnecessary CPU-side conversion overhead.
- rolling('D') computes a moving window over time, not fixed non-overlapping daily buckets, which changes the semantics of the aggregation.
- resample('D').apply() with a Python lambda forces row-at-a-time execution and is far slower than the built-in vectorized .mean(), and is not the preferred GPU path.

</details>

---

**Q7.** Which statement correctly differentiates cuDF's null representation for string columns from pandas' typical handling of missing string data?

A) cuDF string columns carry a native Arrow-style validity bitmask, so missing values are true nulls with no sentinel value needed
B) cuDF string columns use a sentinel empty string '' internally to represent missing values, just like older pandas versions
C) cuDF strings cannot represent missing values at all; every string cell must contain data
D) cuDF stores missing strings as NaN floats internally, identical to pandas object columns

<details>
<summary>Answer</summary>

**Correct: A**

cuDF's columnar layout follows Apache Arrow conventions, where every column (including strings) has an associated validity bitmask marking nulls independently of the data buffer, so missing string values are represented natively without needing a sentinel string or float NaN.

**Why the others are wrong:**
- Using '' as a sentinel would make legitimate empty strings indistinguishable from missing data, which Arrow/cuDF explicitly avoids via the bitmask.
- cuDF fully supports null strings; this is a core feature exercised heavily by isnull/dropna/fillna operations on string columns.
- Strings cannot be represented as float NaN; that pattern only applies to pandas object columns using Python None/np.nan, not to cuDF's Arrow-backed buffers.

</details>

---

**Q8.** A cudf string column contains values '123', '456', and 'abc'. You need a numeric column where unparseable entries become null instead of raising an error. Which call accomplishes this?

A) series.astype('int32')
B) series.str.to_numeric()
C) series.astype('int32', errors='ignore')
D) cudf.to_numeric(series, errors='coerce')

<details>
<summary>Answer</summary>

**Correct: D**

cudf.to_numeric() mirrors pandas' function and supports errors='coerce', which converts unparseable strings like 'abc' into nulls instead of raising, while valid strings are converted to their numeric type.

**Why the others are wrong:**
- astype('int32') will raise an error (or produce undefined results) on the non-numeric 'abc' entry with no coercion option.
- There is no .str.to_numeric() accessor method in cuDF; numeric conversion is done via cudf.to_numeric() or astype(), not the string accessor.
- astype() does not accept an errors='ignore' parameter in cuDF the way to_numeric() accepts errors='coerce'; this call would raise a TypeError.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

