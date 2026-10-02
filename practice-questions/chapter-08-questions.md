## Chapter 8: EDA, Time-Series Analysis, and Anomaly Detection at GPU Scale — Practice Questions

**Q1.** A pipeline ingests 800 million single-sensor temperature readings per day into a cuDF DataFrame. The distribution is approximately Gaussian. Which anomaly detection approach BEST balances speed and accuracy at this scale?

A) cuML DBSCAN clustering on the raw values
B) Z-score computed via cuDF vectorized mean/std with threshold |z|>3
C) cuML Isolation Forest run per batch
D) PCA reconstruction error on the single feature

<details>
<summary>Answer</summary>

**Correct: B**

Z-score on a single near-Gaussian feature is a fully vectorized O(n) GPU computation using mean/std — no distance matrix or iterative clustering required, and it's statistically appropriate for normal 1-D data.

**Why the others are wrong:**
- DBSCAN needs pairwise distance computation and eps/min_samples tuning — unnecessary overhead for 1-D Gaussian data and scales worse than vectorized stats.
- cuML does not ship a native Isolation Forest implementation; it's not an available option and adds needless ensemble overhead here.
- PCA reconstruction error needs multiple correlated features to be meaningful; a single feature has no dimensionality to reduce.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** Which single method call on a cuDF DataFrame raises an error if you request anything other than the default correlation method?

A) df.describe(method='spearman')
B) df.corr(method='spearman')
C) df.rolling(method='spearman')
D) df.resample(method='spearman')

<details>
<summary>Answer</summary>

**Correct: B**

cuDF's corr() only implements Pearson correlation on GPU; passing method='spearman' or 'kendall' raises NotImplementedError, unlike pandas which supports all three methods.

**Why the others are wrong:**
- describe() has no method parameter at all — it takes percentiles, not correlation method.
- rolling() takes window/min_periods, not a correlation method parameter.
- resample() takes a frequency/offset rule string, not a correlation method parameter.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** You call sensor_df.rolling(window=10, min_periods=3).mean() on a cuDF Series. What happens for the first 2 rows?

A) They are dropped entirely from the output
B) They are backfilled with the Series mean
C) They raise a ValueError immediately
D) They return NaN because fewer than min_periods values are available

<details>
<summary>Answer</summary>

**Correct: D**

min_periods=3 requires at least 3 observations in the window; rows 0 and 1 only have 1 and 2 values respectively, so cuDF returns NaN there, matching pandas rolling semantics.

**Why the others are wrong:**
- Rolling operations preserve the original row count/index; they never drop rows.
- cuDF never auto-backfills rolling results; that requires an explicit fillna() call.
- Insufficient window data is a normal, expected NaN outcome, not an error condition.

</details>

---

**Q4.** A retailer wants daily average revenue from second-level POS timestamps stored in a cuDF DataFrame with a DatetimeIndex. Which single operation gets this fastest on GPU?

A) df.groupby(df.index.date).mean()
B) df.resample('D').mean()
C) df.rolling(window=86400).mean()
D) Convert to pandas, then use pandas resample('D')

<details>
<summary>Answer</summary>

**Correct: B**

resample('D') is purpose-built for fixed-frequency time bucketing on a DatetimeIndex and runs natively on GPU, producing one aggregated row per calendar day.

**Why the others are wrong:**
- Works, but requires deriving a date column first and is less efficient than the native resample path.
- rolling(window=86400) gives a moving 24-hour window ending at every second-level row, not one row per day — wrong output shape.
- Moving data off the GPU to pandas adds a costly transfer and defeats the purpose of GPU acceleration.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q5.** A dataset has 200 numeric features with complex non-linear multivariate outliers that don't appear when inspecting any single column with z-score. Which cuML-based approach is BEST suited to flag these records?

A) cuML DBSCAN with points labeled -1 treated as anomalies
B) IQR bounds computed per column independently
C) cudf.DataFrame.describe() percentile inspection
D) Z-score with threshold 3 on each column separately

<details>
<summary>Answer</summary>

**Correct: A**

DBSCAN is density-based and multivariate — it naturally flags points that don't belong to any dense cluster (label -1) even when no single column looks anomalous, matching the described multivariate case.

**Why the others are wrong:**
- IQR per column is univariate and misses anomalies that are only unusual in combination across features.
- describe() is a descriptive summary tool, not a detection method — it flags no specific records.
- Per-column z-score has the same univariate blind spot — a point can look normal on every axis yet be a multivariate outlier.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q6.** Using cuDF's describe() on a numeric Series, what are the default percentile values returned, matching pandas behavior?

A) 25%, 50%, 75%
B) 10%, 50%, 90%
C) 5%, 50%, 95%
D) 33%, 66%, 100%

<details>
<summary>Answer</summary>

**Correct: A**

cuDF mirrors pandas' default describe() percentiles of the 25th, 50th (median), and 75th percentile alongside count, mean, std, min, max.

**Why the others are wrong:**
- 10/50/90 is not the default; it requires explicit percentiles=[0.1,0.5,0.9].
- 5/50/95 is also non-default and requires explicit percentiles=[0.05,0.5,0.95].
- 33/66/100 is not a valid default, and 100% would just duplicate max.

</details>

---

**Q7.** A team's raw log dataset grows to 400GB, exceeding the memory of any single GPU in their cluster, but they still want cuDF-style dataframe operations for EDA. What should they use?

A) Increase min_periods on all rolling windows to reduce memory
B) Switch every operation back to pandas on CPU
C) Dask-cuDF to distribute the dataframe across multiple GPUs
D) Reduce precision of all columns to float16 and hope it fits

<details>
<summary>Answer</summary>

**Correct: C**

Dask-cuDF partitions the dataframe across multiple GPUs/nodes while preserving the cuDF-like API, which is the standard RAPIDS answer when data exceeds single-GPU VRAM.

**Why the others are wrong:**
- min_periods only affects rolling window NaN behavior, not overall dataset memory footprint.
- Abandoning GPU acceleration entirely is dramatically slower at this scale and not the intended solution.
- Downcasting precision may help somewhat but isn't guaranteed to fit and isn't the robust scale-out answer expected here.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q8.** What is the primary reason cuDF's corr() computes correlations faster than pandas at large row counts?

A) It computes correlations lazily and never materializes results
B) It automatically samples only 10% of rows before computing correlation
C) It parallelizes the pairwise sum-of-products computations across thousands of GPU cores
D) It caches previous correlation results and reuses them

<details>
<summary>Answer</summary>

**Correct: C**

Correlation is fundamentally element-wise sums and products across columns, which parallelizes extremely well across thousands of GPU cores compared to serial CPU execution in pandas.

**Why the others are wrong:**
- cuDF corr() returns a materialized DataFrame of coefficients, not a lazy object.
- cuDF does not silently subsample data for corr(); it computes over the full dataset unless the user explicitly samples.
- There is no built-in result caching for corr() — each call recomputes from current data.

</details>

---

