## Chapter 8: EDA, Time-Series Analysis, and Anomaly Detection at GPU Scale — Flashcards

| # | Front (Question) | Back (Answer) |
|---|-----------------|---------------|
| 1 | cuDF corr() supported correlation method(s)? | Only Pearson. method='spearman' or 'kendall' raises NotImplementedError, unlike pandas. Exam tip: cuDF corr() = Pearson-only. |
| 2 | Default percentiles in cuDF/pandas describe()? | 25%, 50%, 75% (plus count, mean, std, min, max). Exam tip: override via percentiles=[...] param. |
| 3 | IQR outlier bounds formula? | Lower = Q1 - 1.5*IQR, Upper = Q3 + 1.5*IQR, where IQR = Q3 - Q1. Exam tip: robust to skew, preferred over z-score for non-normal data. |
| 4 | Common z-score anomaly threshold on exam scenarios? | \|z\| > 3 flags an anomaly (~99.7% of normal data within ±3σ). Exam tip: only valid for roughly Gaussian data. |
| 5 | cuDF resample() prerequisite? | Requires a DatetimeIndex on the DataFrame/Series. Exam tip: set_index() on the timestamp column first, or it errors. |
| 6 | rolling(window=N, min_periods=M).mean() when fewer than M values available? | Returns NaN for that row — matches pandas, never drops the row. Exam tip: NaN, not removal. |
| 7 | cuML DBSCAN — how are anomalies labeled? | Points not assigned to any dense cluster get label -1. Exam tip: treat -1-labeled rows as multivariate outliers. |
| 8 | Dataset exceeds single GPU VRAM for EDA — what's the scale-out solution? | Dask-cuDF partitions the DataFrame across multiple GPUs/nodes while keeping cuDF-style API. Exam tip: don't pick 'downcast precision' as the answer. |
| 9 | z-score vs IQR — when to prefer which? | Z-score: roughly normal/Gaussian single feature. IQR: skewed or unknown distribution — more robust to non-normality and existing outliers. |
| 10 | Lag feature creation on a cuDF Series? | df['col'].shift(1) creates a 1-step lag feature. Exam tip: pair with rolling().mean() for lag + rolling-mean feature engineering. |
| 11 | Univariate stats vs DBSCAN for anomaly detection — key exam distinction? | z-score/IQR check one column at a time and miss multivariate outliers; DBSCAN evaluates density across multiple features and catches outliers invisible on any single axis. |
| 12 | Why is cuDF corr()/describe() faster than pandas at scale? | GPU parallelizes the underlying sum/mean/variance/product reductions across thousands of cores vs serial CPU execution — no laziness or sampling involved. |
