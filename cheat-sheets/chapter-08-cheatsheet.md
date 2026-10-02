# Chapter 8 Cheat Sheet: GPU-Scale EDA, Time-Series, Anomaly Detection

## Summary Stats & Correlation (cuDF)
| Op | Notes |
|---|---|
| `df.describe()` | Default percentiles: 25%, 50%, 75% + count/mean/std/min/max. Override with `percentiles=[...]` |
| `df.corr()` | **Pearson only** on GPU. `method='spearman'`/`'kendall'` → `NotImplementedError` (pandas supports all three) |
| `df.value_counts()`, `df.unique()` | Vectorized GPU reductions, big speedup vs pandas at high row counts |

## Time-Series Ops (cuDF)
| Op | Requirement |
|---|---|
| `df.resample(rule)` | DataFrame/Series **must have a DatetimeIndex** |
| `df.rolling(window, min_periods)` | `window` = int size; rows with fewer than `min_periods` obs → `NaN`, never dropped |
| `series.shift(n)` | Lag feature; combine with `.rolling(k).mean()` for lag+rolling-mean features |

## Anomaly Detection: Choosing the Right Approach
| Data shape | Best approach | Why |
|---|---|---|
| Single feature, ~Gaussian | Z-score (\|z\|>3), cuDF vectorized mean/std | O(n), fastest, no tuning |
| Single feature, skewed/unknown dist | IQR (Q1-1.5×IQR / Q3+1.5×IQR) | Robust to non-normality |
| Multivariate, non-linear relationships | cuML **DBSCAN** (label `-1` = outlier) | Density-based, catches combo-outliers invisible per-column |
| High-dim correlated features | PCA reconstruction error | Needs ≥2 correlated features to be meaningful |
| Data > single GPU VRAM | Dask-cuDF (multi-GPU partitioning) | Scale-out, keeps cuDF-like API |

## Key Numbers
- Z-score anomaly threshold: **|z| > 3** (~99.7% within ±3σ for normal data)
- IQR bounds: **Q1 − 1.5×IQR** to **Q3 + 1.5×IQR**
- describe() default percentiles: **25/50/75**
- DBSCAN noise/outlier label: **-1**

## Common Exam Traps
1. Assuming `cuDF.corr()` supports Spearman/Kendall like pandas — **it doesn't** (Pearson only).
2. Picking a clustering algorithm (DBSCAN/PCA) for simple **univariate** Gaussian anomaly detection when vectorized z-score is faster and sufficient.
3. Forgetting `resample()` requires a **DatetimeIndex** — will error on a plain integer/RangeIndex.
4. Choosing "downcast to float16" or "switch to pandas" instead of **Dask-cuDF** when data exceeds single-GPU memory.
5. Confusing `rolling(window=86400)` (row-count window) with `resample('D')` (calendar-day bucketing) — different output shapes entirely.
6. Forgetting that `min_periods` produces `NaN`, not dropped rows, at the start of a rolling series.

## Quick Decision Rule
Single column + normal → **z-score**. Single column + skewed → **IQR**. Multi-column + non-linear → **DBSCAN**. Data doesn't fit one GPU → **Dask-cuDF**.
