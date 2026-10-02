# Chapter 4 Cheat Sheet — Strings, Datetimes, Nulls, Categoricals

## .str Accessor — Common Methods
| Method | Purpose | Notes |
|---|---|---|
| `.str.contains(pat, regex=True)` | boolean substring/regex match | set `regex=False` for literal, faster |
| `.str.extract(pat)` | regex capture groups → DataFrame | one column per group |
| `.str.replace(pat, repl, regex=True)` | pattern replace | falls back to CPU needed for unsupported regex |
| `.str.split(pat, expand=True)` | split into columns | delimiter or regex |
| `.str.lower()/.upper()/.strip()` | case/whitespace ops | fully GPU-vectorized, cheap |
| `.str.len()` | string length per row | int32 result |

## Regex Support Limits
- cuDF regex engine = GPU-optimized **subset** of full PCRE.
- **Not supported:** variable-length lookbehind, some backreference patterns.
- **Supported:** most anchors, character classes, quantifiers, named/unnamed capture groups.
- **Fallback pattern:** `.to_pandas()` → `re`/pandas ops → `cudf.from_pandas()` for unsupported patterns on manageable data sizes.

## Null Handling: cuDF vs pandas
| Aspect | pandas | cuDF |
|---|---|---|
| Float NaN | real float value, `isna()`→True | constructor default `nan_as_null=True` converts NaN → true null in bitmask |
| Missing repr | sentinel NaN/None (object dtype) | Arrow-style validity **bitmask** on every column type, incl. strings |
| String nulls | `None`/NaN in object array | native null bit, no sentinel string |
| Key methods | `isna()`, `fillna()`, `dropna()` | `isnull()`, `fillna()`, `dropna()` — same names, GPU-executed |

## Categorical Dtype — Code Size Rule
| # Unique Categories | Code Storage Type |
|---|---|
| ≤ 127 | int8 |
| ≤ 32,767 | int16 |
| > 32,767 | int32 |

Saves memory: dictionary stores unique values once; column stores small integer codes referencing dictionary. Best for **low-cardinality, high-row-count** columns (e.g., country, category, status).

## Datetime
- `df.resample('D').mean()` — standard offset aliases (`'D'`, `'H'`, `'M'`, etc.) supported on GPU.
- Fewer custom offset/holiday-calendar features than pandas — verify support before relying on exotic aliases.
- `.rolling(window)` = moving window ≠ `.resample()` = fixed bucket aggregation.

## String ↔ Numeric Conversion
- `cudf.to_numeric(series, errors='coerce')` → invalid parses become null.
- `series.astype('int32')` → raises/undefined on bad strings, no coerce option.

## When to Stay on CPU
- Regex needing unsupported features (lookbehind, complex backreferences).
- Heavy per-row Python UDFs / NLP-style token processing.
- Small datasets where GPU kernel-launch overhead > compute savings.

## Exam Callouts
- **Heavily tested:** `.str` accessor methods, nan_as_null default behavior, categorical code-size rule.
- **Common trap:** assuming cuDF NaN behaves identically to pandas NaN (it's converted to null by default).
- **Common trap:** assuming cuDF regex is full PCRE — it silently fails or errors on advanced constructs.
