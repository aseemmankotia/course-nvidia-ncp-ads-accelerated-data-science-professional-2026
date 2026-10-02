## Chapter 10 Cheat Sheet: cuML, GPU XGBoost, DALI

### cuML Module Map (mirrors scikit-learn API)
| Module | Key Classes | sklearn Equivalent |
|---|---|---|
| cuml.linear_model | LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet | sklearn.linear_model |
| cuml.ensemble | RandomForestClassifier/Regressor | sklearn.ensemble |
| cuml.cluster | KMeans, DBSCAN, AgglomerativeClustering | sklearn.cluster |
| cuml.decomposition | PCA, TruncatedSVD | sklearn.decomposition |
| cuml.manifold | UMAP, TSNE | sklearn/umap-learn |
| cuml.svm | SVC, SVR | sklearn.svm |

**API parity caveat:** same fit/predict/transform methods, but not every solver/parameter is implemented (e.g., LogisticRegression solver options are limited vs sklearn).

### cuML Output Control
- Default `output_type='input'` — matches input type (cuDF in → cuDF out).
- Force GPU-resident output: `output_type='cudf'` or `'cupy'` per estimator, or globally via `cuml.set_global_output_type('cudf')`.
- Avoids implicit conversion to NumPy (CPU round trip).

### XGBoost GPU Tree Methods
| Setting | Status | Notes |
|---|---|---|
| `tree_method='gpu_hist'` | Deprecated (since XGBoost 2.0) | Still functional in some builds, legacy exam trap |
| `tree_method='hist', device='cuda'` | Current recommended | Unified hist algorithm, GPU-executed |
| `tree_method='exact'` | CPU-oriented | Not GPU histogram-optimized, slow at scale |

### DMatrix from cuDF
```python
import xgboost as xgb
dtrain = xgb.DMatrix(cudf_df, label=cudf_label)      # direct, no host copy
dtrain_q = xgb.QuantileDMatrix(cudf_df, label=cudf_label)  # memory-efficient for hist/GPU
```

### cuML vs scikit-learn Decision Rule
- **>100K–1M rows:** cuML/GPU wins — transfer overhead amortized.
- **<10K rows:** scikit-learn (CPU) often faster — GPU transfer/launch overhead dominates.
- **Boosted trees at scale:** XGBoost (hist/cuda) > cuML RandomForest for accuracy-critical large tabular tasks.
- **Clustering:** KMeans (need k, centroid-based) vs DBSCAN (density-based, auto cluster count).

### Zero-Copy Handoff Patterns
| From | To | Mechanism |
|---|---|---|
| cuDF/cuPy | PyTorch tensor | DLPack (`to_dlpack()`) or `__cuda_array_interface__` |
| cuDF | cuML estimator | Direct — no conversion needed |
| cuDF/cuML output | XGBoost DMatrix | Direct pass-through |
| Deep learning data pipeline | PyTorch/TensorFlow | DALI GPU-accelerated loader/augmentation |

### RAPIDS → Deep Learning Handoff
1. cuDF: ETL/feature engineering on GPU.
2. cuML: classical ML (regression/classification/clustering) OR XGBoost: boosted trees.
3. DALI: GPU-native data loading/augmentation for image/audio/video pipelines feeding PyTorch/TensorFlow.
4. Zero-copy handoff via DLPack/`__cuda_array_interface__` — never round-trip through host/NumPy/pandas.

### Hyperparameter Tuning at Scale
- Multi-GPU: Dask-CUDA + Optuna (GPU-aware parallel trials).
- Avoid: sklearn GridSearchCV/RandomizedSearchCV `n_jobs` — this parallelizes CPU processes, not GPUs.

### Common Exam Traps
1. Choosing `gpu_hist` as "the only correct answer" when `hist`+`device='cuda'` is the current standard.
2. Forgetting cuML output defaults to input type — assuming it always returns NumPy.
3. Using tiny datasets with cuML/GPU XGBoost when CPU scikit-learn would be faster and simpler.
