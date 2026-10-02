## Chapter 10: cuML, GPU-Accelerated XGBoost, and Rapid Hyperparameter Tuning — Practice Questions

**Q1.** A data science team has 60 million rows of tabular data on a single A100 GPU and needs a gradient-boosted tree model with maximum accuracy in minimum training time. Which configuration BEST meets this requirement?

A) cuml.ensemble.RandomForestClassifier with default parameters
B) XGBoost with tree_method='exact'
C) XGBoost with tree_method='hist' and device='cuda'
D) scikit-learn's GradientBoostingClassifier with n_jobs=-1

<details>
<summary>Answer</summary>

**Correct: C**

For large tabular datasets, GPU-accelerated boosted trees via tree_method='hist' with device='cuda' (the current XGBoost 2.x GPU histogram method) give the best accuracy/speed tradeoff and scale to tens of millions of rows on GPU memory.

**Why the others are wrong:**
- Random forest is a different ensemble strategy (bagging, not boosting) and typically underperforms boosted trees on accuracy for this class of problem.
- tree_method='exact' builds exact split points on every iteration and does not benefit from GPU histogram binning — it becomes prohibitively slow at this scale.
- n_jobs=-1 only parallelizes across CPU cores; scikit-learn's GradientBoostingClassifier has no GPU support at all.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** You need to cluster 5 million GPU-resident points into exactly 20 groups as fast as possible using RAPIDS. Which class should you use?

A) cuml.cluster.DBSCAN
B) sklearn.cluster.KMeans
C) cuml.cluster.KMeans
D) cuml.cluster.AgglomerativeClustering

<details>
<summary>Answer</summary>

**Correct: C**

cuml.cluster.KMeans directly accepts a fixed k, runs entirely on GPU, and is the RAPIDS-native counterpart for centroid-based clustering at this scale.

**Why the others are wrong:**
- DBSCAN is density-based and does not take a target number of clusters — it discovers clusters based on density parameters (eps, min_samples).
- scikit-learn's KMeans runs on CPU only, forcing a slow GPU-to-host transfer and losing the acceleration benefit.
- AgglomerativeClustering is hierarchical and has much higher memory/compute cost, making it impractical at 5 million points even on GPU.

</details>

---

**Q3.** Which statement accurately describes cuML's API relationship to scikit-learn?

A) cuML requires entirely different method names (e.g., 'train' instead of 'fit') because it targets GPU execution
B) cuML estimators expose the same fit/predict/transform interface as scikit-learn but accept and return GPU-resident data (cuDF/cuPy) by default
C) cuML is a wrapper that transparently sends scikit-learn calls to the GPU without any API changes
D) cuML only supports unsupervised learning; supervised estimators must still use scikit-learn

<details>
<summary>Answer</summary>

**Correct: B**

cuML deliberately mirrors the scikit-learn estimator API (fit/predict/transform, similar parameter names) for near drop-in compatibility, while operating on cuDF DataFrames or cuPy arrays and returning GPU-resident results controlled by the output_type setting.

**Why the others are wrong:**
- cuML intentionally keeps the same method names as scikit-learn — this is the opposite of what the exam tests.
- cuML is a separate library with its own GPU-native implementations, not a transparent scikit-learn wrapper.
- cuML supports both supervised (linear_model, ensemble, svm) and unsupervised (cluster, decomposition, manifold) algorithms.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q4.** You have preprocessed a 30 GB dataset entirely in cuDF and now need to train an XGBoost model on GPU without incurring a CPU round trip. What should you do?

A) Convert the cuDF DataFrame to pandas, then to NumPy, then build the DMatrix
B) Pass the cuDF DataFrame directly into xgb.DMatrix() (or xgb.QuantileDMatrix()) along with the cuDF label column
C) Write the cuDF DataFrame to CSV and reload it with pandas before building the DMatrix
D) Convert the cuDF DataFrame to a Python list of dictionaries before building the DMatrix

<details>
<summary>Answer</summary>

**Correct: B**

XGBoost's DMatrix (and the memory-efficient QuantileDMatrix used with the hist/gpu tree method) accepts cuDF DataFrames and cuDF label Series directly, keeping data on GPU and eliminating a host round trip.

**Why the others are wrong:**
- Converting through pandas/NumPy forces the entire dataset off the GPU onto host memory, defeating the purpose of the RAPIDS pipeline.
- Writing to CSV and reloading with pandas is even slower and reintroduces disk I/O plus a CPU round trip.
- Converting to a list of dictionaries is neither GPU-resident nor an efficient input format for DMatrix.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q5.** A team needs to run hundreds of hyperparameter trials for a cuML/XGBoost model across 4 GPUs on one node as fast as possible. Which approach is BEST?

A) scikit-learn's GridSearchCV with n_jobs=-1
B) A single-GPU sequential Python for-loop over hyperparameter combinations
C) RandomizedSearchCV with cv=10 and n_jobs=-1
D) Dask-CUDA to distribute trials across the 4 GPUs, coordinated with a GPU-aware tuning framework such as Optuna

<details>
<summary>Answer</summary>

**Correct: D**

Dask-CUDA lets multiple GPUs run trials concurrently, and pairing it with a tuning framework like Optuna enables parallel, GPU-aware hyperparameter search — the recommended RAPIDS pattern for rapid multi-GPU tuning.

**Why the others are wrong:**
- GridSearchCV's n_jobs parallelizes across CPU processes, not GPUs, and does not use RAPIDS acceleration.
- A sequential loop wastes 3 of the 4 available GPUs and is the slowest option listed.
- RandomizedSearchCV's n_jobs is also CPU-process based; it does not dispatch cuML/XGBoost work across multiple GPUs natively.

</details>

---

**Q6.** What is the primary role of NVIDIA DALI in a RAPIDS-to-deep-learning pipeline?

A) It performs GPU-accelerated hyperparameter search for cuML estimators
B) It replaces cuDF as the tabular data manipulation library
C) It trains gradient-boosted tree models on GPU instead of XGBoost
D) It provides GPU-accelerated data loading and augmentation that feeds batches directly into PyTorch or TensorFlow training loops

<details>
<summary>Answer</summary>

**Correct: D**

DALI (Data Loading Library) builds GPU-resident data pipelines — decoding, augmenting, and batching data — and hands batches directly to deep learning frameworks via DALI iterators, removing CPU-bound dataloader bottlenecks.

**Why the others are wrong:**
- Hyperparameter search is handled by tools like Optuna or Dask-ML, not DALI.
- cuDF remains the tabular/GPU DataFrame library; DALI targets image/audio/video pipelines for deep learning, not tabular manipulation.
- DALI is not a model training library — XGBoost (or a deep learning framework) still performs the actual training.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** After computing embeddings with cuML on GPU, you need to feed them into a PyTorch model for fine-tuning with zero CPU copy. Which mechanism enables this handoff?

A) Use DLPack or the __cuda_array_interface__ protocol to convert the cuPy/cuDF output directly into a PyTorch GPU tensor
B) Serialize the cuML output to a pickle file and load it in PyTorch
C) Convert cuML output to a pandas DataFrame, then to a PyTorch tensor
D) Write the embeddings to a CSV file and load with torch.from_numpy()

<details>
<summary>Answer</summary>

**Correct: A**

cuML outputs (as cuPy arrays or cuDF objects) expose the __cuda_array_interface__ and support DLPack export, allowing zero-copy conversion to PyTorch tensors while data remains on the GPU — the standard RAPIDS-to-deep-learning handoff point.

**Why the others are wrong:**
- Pickling forces a serialize/deserialize round trip through host memory, adding CPU overhead and latency.
- Converting through pandas moves data to host memory, which is exactly the CPU round trip the exam expects you to avoid.
- torch.from_numpy() only works on host (CPU) NumPy arrays, requiring a GPU-to-host transfer first.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q8.** A dataset has only 8,000 rows and 15 features. Which approach is MOST appropriate for training a linear regression model?

A) cuml.linear_model.LinearRegression, because GPU acceleration always outperforms CPU regardless of dataset size
B) XGBoost with tree_method='hist' and device='cuda'
C) cuml.ensemble.RandomForestRegressor with 1000 trees
D) scikit-learn's LinearRegression, because the dataset is too small for GPU transfer overhead to pay off

<details>
<summary>Answer</summary>

**Correct: D**

At only 8,000 rows, the cost of transferring data to GPU memory and launching CUDA kernels typically exceeds the compute savings; scikit-learn's CPU implementation is faster for small datasets. cuML's benefit grows with scale (typically >100K–1M rows).

**Why the others are wrong:**
- GPU acceleration is not universally faster — small datasets suffer from fixed transfer/launch overhead that dominates total time.
- This is the correct choice, not a wrong option — see correct_index.
- XGBoost is a boosted-tree method, not a linear regression solution, and is unnecessary overkill for a simple 15-feature linear problem.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

