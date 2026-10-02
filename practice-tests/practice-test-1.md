# Practice Test 1: NCP-ADS: NVIDIA Accelerated Data Science Prep

- **Questions:** 45
- **Time limit:** 90 minutes
- **Passing score:** 70%%

> Answers are in `answer-key-1.md`. Track results in `score-tracker.md`.

---

**Q1.** A data scientist needs to run a single quick aggregation on a 5,000-row CSV file, one time only, as part of an ad-hoc exploratory analysis. Which approach is BEST?

A) Load the CSV into cuDF and run the aggregation on GPU
B) Convert the file to Dask-cuDF and distribute it across multiple GPUs
C) Load the CSV into pandas and run the aggregation on CPU
D) Load the CSV into cuML and run the aggregation as a GPU-accelerated model

---

**Q2.** Which RAPIDS library provides GPU-accelerated graph analytics algorithms such as PageRank and breadth-first search (BFS)?

A) cuML
B) cuDF
C) cuSpatial
D) cuGraph

---

**Q3.** What is the primary architectural difference between a GPU and a CPU that enables GPU acceleration for data-parallel workloads?

A) GPUs have thousands of smaller cores optimized for throughput, while CPUs have few cores optimized for low-latency sequential execution
B) GPUs have larger per-core L1 cache than CPUs
C) GPUs run at significantly higher clock speeds than CPUs
D) GPUs use unified virtual addressing while CPUs cannot address memory directly

---

**Q4.** Which conda command correctly installs a RAPIDS environment targeting CUDA 12.x from the official channels?

A) pip install rapids-gpu
B) conda install -c anaconda rapids
C) conda install -c rapidsai -c conda-forge -c nvidia rapids=25.02 python=3.11 cuda-version=12.2
D) conda create -n rapids rapids-cudf rapids-cuml

---

**Q5.** A team needs to run cuDF-based ETL on a 200 GB dataset, but each available GPU has only 32 GB of memory. Which approach BEST addresses this constraint?

A) Use Dask-cuDF to partition and distribute the dataset across multiple GPUs
B) Increase the batch size in cuDF's read_csv function to fit more data per read
C) Enable spill-to-disk configuration and process the entire dataset on a single GPU
D) Convert the dataset to pandas and process it in chunks on CPU instead

---

**Q6.** A team wants a reproducible, pre-configured RAPIDS environment with matched CUDA drivers and dependencies, deployable consistently across on-prem servers and cloud GPU instances without dependency conflicts. What is the BEST approach?

A) Install RAPIDS via pip inside a plain Python virtualenv on each machine
B) Manually install the CUDA toolkit and build RAPIDS from source on each target machine
C) Pull the official RAPIDS container image from NVIDIA NGC
D) Use 'conda install' with only the default base channel enabled

---

**Q7.** A data science team runs a pandas job on a dataset containing 20,000 rows and 15 columns. The job completes in 0.8 seconds on a laptop CPU. Management wants the team to port this workload to cuDF running on a single A100 GPU to 'make it faster.' What is the BEST recommendation?

A) Port the code to cuDF and run it on the A100; GPU acceleration always outperforms CPU pandas regardless of data size
B) Rewrite the workload using cuML instead of cuDF, since cuML is NVIDIA's recommended replacement for all pandas operations
C) Keep the workload on CPU pandas; at this row count, host-to-device transfer and kernel launch overhead will likely make cuDF slower, not faster
D) Deploy Dask-cuDF across a multi-GPU cluster to maximize parallelism for this dataset

---

**Q8.** A system has CUDA 12.2 installed and the team wants to create an isolated conda environment with the full RAPIDS 24.10 suite (cuDF, cuML, cuGraph) and Python 3.11. Which command correctly accomplishes this per RAPIDS installation guidance?

A) conda create -n rapids -c pytorch -c nvidia rapids=24.10 python=3.11 cuda-version=12.2
B) conda create -n rapids -c rapidsai -c conda-forge -c nvidia rapids=24.10 python=3.11 cuda-version=12.2
C) pip install rapids==24.10
D) docker pull nvcr.io/nvidia/rapidsai/base:24.10-cuda12.2-py3.11 and then run conda install rapids=24.10 inside the running container

---

**Q9.** A team has a 4,000-line pandas ETL script that performs well on CPU but now needs to run faster on a workstation with an NVIDIA GPU. Management wants the speedup with zero code changes to the script itself. Which approach BEST meets this requirement?

A) Rewrite every DataFrame call to use the cudf module explicitly (import cudf as pd)
B) Convert the script to use Dask DataFrame with a distributed CPU cluster
C) Manually replace only the groupby and merge calls with cuDF equivalents
D) Run the script with `python -m cudf.pandas your_script.py` to enable the GPU accelerator for pandas

---

**Q10.** An engineer runs `df['phone'].str.extract(r'(?<!\d)(\d{3}-\d{4})')` on a cuDF Series and it raises a regex compilation error, even though the identical pattern works fine on the same column in pandas. What is the MOST likely cause?

A) cuDF string columns cannot be searched with any regular expression
B) str.extract requires the column to be converted to categorical dtype first
C) cuDF requires regex patterns to be pre-compiled with the `re` module before use
D) The lookbehind assertion `(?<!\d)` is not supported by cuDF's libcudf regex engine

---

**Q11.** A data scientist prints a cuDF DataFrame column that contains missing values and sees `<NA>` instead of `NaN` in the output, differing from the pandas display she is used to. What does this indicate about how cuDF represents missing data internally?

A) The column has been corrupted and must be reloaded from source
B) cuDF automatically converted the column to string dtype, which always displays <NA>
C) cuDF stores missing values using a separate validity bitmask (Arrow-style nulls) rather than encoding them as NaN sentinel values in the data buffer
D) <NA> indicates the operation ran on CPU fallback instead of GPU

---

**Q12.** A cuDF DataFrame has a 'region' column containing only 8 distinct string values but 50 million rows, and it is used repeatedly as a groupby key. Which transformation should be applied BEFORE the groupby to reduce memory footprint and speed up the operation?

A) Convert the column to a Python list and iterate manually
B) Convert the column to categorical dtype with `df['region'] = df['region'].astype('category')`
C) Convert the column to 64-bit floating point using `astype('float64')`
D) Drop the column and re-derive it after the groupby completes

---

**Q13.** A developer needs to quickly generate a large synthetic time-series DataFrame directly on the GPU for benchmarking a new cuDF pipeline, without writing custom random-generation code. Which built-in utility should they use?

A) numpy.random.default_rng().normal()
B) pandas.util.testing.makeTimeDataFrame()
C) cudf.datasets.timeseries()
D) cudf.read_csv() pointed at an empty file

---

**Q14.** A team profiles a cudf.pandas-accelerated script and finds that a `.groupby('id').apply(custom_func)` call, where custom_func performs arbitrary Python object manipulation and string formatting, is running almost entirely on CPU with no GPU speedup. What is the MOST likely explanation?

A) The custom_func uses Python-object-level logic that cuDF cannot execute on GPU, so cudf.pandas automatically falls back to pandas for that call
B) groupby operations are disabled entirely in cudf.pandas mode
C) The DataFrame must be re-sorted by 'id' before groupby will run on GPU
D) Applying any function inside groupby always forces the entire script to run on CPU from that point forward

---

**Q15.** A data engineer loads a CSV into cuDF and needs to impute missing values in a float64 column with the column's median while keeping the entire operation on the GPU (no round-trips to pandas). Which line of code correctly meets this requirement?

A) df = df.to_pandas(); df['col'] = df['col'].fillna(df['col'].median()); df = cudf.from_pandas(df)
B) df['col'] = df['col'].fillna(df['col'].median())
C) df['col'].dropna(inplace=True)
D) df['col'] = df['col'].where(df['col'].notnull(), 0)

---

**Q16.** A team has a large legacy pandas ETL script containing hundreds of calls, including a few operations not yet supported by cuDF (e.g., certain custom rolling-window apply functions). They want GPU acceleration with the fewest code changes possible and automatic fallback to CPU pandas for unsupported calls. What should they do?

A) Manually replace every 'import pandas as pd' with 'import cudf as pd' throughout the script
B) Wrap every DataFrame operation individually in try/except blocks that call .to_pandas() on failure
C) Rewrite the pipeline using Dask-cuDF, since it automatically falls back to pandas for any unsupported operation
D) Run the script as `python -m cudf.pandas etl_script.py` (or load the `cudf.pandas` extension), which proxies pandas calls to cuDF and falls back to CPU pandas automatically

---

**Q17.** You are merging two large cuDF DataFrames on a high-cardinality string column and the join is slower than expected. Which configuration change is most likely to improve join throughput?

A) Convert the join key column to Python object dtype using .astype(object)
B) Convert the join key column in both DataFrames to the 'category' dtype before merging
C) Increase host system RAM without changing any column dtypes
D) Switch from cudf.merge() to pandas.merge() for this operation

---

**Q18.** Which statement correctly describes how cuDF handles string columns?

A) cuDF stores strings as GPU-resident variable-length UTF-8 data using an offsets-and-characters layout, and supports GPU-accelerated vectorized operations like .str.contains() and .str.replace() without host transfer
B) cuDF stores strings as a Python list of str objects, identical to pandas' object dtype
C) cuDF cannot execute any string operations on the GPU and always transparently falls back to pandas for .str accessor calls
D) All regex-based string operations in cuDF run with identical performance to simple substring matches, since both use the same GPU kernel

---

**Q19.** A data science team must process a 500GB CSV dataset on a workstation with a single 32GB GPU. The transformations include groupby aggregations and joins. Which approach BEST satisfies this requirement?

A) Load the entire dataset into a single cuDF DataFrame using unified memory oversubscription
B) Convert the dataset to pandas and run entirely on CPU
C) Use dask-cuDF with a LocalCUDACluster to partition the dataset into GPU-memory-sized chunks processed out-of-core
D) Load a random 32GB sample into cuDF and extrapolate results

---

**Q20.** A single server has 4 NVIDIA A100 GPUs. The team wants to accelerate a large ETL pipeline consisting of filtering, joins, and groupby operations across all 4 GPUs. Which setup is the correct choice?

A) dask-cuDF with LocalCUDACluster spanning all 4 GPUs
B) cuDF DataFrame within a single Python process
C) cuML with a single GPU device context
D) cuGraph running on a CPU-only backend

---

**Q21.** When repartitioning a dask-cuDF DataFrame prior to a distributed ETL job, what is the recommended target size per partition to balance parallelism and GPU memory constraints?

A) 1-2 KB per partition to maximize task count
B) Roughly 150 MB-1GB per partition, small enough to comfortably fit in GPU memory alongside working memory overhead
C) As large as possible, ideally exceeding total GPU memory to reduce partition count
D) Exactly equal to total dataset size divided by number of CPU cores

---

**Q22.** An engineer needs to read only 3 columns from a 200GB partitioned Parquet dataset using dask-cuDF, filtering rows where a partition-key date column falls in the last 30 days. Which practice BEST minimizes I/O and GPU memory use?

A) Use dask_cudf.read_parquet() with the columns parameter to select the needed columns and the filters parameter to push down the row predicate
B) Convert Parquet to CSV first for row-based filtering then load with cudf.read_csv()
C) Read the entire dataset with dask_cudf.read_parquet() with no arguments, then filter columns and rows afterward
D) Read the dataset with pandas.read_parquet() to leverage predicate pushdown on CPU, then transfer needed columns to cuDF

---

**Q23.** A workload requires processing 50GB of tabular data with moderate compute (filtering and simple aggregations), and the deployment is a single workstation with one 48GB GPU. Which topology is MOST appropriate?

A) Single GPU using cuDF directly without Dask
B) Multi-node Dask cluster with 8 nodes
C) CPU-only pandas processing
D) Multi-GPU LocalCUDACluster with 4 GPUs

---

**Q24.** A dask-cuDF pipeline computes an expensive intermediate DataFrame that is reused in three downstream computations (train/test split, feature engineering, and model input). Currently the lazy task graph recomputes the intermediate DataFrame each time it is referenced, tripling runtime. Which change BEST resolves this inefficiency?

A) Call .compute() on the intermediate DataFrame to convert it into a single local cuDF DataFrame
B) Re-run the same read_parquet() and initial transformation for each downstream branch
C) Call .persist() on the intermediate dask-cuDF DataFrame to materialize it in distributed GPU memory once, then reference it in each downstream computation
D) Wrap the intermediate computation in a Python function decorated with functools.lru_cache

---

**Q25.** A team must run a groupby-aggregation pipeline on a 500GB Parquet dataset. They have access to a workstation with a single 32GB GPU. Which approach BEST allows this workload to complete without running out of GPU memory?

A) Load the entire 500GB dataset into a single cuDF DataFrame using CUDA unified memory to oversubscribe the GPU
B) Convert the dataset to pandas and run the pipeline entirely on Dask with CPU workers instead
C) Use dask-cuDF with multiple partitions distributed across a multi-GPU cluster, enabling device-to-host spilling for out-of-core processing
D) Read the Parquet files in chunks with cuDF, manually concatenate all chunks into one DataFrame, then run the groupby

---

**Q26.** When partitioning a large Parquet dataset for processing with dask-cuDF on GPUs with 32-40GB of memory each, what is the recommended target size per partition to balance parallelism against per-partition overhead?

A) 10-50 MB per partition, matching typical CPU-only Spark partition sizing
B) Roughly 1-3 GB per partition, since GPU kernel-launch and scheduling overhead makes CPU-sized partitions too small to be efficient
C) 100-200 MB per partition, matching typical CPU Dask DataFrame guidance
D) 20-30 GB per partition, sized so exactly one partition fits per GPU

---

**Q27.** A team must process a 3TB Parquet dataset within a strict 1-hour SLA. Benchmarks show a single node with 4x GPUs (40GB HBM each) connected via NVLink can complete the job in 55 minutes at ~90% GPU utilization, leaving no headroom. The dataset is expected to grow to 8TB within the quarter. An alternative is a 4-node cluster (16 GPUs total) connected via InfiniBand, running Dask-CUDA workers with UCX. Which topology BEST satisfies both the current SLA and the near-term growth requirement?

A) Keep the single node, since NVLink bandwidth always outperforms InfiniBand regardless of aggregate capacity
B) Keep the single node but reduce the number of partitions to exactly one per GPU to simplify scheduling
C) Fall back to a single-GPU pandas/CPU pipeline to reduce infrastructure cost while still meeting the SLA
D) Migrate to the 4-node cluster using Dask-CUDA workers with UCX over InfiniBand, providing additional GPU memory and compute headroom to scale horizontally as data volume grows

---

**Q28.** A payment processor wants to identify rings of colluding accounts that pass money back and forth in circular patterns to launder funds. Which approach BEST fits this requirement?

A) Use cuML's ARIMA model to detect transaction volume spikes per account
B) Build a graph with accounts as nodes and transactions as edges, then run cuGraph's Louvain community detection to find tightly-connected clusters
C) Use cuDF rolling window statistics on transaction amounts to flag per-account outliers
D) Use cuSpatial to compute the geographic distance between transaction origin points

---

**Q29.** You need to rank nodes in a large web-crawl graph by importance based on the structure of incoming links, where a link from an important page counts more than one from an unimportant page. Which cuGraph algorithm should you use?

A) cugraph.louvain
B) cugraph.betweenness_centrality
C) cugraph.pagerank
D) cugraph.k_truss

---

**Q30.** You run cugraph.louvain() on a fraud network and get only 3 very large communities, which is too coarse to be actionable. You want the algorithm to produce more, smaller, granular communities. Which parameter should you increase?

A) resolution
B) max_iter
C) threshold
D) max_level

---

**Q31.** You have a dataset of millions of GPS delivery-drop points and a set of polygons representing delivery zones. You need to determine, for each point, which zone polygon it falls inside. Which cuSpatial function is purpose-built for this?

A) cuspatial.haversine_distance
B) cuspatial.point_in_polygon
C) cuspatial.quadtree_point_to_nearest_polyline
D) cuspatial.trajectory_distances_and_speeds

---

**Q32.** In a suspected fraud network graph, you want to identify accounts that act as critical 'bridges' connecting otherwise separate clusters of accounts, because removing them would disconnect the network. Which cuGraph centrality measure is designed to quantify this?

A) degree_centrality
B) betweenness_centrality
C) katz_centrality
D) pagerank

---

**Q33.** You need to create a target-encoded feature (mean of the target per category) for a high-cardinality categorical column across a multi-GPU training set, and you must avoid leakage from including a row's own target value in its encoding. Which approach fits the RAPIDS/Dask ecosystem covered on the exam?

A) Compute cudf.DataFrame.groupby(category_col)[target_col].transform('mean') once across the entire training set and apply it to all rows, including the rows used to compute it
B) Compute per-fold groupby aggregations using dask_cudf/cudf and map each fold's held-out rows to means computed from the other folds only
C) Convert the data to pandas, run groupby on CPU, then load the result back into cuDF for training
D) Replace target encoding with cuml.preprocessing.OneHotEncoder because GPU-accelerated groupby aggregations are not available in RAPIDS

---

**Q34.** A financial services company wants to identify tightly-knit clusters of accounts that frequently transact with each other, which may indicate coordinated fraud rings, within a graph of 50 million accounts and 200 million transaction edges. Which cuGraph algorithm BEST identifies these fraud rings?

A) Louvain community detection
B) PageRank
C) Single-source shortest path (SSSP)
D) Betweenness centrality

---

**Q35.** A team needs to detect anomalies in sensor readings from 100,000 IoT devices, each producing a time-series of 1 million readings, by computing a rolling z-score over a sliding window of 500 readings per device. Which approach BEST leverages GPU acceleration for this workload?

A) Load the data into a pandas DataFrame, use pandas.rolling(), then transfer the results to the GPU for z-score scoring
B) Use cuGraph's Louvain community detection to group similar devices before computing rolling statistics
C) Use cuSpatial's spatial join to correlate device locations before computing rolling statistics
D) Use cuDF's groupby-rolling window functions, grouped by device_id, to compute rolling mean and standard deviation directly on the GPU

---

**Q36.** A data science team has a dataset of 50 million rows that fits entirely within a single A100 GPU's 80GB memory. They need to perform density-based clustering to identify arbitrarily shaped clusters and flag outliers, without specifying the number of clusters in advance. Which cuML algorithm BEST meets these requirements?

A) cuML KMeans
B) cuML RandomForestClassifier
C) cuML PCA
D) cuML DBSCAN

---

**Q37.** A team wants to train an XGBoost 2.x model using GPU acceleration for gradient boosted tree construction. Which configuration correctly enables GPU-based training?

A) tree_method='exact', device='cpu'
B) booster='dart', n_jobs=-1
C) tree_method='approx', device='cpu'
D) tree_method='hist', device='cuda'

---

**Q38.** A team is hyperparameter tuning a cuML RandomForestClassifier and wants to run many trials in parallel across a multi-GPU Dask cluster to minimize total tuning time. Which approach BEST accomplishes this?

A) Manually loop through hyperparameter combinations sequentially on a single GPU
B) Use scikit-learn's GridSearchCV directly on the cuML estimator without configuring a Dask client
C) Convert the cuML model to a NumPy-based scikit-learn equivalent before tuning
D) Use Optuna together with a Dask-CUDA cluster to distribute trials across the available GPUs

---

**Q39.** What is the primary architectural advantage of cuML compared to standard scikit-learn for GPU-accelerated data science workflows?

A) cuML provides deep learning-specific neural network layers not found in scikit-learn
B) cuML eliminates the need for any data preprocessing before model training
C) cuML only supports unsupervised learning algorithms
D) cuML mirrors the scikit-learn API while executing algorithms on the GPU for significant speedups on large tabular datasets

---

**Q40.** An engineering team training an image classification model in PyTorch observes that CPU-based JPEG decoding and augmentation are bottlenecking the pipeline, leaving GPUs underutilized during training. Which solution BEST resolves this bottleneck?

A) Increase the batch size to reduce the total number of training steps
B) Use NVIDIA DALI to build a data loading pipeline that performs JPEG decoding and augmentation directly on the GPU
C) Reduce the stored image resolution before writing images to the dataset
D) Switch from PyTorch to TensorFlow to use a different native data loader

---

**Q41.** A data scientist notices via Nsight Systems that a significant portion of pipeline time is consumed by cudaMemcpy calls between host and device, even though the workflow uses cuDF for all transformations. Which change would BEST reduce this overhead?

A) Increase the RMM pool size to pre-allocate GPU memory
B) Switch from Nsight Systems to DLProf to get more granular kernel timings
C) Enable mixed precision training to reduce transfer size
D) Ensure the entire pipeline is chained using cuDF/cuML operations without calling .to_pandas() or .values on intermediate results, keeping data resident on the GPU

---

**Q42.** A Dask-cuDF workflow processing a 200GB dataset on a cluster of GPUs with 32GB memory each begins spilling data to host memory partway through execution, causing a 5x slowdown. Nsight Systems shows large gaps where GPUs are idle waiting on host-to-device copies. What is the BEST first optimization to try?

A) Convert the workflow to use plain pandas DataFrames distributed across the cluster
B) Reduce the number of GPU workers to force larger partitions per worker
C) Configure RMM with a pool allocator and set an appropriate device_memory_limit for the Dask-CUDA workers to reduce fragmentation and control when spilling triggers
D) Disable UCX communication between workers to save memory

---

**Q43.** An engineer wants to determine whether a specific custom CUDA kernel invoked by a cuML algorithm is compute-bound or memory-bandwidth-bound, in order to decide whether to optimize arithmetic intensity or memory access patterns. Which tool should be used?

A) Nsight Compute, to perform detailed kernel-level analysis including roofline metrics
B) Nsight Systems, to view the overall application timeline
C) DLProf, to inspect TensorRT layer statistics
D) Dask dashboard, to monitor worker task streams

---

**Q44.** To ensure a RAPIDS-based accelerated data science pipeline produces consistent results when re-run months later on a different machine, which practice provides the STRONGEST guarantee of reproducibility?

A) Pin exact versions of RAPIDS, CUDA toolkit, and driver compatibility in a conda environment YAML or container image, and record the GPU architecture used
B) Set a global random seed at the start of the script
C) Run the pipeline with cudf.set_option('mode.chained_assignment', None)
D) Use the latest RAPIDS release each time to benefit from bug fixes

---

**Q45.** What is the primary benefit of using mixed precision (e.g., FP16/TF32) instead of full FP32 precision in a GPU-accelerated deep learning training pipeline?

A) It leverages Tensor Cores to increase throughput and reduce memory footprint, at the cost of some numerical precision, often maintained via loss scaling
B) It increases the numerical precision of gradient calculations for better model accuracy
C) It eliminates the need for a GPU by allowing computation to run efficiently on CPU
D) It guarantees identical results to FP32 training with no configuration changes

---

