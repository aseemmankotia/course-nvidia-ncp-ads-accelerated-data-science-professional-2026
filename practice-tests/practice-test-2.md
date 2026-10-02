# Practice Test 2: NCP-ADS: NVIDIA Accelerated Data Science Prep

- **Questions:** 45
- **Time limit:** 90 minutes
- **Passing score:** 70%%

> Answers are in `answer-key-2.md`. Track results in `score-tracker.md`.

---

**Q1.** A data pipeline moves NumPy arrays from host to a GPU-backed cuDF workflow using thousands of separate transfer calls, each moving less than 1MB. Profiling shows total runtime is dominated by transfer overhead rather than the achievable PCIe bandwidth. Which change BEST addresses this bottleneck?

A) Read the source data directly with cuDF's GPU CSV reader instead of using NumPy at all
B) Overlap kernel execution using multiple CUDA streams while keeping the transfer pattern unchanged
C) Increase the RMM pool allocator size to cover the full GPU memory
D) Batch the small arrays into fewer, larger transfers and use pinned (page-locked) host memory buffers for the copy

---

**Q2.** A team must deploy an identical RAPIDS environment across on-prem workstations and cloud GPU instances that run different host OS versions, and they need CUDA toolkit and driver compatibility that is validated and supported by NVIDIA. Which deployment approach BEST satisfies this requirement?

A) Install RAPIDS with pip into a Python virtual environment on each host individually
B) Pull and run the official RAPIDS container image from NGC matching the host's CUDA driver version
C) Build cuDF and cuML from source on each host to match the local CUDA driver exactly
D) Share a single conda environment.yml file via version control for each team member to install locally

---

**Q3.** You initialize RMM with a pool allocator to reduce allocation fragmentation, but you want the pool to be able to grow beyond its initial size if the workload needs more memory, rather than failing or falling back to raw cudaMalloc. Which RMM configuration achieves this?

A) Call rmm.reinitialize(managed_memory=True, pool_allocator=False) to rely on CUDA Unified Memory instead
B) Call rmm.reinitialize(pool_allocator=False) so every allocation goes directly to the CUDA driver
C) Call rmm.reinitialize(pool_allocator=True, initial_pool_size=<value>, maximum_pool_size=<value>) with maximum_pool_size set higher than initial_pool_size
D) Set initial_pool_size equal to maximum_pool_size so the pool is fixed at 100% of GPU memory

---

**Q4.** A cuDF workflow allocates a working set larger than physical GPU memory using CUDA Unified Memory (RMM managed_memory=True). When the workflow accesses data pages that are not currently resident on the GPU, what actually happens?

A) The allocation call fails immediately with an out-of-memory error before any computation runs
B) The runtime automatically migrates the needed pages from host memory to device memory on-demand via page faults, adding latency
C) The oversized portion of the data is silently truncated to fit within available GPU memory
D) Execution automatically falls back to equivalent pandas CPU kernels for the oversized data

---

**Q5.** A pandas workflow performs groupby and merge operations on a 50MB CSV and completes in about 2 seconds on a single CPU core. A junior engineer proposes porting this exact workflow to cuDF on an A100 GPU expecting a speedup. What is the MOST accurate assessment and recommendation?

A) cuDF will be faster regardless of size because GPU cores vastly outnumber CPU cores, so the migration should proceed immediately
B) The dataset is too small for cuDF to even load; cuDF requires a minimum file size of 100MB to function
C) The workflow must first be rewritten using cuGraph before any GPU acceleration is possible
D) Host-to-device transfer and kernel launch overhead will likely make the GPU version slower for this data size; keep pandas unless data volume grows substantially

---

**Q6.** A Dask-CUDA cluster (LocalCUDACluster) spans 4 GPUs and runs a groupby-aggregate job on a dataset larger than the combined GPU memory across all 4 devices. The job repeatedly fails with GPU out-of-memory errors before completing, even though total GPU memory is close to the dataset size. Which change most directly resolves this without reducing the dataset size?

A) Enable device-to-host spilling (e.g., jit-unspill / device_memory_limit) so data exceeding GPU capacity is moved to host memory as needed
B) Increase the number of Dask-CUDA workers assigned per GPU to parallelize the aggregation further
C) Convert the aggregation step from cuDF operations to pandas operations inside the Dask graph
D) Restrict the cluster to a single GPU via CUDA_VISIBLE_DEVICES to simplify memory management

---

**Q7.** A data scientist has a 50MB CSV file and needs to run a single one-time groupby aggregation. Their existing pandas script already completes the job in about 2 seconds. Which approach is BEST for this specific job?

A) Load the CSV directly into a cuDF DataFrame on the GPU and rerun the same groupby — GPU acceleration will make the one-time job faster
B) Convert the script to use Dask-cuDF across all available GPUs to parallelize the groupby operation
C) Keep the existing pandas script unchanged, since the host-to-device transfer and CUDA context initialization overhead for a 50MB one-time job will exceed the compute time saved
D) Wrap the pandas call in cuDF's pandas accelerator mode (cudf.pandas) so RAPIDS automatically picks the fastest device

---

**Q8.** A workstation has NVIDIA GPU drivers installed that only support up to CUDA 11.8 at the driver level, and the drivers cannot be upgraded due to organizational policy. Which command correctly installs a working RAPIDS environment via conda on this host?

A) conda install -c rapidsai -c conda-forge -c nvidia rapids=24.02 python=3.10 cuda-version=11.8
B) conda install -c rapidsai -c conda-forge -c nvidia rapids=24.02 python=3.10 cuda-version=12.0
C) pip install cudf-cu12 cuml-cu12 dask-cudf-cu12
D) conda create -n rapids rapids=24.02 cudatoolkit=11.2 -c rapidsai -c nvidia -c conda-forge

---

**Q9.** You have enabled GPU acceleration with `%load_ext cudf.pandas` in a Jupyter notebook. All prior cell operations ran on a DataFrame `df` created after the extension loaded. You then call `df.style.background_gradient()`, an accessor cuDF does not implement natively. What actually happens?

A) cudf.pandas seamlessly falls back to the real pandas implementation for the unsupported .style accessor, transferring data to host memory automatically with no code change required
B) The call raises a hard error because Styler objects cannot be produced from GPU-backed frames
C) cudf.pandas silently ignores the unsupported call and returns the DataFrame unmodified
D) You must manually call .to_pandas() first, since cudf.pandas only accelerates method calls, never attribute-based accessors

---

**Q10.** You run `cudf.Series([1, 2, None, 4])`. What is the resulting dtype, and how does this differ from pandas' default behavior for the same input list?

A) float64 with NaN representing the missing value; this is identical to pandas' default behavior
B) int64 with the null represented via pd.NA; pandas would also default to nullable int64
C) object dtype holding a literal Python None; pandas would raise a TypeError for the same input
D) int64 with a validity bitmask marking the null entry; pandas, by default, would upcast the column to float64 and represent the missing value as NaN

---

**Q11.** You need to split a cuDF string column on a comma where the number of resulting substrings varies per row, and you only need the first two resulting tokens as separate columns. Which approach is correct and GPU-efficient?

A) Iterate rows in a Python for-loop, calling `.split(",")` on each extracted Python string, then reassemble the results into new columns
B) Convert the column to pandas first, because cuDF cannot handle ragged/variable-length string splits
C) Call `series.str.split(pat=",", n=2, expand=True)` and select the first two resulting columns
D) Use `series.str.rsplit()` with a variable-length regex lookbehind to bound the token count to two

---

**Q12.** A cuDF DataFrame has a column 'transaction_id' containing 50 million unique string values and a column 'country' containing only 15 unique string values repeated across all 50 million rows. Before a large groupby, which single dtype change correctly targets GPU memory reduction?

A) Convert 'transaction_id' to category dtype
B) Convert only 'country' to category dtype
C) Convert both 'transaction_id' and 'country' to category dtype
D) Category dtype conversion is unsupported in cuDF prior to a groupby operation

---

**Q13.** Without using cudf.pandas, you call `.dt.to_period('M')` on a cuDF datetime64 column to bucket records into calendar months. What is the actual outcome?

A) It works identically to pandas and returns a cuDF column with PeriodDtype
B) It silently converts the column to a plain string column representing the period label
C) It executes correctly because cudf.pandas is active by default in any RAPIDS environment, transparently routing the call to CPU pandas
D) It raises a NotImplementedError because cuDF does not implement pandas' Period extension dtype; you should derive year/month components (e.g., via .dt.year and .dt.month, or a truncated timestamp) instead

---

**Q14.** Which RAPIDS utility generates a synthetic, GPU-resident time-indexed cuDF DataFrame (with configurable id, x, and y columns) that is commonly used to benchmark cuDF groupby and join performance without loading external datasets?

A) cuml.datasets.make_blobs() — generates GPU-resident clustering blobs for ML algorithms, not time-indexed tabular records
B) cudf.datasets.timeseries() — generates a synthetic time-indexed cuDF DataFrame with configurable id/x/y columns entirely on GPU
C) dask_cudf.datasets.make_timeseries() — not an existing RAPIDS API
D) cupy.random.default_rng().synthetic_frame() — not a real cuPy API

---

**Q15.** A team enables cudf.pandas (`%load_ext cudf.pandas`) to accelerate a legacy ETL script without code changes. The script calls `.apply()` with a complex Python function that does conditional string parsing not expressible as vectorized cuDF operations. After enabling cudf.pandas, the script still finishes with correct results, but the built-in profiler shows this single `.apply()` line consuming 90% of total runtime. What is the BEST explanation and remedy?

A) The `.apply()` logic isn't supported by libcudf kernels, so cudf.pandas automatically fell back to CPU pandas for that call (with data copied host-to-device each time); rewrite the logic using vectorized cuDF string/column operations to keep it GPU-resident
B) cudf.pandas cannot execute `.apply()` at all and the script must have crashed rather than completed, so the profiler output is invalid
C) The slowdown indicates a GPU memory spill event; setting `CUDF_SPILL=on` will resolve the `.apply()` performance issue
D) cudf.pandas only accelerates `read_csv` and `groupby` calls — all `.apply()` usage is always routed to CPU pandas and cannot be optimized

---

**Q16.** An engineer merges two cuDF DataFrames, `orders` and `customers`, on a categorical column `region`. Both columns contain the same underlying string values, but the categorical dtypes were built independently, so their `categories` arrays have different orderings (and therefore different integer codes for the same string). The merge silently returns zero matching rows for records that clearly share the same region string in both frames. What is the root cause and correct fix?

A) cuDF merge compares underlying integer category codes, not the string values, for `category` dtype columns; mismatched category orderings map identical strings to different codes, causing missed matches — fix by casting both join columns to `str` or unifying categories via `.cat.set_categories()` on both sides before merging
B) Categorical column merges are entirely unsupported in cuDF and always return zero rows regardless of category alignment
C) The engineer forgot to pass `sort=True` to `merge()`, which is required whenever either join column is categorical
D) cuDF treats any category-code mismatch between two categorical columns as a null value, and null keys are dropped by default during merge

---

**Q17.** A pandas script runs `df.groupby('key', dropna=True).sum()`. When porting this line directly to cuDF (`gdf`), which cuDF call reproduces IDENTICAL null-key handling behavior?

A) `gdf.groupby('key', dropna=False).sum()`
B) `gdf.groupby('key').sum(skipna=False)`
C) cuDF always retains null keys as their own group, so an explicit `.dropna()` call must be chained after `.groupby()` regardless of parameters
D) `gdf.groupby('key').sum()`

---

**Q18.** A single-GPU cuDF workflow performs a multi-way merge that occasionally exceeds available GPU memory. The team wants cuDF itself to automatically move column buffers between GPU and host memory as memory pressure occurs, without manually chunking the data or migrating the script to a distributed Dask-cuDF cluster. Which configuration achieves this?

A) Create an RMM `PoolMemoryResource` with `initial_pool_size` set to the amount of available host RAM
B) Set `CUDF_HOST_MEMORY=unified` so CUDA Unified Memory pages data between GPU and host automatically
C) Enable cuDF's built-in spill manager with `cudf.set_option("spill", True)` (equivalently, the `CUDF_SPILL=on` environment variable)
D) Migrate the script to Dask-cuDF so its out-of-core task scheduler handles memory pressure

---

**Q19.** A team must join two datasets totaling 1.5TB on a single DGX node with 8x A100 80GB GPUs (640GB aggregate GPU memory). The join triggers a large shuffle that temporarily needs more memory than fits on any single GPU. Which configuration BEST handles this on the single node?

A) LocalCUDACluster across all 8 GPUs with UCX/NVLink enabled, RMM pool allocator, and device-to-host spilling configured for the shuffle buffers
B) Load both datasets fully into one A100 80GB GPU using cuDF and perform the join in a single process
C) Provision a second DGX node and build a multi-node dask-cuda cluster over InfiniBand, since a single node cannot handle a join of this size
D) Convert both datasets to pandas DataFrames and run the join with a CPU Dask cluster to avoid GPU memory constraints

---

**Q20.** An out-of-core groupby-aggregation on a dataset larger than aggregate GPU memory keeps failing with GPU out-of-memory errors on a dask-cuda LocalCUDACluster. Which cluster configuration correctly enables automatic spilling from GPU memory to host RAM and then to disk as needed?

A) LocalCUDACluster(rmm_pool_size='90GB') with no other memory limits set, to maximize the amount of data each GPU can hold
B) LocalCUDACluster(device_memory_limit=0) to disable any GPU memory ceiling and let RMM manage allocation directly
C) LocalCUDACluster(device_memory_limit='0.8', jit_unspill=True) so workers spill to host memory at 80% GPU usage and to disk when host memory is also exhausted
D) LocalCUDACluster(memory_limit='0') to give workers unlimited host memory so spilling is never triggered

---

**Q21.** A 500GB dataset will be written to Parquet for later ingestion by dask_cudf on GPUs with 32GB of memory each. Which write/read strategy minimizes GPU memory overhead while maximizing read throughput?

A) Write CSV files instead of Parquet since CSV avoids row-group complexity entirely
B) Write a single Parquet file with one giant row group covering the entire 500GB dataset for simplest metadata handling
C) Write Parquet files with row groups under 5MB each to guarantee the smallest possible per-partition memory footprint
D) Write multiple Parquet files with row groups sized around 128-256MB, and read with dask_cudf.read_parquet using split_row_groups to control partition size

---

**Q22.** An organization needs to run daily ETL processing 50TB of Parquet data with heavy shuffle-based joins and sorts. They have 4 DGX A100 nodes (8 GPUs per node, NVLink intra-node, InfiniBand inter-node). Which setup best supports this throughput requirement?

A) A single-node LocalCUDACluster using only the 8 GPUs on one DGX node, repeating the job four times sequentially across nodes
B) A multi-node dask-cuda cluster spanning all 4 DGX nodes, using UCX for communication so NVLink is used intra-node and InfiniBand is used inter-node
C) A CPU-only Dask cluster spanning all 4 nodes to avoid GPU memory limitations during the large shuffle
D) Apache Spark with a generic UDF layer calling cuDF functions row-by-row on each of the 4 nodes

---

**Q23.** When repartitioning a dask_cudf DataFrame before a memory-intensive shuffle operation like a sort-based join, what partition sizing guideline best balances GPU utilization against out-of-memory risk during the shuffle?

A) Size partitions equal to the full memory capacity of a single GPU, so each partition maximizes hardware utilization
B) Size partitions to roughly 1/4 to 1/3 of a single GPU's memory, leaving headroom for intermediate shuffle buffers and temporary allocations
C) Size partitions as small as a few megabytes to guarantee the operation never runs out of memory regardless of GPU size
D) Partition size does not matter for shuffle operations as long as the total dataset fits in aggregate cluster GPU memory

---

**Q24.** A 2TB ORC dataset has 50 columns, but a downstream GPU pipeline only needs 3 columns and rows within a specific date range. Which approach minimizes I/O and GPU memory usage when loading this with dask_cudf?

A) Use dask_cudf.read_orc with the columns parameter to project only the needed columns, combined with partition-level filtering on the date range after minimal projected load
B) Load the entire ORC dataset into cuDF first, then drop unneeded columns and filter rows after loading
C) Convert the ORC file to CSV first, since CSV supports easier column selection during read
D) Manually rewrite the ORC file as a single stripe containing only the 3 needed columns before every pipeline run

---

**Q25.** A team has 20TB of Parquet transaction data requiring wide groupby aggregations. They have access to 8 nodes, each with 4x A100 40GB GPUs connected via NVLink within nodes and InfiniBand across nodes. Which cluster topology and configuration BEST meets this throughput and data-volume requirement?

A) Launch a multi-node dask-cuda cluster spanning all 8 nodes with UCX enabled to leverage NVLink/InfiniBand, and configure JIT-unspill so intermediate results larger than GPU memory spill to host memory and disk
B) Deploy a single-node LocalCUDACluster using all 4 local GPUs and rely on CUDA Unified Memory oversubscription to page the remaining data from host RAM as needed
C) Use a single GPU with cuDF and manually chunk the 20TB dataset into sequential batches read and processed one at a time
D) Run the job as a CPU-based Dask+pandas cluster across all 8 nodes for reliable out-of-core handling, converting only the final aggregated result to cuDF

---

**Q26.** During a large distributed merge, intermediate dask_cudf partitions occasionally exceed available GPU memory on individual workers. Which dask-cuda LocalCUDACluster configuration correctly enables spilling of oversized GPU objects to host memory (and then disk) to avoid worker OOM crashes?

A) LocalCUDACluster(device_memory_limit="0.8", jit_unspill=True) to trigger spilling at 80% GPU memory usage and just-in-time unspill objects from host memory/disk only when needed
B) LocalCUDACluster(memory_limit=None) to remove worker memory limits entirely so no spilling decision is ever triggered
C) LocalCUDACluster(enable_tcp_over_ucx=True) to route all inter-worker transfers over TCP instead of NVLink
D) LocalCUDACluster(rmm_pool_size="0") to disable the RMM memory pool so allocations always go directly to the driver

---

**Q27.** A dataset of 5TB of Parquet files, partitioned by date with ~128MB row groups, must be queried by a dask_cudf job that only needs the date, amount, and txn_id columns for January-March 2026 records where amount > 10000. Which approach minimizes both I/O and GPU memory usage for this query?

A) dask_cudf.read_parquet(path) to load the full dataset into GPU memory, then apply .query("amount > 10000") afterward
B) Convert the Parquet files to CSV first, then use cudf.read_csv with a row filter applied during the read
C) Load the Parquet files using pandas.read_parquet on the CPU, filter with boolean indexing, then convert the filtered result to a cudf.DataFrame
D) dask_cudf.read_parquet(path, columns=["date","amount","txn_id"], filters=[[("date",">=","2026-01-01"),("date","<=","2026-03-31")],[("amount",">",10000)]]) to leverage row-group statistics for pruning and column projection

---

**Q28.** A financial services company runs fraud detection on a transaction graph with 50M nodes and 200M edges on a DGX cluster. Analysts need to surface tightly-connected clusters of accounts that transact heavily with each other, indicative of collusive fraud rings, rather than ranking individual account importance. Which cuGraph algorithm BEST accomplishes this?

A) cugraph.bfs
B) cugraph.pagerank
C) cugraph.sssp
D) cugraph.louvain

---

**Q29.** In a social network graph, an analyst wants to identify accounts that act as bridges connecting otherwise separate communities, since these bottleneck accounts are critical for controlling information flow between clusters. Which cuGraph centrality measure is specifically designed to score nodes based on the fraction of all shortest paths that pass through them?

A) cugraph.degree_centrality
B) cugraph.pagerank
C) cugraph.katz_centrality
D) cugraph.betweenness_centrality

---

**Q30.** You must flag anomalous readings in a 500M-row GPU dataframe of sensor telemetry sampled every 1 minute, using a trailing 24-hour rolling z-score in cuDF for a real-time monitoring pipeline. The window must use only past and current data with no look-ahead bias. Which rolling configuration is correct?

A) df.rolling(window=1440, min_periods=1, center=True)
B) df.rolling(window=24, min_periods=1, center=False)
C) df.expanding().mean()
D) df.rolling(window=1440, min_periods=1, center=False)

---

**Q31.** A geospatial team needs to determine, for millions of GPS pings, which delivery zone polygon each ping falls inside, using GPU acceleration. Which capability does cuSpatial provide to support this at an awareness level within the RAPIDS ecosystem?

A) Real-time decoding of geospatial video streams
B) Native training of graph neural networks on map data
C) GPU-accelerated point-in-polygon testing and spatial joins
D) Distributed file-system indexing for GeoTIFF raster files

---

**Q32.** While engineering features for a 5M-row tabular dataset in cuDF, a data scientist computes a target-mean encoding for a high-cardinality categorical column by running a single groupby on the target across the entire dataset and merging the result back onto every row before training. Model validation accuracy is suspiciously high. Which approach correctly addresses the underlying problem?

A) Replace the categorical column with one-hot encoding computed the same way
B) Keep the full-dataset groupby mean encoding but merge it after train/test split instead of before
C) Compute the target-mean encoding using k-fold cross-validation, excluding each row's own fold when computing its encoded value
D) Replace the encoding with alphabetical label encoding of the category values

---

**Q33.** A recommendation engine needs influence scores that are biased toward relevance to a specific seed set of users who purchased Product X, rather than a global importance ranking across the entire graph. Which cuGraph approach is designed for this personalized use case?

A) cugraph.pagerank with a personalization vector specifying the seed nodes
B) cugraph.triangle_count
C) cugraph.katz_centrality with default parameters
D) cugraph.pagerank with no personalization vector

---

**Q34.** A fintech company suspects a fraud ring: a group of accounts that transact heavily with each other but rarely with outside accounts, in a graph of 40 million account nodes and 300 million transaction edges. The analytics team needs to GPU-accelerate the detection of these tightly-knit account clusters using cuGraph. Which algorithm should they run?

A) cuGraph PageRank, since fraud rings will have the highest PageRank scores in the graph
B) cuGraph Single-Source Shortest Path (SSSP) from a known fraudulent account to find related accounts
C) cuGraph Louvain community detection to partition the graph into densely connected subgraphs
D) cuGraph Betweenness Centrality to identify accounts that bridge different parts of the network

---

**Q35.** A team has 5 years of minute-level sensor readings (over 2 billion rows) stored in Parquet and needs to flag anomalous readings using a rolling 60-minute z-score per sensor, then feed flagged windows into a downstream cuML model. The pipeline must stay entirely GPU-accelerated for performance. Which approach is correct?

A) Load the data into cuDF, then convert to a pandas DataFrame to use scipy.stats.zscore with a rolling window, since cuDF does not support windowed statistics
B) Load the data into cuGraph as a graph where each sensor reading is a node, then run PageRank to surface anomalous readings
C) Load the data into cuSpatial and use a point-in-polygon query to identify anomalous sensor locations
D) Load the data into cuDF, group by sensor_id, use cuDF's rolling() window operations to compute rolling mean and standard deviation, then compute z-scores directly on the GPU dataframe

---

**Q36.** A data science team must train a gradient boosted tree model on a 500 GB tabular dataset distributed across 8 GPUs in a single node, where the dataset does not fit into the memory of any single GPU. Which configuration BEST meets these requirements?

A) Use cuML's RandomForestClassifier with n_streams=8 to parallelize training across the GPUs
B) Convert the dataset to a single pandas DataFrame and train scikit-learn's GradientBoostingClassifier with n_jobs=-1
C) Launch a LocalCUDACluster with dask_cuda, load the data into a dask_cudf.DataFrame, and train using the Dask API of XGBoost with tree_method='hist' and device='cuda' via a distributed DaskDMatrix
D) Load the entire dataset onto GPU 0 using cudf.read_parquet and train XGBoost with a single-GPU DMatrix

---

**Q37.** Which cuML clustering algorithm exposes a max_mbytes_per_batch parameter specifically designed to process datasets whose full pairwise distance matrix would exceed available GPU memory, by computing distances in batches?

A) cuML KMeans
B) cuML DBSCAN
C) cuML AgglomerativeClustering
D) cuML HDBSCAN

---

**Q38.** A team is building a DALI pipeline to feed decoded and augmented image batches into a PyTorch training loop and wants JPEG decoding to run on the GPU using nvJPEG rather than on the CPU. Which operator configuration achieves this?

A) fn.decoders.image with device='mixed', which decodes JPEGs on the GPU via nvJPEG and outputs device tensors to subsequent GPU operators
B) fn.decoders.image with device='cpu', followed by fn.resize with device='gpu'
C) fn.readers.file followed by PIL-based decoding inside an external_source operator
D) fn.decoders.image with device='cpu' only, since DALI does not support GPU-accelerated JPEG decoding

---

**Q39.** A data scientist needs to run a randomized search over 200 candidate hyperparameter configurations for a cuML RandomForestClassifier on a dataset that fits entirely in the memory of one GPU. The search must complete as fast as possible on a workstation with 4 GPUs. Which approach BEST achieves this?

A) Use Dask-ML's RandomizedSearchCV backed by a LocalCUDACluster with 4 workers (one per GPU), so multiple candidate models train concurrently on separate GPUs
B) Use scikit-learn's RandomizedSearchCV directly on the cuML estimator with n_jobs=4
C) Manually loop over all 200 configurations sequentially on a single GPU and record the best score
D) Use Optuna's TPE sampler running purely on the CPU, leaving all 4 GPUs idle, since hyperparameter search must be sequential

---

**Q40.** In current XGBoost releases, which configuration is the recommended way to enable GPU-accelerated histogram-based tree construction, replacing the deprecated tree_method='gpu_hist' parameter?

A) booster='gblinear' combined with device='cuda'
B) tree_method='hist' combined with device='cuda'
C) tree_method='approx' combined with n_gpus=1
D) tree_method='gpu_exact' combined with predictor='gpu_predictor'

---

**Q41.** A cuDF ETL job reads 50,000 small CSV files inside a Python loop, converting each to a cuDF DataFrame and immediately appending it to a list before a final concat. Nsight Systems shows the GPU idle for most of the wall-clock time while the CPU is busy issuing many small H2D memory copies. Which change will have the LARGEST impact on throughput?

A) Increase the RMM pool allocator size so cuDF can allocate larger buffers per file
B) Wrap each read_csv call in its own CUDA stream to overlap the copies
C) Read the files with cudf.read_csv on the host in a single batched call (or use dask_cudf.read_csv with globbing) so far fewer, larger host-to-device transfers occur
D) Switch each file's dtype from float64 to float32 before transferring

---

**Q42.** An Nsight Systems timeline of a custom CUDA-accelerated feature pipeline shows repeating pairs of short kernels separated by synchronous cudaMemcpyHtoD calls, each moving a small NumPy array built inside a Python for-loop before being pushed to the GPU. Which fix directly addresses this specific pattern?

A) Enable automatic mixed precision (AMP) for the kernels to shrink their execution time
B) Consolidate the small arrays into one contiguous host buffer, allocate it as pinned (page-locked) memory, and issue a single asynchronous transfer on a CUDA stream before the kernel loop runs
C) Call cudf.DataFrame.to_pandas() after each kernel to move results back for inspection
D) Increase CUDA_VISIBLE_DEVICES to expose additional GPUs to the process

---

**Q43.** A team trains a PyTorch model with torch.cuda.amp and needs to confirm which layers are actually falling back to FP32 and what percentage of GPU time is spent in Tensor Core-eligible kernels, visualized as an interactive report. Which tool is purpose-built for this task?

A) nvidia-smi dmon
B) DLProf with the TensorBoard plugin
C) Nsight Compute in isolation
D) cuML's built-in profiler context manager

---

**Q44.** A cuML RandomForestClassifier trained with a fixed random_state produces slightly different accuracy and split points on repeated runs against the same dataset on the same GPU. Which configuration change makes the training run deterministic?

A) Set the environment variable PYTHONHASHSEED=0 before launching the script
B) Set n_streams=1 on the RandomForestClassifier in addition to the fixed random_state
C) Set CUDA_LAUNCH_BLOCKING=1 to force synchronous kernel execution
D) Cast the training data to float64 instead of float32 before fitting

---

**Q45.** On an NVIDIA A100 GPU, which precision mode requires NO changes to model code (no autocast, no GradScaler) yet automatically accelerates FP32 matrix multiplications and convolutions on Tensor Cores, while data remains stored in 32-bit memory?

A) BF16 (bfloat16)
B) FP16 with automatic mixed precision (AMP)
C) TF32 (TensorFoat-32)
D) INT8 post-training quantization

---

