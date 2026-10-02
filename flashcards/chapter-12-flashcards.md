## Chapter 12: Full Exam Simulation and Time-Management Strategy — Flashcards

| # | Front (Question) | Back (Answer) |
|---|-----------------|---------------|
| 1 | Recommended time budget per question on a ~65-70 question, 120-minute exam? | Roughly 1.5-2 minutes per question. Exam tip: reserve the last 5-10 minutes purely for reviewing flagged questions. |
| 2 | How long is an NVIDIA Certified Professional credential valid before renewal? | 2 years. Exam tip: after expiry you must retake the current version of the exam to stay certified. |
| 3 | pandas vs cuDF — when to choose which? | pandas: small datasets, no GPU available, exploratory work. cuDF: large tabular data that fits in single-GPU memory. Exam tip: if data size is small, GPU overhead makes pandas equally or more efficient. |
| 4 | single-GPU cuDF vs dask-cuDF — decision rule? | If dataset fits comfortably in one GPU's memory, use cuDF alone. If data exceeds GPU memory or spans multiple GPUs/nodes, use dask-cuDF for partitioning and spilling. |
| 5 | cuML vs scikit-learn — decision rule? | cuML: large datasets where GPU acceleration reduces training time meaningfully. scikit-learn: small/medium datasets, CPU-only environments, or when broad algorithm/library support matters more than raw speed. |
| 6 | GPU XGBoost vs CPU XGBoost — decision rule? | Use tree_method='hist' with device='cuda' for large row counts needing fastest training. Use CPU hist mode when GPU is unavailable or dataset is small enough that GPU overhead isn't worth it. |
| 7 | Syntax: read a large CSV across GPUs with dask-cuDF | import dask_cudf; ddf = dask_cudf.read_csv('file*.csv'). Exam tip: dask-cuDF partitions large files automatically for out-of-core / multi-GPU processing. |
| 8 | Syntax: enable GPU acceleration in current XGBoost versions | xgb.train(params={'tree_method':'hist','device':'cuda'}, ...). Exam tip: 'gpu_hist' is the deprecated legacy parameter; current versions use device='cuda' with tree_method='hist'. |
| 9 | What should you do with a scenario question you're unsure about mid-exam? | Select your best current guess, flag it, and move on. Exam tip: never leave a question fully blank — a locked-in guess protects your score if you run out of time. |
| 10 | How should you interpret 'BEST' or 'MOST efficient' in a question stem? | Multiple options may be technically valid, but only one is optimized for the exact constraint given (data size, hardware, latency). Exam tip: eliminate technically-correct-but-suboptimal distractors first. |
| 11 | How should the final review pass before submitting the exam be structured? | Revisit flagged questions first, then do a rapid pass through each of the six domains' highest-yield facts (key tool decision pairs, limits, syntax). Exam tip: don't re-read every question — target only flags and weak-confidence areas. |
| 12 | What should you verify before scheduling the NCP-ADS exam? | Current exam fee, retake policy, and scheduling terms directly on NVIDIA's official certification portal. Exam tip: fees and policies change over time — never rely on outdated third-party info. |
