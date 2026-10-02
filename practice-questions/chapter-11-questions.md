## Chapter 11: Profiling and Optimization: DLProf, Nsight, and Memory Efficiency — Practice Questions

**Q1.** A training pipeline shows periodic dips in GPU utilization on the timeline, correlating with large blocks of host-to-device memory copies between iterations. Which tool should be used FIRST to confirm this bottleneck?

A) Nsight Compute, to inspect kernel occupancy
B) DLProf, to get aggregated Tensor Core utilization
C) Nsight Systems, to view the system-wide timeline including memcpy events
D) RMM pool allocator logs

<details>
<summary>Answer</summary>

**Correct: C**

Nsight Systems provides a pipeline-level timeline showing CPU activity, GPU activity, and memcpy (H2D/D2H) events together, making it the correct tool to confirm transfer-related stalls across the pipeline.

**Why the others are wrong:**
- Nsight Compute profiles inside a single kernel launch and does not show pipeline-wide timeline gaps caused by transfers.
- DLProf gives aggregated per-run training metrics (iteration time, Tensor Core %) but not a visual timeline correlating stalls to specific memcpy events.
- RMM logs relate to device memory pool allocation, not host-to-device transfer timing.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** Profiling reveals that a specific custom CUDA kernel used in a data-loading step is running with low warp occupancy and high register usage. Which tool provides the granularity needed to diagnose this?

A) Nsight Systems
B) Nsight Compute
C) DLProf
D) RMM statistics logger

<details>
<summary>Answer</summary>

**Correct: B**

Nsight Compute performs per-kernel deep-dive analysis, reporting occupancy, register usage, and memory throughput for an individual kernel launch — exactly what's needed here.

**Why the others are wrong:**
- Nsight Systems shows timeline-level activity across the whole pipeline, not per-kernel register/occupancy detail.
- DLProf reports aggregated deep learning training metrics, not individual custom kernel internals.
- RMM statistics track memory pool allocations, not kernel execution efficiency.

</details>

---

**Q3.** A team wants a single report showing overall GPU utilization percentage, Tensor Core usage percentage, and average iteration time across an entire PyTorch training run. Which tool is BEST suited?

A) DLProf
B) Nsight Compute
C) Nsight Systems
D) cuDF profiler

<details>
<summary>Answer</summary>

**Correct: A**

DLProf is purpose-built for deep learning training profiling, producing aggregated metrics such as GPU utilization, Tensor Core usage %, and iteration time, and can export a detailed timeline viewable in Nsight Systems.

**Why the others are wrong:**
- Nsight Compute is kernel-level and would require manually aggregating many individual kernel reports, not designed for run-level summaries.
- Nsight Systems shows a raw timeline but does not natively compute aggregated Tensor Core usage percentages the way DLProf does.
- There is no standard tool called 'cuDF profiler'; this is a distractor.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q4.** A RAPIDS cuDF workflow performs thousands of small, frequent GPU memory allocations during ETL operations, causing high overhead from repeated cudaMalloc/cudaFree calls. Which configuration change addresses this?

A) Switch the workload to FP16 mixed precision
B) Increase the batch size of the ETL job
C) Profile with Nsight Compute to find the slow kernel
D) Enable RMM pool allocation with a pre-allocated memory pool

<details>
<summary>Answer</summary>

**Correct: D**

RMM's pool allocator pre-allocates a large device memory block and services allocations from it via a sub-allocator, eliminating repeated driver calls for small/frequent allocations — the exact scenario described.

**Why the others are wrong:**
- Mixed precision reduces memory footprint per value but does not address allocation call overhead.
- Batch size affects training throughput, not the allocator overhead of an ETL workflow.
- Nsight Compute profiles kernel execution, not memory allocation call overhead; it would not directly fix this problem.

</details>

---

**Q5.** A model trained with FP16 mixed precision experiences vanishing gradients during backpropagation, where small gradient values underflow to zero. Which technique should be applied to fix this?

A) Apply loss scaling before the backward pass and unscale gradients before the optimizer step
B) Switch to TF32 for all matrix multiplications
C) Reduce the batch size to lower memory pressure
D) Enable RMM pool allocation for gradient buffers

<details>
<summary>Answer</summary>

**Correct: A**

Loss scaling multiplies the loss by a scale factor to shift small gradient values into the representable FP16 range before backpropagation, then unscales gradients prior to the optimizer step — the standard fix for FP16 underflow.

**Why the others are wrong:**
- TF32 avoids the underflow issue by design but switching to it changes the precision scheme entirely rather than fixing FP16 training as asked; the described fix must work within FP16.
- Reducing batch size affects memory usage, not gradient underflow caused by limited FP16 exponent range.
- RMM pool allocation addresses allocation overhead, not numerical precision issues.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q6.** Nsight Systems timeline analysis shows a GPU utilization of only 40%, with many short gaps between kernel launches during training. What is the MOST likely root cause and fix?

A) The model needs to be converted to TF32 to increase Tensor Core throughput
B) The RMM memory pool is undersized and needs to be increased
C) Nsight Compute must be used instead to fix the utilization gaps
D) Batch size is too small, causing kernel launch overhead to dominate; increase the batch size

<details>
<summary>Answer</summary>

**Correct: D**

Frequent short gaps between kernel launches at low utilization is a classic symptom of small batch sizes, where per-batch kernel launch overhead is not amortized over enough computation. Increasing batch size improves GPU utilization.

**Why the others are wrong:**
- TF32 conversion improves numerical throughput per kernel but does not close timeline gaps caused by launch overhead from small batches.
- RMM pool sizing affects allocation overhead, not kernel launch frequency gaps on the compute timeline.
- Nsight Compute is used for kernel-level diagnosis, not as a fix for utilization; it's a profiling tool, not a corrective action.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** A production ADS pipeline runs correctly in staging but produces different results when redeployed on a different node using the container tagged 'latest'. What practice would MOST directly prevent this issue?

A) Enable mixed precision training on all nodes
B) Increase the RMM pool size on the new node
C) Pin the container image to a specific version tag and pin all library versions instead of using 'latest'
D) Switch profiling from DLProf to Nsight Systems on the new node

<details>
<summary>Answer</summary>

**Correct: C**

Using the 'latest' tag means the container and its dependencies can differ silently between deployments. Pinning exact image versions and library versions ensures a reproducible, consistent environment.

**Why the others are wrong:**
- Mixed precision affects numerical computation, not environment/version drift causing inconsistent deployments.
- RMM pool size affects memory allocation performance, not result reproducibility across environments.
- Choice of profiling tool has no bearing on environment consistency between deployments.

</details>

---

**Q8.** Which statement correctly differentiates DLProf from Nsight Systems in the context of deep learning training profiling?

A) DLProf wraps a training script to produce DL-specific aggregated metrics and can export a timeline viewable in Nsight Systems, while Nsight Systems independently captures the full system-wide timeline
B) DLProf profiles individual CUDA kernels, while Nsight Systems provides aggregated Tensor Core metrics
C) DLProf and Nsight Systems are interchangeable and provide identical outputs
D) Nsight Systems is only usable with TensorFlow, while DLProf works with any framework

<details>
<summary>Answer</summary>

**Correct: A**

DLProf is a deep-learning-specific wrapper that produces aggregated metrics (GPU utilization, Tensor Core usage, iteration time) for a training run and generates output that can be viewed inside Nsight Systems, while Nsight Systems independently captures a full system-wide timeline for any CUDA application.

**Why the others are wrong:**
- This reverses the actual roles — Nsight Compute, not Nsight Systems, profiles individual kernels.
- The two tools serve different purposes and are not interchangeable.
- Nsight Systems is framework-agnostic and works with any CUDA application, not just TensorFlow.

</details>

---

