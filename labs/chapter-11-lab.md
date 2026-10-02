# Chapter 11 Lab: Profiling and Optimization: DLProf, Nsight, and Memory Efficiency

**Estimated time:** 20 minutes

## Exercise

Walkthrough: profile a slow RAPIDS pipeline with Nsight Systems to identify a transfer-bound stage, then profile a model training loop with DLProf to identify a GPU-utilization gap

## Key takeaway

Pick Nsight for pipeline/kernel-level bottlenecks and DLProf for deep learning training profiling, and confirm root cause before optimizing.
