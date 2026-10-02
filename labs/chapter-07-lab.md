# Chapter 7 Lab: Out-of-Core Workflows and Efficient Parquet/ORC I/O

**Estimated time:** 22 minutes

## Exercise

Walkthrough: read a multi-GB partitioned Parquet dataset larger than a single GPU's memory using dask_cudf with spilling enabled, and inspect memory spill logs

## Key takeaway

When data exceeds GPU memory, use dask_cudf with spilling and format-aware I/O rather than forcing a single-GPU load.
