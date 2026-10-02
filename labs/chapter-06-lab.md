# Chapter 6 Lab: ETL Design and Multi-GPU Scaling with Dask-cuDF

**Estimated time:** 25 minutes

## Exercise

Walkthrough: launch a LocalCUDACluster across multiple GPUs, load a partitioned dataset with dask_cudf, and run a distributed groupby, comparing against single-GPU cuDF

## Key takeaway

dask-cuDF scales cuDF across GPUs, but partition sizing and shuffle-heavy operations decide whether that scaling actually pays off.
