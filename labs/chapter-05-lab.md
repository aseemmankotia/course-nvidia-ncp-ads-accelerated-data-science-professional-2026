# Chapter 5 Lab: cuDF Interoperability, Synthetic Data, and Pipeline Efficiency

**Estimated time:** 20 minutes

## Exercise

Walkthrough: build a cuDF pipeline that generates synthetic transaction data, benchmark a full GPU pipeline against one with an unnecessary to_pandas() round trip in the middle

## Key takeaway

Every pandas round trip has a real transfer cost; keep data on the GPU end-to-end and prune columns early.
