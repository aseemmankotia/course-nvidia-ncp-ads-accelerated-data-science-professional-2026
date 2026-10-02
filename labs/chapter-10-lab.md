# Chapter 10 Lab: cuML, GPU-Accelerated XGBoost, and Rapid Hyperparameter Tuning

**Estimated time:** 24 minutes

## Exercise

Walkthrough: train a cuML RandomForestClassifier and a GPU XGBoost model directly on a cuDF DataFrame, comparing training time to a scikit-learn/CPU-XGBoost baseline

## Key takeaway

Train directly on cuDF-resident data with cuML or GPU XGBoost; any CPU conversion in between erases the acceleration.
