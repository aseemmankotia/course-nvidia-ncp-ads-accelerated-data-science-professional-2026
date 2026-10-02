## Chapter 12 Cheat Sheet — Full Exam Simulation & Time Management

### Pacing Formula
| Exam Length | Question Count | Time/Question | Review Buffer |
|---|---|---|---|
| 120 min | ~65-70 | 1.5-2 min | Last 5-10 min |

Rule: If a question exceeds ~2.5 min of deliberation, flag your best guess and move on.

### Recurring Tool Decision Pairs
| Decision | Choose Left When... | Choose Right When... |
|---|---|---|
| pandas vs cuDF | Small data, no GPU, exploration | Large tabular data fits single-GPU memory |
| Single-GPU cuDF vs dask-cuDF | Data fits in one GPU's memory | Data exceeds GPU memory or spans GPUs/nodes (spilling/partitioning) |
| cuML vs scikit-learn | Large data, GPU speedup matters | Small/medium data, CPU-only, broad algorithm support/interpretability |
| GPU XGBoost vs CPU XGBoost | Large row counts, fastest training needed | No GPU available or dataset too small to benefit |

### Distractor Elimination Checklist
1. Does the option ignore an explicit hardware constraint (e.g., GPU memory size)? → Eliminate.
2. Does the option use a deprecated/incorrect API (e.g., 'gpu_hist' alone)? → Eliminate.
3. Does the option add unnecessary complexity for the stated data scale? → Eliminate.
4. Does the remaining option match BOTH the scale AND the hardware given in the stem? → Likely correct.

### Keyword Cues in Question Stems
| Keyword | What It Signals |
|---|---|
| BEST | Multiple valid options exist; pick the most optimal for stated constraints |
| MOST efficient | Focus on compute/time performance, not just correctness |
| MINIMIZE cost | Cost becomes the deciding factor over raw performance |
| REQUIRES | Look for a hard constraint that eliminates most options immediately |

### Final Review Pass Strategy
1. Revisit all flagged questions first.
2. Do one fast pass per domain on highest-yield facts (tool decision pairs, key syntax, limits).
3. Do NOT reread every answered question — this wastes remaining time.

### Post-Exam Next Steps
- Certification validity: **2 years** from issue date; recertify with the then-current exam version before expiry.
- Confirm current exam fee, retake policy, and scheduling rules on NVIDIA's official certification portal **before** registering — these can change.
- Save your score report and digital badge immediately after passing for portfolio/LinkedIn use.
