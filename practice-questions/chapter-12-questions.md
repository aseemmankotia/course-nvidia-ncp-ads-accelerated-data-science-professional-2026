## Chapter 12: Full Exam Simulation and Time-Management Strategy — Practice Questions

**Q1.** A candidate must process a 200GB tabular dataset on a workstation with a single NVIDIA A100 40GB GPU. Which approach BEST meets this requirement?

A) Load the full dataset into pandas and rely on CPU RAM
B) Load the full dataset directly into cuDF on the single GPU
C) Use dask-cuDF with GPU memory spilling to disk/host across partitions
D) Use cuML directly on the raw CSV without a dataframe library

<details>
<summary>Answer</summary>

**Correct: C**

200GB exceeds the 40GB GPU memory, so dask-cuDF partitions the data and spills excess partitions to host memory/disk, allowing GPU-accelerated processing without an out-of-memory failure.

**Why the others are wrong:**
- pandas would work but abandons GPU acceleration entirely, failing the 'BEST' efficiency requirement.
- Plain cuDF has no spilling mechanism and will throw an out-of-memory error once GPU memory is exceeded.
- cuML is a modeling library, not a dataframe/ETL tool, so it cannot load or manage the raw dataset.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** You have 66 questions to complete in 120 minutes on the exam. Which pacing strategy is MOST appropriate?

A) Spend as long as needed on each question in order, since there is no partial credit
B) Budget roughly 1.5-2 minutes per question, flag uncertain ones, and reserve the final minutes to review flagged items
C) Answer only the questions you are 100% certain about and leave the rest blank
D) Skip all scenario-based questions first and answer them last regardless of remaining time

<details>
<summary>Answer</summary>

**Correct: B**

120 minutes across ~66 questions gives roughly 1.5-2 minutes each; flagging uncertain questions and reserving a review buffer maximizes the number of confidently-answered questions before time runs out.

**Why the others are wrong:**
- Spending unlimited time per question risks running out of time before reaching later questions.
- Leaving questions blank guarantees a zero score on them versus a chance at partial success from elimination.
- Arbitrarily deferring an entire question type ignores per-question time budgeting and risks a rushed final stretch.

</details>

---

**Q3.** A question stem reads: 'Which approach is the MOST efficient way to train a gradient boosting model on 50 million rows?' What does the word 'MOST' signal to the test-taker?

A) Any technically correct answer can be selected, since MOST is filler language
B) Multiple options may be valid, but one option is optimized further than the others for the stated constraint
C) The question is asking for the least expensive option regardless of performance
D) The question requires selecting two answers that together are efficient

<details>
<summary>Answer</summary>

**Correct: B**

Superlative cues like 'MOST' or 'BEST' indicate several distractors are technically valid, but only one option is optimal for the specific scale/constraint described in the stem.

**Why the others are wrong:**
- Superlatives are never filler — they explicitly narrow the correct choice among plausible options.
- Efficiency in these stems typically refers to compute/time performance, not cost, unless cost is explicitly stated.
- Every exam question in this course format has exactly one correct answer; superlative wording does not imply multi-select.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q4.** A team has a 5,000-row labeled dataset and needs a highly interpretable classification model with broad community library support. Which tool is the BEST fit?

A) cuML on GPU
B) dask-cuDF with cuML
C) XGBoost with device='cuda'
D) scikit-learn on CPU

<details>
<summary>Answer</summary>

**Correct: D**

At 5,000 rows the dataset is too small to benefit from GPU acceleration; the data transfer and kernel-launch overhead of cuML/GPU tools outweighs any speedup, so CPU-based scikit-learn is the practical and BEST choice, especially given the interpretability and ecosystem requirement.

**Why the others are wrong:**
- cuML's GPU acceleration provides no meaningful benefit at this small scale and adds unnecessary overhead.
- dask-cuDF is designed for distributed/large-scale data and is unnecessary overhead for 5,000 rows.
- GPU XGBoost is a boosting library, not a general interpretable-model toolkit with broad library support like scikit-learn.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q5.** A dataset of 10 million rows requires the fastest possible gradient boosting training time on available hardware. Which configuration is the BEST choice?

A) XGBoost with tree_method='hist' on CPU
B) scikit-learn GradientBoostingClassifier
C) XGBoost with tree_method='hist', device='cuda'
D) cuML DBSCAN

<details>
<summary>Answer</summary>

**Correct: C**

For large row counts, GPU-accelerated histogram-based tree construction (tree_method='hist' with device='cuda') dramatically reduces training time compared to CPU-only execution.

**Why the others are wrong:**
- CPU-only hist mode is correct algorithmically but ignores available GPU acceleration, so it is not the fastest option.
- scikit-learn's gradient boosting implementation has no native GPU support and scales poorly at 10 million rows.
- DBSCAN is a clustering algorithm, not a gradient boosting method, so it does not satisfy the requirement at all.

</details>

---

**Q6.** During the exam you encounter a scenario question you are unsure about. What is the recommended action?

A) Guess immediately and move on without marking the question
B) Stop the exam timer mentally and research the concept before answering
C) Leave the question blank and return to it only if time allows
D) Flag the question, select your best current guess, and move on to avoid losing time

<details>
<summary>Answer</summary>

**Correct: D**

Flagging with a best guess locked in protects your score even if you run out of time to revisit it, while still allowing a focused review pass later.

**Why the others are wrong:**
- Guessing without flagging forfeits the chance to revisit the question with fresh eyes during the review pass.
- There is no ability to pause exam time for research; this describes an action not available during a proctored exam.
- Leaving it blank risks losing the answer entirely if time runs out before returning to it, unlike a locked-in guess.

</details>

---

**Q7.** How long is an NVIDIA Certified Professional credential typically valid before renewal is required?

A) One year
B) Two years
C) Three years
D) The credential never expires

<details>
<summary>Answer</summary>

**Correct: B**

NVIDIA certifications, including NCP-level credentials, carry a two-year validity period, after which the credential holder must recertify to keep the credential active.

**Why the others are wrong:**
- One year is shorter than NVIDIA's stated renewal cycle for this credential tier.
- Three years overstates the actual renewal window for NVIDIA certifications.
- NVIDIA certifications are not permanent; they require periodic renewal to reflect current platform knowledge.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q8.** Before scheduling the exam, what should a candidate do regarding the exam fee and policies?

A) Assume the fee and policies are fixed and identical to what was true last year
B) Check NVIDIA's official certification portal for the current fee, retake policy, and scheduling terms
C) Rely solely on third-party training providers for pricing information
D) Skip fee verification since NVIDIA certifications are always free

<details>
<summary>Answer</summary>

**Correct: B**

Certification fees and policies (retake windows, rescheduling rules) can change over time, so the only reliable source is NVIDIA's official certification page checked immediately before registering.

**Why the others are wrong:**
- Fees and policies are subject to change, so assuming prior-year terms still apply can lead to scheduling or payment errors.
- Third-party providers may have outdated or inaccurate pricing; only the official NVIDIA source is authoritative.
- NVIDIA certification exams are not free; a fee applies and must be confirmed before scheduling.

</details>

---

