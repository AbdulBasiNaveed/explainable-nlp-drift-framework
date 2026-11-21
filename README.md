# An Explainable Machine Learning Framework for Detecting and Managing Data Drift in Predictive NLP Models

**Author:** Abdul Basit Naveed (DS-035/2025)  
**Supervisor:** Dr. Muhammad Umer Farooq  
**Institution:** Department of Computer Science & Information Technology, NED University of Engineering and Technology, Karachi

## Overview
Implementation code for detecting, quantifying, explaining, and managing temporal data drift in fine-tuned DistilBERT sentiment classifiers using Kolmogorov-Smirnov (KS) tests, Population Stability Index (PSI), ADWIN, a composite severity score ($S$), and PartitionSHAP with ablation validation.

## Environment Setup
Tested on **Python 3.10.12**. Install dependencies via:
```bash
pip install -r requirements.txt

## Notebook Execution Order (`seed = 42`)
1. `01_Data_Loading_Exploration.ipynb` — Loads and cleans the Amazon Electronics Reviews dataset (1999–2023).
2. `02_Temporal_Split_Baseline_Models.ipynb` — Creates the chronological 2018 split and evaluates Logistic Regression (**74.97%**) and Random Forest (**83.42%**) baselines.
3. `03_DistilBERT_FineTuning.ipynb` — Fine-tunes `distilbert-base-uncased` on pre-2019 data (Baseline Accuracy: **88.00%**).
4. `03_Drift_Detection.ipynb` — Computes annual KS feature drift, 10-bin softmax PSI (`0.0068` to `0.1453`), and River ADWIN (`delta=0.002`) signals (2019–2023).
5. `04_SHAP_Explainability.ipynb` — Runs PartitionSHAP on 100 samples/year and performs top-5 vs. random-5 token masking ablation (**8.90x** mean accuracy drop ratio).
6. `05_SHAP_Explanation_Management.ipynb` — Computes composite severity scores ($S = 0.257$ in 2019 $\rightarrow 0.588$ in 2023) and maps cohorts to *Monitor*, *Partial Retrain*, or *Full Retrain* tiers.
7. `06_Evaluation.ipynb` — Runs statistical significance tests (Wilcoxon, McNemar, one-sample $t$-test) and computational cost benchmarks.
