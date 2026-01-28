# Protein Mutation Effects

This project predicts the functional impact of single amino acid substitutions in proteins using machine learning and physicochemical feature representations.

The goal is to build a clean, interpretable baseline pipeline integrating biological priors with modern ML workflows.

---

# Motivation

Predicting whether a mutation stabilizes or destabilizes a protein is central to protein engineering and disease biology.

This project investigates whether physicochemical descriptors (AAindex) and delta-encoding strategies (mutant - wild-type) provide useful inductive bias for learning mutation effects, beyond simple categorical mutation representations.

---

# Data

Single-point protein mutations with experimentally measured stability labels.

Preprocessing steps:
- Removal of invalid or ambiguous mutations.
- Stratified train/test split on stability labels.
- Separate wild-type and mutant residue information.

---

# Feature Engineering

- Baseline features:
    - One-hot encoding of wild-type and mutant amino acids.
    - Mutation position.
    - Structural/contextual metadata (B-factor, conservation scores).

- Physicochemical features:
    - AAindex descriptors for wild-type and mutant residues.
    - Delta encoding (mutant - wild-type).

Feature generation is modularized in `src/feature_utils.py`.

---

# Modeling

- Supervised classification model (fully-connected neural network).
- Clean train/validation/test separation.
- Class imbalance handled via class weighting or downsampling.
- Metrics tracked: accuracy, precision, recall, ROC-AUC, and PR.
- Evaluation plots and metrics saved in results/ for reproducibility.

---

# Project Structure

```
protein-mutation-effects/
├── data/
├── notebooks/
│   └── final_analysis.ipynb
├── results/
│   └── figures/
│   └── metrics/
├── src/
│   └── __init__.py
│   └── analysis_utils.py
│   └── config.py
│   └── data_utils.py
│   └── feature_utils.py
│   └── model_utils.py
│   └── train_utils.py
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Usage

1. Install dependencies:
`pip install -r requirements.txt`.

2. Run the final analysis notebook:
`jupyter notebook notebooks/final_analysis.ipynb`.

---

# Notes

- The repository is a baseline reference for protein mutation effect prediction.
- Designed for reproducibility and clarity, not competition-level performance.