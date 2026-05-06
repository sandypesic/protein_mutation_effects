# Protein Mutation Effects

A machine learning pipeline for predicting the stability impact of single amino acid substitutions in proteins, using physicochemical feature representations and experimentally measured ΔΔG labels.

---

## Motivation

Predicting whether a mutation stabilizes or destabilizes a protein is central to protein engineering and disease biology. Random mutations are overwhelmingly destabilizing in nature, but curated experimental databases like FireProtDB reflect the opposite distribution, as researchers preferentially study and publish stabilizing mutations. This selection bias is an important consideration when interpreting model performance and generalizability.

This project investigates whether physicochemical descriptors (AAindex) and delta-encoding strategies (mutant - wild-type) provide useful inductive bias for learning mutation effects, beyond simple categorical mutation representations.

---

## Data

Source: [FireProtDB](https://loschmidt.chemi.muni.cz/fireprotdb/) — a curated database of experimentally measured protein stability changes for single-point mutations.

- ~412,000 single-site substitutions after filtering
- Binary stability label derived from ΔΔG (negative = stabilizing)
- Class distribution: ~78% stabilizing, ~22% destabilizing (reflects experimental selection bias, not the broader mutational landscape)

Preprocessing steps:
- Filtering to well-formed single-site substitutions
- Stratified train/test split (80/20) preserving class ratio
- Class imbalance handled via class weighting (preferred) or downsampling

---

## Feature Engineering

**Baseline features:**
- One-hot encoding of wild-type and mutant amino acids
- Mutation position
- Structural/contextual metadata: B-factor, conservation scores

**Physicochemical features (AAindex):**
- Five descriptors selected for biological relevance: hydrophobicity (KYTJ820101), secondary structure propensity (FAUJ880111), volume (CHAM810101), polarity (GRAR740102), hydrogen bonding (HOPT810101)
- Delta encoding (mutant − wild-type) to capture the direction and magnitude of physicochemical change

Feature generation is modularized in `src/feature_utils.py`.

---

## Modeling

A fully-connected neural network (128 → 64 → 1) with batch normalization, dropout regularization, and adaptive learning rate scheduling. A neural network was chosen over gradient boosting to capture potential nonlinear interactions between physicochemical features — an assumption validated by the performance gap over logistic regression.

**Results:**

| Model | AUC |
|---|---|
| Neural Network (this project) | 0.7217 |
| Random Forest (baseline) | 0.7213 |
| Logistic Regression (baseline) | 0.6975 |
| Majority Class (floor) | 0.500 |

The neural network outperforms all baselines. The narrow margin over Random Forest is expected on tabular data with hand-crafted features; the more meaningful result is the gap over logistic regression suggesting the model captures nonlinear physicochemical interactions that a linear model cannot.

---

## Methods Rationale

- Train/test split is randomized across mutations rather than proteins; with only 186 unique proteins in FireProtDB, a protein-aware split produced insufficient test samples (~1,400 vs ~82,000) due to high variance in mutations per protein (median 9, mean 51); a protein-aware evaluation would be more rigorous but requires a dataset with greater protein diversity
- Ablation study results suggest that one-hot encodings and structural metadata (B-factor, conservation) drive the majority of predictive signal; AAindex physicochemical delta features contribute marginally to the full model, suggesting redundancy with the categorical amino acid encodings; this motivates future work incorporating richer structural representations (e.g. ESM embeddings) that may capture signal not encoded by identity alone
- Permutation importance analysis reveals that structural context features — B-factor (local flexibility) and conservation score (evolutionary constraint) — contribute the most predictive signal, suggesting that *where* a mutation occurs matters more than *what* the substitution is; this motivates future incorporation of richer positional features such as solvent accessibility and contact number

___

## Project Structure

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

## Usage

1. Install dependencies: `pip install -r requirements.txt`.

2. TensorFlow installation:
    1. **Standard installation (Windows/Linux/Intel Macs):** `pip install tensorflow==2.16.2`.
    2. **Apple Silicon (M-series Macs):** `pip install tensorflow-macos tensorflow-metal`.
        1. `tensorflow-metal` enables GPU acceleration through Apple's Metal backend.
        2. See [Apple TensorFlow guide](https://developer.apple.com/metal/tensorflow-plugin/).

3. Run the final analysis notebook: `jupyter notebook notebooks/final_analysis.ipynb`.

---

## Limitations & Future Work

- Features are sequence- and structure-agnostic beyond B-factor and conservation — incorporating 3D structural context (e.g. via ESM embeddings or graph neural networks) would likely improve performance meaningfully
- The dataset reflects experimental selection bias; performance on a more representative mutation distribution is unknown
- Hyperparameter search was not performed; systematic tuning would be a natural next step