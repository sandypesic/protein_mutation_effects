import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.utils import compute_class_weight

def parse_numeric_list(x):
    """
    ----
    Parse a comma-separated numeric string and return its mean.
    ----
    Used for columns like CONSERVATION and B_FACTOR that may store lists of values as strings.
    """
    if pd.isna(x):
        return np.nan
    try:
        parts = [float(v) for v in str(x).split(",") if v.strip()]
        return np.mean(parts) if parts else np.nan
    except ValueError:
        return np.nan

def load_and_clean_data(
    csv_path,
    needed_cols,
    use_sample=False,
    sample_size=None,
    processed_path=None,
    random_state=42,
    verbose=True
):
    """
    ----
    Load and clean protein mutation ΔΔG data.
    ----
    Steps:
        1) Load required columns from CSV.
        2) Filter to single-site substitutions.
        3) Parse numeric ΔΔG values.
        4) Derive WT, mutant, position, and stabilizing label.
        5) Clean CONSERVATION and B_FACTOR columns.
        6) Optionally subsample for faster experimentation
    Return cleaned pandas DataFrame.
    """

    # Load cached processed file if available.
    if use_sample and processed_path and processed_path.exists():
        df = pd.read_csv(processed_path)
        if verbose:
            print(f"Loaded cached dataset: {processed_path}")
            print("Stabilizing label distribution:")
            print(df['stabilizing'].value_counts(normalize=True))
        return df

    df = pd.read_csv(csv_path, usecols=lambda c: c in needed_cols)

    # Keep only single-site substitutions.
    mask = df['SUBSTITUTION'].astype(str).str.match(r'^[A-Z]\d+[A-Z]$')
    df = df[mask].copy()

    # Parse ΔΔG values.
    df['DDG'] = pd.to_numeric(df['DDG'], errors='coerce')
    df = df[df['DDG'].notna()].copy()

    # Extract mutation components.
    df['pos'] = df['SUBSTITUTION'].str.extract(r'^[A-Z](\d+)[A-Z]$')[0].astype(int)
    df['WT'] = df['SUBSTITUTION'].str[0]
    df['mutant'] = df['SUBSTITUTION'].str[-1]

    # Binary classification target.
    df['stabilizing'] = df['DDG'] < 0

    # Clean numeric list-like columns.
    for col in ['CONSERVATION', 'B_FACTOR']:
        df[col + '_clean'] = df[col].apply(parse_numeric_list)

    # Optional subsampling.
    if use_sample and sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=random_state)

    df = df.reset_index(drop=True)

    if verbose:
        print(f"Final dataset size: {len(df)}")
        print("Stabilizing label distribution:")
        print(df['stabilizing'].value_counts(normalize=True))

    return df

def apply_downsampling(df, random_state=42):
    """
    ----
    Downsample the majority class to match the minority class size.
    ----
    Used as an alternative to class weighting.
    """

    df_major = df[df['stabilizing'] == False]
    df_minor = df[df['stabilizing'] == True]

    if len(df_major) > len(df_minor):
        df_major_down = df_major.sample(n=len(df_minor), random_state=random_state)
        df_balanced = pd.concat([df_major_down, df_minor])
    else:
        df_minor_down = df_minor.sample(n=len(df_major), random_state=random_state)
        df_balanced = pd.concat([df_major, df_minor_down])

    return df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

def compute_class_weights(y):
    """
    ----
    Compute balanced class weights for binary classification.
    ----
    Useful for handling class imbalance without downsampling.
    """
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )
    return dict(zip(classes, weights))