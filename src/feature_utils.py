import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ----
    Build simple baseline features.
    ----
    One-hot encode wild-type and mutant amino acids.
    Basic numeric position- and structure-related features.
    """

    # Numeric features with missing values treated as neutral.
    numeric = (
        df[['pos', 'CONSERVATION_clean', 'B_FACTOR_clean']]
        .fillna(0)
    )

    # One-hot encode amino acid identities.
    wt = pd.get_dummies(df['WT'], prefix='WT').astype(int)
    mut = pd.get_dummies(df['mutant'], prefix='mutant').astype(int)

    return pd.concat([wt, mut, numeric], axis=1)

# Standard 20 amino acids in AAindex ordering.
AA_LIST = list("ARNDCQEGHILKMFPSTWYV")

def load_aaindex_safe(path: str) -> dict:
    """
    ----
    Parse an AAindex file into a dictionary.
    ----
    Form:
    {
        index_id: {AA: value, ...}
    }
    The parser is intentionally defensive:
        1) Skip malformed entries.
        2) Ignore non-numeric tokens.
        3) Only keep indices with complete 20-AA vectors.
    """
    
    aaindex = {}

    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Header line marks a new AAindex entry.
        if line.startswith("H "):
            entry_id = line.split()[1]
            nums = []

            # Skip to the numeric data block.
            i += 1
            while i < len(lines) and not lines[i].startswith("I "):
                i += 1

            # Read numeric values until end-of-entry marker.
            i += 1
            while i < len(lines) and not lines[i].startswith("//"):
                parts = lines[i].replace(",", " ").split()
                for token in parts:
                    try:
                        nums.append(float(token))
                    except ValueError:
                        continue
                i += 1

            # Keep only well-formed AAindex entries.
            if len(nums) == 20:
                aaindex[entry_id] = dict(zip(AA_LIST, nums))

        i += 1

    return aaindex

def aaindex_features(
    df: pd.DataFrame,
    aaindex: dict,
    index_ids: list
) -> pd.DataFrame:
    """
    ----
    Construct AAindex-based features for a set of indices.
    ----
    For each index:
        1) WT value.
        2) Mutant value.
        3) Delta (mutant - WT).
    """

    blocks = []

    for idx in index_ids:
        if idx not in aaindex:
            print(f"Warning: AAindex {idx} not found, skipping.")
            continue

        table = aaindex[idx]

        wt_vals = df['WT'].map(lambda aa: table.get(aa, np.nan))
        mut_vals = df['mutant'].map(lambda aa: table.get(aa, np.nan))

        block = pd.DataFrame({
            f"{idx}_wt": wt_vals,
            f"{idx}_mut": mut_vals,
            f"{idx}_delta": mut_vals - wt_vals,
        })

        blocks.append(block)

    return pd.concat(blocks, axis=1)

def build_aaindex_features_raw(
    df: pd.DataFrame,
    aaindex_path: str,
    index_ids: list
) -> pd.DataFrame:
    """
    ----
    Build AAindex-based features without scaling.
    ----
    This function is pure and safe to use on any split.
    """
    aaindex = load_aaindex_safe(aaindex_path)
    feats = aaindex_features(df, aaindex, index_ids)

    # Missing AAindex values treated as neutral.
    return feats.fillna(0)

def fit_feature_scaler(X: pd.DataFrame) -> StandardScaler:
    """
    ----
    Fit a StandardScaler on training features only.
    ----
    """
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

def apply_feature_scaler(
    X: pd.DataFrame,
    scaler: StandardScaler
) -> pd.DataFrame:
    """
    ----
    Apply a fitted scaler to features.
    ----
    """
    X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)