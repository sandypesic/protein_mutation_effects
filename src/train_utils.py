import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def train_test_split_stratified_df(df: pd.DataFrame, target_col: str, test_size=0.2, random_state=42):
    """
    ----
    Split a DataFrame into stratified train/test sets based on target_col.
    ----
    Return df_train, df_test (both as DataFrames).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y))

    df_train = df.iloc[train_idx]
    df_test  = df.iloc[test_idx]

    return df_train, df_test