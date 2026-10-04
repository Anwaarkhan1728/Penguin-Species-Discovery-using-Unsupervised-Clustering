"""
utils.py — Helper functions for Penguin Clustering App
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA


# ==========================================================
# Constants
# ==========================================================
FEATURE_COLUMNS = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

CATEGORICAL_COLUMNS = ["island", "sex"]

CLUSTER_COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
    -1: "#7f7f7f",   # DBSCAN noise
}


# ==========================================================
# Preprocessing
# ==========================================================
def preprocess_dataframe(df: pd.DataFrame,
                         feature_cols=None,
                         categorical_cols=None) -> pd.DataFrame:
    """
    Clean and prepare raw penguin dataframe:
      - Drop missing values
      - Encode categorical features
      - Return only numeric columns
    """
    feature_cols = feature_cols or FEATURE_COLUMNS
    categorical_cols = categorical_cols or CATEGORICAL_COLUMNS

    df = df.copy()

    # Drop target column if present (we don't need it for clustering)
    for col in ["species", "Unnamed: 0"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Drop rows with missing values
    df = df.dropna().reset_index(drop=True)

    # Encode categoricals
    for col in categorical_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Keep only numeric
    df = df.select_dtypes(include=[np.number])

    return df


def prepare_input(feature_dict: dict,
                  feature_cols,
                  scaler,
                  pca) -> np.ndarray:
    """
    Take a dict of user inputs, scale and reduce dimensions.
    """
    X = pd.DataFrame([feature_dict])[feature_cols]
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return X_pca


# ==========================================================
# IO helpers
# ==========================================================
def save_artifact(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_artifact(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ==========================================================
# Formatting
# ==========================================================
def format_metrics(md: dict) -> pd.DataFrame:
    """Convert metrics dict into a clean dataframe."""
    rows = []
    for k, v in md.items():
        if isinstance(v, float):
            rows.append({"Metric": k, "Value": f"{v:.4f}"})
        else:
            rows.append({"Metric": k, "Value": v})
    return pd.DataFrame(rows)


def cluster_summary(df: pd.DataFrame, labels, feature_cols) -> pd.DataFrame:
    """Return mean of features per cluster."""
    tmp = df[feature_cols].copy()
    tmp["cluster"] = labels
    return tmp.groupby("cluster").mean().round(2)
