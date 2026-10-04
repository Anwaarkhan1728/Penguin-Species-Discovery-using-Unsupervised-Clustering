
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder

FEATURE_COLUMNS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_COLUMNS = ["island", "sex"]
CLUSTER_COLORS = {0:"#1f77b4", 1:"#ff7f0e", 2:"#2ca02c", 3:"#d62728", 4:"#9467bd", -1:"#7f7f7f"}

def preprocess_dataframe(df, feature_cols=None, categorical_cols=None):
    feature_cols = feature_cols or FEATURE_COLUMNS
    categorical_cols = categorical_cols or CATEGORICAL_COLUMNS
    df = df.copy()
    for col in ["species", "Unnamed: 0"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df = df.dropna().reset_index(drop=True)
    for col in categorical_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df.select_dtypes(include=[np.number])

def prepare_input(feature_dict, feature_cols, scaler, pca):
    X = pd.DataFrame([feature_dict])[feature_cols]
    X_scaled = scaler.transform(X)
    return pca.transform(X_scaled)

def save_artifact(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_artifact(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        return pickle.load(f)

def cluster_summary(df, labels, feature_cols):
    tmp = df[feature_cols].copy()
    tmp["cluster"] = labels
    return tmp.groupby("cluster").mean().round(2)
