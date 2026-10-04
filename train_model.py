"""
train_model.py — Trains KMeans, Agglomerative, DBSCAN and saves best model.

Run:
    python train_model.py --data data/penguins.csv --out models
"""

import argparse
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score,
    normalized_mutual_info_score,
)

from utils import (
    FEATURE_COLUMNS,
    preprocess_dataframe,
    save_artifact,
)


# ==========================================================
def find_best_k(X, k_range=range(2, 11)):
    best_k, best_score = 2, -1
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        s = silhouette_score(X, labels)
        scores[k] = s
        if s > best_score:
            best_score, best_k = s, k
    return best_k, scores


def evaluate(X, labels, y_true=None):
    mask = labels != -1  # ignore DBSCAN noise
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2 or mask.sum() < 2:
        return {"Silhouette": np.nan, "Davies-Bouldin": np.nan,
                "Calinski-Harabasz": np.nan, "ARI": None, "NMI": None}
    m = {
        "Silhouette": silhouette_score(X[mask], labels[mask]),
        "Davies-Bouldin": davies_bouldin_score(X[mask], labels[mask]),
        "Calinski-Harabasz": calinski_harabasz_score(X[mask], labels[mask]),
        "ARI": adjusted_rand_score(y_true[mask], labels[mask]) if y_true is not None else None,
        "NMI": normalized_mutual_info_score(y_true[mask], labels[mask]) if y_true is not None else None,
    }
    return m


# ==========================================================
def main(args):
    print("🔹 Loading data ...")
    raw_df = pd.read_csv(args.data)
    print("Raw shape:", raw_df.shape)

    # Save ground truth for evaluation
    y_true = None
    if "species" in raw_df.columns:
        y_true = raw_df["species"].astype("category").cat.codes.values

    # Preprocess
    df = preprocess_dataframe(raw_df)
    # Remove 'year' if present (not useful for clustering)
    df = df.drop(columns=[c for c in ["year"] if c in df.columns])
    print("Cleaned shape:", df.shape)
    print("Columns used:", df.columns.tolist())

    # Align y_true length with cleaned df
    # (we already dropped NaNs inside preprocess; simplest: re-drop)
    raw = raw_df.dropna().reset_index(drop=True)
    y_true = raw["species"].astype("category").cat.codes.values if "species" in raw.columns else None

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # PCA (95% variance)
    pca_full = PCA().fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.argmax(cum_var >= 0.95) + 1)
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA components kept: {n_comp}")

    # Find best k
    best_k, _ = find_best_k(X_pca)
    print("Best k:", best_k)

    # =========================================================
    # Train 3 models
    # =========================================================
    print("\n🔹 Training KMeans ...")
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels_km = km.fit_predict(X_pca)
    metrics_km = evaluate(X_pca, labels_km, y_true)
    metrics_km["Model"] = "KMeans"

    print("🔹 Training Agglomerative ...")
    agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    labels_agg = agg.fit_predict(X_pca)
    metrics_agg = evaluate(X_pca, labels_agg, y_true)
    metrics_agg["Model"] = "Agglomerative"

    print("🔹 Training DBSCAN ...")
    db = DBSCAN(eps=args.eps, min_samples=args.min_samples)
    labels_db = db.fit_predict(X_pca)
    metrics_db = evaluate(X_pca, labels_db, y_true)
    metrics_db["Model"] = "DBSCAN"

    # Compare
    results = pd.DataFrame([metrics_km, metrics_agg, metrics_db])
    print("\n📊 Model Comparison:\n", results.to_string(index=False))

    # Pick best (by Silhouette)
    results_valid = results.dropna(subset=["Silhouette"])
    best_row = results_valid.sort_values("Silhouette", ascending=False).iloc[0]
    best_name = best_row["Model"]
    print(f"\n🏆 Best model: {best_name} (silhouette={best_row['Silhouette']:.4f})")

    model_lookup = {
        "KMeans": km,
        "Agglomerative": agg,
        "DBSCAN": db,
    }
    best_model = model_lookup[best_name]

    # =========================================================
    # Save artifacts
    # =========================================================
    os.makedirs(args.out, exist_ok=True)

    # Save full "best" artifact
    artifact = {
        "model_name": best_name,
        "model": best_model,
        "scaler": scaler,
        "pca": pca,
        "feature_columns": df.columns.tolist(),
        "n_clusters": int(best_k),
        "n_components": int(n_comp),
        "metrics": best_row.to_dict(),
    }
    save_artifact(os.path.join(args.out, "penguin_clustering_best.pkl"), artifact)

    # Save each model individually
    for name, mdl in model_lookup.items():
        save_artifact(
            os.path.join(args.out, f"penguin_{name.lower()}.pkl"),
            {"model": mdl, "scaler": scaler, "pca": pca},
        )

    # Save metadata / comparison
    save_artifact(
        os.path.join(args.out, "metadata.pkl"),
        {
            "comparison": results,
            "best_k": int(best_k),
            "n_components": int(n_comp),
            "feature_columns": df.columns.tolist(),
        },
    )

    print(f"\n✅ Saved all artifacts to: {args.out}")
    for f in sorted(os.listdir(args.out)):
        print("   •", f)


# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/penguins.csv")
    parser.add_argument("--out", type=str, default="models")
    parser.add_argument("--eps", type=float, default=0.7)
    parser.add_argument("--min_samples", type=int, default=5)
    args = parser.parse_args()
    main(args)
