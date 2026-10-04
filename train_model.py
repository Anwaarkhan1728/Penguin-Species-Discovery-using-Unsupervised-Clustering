
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score)
from utils import preprocess_dataframe, save_artifact

DATA_PATH = "data/penguins.csv"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading data...")
raw = pd.read_csv(DATA_PATH)
print("Shape:", raw.shape)

df = preprocess_dataframe(raw)
df = df.drop(columns=[c for c in ["year"] if c in df.columns])
print("Cleaned:", df.shape)

raw_clean = raw.dropna().reset_index(drop=True)
y_true = raw_clean["species"].astype("category").cat.codes.values if "species" in raw_clean.columns else None

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

pca_full = PCA().fit(X_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_comp = int(np.argmax(cum_var >= 0.95) + 1)
pca = PCA(n_components=n_comp, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("PCA components:", n_comp)

best_k, best_s = 2, -1
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lb = km.fit_predict(X_pca)
    s = silhouette_score(X_pca, lb)
    if s > best_s:
        best_s, best_k = s, k
print("Best k:", best_k, "sil:", round(best_s,4))

def ev(X, lb, yt):
    mask = lb != -1
    nc = len(set(lb)) - (1 if -1 in lb else 0)
    if nc < 2 or mask.sum() < 2:
        return {"Silhouette": np.nan, "Davies-Bouldin": np.nan,
                "Calinski-Harabasz": np.nan, "ARI": None, "NMI": None}
    return {
        "Silhouette": silhouette_score(X[mask], lb[mask]),
        "Davies-Bouldin": davies_bouldin_score(X[mask], lb[mask]),
        "Calinski-Harabasz": calinski_harabasz_score(X[mask], lb[mask]),
        "ARI": adjusted_rand_score(yt[mask], lb[mask]) if yt is not None else None,
        "NMI": normalized_mutual_info_score(yt[mask], lb[mask]) if yt is not None else None,
    }

km = KMeans(n_clusters=best_k, random_state=42, n_init=10); lb_km = km.fit_predict(X_pca)
agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward"); lb_agg = agg.fit_predict(X_pca)
db = DBSCAN(eps=0.7, min_samples=5); lb_db = db.fit_predict(X_pca)

m_km = ev(X_pca, lb_km, y_true); m_km["Model"] = "KMeans"
m_agg = ev(X_pca, lb_agg, y_true); m_agg["Model"] = "Agglomerative"
m_db = ev(X_pca, lb_db, y_true); m_db["Model"] = "DBSCAN"

results = pd.DataFrame([m_km, m_agg, m_db])
print(results.to_string(index=False))

best_row = results.dropna(subset=["Silhouette"]).sort_values("Silhouette", ascending=False).iloc[0]
best_name = best_row["Model"]
print("Best model:", best_name)

model_lookup = {"KMeans": km, "Agglomerative": agg, "DBSCAN": db}
best_model = model_lookup[best_name]

save_artifact(os.path.join(OUT_DIR, "penguin_clustering_best.pkl"), {
    "model_name": best_name, "model": best_model,
    "scaler": scaler, "pca": pca,
    "feature_columns": df.columns.tolist(),
    "n_clusters": int(best_k), "n_components": int(n_comp),
    "metrics": best_row.to_dict(),
})

for name, mdl in model_lookup.items():
    save_artifact(os.path.join(OUT_DIR, f"penguin_{name.lower()}.pkl"),
                  {"model": mdl, "scaler": scaler, "pca": pca})

save_artifact(os.path.join(OUT_DIR, "metadata.pkl"),
              {"comparison": results, "best_k": int(best_k),
               "n_components": int(n_comp),
               "feature_columns": df.columns.tolist()})

print("Saved:", os.listdir(OUT_DIR))
