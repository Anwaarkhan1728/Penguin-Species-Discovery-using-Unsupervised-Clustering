# ==========================================================
# app.py — Penguin Clustering (Self-Contained)
# ==========================================================
import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score,
    normalized_mutual_info_score,
)

# ==========================================================
# Config
# ==========================================================
st.set_page_config(
    page_title="Penguin Clustering App",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_COLUMNS = ["bill_length_mm", "bill_depth_mm",
                   "flipper_length_mm", "body_mass_g"]
CATEGORICAL_COLUMNS = ["island", "sex"]

# Paths — try multiple locations for Streamlit Cloud + local
CANDIDATE_DATA_PATHS = [
    "data/penguins.csv",
    "penguins.csv",
    "/mount/src/penguin-species-discovery-using-unsupervised-clustering/data/penguins.csv",
    "/kaggle/input/datasets/anwarkhanniazi/unsupervised-clustering/penguins.csv",
]

MODELS_DIR = "models"
BEST_PKL = os.path.join(MODELS_DIR, "penguin_clustering_best.pkl")


# ==========================================================
# Helpers
# ==========================================================
def find_data_file():
    for p in CANDIDATE_DATA_PATHS:
        if os.path.exists(p):
            return p
    return None


def preprocess(df):
    df = df.copy()
    for col in ["species", "Unnamed: 0"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df = df.dropna().reset_index(drop=True)
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df.select_dtypes(include=[np.number])


def evaluate(X, labels, y_true=None):
    mask = labels != -1
    nc = len(set(labels)) - (1 if -1 in labels else 0)
    if nc < 2 or mask.sum() < 2:
        return {"Silhouette": np.nan, "Davies-Bouldin": np.nan,
                "Calinski-Harabasz": np.nan, "ARI": None, "NMI": None}
    return {
        "Silhouette": silhouette_score(X[mask], labels[mask]),
        "Davies-Bouldin": davies_bouldin_score(X[mask], labels[mask]),
        "Calinski-Harabasz": calinski_harabasz_score(X[mask], labels[mask]),
        "ARI": adjusted_rand_score(y_true[mask], labels[mask]) if y_true is not None else None,
        "NMI": normalized_mutual_info_score(y_true[mask], labels[mask]) if y_true is not None else None,
    }


# ==========================================================
# Train (in-memory, cached)
# ==========================================================
@st.cache_resource(show_spinner="🐧 Training models (first run only)...")
def train_models(data_path):
    raw = pd.read_csv(data_path)

    raw_clean = raw.dropna().reset_index(drop=True)
    y_true = (raw_clean["species"].astype("category").cat.codes.values
              if "species" in raw_clean.columns else None)

    df = preprocess(raw)
    df = df.drop(columns=[c for c in ["year"] if c in df.columns])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    pca_full = PCA().fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.argmax(cum_var >= 0.95) + 1)
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Best k
    best_k, best_s = 2, -1
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lb = km.fit_predict(X_pca)
        s = silhouette_score(X_pca, lb)
        if s > best_s:
            best_s, best_k = s, k

    # 3 models
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    lb_km = km.fit_predict(X_pca)
    m_km = evaluate(X_pca, lb_km, y_true); m_km["Model"] = "KMeans"

    agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    lb_agg = agg.fit_predict(X_pca)
    m_agg = evaluate(X_pca, lb_agg, y_true); m_agg["Model"] = "Agglomerative"

    db = DBSCAN(eps=0.7, min_samples=5)
    lb_db = db.fit_predict(X_pca)
    m_db = evaluate(X_pca, lb_db, y_true); m_db["Model"] = "DBSCAN"

    results = pd.DataFrame([m_km, m_agg, m_db])
    valid = results.dropna(subset=["Silhouette"])
    best_row = valid.sort_values("Silhouette", ascending=False).iloc[0]
    best_name = best_row["Model"]

    model_lookup = {"KMeans": km, "Agglomerative": agg, "DBSCAN": db}

    return {
        "model_name": best_name,
        "model": model_lookup[best_name],
        "models": model_lookup,
        "scaler": scaler,
        "pca": pca,
        "feature_columns": df.columns.tolist(),
        "n_clusters": int(best_k),
        "n_components": int(n_comp),
        "metrics": best_row.to_dict(),
        "comparison": results,
        "df_clean": df,
        "X_pca": X_pca,
        "labels": {"KMeans": lb_km, "Agglomerative": lb_agg, "DBSCAN": lb_db},
    }


# ==========================================================
# Load or train
# ==========================================================
DATA_PATH = find_data_file()

if DATA_PATH is None:
    st.error("❌ **Dataset not found.** Please upload `penguins.csv` in the `data/` folder.")
    st.code("""Project structure needed:
penguin-clustering-app/
├── app.py
├── requirements.txt
├── data/penguins.csv       ← REQUIRED
└── models/                 ← optional (auto-created)
""")
    st.stop()

# Try loading pre-trained pickle first, otherwise train on-the-fly
artifact = None
if os.path.exists(BEST_PKL):
    try:
        with open(BEST_PKL, "rb") as f:
            artifact = pickle.load(f)
        # Add derived data for visualization
        df_clean = preprocess(pd.read_csv(DATA_PATH))
        df_clean = df_clean.drop(columns=[c for c in ["year"] if c in df_clean.columns])
        artifact["df_clean"] = df_clean
        artifact["X_pca"] = artifact["pca"].transform(
            artifact["scaler"].transform(df_clean))
        # reconstruct labels
        labels = {}
        for name, mdl in artifact.get("models", {}).items():
            if name == "DBSCAN":
                labels[name] = mdl.fit_predict(artifact["X_pca"])
            else:
                labels[name] = mdl.predict(artifact["X_pca"])
        artifact["labels"] = labels
        # comparison fallback
        if "comparison" not in artifact:
            artifact["comparison"] = pd.DataFrame([artifact["metrics"]])
    except Exception as e:
        st.warning(f"⚠️ Could not load pickle ({e}). Training fresh...")
        artifact = None

if artifact is None:
    artifact = train_models(DATA_PATH)


# ==========================================================
# Sidebar
# ==========================================================
st.sidebar.title("🐧 Penguin Clustering")
st.sidebar.markdown("**CASMI26 — Unsupervised Learning**")
st.sidebar.markdown(f"📂 Data: `{DATA_PATH}`")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Data Explorer", "🎯 Prediction",
     "📈 Model Insights", "ℹ️ About"],
)


# ==========================================================
# Header
# ==========================================================
st.title("🐧 Penguin Species Discovery via Unsupervised Clustering")
st.caption("KMeans · Agglomerative · DBSCAN | PCA-reduced features")


# ==========================================================
# HOME
# ==========================================================
if page == "🏠 Home":
    st.markdown("""
    ### Welcome 👋
    Unsupervised learning on the *Palmer Archipelago Penguins* dataset.
    Species labels are **not** used during training — only for validation.

    #### 🔍 What you can do
    - Explore the dataset distribution
    - Predict the cluster for a new penguin
    - Compare three clustering algorithms
    - Inspect cluster profiles + PCA projection
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", artifact["model_name"])
    c2.metric("Clusters", artifact["n_clusters"])
    c3.metric("PCA Components", artifact["n_components"])
    sil = artifact["metrics"].get("Silhouette", float("nan"))
    c4.metric("Silhouette", f"{sil:.3f}" if not pd.isna(sil) else "—")

    st.markdown("---")
    st.subheader("📁 Dataset Preview")
    st.dataframe(pd.read_csv(DATA_PATH).head(20), use_container_width=True)


# ==========================================================
# DATA EXPLORER
# ==========================================================
elif page == "📊 Data Explorer":
    st.subheader("📊 Exploratory Data Analysis")
    df = pd.read_csv(DATA_PATH).dropna().reset_index(drop=True)

    if "species" in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Species Count**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.countplot(x="species", data=df, palette="Set2", ax=ax)
            st.pyplot(fig)
        with col2:
            st.markdown("**Island Count**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.countplot(x="island", data=df, palette="Set3", ax=ax)
            st.pyplot(fig)

    st.markdown("**Correlation Heatmap**")
    num_df = df.select_dtypes(include=np.number).drop(
        columns=[c for c in ["Unnamed: 0", "year"] if c in df.columns])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.markdown("**Pairplot**")
    cols_plot = [c for c in FEATURE_COLUMNS if c in df.columns]
    if "species" in df.columns:
        fig = sns.pairplot(df[cols_plot + ["species"]],
                           hue="species", palette="husl", height=2)
        st.pyplot(fig)

    st.markdown("**Summary Statistics**")
    st.dataframe(df.describe().T, use_container_width=True)


# ==========================================================
# PREDICTION
# ==========================================================
elif page == "🎯 Prediction":
    st.subheader("🎯 Predict Cluster for a New Penguin")

    feature_cols = artifact["feature_columns"]
    defaults = {"bill_length_mm": 44.0, "bill_depth_mm": 17.0,
                "flipper_length_mm": 200.0, "body_mass_g": 4200.0,
                "island": 0, "sex": 0}
    mins = {"bill_length_mm": 30.0, "bill_depth_mm": 12.0,
            "flipper_length_mm": 170.0, "body_mass_g": 2500.0,
            "island": 0, "sex": 0}
    maxs = {"bill_length_mm": 60.0, "bill_depth_mm": 22.0,
            "flipper_length_mm": 235.0, "body_mass_g": 6400.0,
            "island": 2, "sex": 1}

    inputs = {}
    cols = st.columns(len(feature_cols))
    for i, c in enumerate(feature_cols):
        with cols[i]:
            inputs[c] = st.number_input(
                c,
                min_value=float(mins.get(c, 0)),
                max_value=float(maxs.get(c, 100)),
                value=float(defaults.get(c, 0)),
                step=0.1 if c != "body_mass_g" else 10.0,
            )

    if st.button("🔮 Predict Cluster", type="primary", use_container_width=True):
        X = pd.DataFrame([inputs])[feature_cols]
        Xs = artifact["scaler"].transform(X)
        Xp = artifact["pca"].transform(Xs)
        mdl = artifact["model"]
        if artifact["model_name"] == "DBSCAN":
            label = mdl.fit_predict(Xp)[0]
        else:
            label = mdl.predict(Xp)[0]

        if label == -1:
            st.warning("🔊 Noise point — DBSCAN could not assign this sample.")
        else:
            st.success(f"### 🐧 Predicted Cluster: **Cluster {label}**")
            st.info(f"Model: **{artifact['model_name']}** · "
                    f"Total clusters: **{artifact['n_clusters']}**")

        st.markdown("**Input Summary**")
        st.dataframe(pd.DataFrame([inputs]), use_container_width=True)


# ==========================================================
# MODEL INSIGHTS
# ==========================================================
elif page == "📈 Model Insights":
    st.subheader("📈 Model Comparison & Cluster Profiles")

    # Comparison table
    st.markdown("### 🔬 Algorithm Comparison")
    comp = artifact["comparison"].copy()
    for col in ["Silhouette", "Davies-Bouldin",
                "Calinski-Harabasz", "ARI", "NMI"]:
        if col in comp.columns:
            comp[col] = comp[col].apply(
                lambda x: f"{x:.4f}"
                if isinstance(x, (int, float, np.floating)) and not pd.isna(x)
                else x)
    st.dataframe(comp, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.barplot(x="Model", y="Silhouette",
                data=artifact["comparison"], palette="viridis", ax=ax)
    ax.set_title("Silhouette Score by Model")
    st.pyplot(fig)

    # Cluster profiles
    st.markdown("### 🧬 Cluster Profiles (mean values)")
    df_clean = artifact["df_clean"]
    labels = artifact["labels"][artifact["model_name"]]
    tmp = df_clean[artifact["feature_columns"]].copy()
    tmp["cluster"] = labels
    st.dataframe(tmp.groupby("cluster").mean().round(2),
                 use_container_width=True)

    # PCA projection
    st.markdown("### 🌌 PCA Projection with Clusters")
    X_pca = artifact["X_pca"]
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                         c=labels, cmap="tab10", alpha=0.75, s=30)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"Clusters — {artifact['model_name']}")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    st.pyplot(fig)


# ==========================================================
# ABOUT
# ==========================================================
elif page == "ℹ️ About":
    st.subheader("ℹ️ About This Project")
    st.markdown("""
    ### 🎯 Objective
    Discover natural groupings in the Palmer Archipelago Penguins dataset
    using unsupervised learning.

    ### 🧠 Algorithms
    | Model | Idea |
    |-------|------|
    | **KMeans** | Partition-based; minimizes inertia |
    | **Agglomerative (Ward)** | Hierarchical bottom-up |
    | **DBSCAN** | Density-based; detects noise |

    ### ⚙️ Pipeline
    1. Cleaning + missing value removal
    2. Categorical encoding
    3. Standard scaling
    4. PCA (95% variance)
    5. Train 3 models
    6. Compare (Silhouette, DB, CH)
    7. Best model auto-selected

    ### 📦 Tech Stack
    Python · scikit-learn · pandas · numpy · matplotlib · seaborn · Streamlit

    ### 👤 Author
    CASMI26 — Module Project
    """)
    st.markdown("---")
    st.markdown("### 📦 Loaded Artifact")
    st.write(f"**Model:** {artifact['model_name']}")
    st.write(f"**Clusters:** {artifact['n_clusters']}")
    st.write(f"**PCA Components:** {artifact['n_components']}")
    st.write(f"**Features:** {', '.join(artifact['feature_columns'])}")


st.markdown("---")
st.caption("🐧 Penguin Clustering · CASMI26")
