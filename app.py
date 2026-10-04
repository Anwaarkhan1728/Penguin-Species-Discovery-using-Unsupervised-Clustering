# ==========================================================
# 🐧 PENGUIN CLUSTERING — COMPLETE KAGGLE SETUP (ONE CELL)
# ==========================================================
import os, sys, shutil, subprocess, zipfile

# ----------------------------------------------------------
# 0. Setup folders
# ----------------------------------------------------------
PROJECT_DIR = "/kaggle/working/penguin_clustering_app"
DATA_DIR = f"{PROJECT_DIR}/data"
MODELS_DIR = f"{PROJECT_DIR}/models"
STREAMLIT_DIR = f"{PROJECT_DIR}/.streamlit"

for d in [PROJECT_DIR, DATA_DIR, MODELS_DIR, STREAMLIT_DIR]:
    os.makedirs(d, exist_ok=True)

os.chdir(PROJECT_DIR)
print("📁 Project dir:", os.getcwd())

# ----------------------------------------------------------
# 1. Copy dataset
# ----------------------------------------------------------
DATA_SRC = "/kaggle/input/datasets/anwarkhanniazi/unsupervised-clustering/penguins.csv"
DATA_DST = f"{DATA_DIR}/penguins.csv"

if os.path.exists(DATA_SRC):
    shutil.copy(DATA_SRC, DATA_DST)
    print("✅ Dataset copied:", DATA_DST)
else:
    raise FileNotFoundError(f"Dataset not found at {DATA_SRC}")

# ----------------------------------------------------------
# 2. Write utils.py
# ----------------------------------------------------------
UTILS_CODE = '''
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

FEATURE_COLUMNS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_COLUMNS = ["island", "sex"]

CLUSTER_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c",
                  3: "#d62728", 4: "#9467bd", -1: "#7f7f7f"}


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
    df = df.select_dtypes(include=[np.number])
    return df


def prepare_input(feature_dict, feature_cols, scaler, pca):
    X = pd.DataFrame([feature_dict])[feature_cols]
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return X_pca


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
'''
with open(f"{PROJECT_DIR}/utils.py", "w") as f:
    f.write(UTILS_CODE)
print("✅ utils.py written")

# ----------------------------------------------------------
# 3. Write train_model.py
# ----------------------------------------------------------
TRAIN_CODE = '''
import os, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score)
from utils import preprocess_dataframe, save_artifact


def find_best_k(X, k_range=range(2, 11)):
    best_k, best_score = 2, -1
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lb = km.fit_predict(X)
        s = silhouette_score(X, lb)
        if s > best_score:
            best_score, best_k = s, k
    return best_k, best_score


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


def main(args):
    print("🔹 Loading data ...")
    raw = pd.read_csv(args.data)
    print("Raw shape:", raw.shape)

    y_true = raw["species"].astype("category").cat.codes.values if "species" in raw.columns else None

    df = preprocess_dataframe(raw)
    df = df.drop(columns=[c for c in ["year"] if c in df.columns])
    print("Cleaned shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # align y_true with cleaned df (dropna removed rows)
    raw_clean = raw.dropna().reset_index(drop=True)
    y_true = raw_clean["species"].astype("category").cat.codes.values if "species" in raw_clean.columns else None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    pca_full = PCA().fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.argmax(cum_var >= 0.95) + 1)
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA components kept: {n_comp}")

    best_k, best_s = find_best_k(X_pca)
    print(f"Best k: {best_k}  (silhouette={best_s:.4f})")

    # --- KMeans ---
    print("\\n🔹 Training KMeans ...")
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    lb_km = km.fit_predict(X_pca)
    m_km = evaluate(X_pca, lb_km, y_true); m_km["Model"] = "KMeans"

    # --- Agglomerative ---
    print("🔹 Training Agglomerative ...")
    agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    lb_agg = agg.fit_predict(X_pca)
    m_agg = evaluate(X_pca, lb_agg, y_true); m_agg["Model"] = "Agglomerative"

    # --- DBSCAN ---
    print("🔹 Training DBSCAN ...")
    db = DBSCAN(eps=args.eps, min_samples=args.min_samples)
    lb_db = db.fit_predict(X_pca)
    m_db = evaluate(X_pca, lb_db, y_true); m_db["Model"] = "DBSCAN"

    results = pd.DataFrame([m_km, m_agg, m_db])
    print("\\n📊 Model Comparison:")
    print(results.to_string(index=False))

    # Pick best
    valid = results.dropna(subset=["Silhouette"])
    best_row = valid.sort_values("Silhouette", ascending=False).iloc[0]
    best_name = best_row["Model"]
    print(f"\\n🏆 Best model: {best_name}")

    model_lookup = {"KMeans": km, "Agglomerative": agg, "DBSCAN": db}
    best_model = model_lookup[best_name]

    os.makedirs(args.out, exist_ok=True)

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

    for name, mdl in model_lookup.items():
        save_artifact(os.path.join(args.out, f"penguin_{name.lower()}.pkl"),
                      {"model": mdl, "scaler": scaler, "pca": pca})

    save_artifact(os.path.join(args.out, "metadata.pkl"),
                  {"comparison": results, "best_k": int(best_k),
                   "n_components": int(n_comp),
                   "feature_columns": df.columns.tolist()})

    print("\\n✅ Saved files:")
    for f in sorted(os.listdir(args.out)):
        size = os.path.getsize(os.path.join(args.out, f))
        print(f"   • {f}  ({size/1024:.1f} KB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/penguins.csv")
    p.add_argument("--out", type=str, default="models")
    p.add_argument("--eps", type=float, default=0.7)
    p.add_argument("--min_samples", type=int, default=5)
    args = p.parse_args()
    main(args)
'''
with open(f"{PROJECT_DIR}/train_model.py", "w") as f:
    f.write(TRAIN_CODE)
print("✅ train_model.py written")

# ----------------------------------------------------------
# 4. Write requirements.txt
# ----------------------------------------------------------
REQ = """streamlit==1.38.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
matplotlib==3.9.1
seaborn==0.13.2
scipy==1.14.0
joblib==1.4.2
Pillow==10.4.0
"""
with open(f"{PROJECT_DIR}/requirements.txt", "w") as f:
    f.write(REQ)
print("✅ requirements.txt written")

# ----------------------------------------------------------
# 5. Write .streamlit/config.toml
# ----------------------------------------------------------
CONFIG = """[theme]
primaryColor = "#0E7C86"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F2F4F7"
textColor = "#1A1A1A"
font = "sans serif"

[server]
headless = true
enableCORS = false

[browser]
gatherUsageStats = false
"""
with open(f"{STREAMLIT_DIR}/config.toml", "w") as f:
    f.write(CONFIG)
print("✅ .streamlit/config.toml written")

# ----------------------------------------------------------
# 6. Write app.py
# ----------------------------------------------------------
APP_CODE = r'''
"""
app.py — Penguin Clustering Streamlit App
Run: streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    FEATURE_COLUMNS, CLUSTER_COLORS,
    preprocess_dataframe, prepare_input,
    load_artifact, cluster_summary,
)

st.set_page_config(
    page_title="Penguin Clustering App",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = "models"
BEST_PATH = os.path.join(MODELS_DIR, "penguin_clustering_best.pkl")
META_PATH = os.path.join(MODELS_DIR, "metadata.pkl")
DATA_PATH = "data/penguins.csv"


@st.cache_resource(show_spinner=False)
def get_artifact(path):
    return load_artifact(path)


@st.cache_data(show_spinner=False)
def get_dataset(path):
    return pd.read_csv(path)


# --- Auto-train if missing ---
try:
    artifact = get_artifact(BEST_PATH)
    metadata = get_artifact(META_PATH) if os.path.exists(META_PATH) else None
    models_loaded = True
except FileNotFoundError:
    st.warning("⚠️ No trained model found. Training now (~30s)...")
    import subprocess
    res = subprocess.run(
        ["python", "train_model.py", "--data", DATA_PATH, "--out", MODELS_DIR],
        capture_output=True, text=True,
    )
    st.code(res.stdout + "\n" + res.stderr)
    if res.returncode != 0:
        st.error("❌ Training failed.")
        st.stop()
    st.success("✅ Training complete! Reloading...")
    st.rerun()

dataset_loaded = os.path.exists(DATA_PATH)
df_raw = get_dataset(DATA_PATH) if dataset_loaded else None


# --- Sidebar ---
st.sidebar.title("🐧 Penguin Clustering")
st.sidebar.markdown("**CASMI26 — Unsupervised Learning**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Data Explorer", "🎯 Cluster Prediction",
     "📈 Model Insights", "ℹ️ About"],
)


# --- Header ---
st.title("🐧 Penguin Species Discovery via Unsupervised Clustering")
st.caption("KMeans · Agglomerative · DBSCAN | PCA-reduced features")


# =========================================================
if page == "🏠 Home":
    st.markdown("""
    ### Welcome 👋
    This app uses **unsupervised machine learning** to discover natural
    groupings in the *Palmer Archipelago Penguins* dataset — **without using
    species labels during training**.

    #### 🔍 What you can do
    - **Explore** the dataset distribution
    - **Predict** which cluster a new penguin belongs to
    - **Compare** three clustering algorithms side by side
    - **Inspect** cluster profiles and PCA projection
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", artifact["model_name"])
    col2.metric("Clusters", artifact["n_clusters"])
    col3.metric("PCA Components", artifact["n_components"])
    col4.metric("Silhouette", f"{artifact['metrics']['Silhouette']:.3f}")

    st.markdown("---")
    if dataset_loaded:
        st.subheader("📁 Dataset Preview")
        st.dataframe(df_raw.head(20), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df_raw.shape[0])
        c2.metric("Columns", df_raw.shape[1])
        c3.metric("Missing", int(df_raw.isna().sum().sum()))


# =========================================================
elif page == "📊 Data Explorer":
    if not dataset_loaded:
        st.error("Dataset not found.")
        st.stop()

    st.subheader("📊 Exploratory Data Analysis")
    df = df_raw.dropna().reset_index(drop=True)

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
        fig = sns.pairplot(df[cols_plot + ["species"]], hue="species",
                           palette="husl", height=2)
        st.pyplot(fig)

    st.markdown("**Summary Statistics**")
    st.dataframe(df.describe().T, use_container_width=True)


# =========================================================
elif page == "🎯 Cluster Prediction":
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
        X = prepare_input(inputs, feature_cols, artifact["scaler"], artifact["pca"])
        model = artifact["model"]
        if artifact["model_name"] == "DBSCAN":
            label = model.fit_predict(X)[0]
        else:
            label = model.predict(X)[0]

        if label == -1:
            st.warning("🔊 Noise point — DBSCAN could not assign this sample.")
        else:
            st.success(f"### 🐧 Predicted Cluster: **Cluster {label}**")
            st.info(f"Model: **{artifact['model_name']}** | "
                    f"Total clusters: **{artifact['n_clusters']}**")
        st.markdown("**Input Summary**")
        st.dataframe(pd.DataFrame([inputs]), use_container_width=True)


# =========================================================
elif page == "📈 Model Insights":
    st.subheader("📈 Model Comparison & Cluster Profiles")

    if metadata is not None and "comparison" in metadata:
        st.markdown("### 🔬 Algorithm Comparison")
        comp = metadata["comparison"].copy()
        for col in ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "ARI", "NMI"]:
            if col in comp.columns:
                comp[col] = comp[col].apply(
                    lambda x: f"{x:.4f}" if isinstance(x, (int, float, np.floating)) and not pd.isna(x) else x)
        st.dataframe(comp, use_container_width=True)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.barplot(x="Model", y="Silhouette",
                    data=metadata["comparison"], palette="viridis", ax=ax)
        ax.set_title("Silhouette Score by Model")
        st.pyplot(fig)

    if dataset_loaded:
        st.markdown("### 🧬 Cluster Profiles")
        df_clean = preprocess_dataframe(df_raw)
        df_clean = df_clean.drop(columns=[c for c in ["year"] if c in df_clean.columns])

        X_scaled = artifact["scaler"].transform(df_clean)
        X_pca = artifact["pca"].transform(X_scaled)

        model = artifact["model"]
        if artifact["model_name"] == "DBSCAN":
            labels = model.fit_predict(X_pca)
        else:
            labels = model.predict(X_pca)

        summary = cluster_summary(df_clean, labels, artifact["feature_columns"])
        st.dataframe(summary, use_container_width=True)

        st.markdown("### 🌌 PCA Projection")
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                             c=labels, cmap="tab10", alpha=0.75, s=30)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title(f"Clusters — {artifact['model_name']}")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        st.pyplot(fig)


# =========================================================
elif page == "ℹ️ About":
    st.subheader("ℹ️ About This Project")
    st.markdown("""
    ### 🎯 Objective
    Discover natural groupings in the Palmer Archipelago Penguins dataset
    using unsupervised learning.

    ### 🧠 Algorithms
    | Model | Idea |
    |-------|------|
    | KMeans | Partition-based; minimizes inertia |
    | Agglomerative (Ward) | Hierarchical bottom-up |
    | DBSCAN | Density-based; detects noise |

    ### ⚙️ Pipeline
    1. Cleaning + missing value removal
    2. Categorical encoding
    3. Standard scaling
    4. PCA (95% variance)
    5. Train 3 models
    6. Compare (Silhouette, DB, CH)
    7. Save best as pickle

    ### 📦 Tech Stack
    Python · scikit-learn · pandas · numpy · matplotlib · seaborn
    · Streamlit · pickle

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
st.caption("🐧 Penguin Clustering App · Built with Streamlit · CASMI26")
'''
with open(f"{PROJECT_DIR}/app.py", "w") as f:
    f.write(APP_CODE)
print("✅ app.py written")

# ----------------------------------------------------------
# 7. Write README.md
# ----------------------------------------------------------
README = """# 🐧 Penguin Clustering — Streamlit App

Interactive **unsupervised ML** web app for the Palmer Archipelago Penguins dataset.

## Features
- 📊 EDA (distributions, correlations, pairplots)
- 🎯 Predict cluster for new penguin
- 📈 Compare KMeans / Agglomerative / DBSCAN
- 🧬 Cluster profiles + PCA projection

## Quick Start

```bash
pip install -r requirements.txt
python train_model.py --data data/penguins.csv --out models
streamlit run app.py
