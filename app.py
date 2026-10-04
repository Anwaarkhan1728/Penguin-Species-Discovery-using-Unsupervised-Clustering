"""
app.py — Penguin Clustering Streamlit App

Run:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import (
    FEATURE_COLUMNS,
    CLUSTER_COLORS,
    preprocess_dataframe,
    prepare_input,
    load_artifact,
    cluster_summary,
)


# ==========================================================
# Page config
# ==========================================================
st.set_page_config(
    page_title="Penguin Clustering App",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# Paths
# ==========================================================
MODELS_DIR = "models"
BEST_PATH = os.path.join(MODELS_DIR, "penguin_clustering_best.pkl")
META_PATH = os.path.join(MODELS_DIR, "metadata.pkl")
DATA_PATH = "data/penguins.csv"


# ==========================================================
# Cached loaders
# ==========================================================
@st.cache_resource(show_spinner=False)
def get_artifact(path):
    return load_artifact(path)


@st.cache_data(show_spinner=False)
def get_dataset(path):
    return pd.read_csv(path)


# ==========================================================
# Sidebar
# ==========================================================
st.sidebar.image("https://em-content.zobj.net/source/apple/391/penguin_1f427.png",
                 width=90)
st.sidebar.title("🐧 Penguin Clustering")
st.sidebar.markdown("**CASMI26 — Unsupervised Learning**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Data Explorer", "🎯 Cluster Prediction",
     "📈 Model Insights", "ℹ️ About"],
)


# ==========================================================
# Load artifacts (with graceful error)
# ==========================================================
try:
    artifact = get_artifact(BEST_PATH)
    metadata = get_artifact(META_PATH) if os.path.exists(META_PATH) else None
    models_loaded = True
except FileNotFoundError:
    models_loaded = False
    artifact = None
    metadata = None

dataset_loaded = os.path.exists(DATA_PATH)
df_raw = get_dataset(DATA_PATH) if dataset_loaded else None


# ==========================================================
# Header
# ==========================================================
st.title("🐧 Penguin Species Discovery via Unsupervised Clustering")
st.caption("Powered by KMeans · Agglomerative · DBSCAN | PCA-reduced features")


# ==========================================================
# PAGES
# ==========================================================

# ----------------------------------------------------------
if page == "🏠 Home":
    st.markdown(
        """
        ### Welcome 👋
        This app uses **unsupervised machine learning** to discover natural
        groupings in the *Palmer Archipelago Penguins* dataset — **without using
        species labels during training**.

        #### 🔍 What you can do here
        - **Explore** the dataset and its distribution
        - **Predict** which cluster a new penguin belongs to
        - **Compare** three clustering algorithms side by side
        - **Inspect** cluster profiles and PCA projection

        ---
        """
    )

    if not models_loaded:
        st.error(
            "⚠️ **No trained model found.**\n\n"
            "Please run the training script first:\n"
            "```bash\npython train_model.py --data data/penguins.csv --out models\n```"
        )
        st.stop()

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
        c3.metric("Missing Values", int(df_raw.isna().sum().sum()))


# ----------------------------------------------------------
elif page == "📊 Data Explorer":
    if not dataset_loaded:
        st.error("Dataset not found at `data/penguins.csv`")
        st.stop()

    st.subheader("📊 Exploratory Data Analysis")
    df = df_raw.dropna().reset_index(drop=True)

    # Species distribution
    if "species" in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Species Count**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.countplot(x="species", data=df, palette="Set2", ax=ax)
            ax.set_title("Species Distribution")
            st.pyplot(fig)

        with col2:
            st.markdown("**Island Count**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.countplot(x="island", data=df, palette="Set3", ax=ax)
            ax.set_title("Island Distribution")
            st.pyplot(fig)

    # Correlation heatmap
    st.markdown("**Correlation Heatmap**")
    num_df = df.select_dtypes(include=np.number).drop(
        columns=[c for c in ["Unnamed: 0", "year"] if c in df.columns]
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Pairplot
    st.markdown("**Feature Pairplot**")
    cols_plot = [c for c in FEATURE_COLUMNS if c in df.columns]
    if "species" in df.columns:
        fig = sns.pairplot(df[cols_plot + ["species"]],
                           hue="species", palette="husl", height=2)
        st.pyplot(fig)

    # Statistics
    st.markdown("**Summary Statistics**")
    st.dataframe(df.describe().T, use_container_width=True)


# ----------------------------------------------------------
elif page == "🎯 Cluster Prediction":
    if not models_loaded:
        st.error("Train the model first.")
        st.stop()

    st.subheader("🎯 Predict Cluster for a New Penguin")
    st.markdown(
        "Adjust the sliders below and click **Predict** to see which cluster "
        "the penguin belongs to."
    )

    feature_cols = artifact["feature_columns"]

    # Build sidebar-like inputs
    defaults = {
        "bill_length_mm": 44.0,
        "bill_depth_mm": 17.0,
        "flipper_length_mm": 200.0,
        "body_mass_g": 4200.0,
        "island": 0,
        "sex": 0,
    }
    mins = {
        "bill_length_mm": 30.0, "bill_depth_mm": 12.0,
        "flipper_length_mm": 170.0, "body_mass_g": 2500.0,
        "island": 0, "sex": 0,
    }
    maxs = {
        "bill_length_mm": 60.0, "bill_depth_mm": 22.0,
        "flipper_length_mm": 235.0, "body_mass_g": 6400.0,
        "island": 2, "sex": 1,
    }

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
            st.warning("🔊 **Noise point** — DBSCAN could not assign this sample to any cluster.")
        else:
            st.success(f"### 🐧 Predicted Cluster: **Cluster {label}**")
            st.info(
                f"Model used: **{artifact['model_name']}**  \n"
                f"Total clusters: **{artifact['n_clusters']}**"
            )

        # Show the input row
        st.markdown("**Input Summary**")
        st.dataframe(pd.DataFrame([inputs]), use_container_width=True)


# ----------------------------------------------------------
elif page == "📈 Model Insights":
    if not models_loaded:
        st.error("Train the model first.")
        st.stop()

    st.subheader("📈 Model Comparison & Cluster Profiles")

    # Comparison table
    if metadata is not None and "comparison" in metadata:
        st.markdown("### 🔬 Algorithm Comparison")
        comp = metadata["comparison"].copy()
        for col in ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "ARI", "NMI"]:
            if col in comp.columns:
                comp[col] = comp[col].apply(
                    lambda x: f"{x:.4f}" if isinstance(x, (int, float, np.floating)) and not pd.isna(x) else x
                )
        st.dataframe(comp, use_container_width=True)

        # Bar chart
        comp_raw = metadata["comparison"]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.barplot(x="Model", y="Silhouette", data=comp_raw, palette="viridis", ax=ax)
        ax.set_title("Silhouette Score by Model")
        st.pyplot(fig)

    # Cluster profiles
    if dataset_loaded:
        st.markdown("### 🧬 Cluster Profiles (mean feature values)")
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

        # PCA scatter
        st.markdown("### 🌌 PCA Projection with Clusters")
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=labels, cmap="tab10", alpha=0.75, s=30,
        )
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title(f"Cluster Assignments — {artifact['model_name']}")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        st.pyplot(fig)


# ----------------------------------------------------------
elif page == "ℹ️ About":
    st.subheader("ℹ️ About This Project")

    st.markdown(
        """
        ### 🎯 Objective
        Discover natural groupings in the *Palmer Archipelago Penguins* dataset
        using **unsupervised learning**, and compare three popular clustering
        algorithms.

        ### 🧠 Algorithms Used
        | Model | Idea |
        |-------|------|
        | **KMeans** | Partition data into *k* spherical clusters by minimizing inertia |
        | **Agglomerative (Ward)** | Bottom-up hierarchical merging by minimizing variance |
        | **DBSCAN** | Density-based clustering; finds arbitrary shapes + detects noise |

        ### ⚙️ Pipeline
        1. Data cleaning + missing value removal
        2. Categorical encoding (island, sex)
        3. Standard scaling
        4. PCA to 95% variance
        5. Train 3 clustering models
        6. Compare using Silhouette, Davies-Bouldin, Calinski-Harabasz
        7. Pick the best model → save as **pickle**

        ### 📦 Tech Stack
        - **Python 3.11+**
        - **scikit-learn** — clustering, PCA, scaling
        - **pandas / numpy** — data wrangling
        - **seaborn / matplotlib** — visualization
        - **Streamlit** — interactive web app
        - **pickle** — model persistence

        ### 👤 Author
        CASMI26 — Module Project
        """
    )

    st.markdown("---")
    if models_loaded:
        st.markdown("### 📦 Loaded Artifact Info")
        info = {
            "Model": artifact["model_name"],
            "Clusters": artifact["n_clusters"],
            "PCA Components": artifact["n_components"],
            "Features": ", ".join(artifact["feature_columns"]),
        }
        for k, v in info.items():
            st.write(f"**{k}:** {v}")


# ==========================================================
# Footer
# ==========================================================
st.markdown("---")
st.caption("🐧 Penguin Clustering App · Built with Streamlit · CASMI26")
