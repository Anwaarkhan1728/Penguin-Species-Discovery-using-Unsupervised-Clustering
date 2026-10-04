# Penguin-Species-Discovery-using-Unsupervised-Clustering

An interactive **unsupervised machine learning** web app that discovers natural
groupings in the Palmer Archipelago Penguins dataset using **KMeans**,
**Agglomerative Clustering**, and **DBSCAN**.

---

## ✨ Features

- 📊 Interactive EDA — distributions, correlations, pairplots
- 🎯 Predict cluster for a new penguin (sliders + inputs)
- 📈 Compare 3 clustering algorithms side by side
- 🧬 Inspect cluster profiles and PCA projection
- 💾 Trained models saved as `.pkl` for reuse

---

## 📁 Project Structure

```
penguin-clustering-app/
├── app.py                       # Streamlit app
├── train_model.py               # Trains & saves models
├── utils.py                     # Helper functions
├── requirements.txt
├── README.md
├── .streamlit/config.toml
├── data/penguins.csv            # Kaggle dataset
└── models/                      # Saved pickles
```

---

## 🚀 Quick Start

### 1. Clone / download this project

```bash
cd penguin-clustering-app
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add dataset

Copy `penguins.csv` into the `data/` folder:

```
data/penguins.csv
```

### 5. Train the models (creates `models/*.pkl`)

```bash
python train_model.py --data data/penguins.csv --out models
```

You should see output like:

```
🏆 Best model: KMeans (silhouette=0.5231)
✅ Saved all artifacts to: models
   • penguin_clustering_best.pkl
   • penguin_kmeans.pkl
   • penguin_agglomerative.pkl
   • penguin_dbscan.pkl
   • metadata.pkl
```

### 6. Run the Streamlit app

```bash
streamlit run app.py
```

Open your browser at 👉 `http://localhost:8501`

---

## 🧠 Algorithms Used

| Model | Description |
|-------|-------------|
| **KMeans** | Partition-based; minimizes within-cluster variance |
| **Agglomerative (Ward)** | Hierarchical; merges clusters bottom-up |
| **DBSCAN** | Density-based; handles arbitrary shapes + noise |

---

## 📊 Evaluation Metrics

- **Silhouette Score** — cluster cohesion vs separation (higher = better)
- **Davies–Bouldin Index** — cluster similarity (lower = better)
- **Calinski–Harabasz Index** — variance ratio (higher = better)
- **ARI / NMI** — agreement with true species labels (sanity check only)

---

## 📦 Saved Artifacts

The `.pkl` files contain:

| Key | Description |
|-----|-------------|
| `model_name` | Best model's name |
| `model` | Trained clustering model |
| `scaler` | Fitted `StandardScaler` |
| `pca` | Fitted `PCA` transformer |
| `feature_columns` | Feature order used during training |
| `n_clusters` | Number of clusters found |
| `metrics` | Evaluation scores of the best model |

---

## 🛠️ Tech Stack

- Python 3.11+
- scikit-learn, pandas, numpy
- matplotlib, seaborn
- Streamlit
- pickle

---

## 📜 License

MIT — free to use, modify, and share.

---

## 👤 Author

**CASMI26 — Module Project**
