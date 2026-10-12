print("="*60)
print("MODUL 4: MACHINE LEARNING ENGINEERING")
print("Bagian 3: Unsupervised Learning - CLUSTERING + DIMENSIONALITY REDUCTION")
print("="*60)

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)

output_dir = "c:\\ai-engineer\\modul_04_machine_learning\\output_charts"
os.makedirs(output_dir, exist_ok=True)

sns.set_style("whitegrid")
np.random.seed(42)

print("""
┌─────────────────────────────────────────────────────────────────┐
│       UNSUPERVISED LEARNING: CLUSTERING                        │
├─────────────────────────────────────────────────────────────────┤
│  Tujuan: MEMBENTUK KELOMPOK dari data TANPA LABEL!            │
│                                                                 │
│  ALGORITMA CLUSTERING POPULER:                                 │
│  ┌───────────────────────────────────────────────────────┐    │
│  │ 1. K-Means    → Centroid-based, perlu jumlah k, CEPAT│    │
│  │ 2. K-Medoids  → Median centroid, tahan outlier       │    │
│  │ 3. DBSCAN     → Density-based, outlier detection,    │    │
│  │               → TIDAK perlu jumlah k                  │    │
│  │ 4. Hierarchical → Agglomerative, dendrogram          │    │
│  │ 5. GMM (Gaussian Mixture) → Probabilistik cluster    │    │
│  └───────────────────────────────────────────────────────┘    │
│                                                                 │
│  EVALUASI CLUSTERING (Internal, TANPA label):                  │
│  ✅ Silhouette Score  → -1 s/d +1 (LEBIH BESAR LEBIH BAIK)     │
│  ✅ Calinski-Harabasz  → Rasio variance (LEBIH BESAR LEBIH BAIK)│
│  ❌ Davies-Bouldin     → Mirip Silhouette (LEBIH KECIL LEBIH BAIK)│
└─────────────────────────────────────────────────────────────────┘
""")

print("\n>>> 1. PERSIAPAN DATASET - Segmentasi Customer Mall")
print("-" * 60)

n = 500
np.random.seed(42)

df = pd.DataFrame({
    "Usia": np.concatenate([
        np.random.normal(22, 3, 120),
        np.random.normal(35, 5, 130),
        np.random.normal(50, 6, 100),
        np.random.normal(60, 5, 90),
        np.random.normal(45, 7, 60),
    ]).clip(18, 75),
    "Pendapatan_Thn_Juta": np.concatenate([
        np.random.normal(30, 5, 120),
        np.random.normal(80, 10, 130),
        np.random.normal(150, 15, 100),
        np.random.normal(120, 20, 90),
        np.random.normal(60, 8, 60),
    ]).clip(20, 200),
    "Skor_Belanja_1_100": np.concatenate([
        np.random.normal(75, 8, 120),
        np.random.normal(40, 10, 130),
        np.random.normal(25, 6, 100),
        np.random.normal(70, 9, 90),
        np.random.normal(55, 7, 60),
    ]).clip(1, 100),
    "Jml_Transaksi_Bln": np.concatenate([
        np.random.poisson(8, 120),
        np.random.poisson(4, 130),
        np.random.poisson(2, 100),
        np.random.poisson(7, 90),
        np.random.poisson(5, 60),
    ]).clip(0, 20),
    "Jenis_Kelamin": np.random.choice([0, 1], n),
})

df["Usia"] = df["Usia"].round(0).astype(int)

print(f"Shape: {df.shape}")
print(df.describe().round(2))

cols_for_cluster = ["Usia", "Pendapatan_Thn_Juta", "Skor_Belanja_1_100", "Jml_Transaksi_Bln"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[cols_for_cluster])
print(f"\nData setelah scaling (mean=0, std=1): {X_scaled.shape}")

print("\n>>> 2. ELBOW METHOD + SILHOUETTE untuk Tentukan K Optimal (K-Means)")
print("-" * 60)

inertias = []
sil_scores = []
ch_scores = []
db_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

print("Evaluasi per K:")
for i, k in enumerate(k_range):
    star = "★" if sil_scores[i] == max(sil_scores) else ""
    print(f"  K={k} → Inertia: {inertias[i]:.1f} | Silhouette: {sil_scores[i]:.4f} | CH: {ch_scores[i]:.1f} | DB: {db_scores[i]:.4f} {star}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Penentuan Jumlah K Optimal - K-Means Clustering", fontsize=14, fontweight="bold")

axes[0, 0].plot(list(k_range), inertias, "o-", linewidth=2, markersize=8, color="#3b82f6")
axes[0, 0].set_title("Elbow Method: Inertia vs K", fontweight="bold")
axes[0, 0].set_xlabel("Jumlah Cluster (K)")
axes[0, 0].set_ylabel("Inertia (SSE)")
best_sil_k = list(k_range)[np.argmax(sil_scores)]
axes[0, 0].axvline(best_sil_k, color="red", linestyle="--", label=f"Saran K={best_sil_k}")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(list(k_range), sil_scores, "o-", linewidth=2, markersize=8, color="#16a34a")
axes[0, 1].set_title("Silhouette Score vs K (LEBIH BESAR LEBIH BAIK)", fontweight="bold")
axes[0, 1].set_xlabel("Jumlah Cluster (K)")
axes[0, 1].set_ylabel("Silhouette Score")
axes[0, 1].axhline(0.5, color="orange", linestyle=":", label="Threshold Bagus (0.5)")
axes[0, 1].axvline(best_sil_k, color="red", linestyle="--", label=f"Terbaik K={best_sil_k}")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(list(k_range), ch_scores, "o-", linewidth=2, markersize=8, color="#7c3aed")
axes[1, 0].set_title("Calinski-Harabasz vs K (LEBIH BESAR LEBIH BAIK)", fontweight="bold")
axes[1, 0].set_xlabel("Jumlah Cluster (K)")
axes[1, 0].set_ylabel("CH Score")
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(list(k_range), db_scores, "o-", linewidth=2, markersize=8, color="#dc2626")
axes[1, 1].set_title("Davies-Bouldin vs K (LEBIH KECIL LEBIH BAIK)", fontweight="bold")
axes[1, 1].set_xlabel("Jumlah Cluster (K)")
axes[1, 1].set_ylabel("DB Score")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}\\07_elbow_silhouette_kmeans.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart Elbow + Silhouette tersimpan")

print("\n>>> 3. K-MEANS CLUSTERING dengan K Terbaik + PROFILING CLUSTER")
print("-" * 60)

k_opt = best_sil_k
km_final = KMeans(n_clusters=k_opt, n_init=20, random_state=42)
df["Cluster_KMeans"] = km_final.fit_predict(X_scaled)
print(f"K-Means dengan K={k_opt}:")
print(f"  Jumlah per cluster: {df['Cluster_KMeans'].value_counts().sort_index().to_dict()}")
print(f"  Silhouette Score : {silhouette_score(X_scaled, df['Cluster_KMeans']):.4f}")

print("\n=== PROFIL RATA-RATA PER CLUSTER ===")
profil = df.groupby("Cluster_KMeans")[cols_for_cluster].mean().round(2)
profil["Jumlah_Customer"] = df["Cluster_KMeans"].value_counts().sort_index()
print(profil.to_string())

print(f"\n=== NAMA SEGMENTASI (Interpretasi Bisnis) ===")
nama_segment = {}
for c in range(k_opt):
    row = profil.loc[c]
    nama = []
    if row["Skor_Belanja_1_100"] >= 65:
        nama.append("RAJIN BELANJA")
    elif row["Skor_Belanja_1_100"] <= 40:
        nama.append("JARANG BELANJA")
    if row["Pendapatan_Thn_Juta"] >= 100:
        nama.append("PENDAPATAN TINGGI")
    elif row["Pendapatan_Thn_Juta"] <= 50:
        nama.append("PENDAPATAN RENDAH")
    if row["Usia"] <= 30:
        nama.append("MUDA")
    elif row["Usia"] >= 50:
        nama.append("LANSIA")
    nama_segment[c] = " / ".join(nama) if nama else f"Cluster {c}"
    print(f"  Cluster {c}: {nama_segment[c]} → {int(row['Jumlah_Customer'])} customer")

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sc = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=df["Cluster_KMeans"], cmap="tab10",
                   alpha=0.7, s=45, edgecolors="white")
centroids_2d = pca.transform(km_final.cluster_centers_)
axes[0].scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker="X", s=250,
                c="black", edgecolors="yellow", linewidths=2, label="Centroids")
axes[0].set_title(f"K-Means Clustering (K={k_opt}) di 2D PCA Space", fontweight="bold")
axes[0].legend()
axes[0].legend(*sc.legend_elements(), title="Cluster")

prof_plot = profil[cols_for_cluster]
prof_plot_norm = (prof_plot - prof_plot.min()) / (prof_plot.max() - prof_plot.min() + 1e-9)
sns.heatmap(prof_plot_norm, annot=prof_plot.values, cmap="YlOrRd", fmt=".2f",
            yticklabels=[f"Cluster {c}\n({nama_segment[c][:20]})" for c in range(k_opt)], ax=axes[1])
axes[1].set_title("Profil Heatmap per Cluster (nilai asli + normalisasi warna)", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{output_dir}\\08_kmeans_clusters_profiled.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart K-Means + Profil tersimpan")

print("\n>>> 4. DBSCAN vs HIERARCHICAL vs GMM - Perbandingan Algoritma")
print("-" * 60)

clusterers = {
    f"K-Means (K={k_opt})": KMeans(n_clusters=k_opt, n_init=10, random_state=42),
    "DBSCAN (eps=0.7, minPts=8)": DBSCAN(eps=0.7, min_samples=8),
    "Agglomerative (Ward, k=5)": AgglomerativeClustering(n_clusters=k_opt, linkage="ward"),
    "GMM (n=5, full)": GaussianMixture(n_components=k_opt, covariance_type="full", random_state=42),
}

eval_res = []
for name, clf in clusterers.items():
    labels = clf.fit_predict(X_scaled)
    n_clust = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    valid_mask = labels != -1
    if valid_mask.sum() > k_opt:
        sil = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
        ch = calinski_harabasz_score(X_scaled[valid_mask], labels[valid_mask])
    else:
        sil = float("nan")
        ch = float("nan")
    eval_res.append({"Algoritma": name, "Jumlah_Cluster": n_clust,
                     "Noise_Points": n_noise, "Silhouette": sil, "CH_Score": ch})
    print(f"  {name:<30} → K={n_clust:>2}, Noise={n_noise:>3}, Sil={sil:.4f}, CH={ch:.1f}")

print("\n>>> 5. t-SNE vs PCA - Visualisasi High-Dim")
print("-" * 60)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, init="pca", learning_rate="auto")
X_tsne = tsne.fit_transform(X_scaled)
print(f"t-SNE selesai, shape: {X_tsne.shape}")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sc1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=df["Cluster_KMeans"], cmap="tab10", alpha=0.7, s=35)
axes[0].set_title("PCA (Linear Projection) - 2D", fontweight="bold")
axes[0].legend(*sc1.legend_elements(), title="Cluster", fontsize=8)
sc2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=df["Cluster_KMeans"], cmap="tab10", alpha=0.7, s=35)
axes[1].set_title("t-SNE (Non-Linear Manifold) - 2D", fontweight="bold")
axes[1].legend(*sc2.legend_elements(), title="Cluster", fontsize=8)
plt.tight_layout()
plt.savefig(f"{output_dir}\\09_pca_vs_tsne_viz.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart PCA vs t-SNE tersimpan")

print("""
┌─────────────────────────────────────────────────────────────────┐
│  RANGKUMAN UNSUPERVISED LEARNING:                               │
├─────────────────────────────────────────────────────────────────┤
│  🎯 TUJUAN: Segmentasi customer, deteksi outlier, pattern mining│
│  ─────────────────────────────────────────────────────────────  │
│  K-MEANS       → Cepat, butuh K, spherical cluster,           │
│                  sensitif outlier & inisialisasi              │
│  DBSCAN        → TIDAK butuh K, auto noise, cluster tidak     │
│                  perlu bulat tapi butuh tuning eps/minPts     │
│  HIERARCHICAL  → Dendrogram interpretasi bagus, O(n³) lambat │
│  GMM           → Probabilistik, cluster overlap lebih bagus  │
│  ─────────────────────────────────────────────────────────────  │
│  EVALUASI: Silhouette > 0.5 → BAGUS; > 0.7 → SANGAT BAGUS     │
│  VISUALISASI: PCA cepat global, t-SNE lambat tapi detail      │
└─────────────────────────────────────────────────────────────────┘
""")

print("""\n>>> LATIHAN:
1. Untuk setiap cluster, tentukan STRATEGI MARKETING nya!
   (Contoh: Cluster "Pendapatan Tinggi / Jarang Belanja" →
    Berikan promo Cashback 5% + membership gold untuk
    mendorong frekuensi belanja)
2. Tuning DBSCAN: coba eps=0.5 s/d 1.2, minPts=5 s/d 15
   Dapatkan silhouette tertinggi!
3. Gunakan hasil cluster sebagai FEATURE BARU untuk model
   Supervised Classification (misal prediksi Churn)
4. Tambahkan Hierarchical Clustering dengan DENDROGRAM visual!
""")

print("\n✓ Bagian 3 Modul 4 Selesai: UNSUPERVISED LEARNING (CLUSTERING)")
print("\n" + "="*60)
print("MODUL 4 SELESAI - Machine Learning Engineering")
print("="*60)
