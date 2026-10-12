print("="*60)
print("MODUL 3: DATA PREPARATION & PREPROCESSING")
print("Bagian 3: EDA (Exploratory Data Analysis) & Data Splitting")
print("="*60)

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

output_dir = "c:\\ai-engineer\\modul_03_data_prep\\output_charts"
os.makedirs(output_dir, exist_ok=True)

sns.set_style("whitegrid")
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "Umur": np.random.randint(18, 65, n),
    "Jenis_Kelamin": np.random.choice(["Laki-laki", "Perempuan"], n, p=[0.55, 0.45]),
    "Pendidikan": np.random.choice(["D3", "D4", "S1", "S2", "S3"], n, p=[0.12, 0.08, 0.55, 0.20, 0.05]),
    "Pengalaman_Thn": 0,
    "Gaji": 0,
    "Jml_Pinjaman": np.random.choice([0, 1, 2, 3, 4], n, p=[0.35, 0.35, 0.18, 0.08, 0.04]),
    "Skor_Kredit": np.random.normal(700, 80, n).clip(400, 900),
    "Kota": np.random.choice(["Jakarta", "Bandung", "Surabaya", "Yogyakarta", "Semarang"], n),
    "Status_Nikah": np.random.choice(["Belum", "Kawin", "Cerai"], n, p=[0.45, 0.45, 0.10]),
})

df["Pengalaman_Thn"] = (df["Umur"] - 22 + np.random.normal(0, 3, n)).clip(0, 40).astype(int)
base_gaji = {"S3": 25, "S2": 15, "S1": 9, "D4": 7.5, "D3": 6.5}
for p, b in base_gaji.items():
    mask = df["Pendidikan"] == p
    df.loc[mask, "Gaji"] = (b + df.loc[mask, "Pengalaman_Thn"] * 0.7 + np.random.normal(0, 1.5, mask.sum())).round(2)
    df.loc[mask, "Gaji"] = df.loc[mask, "Gaji"].clip(4, 40) * 1_000_000

skor_gaji_norm = (df["Gaji"] - df["Gaji"].min()) / (df["Gaji"].max() - df["Gaji"].min())
skor_kredit_norm = (df["Skor_Kredit"] - df["Skor_Kredit"].min()) / (df["Skor_Kredit"].max() - df["Skor_Kredit"].min())
skor_umur_norm = (df["Umur"] - df["Umur"].min()) / (df["Umur"].max() - df["Umur"].min())
peluang = 0.15 + (1 - skor_gaji_norm) * 0.35 + (1 - skor_kredit_norm) * 0.30 + df["Jml_Pinjaman"] * 0.05 + skor_umur_norm * 0.05
df["Churn"] = (np.random.rand(n) < peluang.clip(0.05, 0.9)).astype(int)

df["Gaji_juta"] = df["Gaji"] / 1_000_000

print(f"Dataset Customer Bank: {len(df)} baris x {len(df.columns)} kolom")
print(f"Target variable: Churn (0 = Tetap, 1 = Pindah)")
print(f"Distribusi Churn:\n{df['Churn'].value_counts().rename({0: 'Tetap', 1: 'Pindah'})}")
print(f"Churn rate: {df['Churn'].mean()*100:.1f}%")

print("\n>>> 1. EDA - Univariate Analysis (Satu per satu variabel)")
print("-" * 60)

print("Statistik deskriptif kolom numerik:")
print(df[["Umur", "Pengalaman_Thn", "Gaji_juta", "Skor_Kredit", "Jml_Pinjaman"]].describe().round(2))

print("\nDistribusi kategori:")
for col in ["Jenis_Kelamin", "Pendidikan", "Kota", "Status_Nikah"]:
    dist = df[col].value_counts(normalize=True).round(4) * 100
    print(f"\n  {col:<15}: ", end="")
    print(" | ".join([f"{k} {v:.1f}%" for k, v in dist.items()]))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Univariate Analysis - Variabel Numerik", fontsize=15, fontweight="bold")

sns.histplot(data=df, x="Gaji_juta", kde=True, ax=axes[0, 0], bins=30, color="#2563eb")
axes[0, 0].set_title("Distribusi Gaji (juta Rupiah)")
axes[0, 0].axvline(df["Gaji_juta"].mean(), color="red", linestyle="--", label=f"Mean: {df['Gaji_juta'].mean():.1f}")
axes[0, 0].legend()

sns.histplot(data=df, x="Skor_Kredit", kde=True, ax=axes[0, 1], bins=30, color="#16a34a")
axes[0, 1].set_title("Distribusi Skor Kredit")

sns.histplot(data=df, x="Umur", kde=True, ax=axes[1, 0], bins=25, color="#dc2626")
axes[1, 0].set_title("Distribusi Umur Nasabah")

sns.countplot(data=df, x="Jml_Pinjaman", hue="Jml_Pinjaman", ax=axes[1, 1], palette="viridis", legend=False)
axes[1, 1].set_title("Distribusi Jumlah Pinjaman")

plt.tight_layout()
plt.savefig(f"{output_dir}\\01_univariate_numerik.png", dpi=150)
plt.close()
print("\n✓ Chart 01_univariate_numerik.png tersimpan")

print("\n>>> 2. EDA - Bivariate Analysis (Hubungan 2 variabel)")
print("-" * 60)

print("\nChurn rate berdasarkan kategori:")
for col in ["Jenis_Kelamin", "Pendidikan", "Status_Nikah", "Jml_Pinjaman"]:
    cr = df.groupby(col)["Churn"].mean().round(4) * 100
    print(f"\n  {col}:")
    for k, v in cr.sort_values(ascending=False).items():
        print(f"    {k:<15}: {v:.1f}% churn")

corr_target = df.select_dtypes(include=[np.number]).corr()["Churn"].drop("Churn").sort_values(ascending=False).round(4)
print(f"\nKorelasi Pearson dengan Churn:")
for k, v in corr_target.items():
    bar = "█" * int(abs(v) * 100)
    arah = "+" if v > 0 else "-"
    print(f"  {k:<20}: {v:+.4f}  {arah}{bar}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Bivariate Analysis vs Churn (Target)", fontsize=15, fontweight="bold")

sns.boxplot(data=df, x="Churn", y="Gaji_juta", hue="Churn", ax=axes[0, 0], palette=["#22c55e", "#ef4444"], legend=False)
axes[0, 0].set_title("Gaji vs Churn")
axes[0, 0].set_xticklabels(["Tetap", "Pindah"])

sns.boxplot(data=df, x="Churn", y="Skor_Kredit", hue="Churn", ax=axes[0, 1], palette=["#22c55e", "#ef4444"], legend=False)
axes[0, 1].set_title("Skor Kredit vs Churn")
axes[0, 1].set_xticklabels(["Tetap", "Pindah"])

sns.barplot(data=df, x="Pendidikan", y="Churn", hue="Pendidikan", ax=axes[1, 0], palette="coolwarm", errorbar=None, legend=False,
            order=["D3", "D4", "S1", "S2", "S3"])
axes[1, 0].set_title("Churn Rate per Pendidikan")
axes[1, 0].set_ylabel("Churn Rate (%)")

sns.barplot(data=df, x="Jml_Pinjaman", y="Churn", hue="Jml_Pinjaman", ax=axes[1, 1], palette="magma", errorbar=None, legend=False)
axes[1, 1].set_title("Churn Rate per Jumlah Pinjaman")
axes[1, 1].set_ylabel("Churn Rate (%)")

plt.tight_layout()
plt.savefig(f"{output_dir}\\02_bivariate_vs_churn.png", dpi=150)
plt.close()
print("✓ Chart 02_bivariate_vs_churn.png tersimpan")

print("\n>>> 3. EDA - Multivariate Analysis & Correlation Matrix")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
kolom_numerik = ["Umur", "Pengalaman_Thn", "Gaji_juta", "Skor_Kredit", "Jml_Pinjaman", "Churn"]

corr_matrix = df[kolom_numerik].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1,
            ax=axes[0], mask=mask, fmt=".3f", linewidths=0.5)
axes[0].set_title("Correlation Matrix (Pearson)")

sns.scatterplot(data=df.sample(300), x="Pengalaman_Thn", y="Gaji_juta", hue="Churn",
                style="Jenis_Kelamin", ax=axes[1], alpha=0.7, s=80, palette="Set1")
axes[1].set_title("Pengalaman vs Gaji (warna = Churn, style = Gender)")

plt.tight_layout()
plt.savefig(f"{output_dir}\\03_correlation_multivariate.png", dpi=150)
plt.close()
print("✓ Chart 03_correlation_multivariate.png tersimpan")

sns.pairplot(df[["Umur", "Gaji_juta", "Skor_Kredit", "Churn", "Jml_Pinjaman"]].sample(200),
             hue="Churn", palette=["#22c55e", "#ef4444"], plot_kws={"alpha": 0.6})
plt.suptitle("Pairplot (Scatter Matrix) - 4 Variabel Kunci", y=1.02, fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{output_dir}\\04_pairplot_key_vars.png", dpi=150)
plt.close()
print("✓ Chart 04_pairplot_key_vars.png tersimpan")

print("\n>>> 4. EDA - Business Insight & Rekomendasi")
print("-" * 60)
print("""
Contoh Insight dari EDA di atas (hypotetis):

📌 INSIGHT 1: Semakin tinggi pendidikan, semakin rendah churn
   → Hipotesis: Lulusan S2/S3 lebih loyal karena gaji lebih tinggi
   → Action: Program loyalty khusus nasabah D3/S1 (churn tinggi)

📌 INSIGHT 2: Nasabah dengan skor kredit <600 churn 2x lipat
   → Hipotesis: Kesulitan pembayaran → pindah ke kompetitor
   → Action: Restrukturisasi pinjaman, penawaran tenor lebih panjang

📌 INSIGHT 3: Semakin banyak pinjaman → churn naik linear
   → Hipotesis: Beban finansial terlalu berat
   → Action: Debt consolidation, keringanan bunga untuk 3+ pinjaman

📌 INSIGHT 4: Gaji vs Churn korelasi -0.3 (negatif kuat)
   → Action: Segmentasi nasabah bergaji <7jt → program khusus
""")

print("\n>>> 5. DATA SPLITTING - Train / Validation / Test Set")
print("-" * 60)
print("""
MEMBAGI DATA = PEMISAHAN DUNIA:
  ╔═══════════════════════════════════════════════════════════╗
  ║  TRAIN SET (≈70-80%)  → Model BELAJAR dari data ini      ║
  ║   • Fit model, scaling, encoding DI DATA INI SAJA        ║
  ║   • JANGAN sentuh val/test set sampai waktunya           ║
  ╠═══════════════════════════════════════════════════════════╣
  ║  VALIDATION SET (≈10-15%) → TUNING HYPERPARAMETER        ║
  ║   • Cek overfitting, pilih model terbaik                 ║
  ║   • BoleHL melihat berkali-kali (tuning)                  ║
  ╠═══════════════════════════════════════════════════════════╣
  ║  TEST SET (≈10-15%) → FINAL EVALUATION SEKALI SAJA       ║
  ║   • "Kunci di dalam lemari", buka di AKHIR SAJA           ║
  ║   • Inilah performa model di DUNIA NYATA                  ║
  ╚═══════════════════════════════════════════════════════════╝

⚠  3 ATURAN PENTING SPLITTING:
  1. STRATIFY untuk classification (jaga rasio kelas)
  2. SHUFFLE data sebelum split (kecuali time series)
  3. RANDOM_STATE tetap → reproducible
""")

fitur_cols = ["Umur", "Pengalaman_Thn", "Gaji_juta", "Skor_Kredit", "Jml_Pinjaman"]
X = df[fitur_cols].values
y = df["Churn"].values

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, random_state=42, stratify=y_trainval
)

print("Hasil Train/Val/Test Split:")
for name, Xs, ys in [("TRAIN", X_train, y_train), ("VAL", X_val, y_val), ("TEST", X_test, y_test)]:
    print(f"  {name:<6}: {len(ys):>4} baris ({len(ys)/len(y)*100:>5.1f}%) | "
          f"Churn rate: {ys.mean()*100:.1f}% | Shape X: {Xs.shape}")

print(f"\nTotal data terbagi: {len(y_train)+len(y_val)+len(y_test)} (sesuai total asli: {len(y)}? {len(y_train)+len(y_val)+len(y_test)==len(y)})")

print("\n>>> 6. VALIDASI STRATEGI SPLIT - StratifiedShuffleSplit")
print("-" * 60)

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
print("Cross-check 5 fold stratify → churn rate di setiap fold:")
for i, (train_idx, test_idx) in enumerate(sss.split(X, y), 1):
    print(f"  Fold {i}: Train churn={y[train_idx].mean()*100:.2f}% | Test churn={y[test_idx].mean()*100:.2f}%")

print("""
┌─────────────────────────────────────────────────────────────────┐
│   RANGKUMAN MODUL 3 - DATA PREPARATION & PREPROCESSING         │
├─────────────────────────────────────────────────────────────────┤
│  1. Data Understanding → .info(), .describe(), null% check     │
│  2. Data Cleaning     → Duplikat, Invalid, Null imputation     │
│                        Outlier: IQR / Z-Score                  │
│  3. Transformation    → Label, One-Hot, Ordinal, Target Enc    │
│  4. Feature Scaling   → Standard / MinMax / Robust Scaler      │
│  5. Feature Eng       → Rasio, Binning, Interaksi, Aggregate   │
│  6. EDA               → Univariate → Bivariate → Multivariate  │
│  7. Splitting         → 70/15/15, STRATIFY, SHUFFLE, FIXED SEED│
│                                                                 │
│  🎯 Semua langkah ini: 80% waktu AI Engineer di real project!  │
│     Kualitas data = Kualitas model. GIGO = Garbage In Garbage Out
└─────────────────────────────────────────────────────────────────┘
""")

print("""\n>>> LATIHAN MANDIRI:
1. Tambahkan 3 visualisasi EDA lagi (mis: swarmplot Pendidikan vs Gaji hue=Churn)
2. Lakukan splitting dengan 3 metode berbeda → bandingkan distribusi target
3. Deteksi outlier di kolom Gaji menggunakan 2 metode → bandingkan hasilnya
4. Simpan pipeline preprocessing menggunakan joblib/pickle
""")

print("\n✓ Bagian 3 Modul 3 Selesai: EDA & Data Splitting")
print("\n" + "="*60)
print("MODUL 3 SELESAI - Data Preparation & Preprocessing")
print("="*60)
