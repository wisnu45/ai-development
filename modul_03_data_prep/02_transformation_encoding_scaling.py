print("="*60)
print("MODUL 3: DATA PREPARATION & PREPROCESSING")
print("Bagian 2: Data Transformation, Encoding & Feature Scaling")
print("="*60)

import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

data = pd.DataFrame({
    "Umur": np.random.randint(18, 65, n),
    "Gaji": np.random.randint(4_000_000, 25_000_000, n),
    "Pengalaman_Thn": np.random.randint(0, 30, n),
    "Jenis_Kelamin": np.random.choice(["Laki-laki", "Perempuan"], n),
    "Pendidikan": np.random.choice(["D3", "D4", "S1", "S2", "S3"], n, p=[0.15, 0.1, 0.55, 0.15, 0.05]),
    "Kota": np.random.choice(["Jakarta", "Bandung", "Surabaya", "Yogyakarta", "Medan", "Semarang"], n),
    "Status_Nikah": np.random.choice(["Belum", "Kawin", "Cerai"], n, p=[0.45, 0.45, 0.10]),
    "Rating_Kinerja": np.random.choice(["Buruk", "Cukup", "Baik", "Sangat Baik", "Luar Biasa"], n),
    "Produk_Favorit": np.random.choice(["A", "B", "C", "D", "E"], n),
    "Hasil_Churn": np.random.choice([0, 1], n, p=[0.7, 0.3])
})

print(f"Dataset berisi {len(data)} baris, {len(data.columns)} kolom")
print("Sample data:")
print(data.head())

print("\n>>> 1. CATEGORICAL ENCODING")
print("-" * 60)
print("""
JENIS-JENIS ENCODING:
  1. Label Encoding       → Ordinal (punya urutan: Buruk < Cukup < Baik)
  2. One-Hot Encoding     → Nominal (tidak punya urutan: Jakarta, Bandung)
  3. Ordinal Encoding     → Custom order mapping
  4. Binary Encoding      → Kategori banyak (>10), hemat dimensi
  5. Target Encoding      → Berdasarkan mean target variable (bahaya leakage!)
  6. Frequency Encoding   → Berdasarkan frekuensi kemunculan
""")

print("  1a. LABEL ENCODING (gunakan pd.factorize atau sklearn)")
data_encoded = data.copy()
data_encoded["Jenis_Kelamin_Encoded"], _ = pd.factorize(data_encoded["Jenis_Kelamin"])
print(data_encoded[["Jenis_Kelamin", "Jenis_Kelamin_Encoded"]].drop_duplicates())

print("\n  1b. ORDINAL ENCODING (Rating_Kinerja dengan urutan jelas)")
ordinal_map = {
    "Buruk": 0, "Cukup": 1, "Baik": 2, "Sangat Baik": 3, "Luar Biasa": 4
}
data_encoded["Rating_Kinerja_Ordinal"] = data_encoded["Rating_Kinerja"].map(ordinal_map)
print(data_encoded[["Rating_Kinerja", "Rating_Kinerja_Ordinal"]].drop_duplicates().sort_values("Rating_Kinerja_Ordinal"))

print("\n  1c. ONE-HOT ENCODING (Status_Nikah - 3 kategori nominal)")
dummies_nikah = pd.get_dummies(data_encoded["Status_Nikah"], prefix="Nikah", dtype=int)
print(f"One-Hot shape: {dummies_nikah.shape}, Kolom: {dummies_nikah.columns.tolist()}")
print(dummies_nikah.head(5))

print("\n  1d. DUMMY TRAP AVOIDANCE - drop_first=True")
dummies_pendidikan = pd.get_dummies(data_encoded["Pendidikan"], prefix="Pend", drop_first=True, dtype=int)
print(f"Pendidikan asli: {sorted(data['Pendidikan'].unique())}")
print(f"One-Hot kolom (drop first): {dummies_pendidikan.columns.tolist()}")

print("\n  1e. FREQUENCY / COUNT ENCODING (untuk Kota - banyak kategori)")
freq_map = data_encoded["Kota"].value_counts().to_dict()
data_encoded["Kota_FreqEnc"] = data_encoded["Kota"].map(freq_map)
print(freq_map)
print(data_encoded[["Kota", "Kota_FreqEnc"]].drop_duplicates().head())

print("\n  1f. TARGET ENCODING (Perhatikan DATA LEAKAGE!)")
target_map = data_encoded.groupby("Produk_Favorit")["Hasil_Churn"].mean().to_dict()
data_encoded["Produk_TargetEnc"] = data_encoded["Produk_Favorit"].map(target_map)
print("Target Mean (Churn rate) per Produk_Favorit:")
for k, v in sorted(target_map.items()):
    print(f"  Produk {k}: {v:.3f}")

print("\n>>> 2. FEATURE SCALING / NORMALIZATION")
print("-" * 60)
print("""
ALASAN SCALING:
  • Algoritma berbasis JARAK (KNN, SVM, K-Means) → sensitif skala
  • Gradient Descent (Linear Reg, Neural Network) → konvergen lebih cepat
  • PCA → fitur dengan skala besar mendominasi
  
JENIS SCALING:
  ┌──────────────────┬─────────────────────────────────────┐
  │ Standard Scaler  │ (x - mean) / std  → mean=0, std=1   │
  │                  │ Paling umum, ada outlier? Hati-hati │
  ├──────────────────┼─────────────────────────────────────┤
  │ Min-Max Scaler   │ (x - min) / (max - min) → [0,1]     │
  │                  │ Neural Network, image pixel         │
  ├──────────────────┼─────────────────────────────────────┤
  │ Robust Scaler    │ (x - Q2) / IQR  → TAHAN OUTLIER     │
  │                  │ Data banyak outlier                 │
  ├──────────────────┼─────────────────────────────────────┤
  │ Normalizer (L2)  │ Per baris, magnitude vektor = 1     │
  │                  │ Text / embedding similarity         │
  └──────────────────┴─────────────────────────────────────┘
""")

kolom_numerik = ["Umur", "Gaji", "Pengalaman_Thn"]
print("Data sebelum scaling:")
print(data_encoded[kolom_numerik].describe().round(2))

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

sc_std = StandardScaler()
data_scaled = data_encoded.copy()
data_scaled[["Umur_std", "Gaji_std", "Pengalaman_std"]] = sc_std.fit_transform(data_encoded[kolom_numerik])

sc_mm = MinMaxScaler()
data_scaled[["Umur_mm", "Gaji_mm", "Pengalaman_mm"]] = sc_mm.fit_transform(data_encoded[kolom_numerik])

sc_rb = RobustScaler()
data_scaled[["Umur_rb", "Gaji_rb", "Pengalaman_rb"]] = sc_rb.fit_transform(data_encoded[kolom_numerik])

print("\nPerbandingan hasil scaling (Gaji):")
gaji_scaled = data_scaled[["Gaji", "Gaji_std", "Gaji_mm", "Gaji_rb"]].describe().round(3)
print(gaji_scaled)

print("\n>>> 3. FEATURE ENGINEERING - Membuat Fitur Baru")
print("-" * 60)
print("""
FEATURE ENGINEERING = "Proses mengubah data mentah menjadi fitur
                      yang berguna untuk model AI"

SUMBER IDE:
  • Domain Knowledge (pengetahuan bisnis)
  • Interaksi antar fitur (perkalian, pembagian, rasio)
  • Polynomial feature (x², x³, xy)
  • Binning / Discretization
  • Extraction dari text / tanggal
""")

df_fe = data_encoded.copy()

print("  3a. Feature interaksi & rasio bisnis")
df_fe["Gaji_Per_Tahun_Pengalaman"] = df_fe["Gaji"] / (df_fe["Pengalaman_Thn"] + 1)
df_fe["Produktivitas_Gaji"] = (df_fe["Rating_Kinerja_Ordinal"] + 1) / df_fe["Gaji"] * 1_000_000

print("  3b. Binning / Diskretisasi Umur → Kelompok Umur")
bins_umur = [0, 25, 35, 45, 55, 100]
labels_umur = ["Gen-Z", "Muda-Mapan", "Paruh-Baya", "Senior", "Purnabakti"]
df_fe["Kelompok_Umur"] = pd.cut(df_fe["Umur"], bins=bins_umur, labels=labels_umur)
print(df_fe[["Umur", "Kelompok_Umur"]].drop_duplicates().sort_values("Umur").head(8))

print("\n  3c. Gaji Percentile Ranking")
df_fe["Gaji_Percentile"] = df_fe["Gaji"].rank(pct=True).round(4) * 100

print("  3d. Binning Quantile (Gaji → Low/Medium/High)")
df_fe["Gaji_Kuantil"] = pd.qcut(df_fe["Gaji"], q=3, labels=["Gaji_Rendah", "Gaji_Sedang", "Gaji_Tinggi"])
print(f"Distribusi Gaji_Kuantil:")
print(df_fe["Gaji_Kuantil"].value_counts().sort_index())

print("\n  3e. Matematis & Statistik Fitur")
df_fe["Total_Skor_Sosial_Ekonomi"] = (
    df_fe["Umur"].rank(pct=True) * 0.2 +
    df_fe["Gaji"].rank(pct=True) * 0.5 +
    df_fe["Pengalaman_Thn"].rank(pct=True) * 0.3
).round(4)

statistik_pendidikan = df_fe.groupby("Pendidikan")["Gaji"].agg(["mean", "std", "min", "max"])
df_fe = df_fe.merge(statistik_pendidikan.add_prefix("Gaji_Pend_"), left_on="Pendidikan", right_index=True, how="left")

kolom_baru = [c for c in df_fe.columns if c not in data.columns]
print(f"\n✅ Jumlah feature baru dibuat: {len(kolom_baru)}")
print(f"   Daftar: {', '.join(kolom_baru)}")

print("\n>>> 4. FINAL PREPROCESSING PIPELINE (Preparasi untuk Modeling)")
print("-" * 60)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

target = "Hasil_Churn"
fitur_kolom = [c for c in df_fe.columns if c != target and df_fe[c].dtype in [np.int64, np.float64, int, float]]

numerik_cols = df_fe[fitur_kolom].select_dtypes(include=[np.number]).columns.tolist()
kategori_cols = ["Jenis_Kelamin", "Pendidikan", "Kota", "Status_Nikah", "Rating_Kinerja", "Kelompok_Umur", "Gaji_Kuantil"]

numerik_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

kategori_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerik_pipeline, numerik_cols),
        ("cat", kategori_pipeline, kategori_cols)
    ],
    remainder="drop"
)

print(f"Target variable      : {target}")
print(f"Fitur numerik ({len(numerik_cols)})  : {numerik_cols[:8]}...")
print(f"Fitur kategori ({len(kategori_cols)}) : {kategori_cols}")

X_processed = preprocessor.fit_transform(df_fe)
y = df_fe[target].values

feature_names = (
    numerik_cols +
    preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(kategori_cols).tolist()
)

print(f"\n✅ Pipeline selesai!")
print(f"   Shape X setelah preprocess: {X_processed.shape}")
print(f"   Shape y (target)         : {y.shape}")
print(f"   Total fitur (expanded)   : {len(feature_names)}")
print(f"   Contoh 10 nama fitur     : {feature_names[:10]}")

print("""
┌─────────────────────────────────────────────────────────────────┐
│           RANGKUMAN TRANSFORMASI DATA & FEATURE                 │
├─────────────────────────────────────────────────────────────────┤
│  Encoding      →  Label / Ordinal / One-Hot / Target Enc       │
│  Scaling       →  Standard / MinMax / Robust (sesuai algoritma)│
│  Feature Eng   →  Rasio, Binning, Interaksi, Aggregate, dll    │
│  Pipeline      →  Sklearn Pipeline + ColumnTransformer          │
│                 (untuk deployment / reproducibility)            │
│                                                                 │
│  ⚠  Penting: Scaler/Fit di FIT ONLY di TRAIN SET,               │
│     TRANSFORM ke Test/Val set. JANGAN fit keseluruhan data!     │
│     (Cegah DATA LEAKAGE → model overfit palsu)                  │
└─────────────────────────────────────────────────────────────────┘
""")

print("""\n>>> LATIHAN:
1. Ubah Pipeline di atas → ganti StandardScaler dengan RobustScaler
2. Tambahkan PolynomialFeatures degree=2 untuk 3 kolom numerik
3. Hitung berapa banyak fitur yang dihasilkan setelah polynomial
""")

print("\n✓ Bagian 2 Modul 3 Selesai: Transformation, Encoding, Scaling")
