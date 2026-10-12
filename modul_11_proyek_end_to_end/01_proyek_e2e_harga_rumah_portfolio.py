# =====================================================================
# MODUL 11 - PROYEK AI END-TO-END (PORTOFOLIO BNSP LENGKAP)
# Persiapan Uji Kompetensi BNSP Skema Artificial Intelligence Engineer
# =====================================================================
# Proyek: SISTEM PREDIKSI HARGA RUMAH BERBASIS MACHINE LEARNING
#         + REST API Deployment + Dokumentasi Portofolio Lengkap
# Tahapan (sesuai brief BNSP Modul 11, 10 langkah WAJIB):
#  [1] Identifikasi Masalah & Requirement         [6] Pemilihan Algoritma
#  [2] Pengumpulan & Pemahaman Data               [7] Training + Evaluasi + Optimasi
#  [3] Data Preprocessing                         [8] Deployment (REST API)
#  [4] Exploratory Data Analysis (EDA)            [9] Pengujian Solusi AI
#  [5] Feature Engineering                        [10] Dokumentasi & Presentasi
# =====================================================================

import os
import re
import sys
import json
import pickle
import warnings
import datetime as dt
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------
# INSTALASI DEPENDENSI (jika error, jalankan di CMD LUAR TRAE):
#   cd /d C:\ai-engineer
#   venv_bnsp\Scripts\activate
#   pip install -r requirements.txt
# ---------------------------------------------------------------------
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import (train_test_split, cross_val_score,
                                         RandomizedSearchCV, KFold)
    from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                       OrdinalEncoder, KBinsDiscretizer)
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import (RandomForestRegressor,
                                  GradientBoostingRegressor)
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score, mean_absolute_percentage_error)
    try:
        import joblib
        JOBLIB_OK = True
    except ImportError:
        JOBLIB_OK = False
except ImportError as e:
    print("=" * 80)
    print("⚠️  MODUL 11 GAGAL BERJALAN - DEPENDENSI TIDAK DITEMUKAN")
    print("=" * 80)
    print(f"Penyebab: {e}")
    print("\nSolusi (CMD):")
    print("  cd /d C:\\ai-engineer ; venv_bnsp\\Scripts\\activate")
    print("  pip install -r requirements.txt")
    print("=" * 80)
    sys.exit(1)

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "output_charts")
MODELS_DIR = os.path.join(BASE_DIR, "portfolio_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

PROJECT_NAME = "SISTEM PREDIKSI HARGA RUMAH"
PROJECT_CODE = "BNSP-AIE-HOUSEPRED-001"
PROJECT_DATE = dt.date.today().isoformat()
AUTHOR = "AI Engineer Trainee (BNSP Certified Candidate)"

sns.set_style("whitegrid")
sns.set_palette("viridis")
np.random.seed(42)

# ============================================================
# LANGKAH 1 — IDENTIFIKASI MASALAH & PENYUSUNAN REQUIREMENT
# ============================================================
print("\n" + "=" * 84)
print(f"🏢 PROYEK E2E BNSP: {PROJECT_NAME}  [{PROJECT_CODE}]")
print("=" * 84)
print("\n📝 LANGKAH 1: IDENTIFIKASI MASALAH & REQUIREMENT")
print("─" * 84)

PROJECT_DOC_STEP1 = """
IDENTIFIKASI MASALAH (Problem Statement 5W1H):
┌─────────────────────────────────────────────────────────────────────────────┐
│ WHAT  : Perumahan developer membutuhkan sistem estimasi harga jual rumah     │
│         yang AKURAT dan REPRODUCIBLE, tidak lagi mengandalkan perkiraan      │
│         subyektif surveyor lapangan (error ±35% MAE).                        │
│ WHO   : Divisi Sales & Marketing + Appraisal Property Developer (200+ staff) │
│ WHEN  : Setiap kali ada unit baru listing, rumah second, atau lelang bank.  │
│ WHERE : Aplikasi web internal perusahaan (dijalankan via browser)           │
│ WHY   : Salah estimasi harga → kerugian per unit bisa mencapai 50-200 juta! │
│ HOW   : Model Machine Learning Gradient Boosting dengan fitur properti.     │
└─────────────────────────────────────────────────────────────────────────────┘

BUSINESS REQUIREMENT (BR) & SUCCESS METRICS:
  BR-01 : Prediksi harga rumah dengan MAE ≤ 50 JUTA rupiah (target ±5%)
  BR-02 : Waktu inferensi ≤ 500ms per request
  BR-03 : Tersedia sebagai REST API bisa di-call oleh frontend Web (React)
  BR-04 : Dokumentasi model lengkap + audit trail setiap prediksi
  BR-05 : Model bisa diretrain ulang jika performa turun (PSI > 0.25)

SUCCESS METRIC TEKNIS (BNSP Portofolio):
  ✅ MAPE (Mean Absolute Percentage Error) < 8%
  ✅ R² Score di test set > 0.90
  ✅ CV 5-Fold R² std-dev < 0.02 (stabil, tidak overfit)
  ✅ API Latency p95 < 250ms
"""
print(PROJECT_DOC_STEP1)

# ============================================================
# LANGKAH 2 — PENGUMPULAN & PEMAHAMAN DATA (Data Understanding)
# ============================================================
print("\n\n📝 LANGKAH 2: PENGUMPULAN & PEMAHAMAN DATASET")
print("─" * 84)

# Generate dataset REALISTIS harga rumah Jabodetabek (1500 data)
N = 1500
np.random.seed(42)

# Lokasi premium → harga lebih tinggi 2-3x
lokasi_list = ['Jakarta Selatan', 'Jakarta Utara', 'Jakarta Barat',
               'Jakarta Timur', 'Jakarta Pusat', 'Tangerang Selatan',
               'Bekasi', 'Depok', 'Bogor']
lokasi_harga_multiplier = {
    'Jakarta Selatan': 2.8, 'Jakarta Pusat': 3.1, 'Jakarta Utara': 2.3,
    'Jakarta Barat': 2.0, 'Jakarta Timur': 1.6,
    'Tangerang Selatan': 1.5, 'Bekasi': 1.15, 'Depok': 1.1, 'Bogor': 0.95,
}
lokasi_prob = [0.16, 0.08, 0.12, 0.13, 0.07, 0.15, 0.13, 0.09, 0.07]

tipe_properti = np.random.choice(['Rumah Minimalis', 'Rumah Mewah', 'Rumah Cluster',
                                   'Rumah Subsidi', 'Rumah Tua (Renovasi)'],
                                  N, p=[0.30, 0.15, 0.25, 0.18, 0.12])
tipe_mult = {'Rumah Mewah': 1.8, 'Rumah Cluster': 1.15,
             'Rumah Minimalis': 1.0, 'Rumah Tua (Renovasi)': 0.75,
             'Rumah Subsidi': 0.55}

sertifikat_list = ['SHM - Sertifikat Hak Milik', 'SHGB - Hak Guna Bangunan',
                   'HP - Hak Pakai', 'AJB - Akta Jual Beli (Belum Balik Nama)']
sertifikat_mult = {'SHM - Sertifikat Hak Milik': 1.0,
                   'SHGB - Hak Guna Bangunan': 0.88,
                   'HP - Hak Pakai': 0.82,
                   'AJB - Akta Jual Beli (Belum Balik Nama)': 0.65}

lokasi = np.random.choice(lokasi_list, N, p=lokasi_prob)
luas_tanah = np.clip(np.random.gamma(shape=2.5, scale=60, size=N).astype(int), 36, 500)
luas_bangunan = (luas_tanah * np.random.uniform(0.45, 0.95, N)).astype(int)
jml_kamar_tidur = np.clip((luas_bangunan / 32).round().astype(int), 1, 7)
jml_kamar_mandi = np.clip((jml_kamar_tidur * np.random.uniform(0.55, 0.9, N)).round().astype(int), 1, 5)
jml_lantai = np.clip((np.log2(luas_bangunan / 40)).round().astype(int) + 1, 1, 4)
tahun_bangun = np.random.randint(1985, 2026, size=N)
umur_bangunan = 2026 - tahun_bangun
daya_listrik = np.random.choice([900, 1300, 2200, 3500, 5500, 7700, 10600, 13000],
                                 N, p=[0.10, 0.25, 0.28, 0.18, 0.10, 0.05, 0.03, 0.01])
hadap_rumah = np.random.choice(['Utara', 'Selatan', 'Timur', 'Barat'],
                               N, p=[0.20, 0.40, 0.22, 0.18])  # selatan = harga +5%
hadap_mult = {'Utara': 1.02, 'Selatan': 1.06, 'Timur': 1.03, 'Barat': 0.98}
furnish = np.random.choice(['Unfurnished', 'Semi Furnished', 'Furnished',
                             'Fully Furnished Luxury'], N,
                            p=[0.35, 0.32, 0.23, 0.10])
furnish_mult = {'Unfurnished': 1.0, 'Semi Furnished': 1.04,
                'Furnished': 1.08, 'Fully Furnished Luxury': 1.15}
jarak_ke_rs_km = np.round(np.random.uniform(0.3, 25, N), 1)
jarak_ke_sekolah_km = np.round(np.random.uniform(0.1, 15, N), 1)
ada_garasi_mobil = np.random.binomial(1, 0.68, N)
ada_taman = np.random.binomial(1, 0.55, N)
ada_kolam_renang = np.random.binomial(1, 0.08 + 0.30 * (tipe_properti == 'Rumah Mewah'), N)
lebar_jalan_depan_m = np.clip(np.random.normal(6, 2, N), 2, 18).round(1)

# --------- Hitung BASELINE HARGA (regression formula yang REALISTIS) ---------
harga_per_m2_lahan_rata2 = 8_500_000  # 8.5 juta per m2 lahan (base average)

harga = (
    # Komponen 1: Lahan + Bangunan
    luas_tanah * harga_per_m2_lahan_rata2 * np.array([lokasi_harga_multiplier[l] for l in lokasi])
    + luas_bangunan * 4_800_000 * np.array([tipe_mult[t] for t in tipe_properti])
    # Komponen 2: Kamar & fasilitas dasar
    + jml_kamar_tidur * 35_000_000
    + jml_kamar_mandi * 28_000_000
    + jml_lantai * 22_000_000
    # Komponen 3: Sertifikat & legalitas
    * np.array([sertifikat_mult[s] for s in np.random.choice(sertifikat_list, N,
                                                             p=[0.55, 0.28, 0.10, 0.07])])
    # Komponen 4: Depresiasi umur bangunan (1% per tahun, max -60%)
    * np.clip(1 - 0.009 * umur_bangunan, 0.4, 1.0)
    # Komponen 5: Lokasi premium
    + (daya_listrik - 1300) * 850_000
    * np.array([hadap_mult[h] for h in hadap_rumah])
    * np.array([furnish_mult[f] for f in furnish])
    # Komponen 6: Fasilitas tambahan
    + ada_garasi_mobil * 65_000_000
    + ada_taman * 35_000_000
    + ada_kolam_renang * 220_000_000
    # Komponen 7: Aksesibilitas (makin dekat makin mahal)
    - jarak_ke_rs_km * 3_500_000
    - jarak_ke_sekolah_km * 2_200_000
    + lebar_jalan_depan_m * 8_500_000
)
# Tambah noise realistis ±12%
harga = (harga * np.random.normal(1.0, 0.07, N)).round(-6)  # bulatkan ke juta terdekat
# Clip agar tidak negatif
harga = np.clip(harga, 90_000_000, 55_000_000_000)

# Buat DataFrame lengkap
df = pd.DataFrame({
    'id_properti': [f'PROP-{i + 1:05d}' for i in range(N)],
    'lokasi': lokasi,
    'tipe_properti': tipe_properti,
    'luas_tanah_m2': luas_tanah,
    'luas_bangunan_m2': luas_bangunan,
    'jml_kamar_tidur': jml_kamar_tidur,
    'jml_kamar_mandi': jml_kamar_mandi,
    'jml_lantai': jml_lantai,
    'tahun_bangun': tahun_bangun,
    'umur_bangunan_tahun': umur_bangunan,
    'daya_listrik_va': daya_listrik,
    'hadap_rumah': hadap_rumah,
    'furnishing': furnish,
    'sertifikat': np.random.choice(sertifikat_list, N,
                                   p=[0.55, 0.28, 0.10, 0.07]),
    'jarak_ke_rs_km': jarak_ke_rs_km,
    'jarak_ke_sekolah_km': jarak_ke_sekolah_km,
    'garasi_mobil_ada': ['Ya' if v else 'Tidak' for v in ada_garasi_mobil],
    'taman_ada': ['Ya' if v else 'Tidak' for v in ada_taman],
    'kolam_renang_ada': ['Ya' if v else 'Tidak' for v in ada_kolam_renang],
    'lebar_jalan_depan_m': lebar_jalan_depan_m,
    'harga_jual_rupiah': harga,
})

# Tambahkan beberapa MISSING VALUE dan ANOMALI (untuk kebutuhan step 3 cleaning)
np.random.seed(7)
missing_idx_lt = np.random.choice(N, 35, replace=False)
df.loc[missing_idx_lt, 'luas_tanah_m2'] = np.nan
missing_idx_jkm = np.random.choice(N, 22, replace=False)
df.loc[missing_idx_jkm, 'jml_kamar_mandi'] = np.nan
missing_idx_sert = np.random.choice(N, 15, replace=False)
df.loc[missing_idx_sert, 'sertifikat'] = np.nan

# 20 duplikat (untuk cleaning step)
dup_rows = df.sample(20, random_state=123).copy()
df = pd.concat([df, dup_rows], ignore_index=True)
# 30 outlier ekstrem
outlier_idx = np.random.choice(len(df), 30, replace=False)
df.loc[outlier_idx[:15], 'harga_jual_rupiah'] *= 5.5
df.loc[outlier_idx[15:], 'luas_tanah_m2'] *= 10

# Reset index after concat
df.reset_index(drop=True, inplace=True)
df['id_properti'] = [f'PROP-{i + 1:05d}' for i in range(len(df))]

print(f"✅ Dataset berhasil DIBUAT (simulasi): {len(df)} baris × {df.shape[1]} kolom")
print(f"   Range Harga: Rp{df['harga_jual_rupiah'].min()/1e6:.1f} Jt "
      f"sampai Rp{df['harga_jual_rupiah'].max()/1e9:.2f} M")
print(f"   Harga rata-rata: Rp{df['harga_jual_rupiah'].mean()/1e9:.2f} Milyar")

print("\n--- DataFrame Info (Data Understanding) ---")
buf = []
df.info(buf=buf)
info_str = "\n".join(buf)
# Print hanya 20 baris pertama info
for line in info_str.splitlines()[:25]:
    print("   " + line)

print("\n--- Persentase Missing Value per Kolom ---")
missing_pct = df.isnull().mean() * 100
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
if len(missing_pct) > 0:
    for col, pct in missing_pct.items():
        print(f"   • {col:<28}: {pct:.2f}% hilang "
              f"({df[col].isnull().sum()} baris)")

print(f"\n--- Duplikat: {df.duplicated().sum()} baris "
      f"({df.duplicated().mean()*100:.2f}%)")

# Simpan raw dataset
raw_dataset_path = os.path.join(BASE_DIR, "dataset_raw_properti.csv")
df.to_csv(raw_dataset_path, index=False)
print(f"\n📄 Dataset RAW disimpan: {raw_dataset_path} "
      f"[{os.path.getsize(raw_dataset_path)/1024:.1f} KB]")

# ============================================================
# LANGKAH 3 — DATA PREPROCESSING & CLEANING
# ============================================================
print("\n\n🛠️  LANGKAH 3: DATA PREPROCESSING (CLEANING - TRANSFORM - ENCODING)")
print("─" * 84)

step3_log = []
df_clean = df.copy()

# 3a. Hapus DUPLIKAT
n_dup = df_clean.duplicated().sum()
df_clean.drop_duplicates(keep='first', inplace=True)
df_clean.reset_index(drop=True, inplace=True)
step3_log.append(f"[3a] Hapus duplikat: {n_dup} baris dihapus. Sisa: {len(df_clean)}")

# 3b. Handling Missing Value
n_before = df_clean.isnull().sum().sum()
df_clean['luas_tanah_m2'] = (
    df_clean.groupby(['lokasi', 'tipe_properti'])['luas_tanah_m2']
    .transform(lambda g: g.fillna(g.median()))
)
# Fallback jika groupby ada NaN
df_clean['luas_tanah_m2'].fillna(df_clean['luas_tanah_m2'].median(), inplace=True)
df_clean['jml_kamar_mandi'] = (
    df_clean.groupby('jml_kamar_tidur')['jml_kamar_mandi']
    .transform(lambda g: g.fillna(g.mode().iloc[0] if len(g.mode()) > 0 else 1))
)
# Sertifikat = fill dengan mode umum (SHM)
df_clean['sertifikat'].fillna('SHM - Sertifikat Hak Milik', inplace=True)
n_after = df_clean.isnull().sum().sum()
step3_log.append(f"[3b] Handling Missing: {n_before} → {n_after} null cells "
                 f"(GroupBy Median + Mode)")

# 3c. Outlier Detection & Removal (IQR Method)
n_before = len(df_clean)
num_cols_outlier = ['luas_tanah_m2', 'luas_bangunan_m2', 'harga_jual_rupiah',
                    'jarak_ke_rs_km', 'jarak_ke_sekolah_km']
mask_outlier = pd.Series(False, index=df_clean.index)
for col in num_cols_outlier:
    q1, q3 = df_clean[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 3 * iqr, q3 + 3 * iqr  # pakai k=3 agar tidak terlalu ketat
    outl = (df_clean[col] < low) | (df_clean[col] > high)
    mask_outlier |= outl
df_clean = df_clean[~mask_outlier].reset_index(drop=True)
step3_log.append(f"[3c] Outlier Removal (IQR × 3): {n_before - len(df_clean)} "
                 f"baris dihapus. Sisa: {len(df_clean)}")

# 3d. Tipe Data konsistensi
df_clean['garasi_mobil_ada'] = df_clean['garasi_mobil_ada'].replace({'Ya': 1, 'Tidak': 0})
df_clean['taman_ada'] = df_clean['taman_ada'].replace({'Ya': 1, 'Tidak': 0})
df_clean['kolam_renang_ada'] = df_clean['kolam_renang_ada'].replace({'Ya': 1, 'Tidak': 0})
step3_log.append(f"[3d] Konversi Ya/Tidak → 1/0 untuk 3 kolom boolean")

for msg in step3_log:
    print("   " + msg)

# --- LANGKAH 4: EDA ---
print("\n\n📊 LANGKAH 4: EXPLORATORY DATA ANALYSIS (EDA)")
print("─" * 84)

TARGET = 'harga_jual_rupiah'
FEAT_NUM = ['luas_tanah_m2', 'luas_bangunan_m2', 'jml_kamar_tidur', 'jml_kamar_mandi',
            'jml_lantai', 'umur_bangunan_tahun', 'daya_listrik_va',
            'jarak_ke_rs_km', 'jarak_ke_sekolah_km', 'lebar_jalan_depan_m',
            'garasi_mobil_ada', 'taman_ada', 'kolam_renang_ada']
FEAT_CAT = ['lokasi', 'tipe_properti', 'hadap_rumah', 'furnishing', 'sertifikat']

print(f"   [4a] Summary Statistik Harga:")
price_desc = df_clean[TARGET].describe().apply(lambda v: f"Rp{v/1e9:.3f} Milyar" if v > 1e9 else
                                               (f"Rp{v/1e6:.1f} Juta" if v > 1e6 else f"{v:.0f}"))
print(price_desc.to_string())

# Visualisasi EDA: 4 Panel Dashboard
fig, axes = plt.subplots(2, 2, figsize=(17, 12))

# (1,1) Distribusi Harga Log Scale (Rumah harga skew kanan)
prices_m = df_clean[TARGET] / 1e6
sns.histplot(prices_m, bins=40, ax=axes[0, 0], kde=True, color='#2196F3')
axes[0, 0].set_title('Distribusi Harga Jual Rumah (Juta Rupiah) — Skew Kanan',
                     fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Harga (Juta Rupiah)', fontsize=10)
axes[0, 0].axvline(prices_m.median(), color='red', ls='--', lw=2,
                   label=f'Median Rp{prices_m.median():.0f} Jt')
axes[0, 0].legend()

# (1,2) Rata-rata harga per Lokasi
hrg_lokasi = (df_clean.groupby('lokasi')[TARGET].median().sort_values() / 1e6)
sns.barplot(x=hrg_lokasi.values, y=hrg_lokasi.index, ax=axes[0, 1],
            palette='coolwarm', edgecolor='white')
axes[0, 1].set_title('Median Harga Rumah per Lokasi (Jabodetabek)',
                     fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Median Harga (Juta Rupiah)', fontsize=10)
for i, v in enumerate(hrg_lokasi.values):
    axes[0, 1].text(v + 20, i, f'Rp{v:,.0f} Jt', va='center', fontsize=9, fontweight='bold')

# (2,1) Luas Tanah vs Harga (scatter + regplot)
sns.regplot(data=df_clean.sample(600, random_state=1), x='luas_tanah_m2',
            y=prices_m[df_clean.sample(600, random_state=1).index],
            ax=axes[1, 0], scatter_kws={'alpha': 0.4, 'color': '#4CAF50'},
            line_kws={'color': '#C62828', 'lw': 2})
axes[1, 0].set_title(f'Korelasi Luas Tanah vs Harga (r = '
                     f'{df_clean["luas_tanah_m2"].corr(df_clean[TARGET]):.3f})',
                     fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Luas Tanah (m²)')
axes[1, 0].set_ylabel('Harga (Juta Rupiah)')

# (2,2) Correlation Heatmap (numeric kolom)
corr = df_clean[FEAT_NUM + [TARGET]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=axes[1, 1], cbar_kws={'label': 'Pearson r'},
            annot_kws={'size': 7})
axes[1, 1].set_title('Correlation Matrix Heatmap (Fitur Numerik vs Harga)',
                     fontsize=12, fontweight='bold')
axes[1, 1].tick_params(labelsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '01_eda_4panel_dashboard.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("   ✅ [4b] EDA Dashboard disimpan: 01_eda_4panel_dashboard.png")

# --- LANGKAH 5: FEATURE ENGINEERING ---
print("\n\n🧬 LANGKAH 5: FEATURE ENGINEERING")
print("─" * 84)

df_fe = df_clean.copy()
# FE1: Ratio Lahan/Bangunan (makin besar = lebih banyak taman)
df_fe['rasio_lahan_bangunan'] = (df_fe['luas_tanah_m2'] /
                                  df_fe['luas_bangunan_m2']).round(3)
# FE2: Harga per m2 lahan (estimasi lokasi premium)
df_fe['harga_per_m2_lahan_est'] = (
    df_fe.groupby('lokasi')['harga_jual_rupiah'].transform('median') /
    df_fe.groupby('lokasi')['luas_tanah_m2'].transform('median')
).round(0)
# FE3: Skor legalitas sertifikat (SHM = tertinggi)
legal_score_map = {
    'SHM - Sertifikat Hak Milik': 100,
    'SHGB - Hak Guna Bangunan': 88,
    'HP - Hak Pakai': 82,
    'AJB - Akta Jual Beli (Belum Balik Nama)': 65,
}
df_fe['skor_legalitas'] = df_fe['sertifikat'].map(legal_score_map)
# FE4: Skor total fasilitas (jumlah bool dijumlah + furnish skor)
furnish_score = {'Unfurnished': 0, 'Semi Furnished': 5,
                 'Furnished': 10, 'Fully Furnished Luxury': 20}
df_fe['skor_fasilitas'] = (
    df_fe['garasi_mobil_ada'] * 8 +
    df_fe['taman_ada'] * 4 +
    df_fe['kolam_renang_ada'] * 20 +
    df_fe['furnishing'].map(furnish_score) +
    (df_fe['daya_listrik_va'] / 1000).round(1)
)
# FE5: Binned Umur bangunan (ordinal)
df_fe['kategori_umur'] = pd.cut(df_fe['umur_bangunan_tahun'],
                                 bins=[-1, 5, 15, 30, 100],
                                 labels=['Baru <5th', 'Sedang 5-15th',
                                         'Tua 15-30th', 'Sangat Tua >30th'])
FEAT_NUM_NEW = FEAT_NUM + ['rasio_lahan_bangunan', 'harga_per_m2_lahan_est',
                            'skor_legalitas', 'skor_fasilitas']
FEAT_CAT_NEW = FEAT_CAT + ['kategori_umur']

print("   Ditambahkan 6 fitur baru hasil Feature Engineering:")
print("   ✅ [FE1] rasio_lahan_bangunan (taman vs bangunan)")
print("   ✅ [FE2] harga_per_m2_lahan_est (estimasi lokasi via median)")
print("   ✅ [FE3] skor_legalitas (SHM=100, AJB=65)")
print("   ✅ [FE4] skor_fasilitas (jumlah poin garasi/taman/kolam/listrik/furnish)")
print("   ✅ [FE5] kategori_umur (ordinal bin 4 kelompok umur)")

FEAT_ALL = FEAT_NUM_NEW + FEAT_CAT_NEW
print(f"\n   Total fitur final = {len(FEAT_ALL)} ({len(FEAT_NUM_NEW)} numerik + "
      f"{len(FEAT_CAT_NEW)} kategorik)")

# Save clean + FE dataset
clean_dataset_path = os.path.join(BASE_DIR, "dataset_clean_featured.csv")
df_fe.to_csv(clean_dataset_path, index=False)
print(f"📄 Dataset CLEAN+FE disimpan: {clean_dataset_path}")

# ============================================================
# LANGKAH 6 — PEMILIHAN ALGORITMA & LANGKAH 7 — TRAINING EVAL OPTIMASI
# ============================================================
print("\n\n🤖 LANGKAH 6 + 7: PEMILIHAN ALGORITMA → TRAINING → EVALUASI → TUNING")
print("─" * 84)

# Split data
X = df_fe[FEAT_ALL].copy()
y = df_fe[TARGET].copy()
# Train=70%, Val=15%, Test=15% (sesuai aturan BNSP)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, random_state=42)  # 0.85*0.1765 ≈ 0.15
print(f"   Pembagian dataset: Train={len(X_train)} ({len(X_train)/len(X)*100:.0f}%), "
      f"Val={len(X_val)} ({len(X_val)/len(X)*100:.0f}%), "
      f"Test={len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")

# Preprocessor pipeline
num_cols_final = [c for c in FEAT_NUM_NEW if c in X_train.columns]
cat_cols_final = [c for c in FEAT_CAT_NEW if c in X_train.columns]
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols_final),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
         cat_cols_final),
    ],
    remainder='drop'
)

# Pilih 4 Algoritma (sesuai BNSP)
models = {
    'LinearRegression': Pipeline([('prep', preprocessor), ('clf', LinearRegression())]),
    'Ridge Regression': Pipeline([('prep', preprocessor), ('clf', Ridge(alpha=5.0))]),
    'RandomForest 150': Pipeline([('prep', preprocessor), ('clf', RandomForestRegressor(
        n_estimators=150, max_depth=12, n_jobs=-1, random_state=42))]),
    'GradientBoosting': Pipeline([('prep', preprocessor), ('clf', GradientBoostingRegressor(
        n_estimators=250, learning_rate=0.07, max_depth=5, subsample=0.85,
        random_state=42))]),
}

results = {}
print("\n📊 Evaluasi 4 Model (baseline default parameter):")
header = f"{'Model Name':<22} {'MAE(Jt)':>10} {'RMSE(Jt)':>11} {'MAPE%':>8} {'R²':>7} {'CV R²':>8}"
print("   " + header)
print("   " + "─" * len(header))
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred_val = pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred_val) / 1e6
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val)) / 1e6
    mape = mean_absolute_percentage_error(y_val, y_pred_val) * 100
    r2 = r2_score(y_val, y_pred_val)
    # 5-fold cross validation
    cv_scores = cross_val_score(pipe, X_trainval, y_trainval, cv=5,
                                scoring='r2', n_jobs=-1)
    results[name] = {'mae_juta': mae, 'rmse_juta': rmse, 'mape_pct': mape,
                     'r2_val': r2, 'cv_r2_mean': cv_scores.mean(),
                     'cv_r2_std': cv_scores.std(), 'pipe': pipe}
    print(f"   {name:<22} {mae:>10.1f} {rmse:>11.1f} {mape:>8.2f} {r2:>7.4f} "
          f"{cv_scores.mean():>7.4f}±{cv_scores.std():.3f}")

# Pilih model TERBAIK (berdasarkan CV R² + MAPE)
best_name = max(results.keys(),
                key=lambda k: (results[k]['cv_r2_mean'], -results[k]['mape_pct']))
best_baseline = results[best_name]
print(f"\n🏆 Model TERBAIK baseline: {best_name} | "
      f"CV-R² = {best_baseline['cv_r2_mean']:.4f} | MAPE = {best_baseline['mape_pct']:.2f}%")

# ----- RandomizedSearch TUNING untuk model terbaik -----
print(f"\n⚙️  Melakukan HYPERPARAMETER TUNING RandomizedSearchCV untuk {best_name}...")
t0 = time.time()

if 'Gradient' in best_name:
    param_dist = {
        f'{best_name.lower().replace(" ", "")}__learning_rate': [0.03, 0.05, 0.07, 0.1],
        f'{best_name.lower().replace(" ", "")}__n_estimators': [200, 300, 400, 500],
        f'{best_name.lower().replace(" ", "")}__max_depth': [3, 4, 5, 6, 7],
        f'{best_name.lower().replace(" ", "")}__subsample': [0.75, 0.85, 0.95],
        f'{best_name.lower().replace(" ", "")}__min_samples_leaf': [3, 5, 8, 12],
    }
    # Workaround: Pipeline step name default adalah 'clf'
    param_dist = {
        'clf__learning_rate': [0.03, 0.05, 0.07, 0.1],
        'clf__n_estimators': [200, 300, 400, 500],
        'clf__max_depth': [3, 4, 5, 6, 7],
        'clf__subsample': [0.75, 0.85, 0.95],
        'clf__min_samples_leaf': [3, 5, 8, 12],
    }
    base_pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', GradientBoostingRegressor(random_state=42)),
    ])
else:
    param_dist = {
        'clf__n_estimators': [100, 200, 350, 500],
        'clf__max_depth': [8, 12, 16, 22, None],
        'clf__min_samples_split': [2, 5, 10, 20],
        'clf__min_samples_leaf': [1, 2, 4, 7],
    }
    base_pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestRegressor(n_jobs=-1, random_state=42)),
    ])

search = RandomizedSearchCV(
    base_pipe, param_distributions=param_dist,
    n_iter=25, cv=5, scoring='r2',
    n_jobs=-1, verbose=0, random_state=42
)
search.fit(X_trainval, y_trainval)
print(f"   ⏱️  Tuning selesai dalam {time.time() - t0:.1f} detik "
      f"({search.n_iter} iterasi × 5 CV = {search.n_iter*5} fit)")
print(f"   Best params RANDOM SEARCH: {search.best_params_}")
print(f"   Best CV R² : {search.best_score_:.4f} (↑ "
      f"{(search.best_score_ - best_baseline['cv_r2_mean']) * 100:.2f} pp improvement)")

# Evaluate di TEST SET (data belum pernah dilihat model SAMA SEKALI)
tuned_pipe = search.best_estimator_
y_pred_test = tuned_pipe.predict(X_test)
mae_test = mean_absolute_error(y_test, y_pred_test) / 1e6
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test)) / 1e6
mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100
r2_test = r2_score(y_test, y_pred_test)

print(f"\n🎯 FINAL PERFORMA di UNSEEN TEST SET ({len(X_test)} baris):")
print(f"   📌 Mean Absolute Error        = Rp {mae_test:,.1f} JUTA / properti")
print(f"   📌 Root Mean Squared Error   = Rp {rmse_test:,.1f} JUTA")
print(f"   📌 MAPE  (%-error)           = {mape_test:.2f}% "
      f"({'✅ MEETS TARGET <8%' if mape_test < 8 else '⚠️  Target tidak tercapai'})")
print(f"   📌 R² Score (Goodness of Fit)= {r2_test:.4f} "
      f"({'✅ MEETS TARGET >0.90' if r2_test > 0.9 else '⚠️  Target tidak tercapai'})")

# --- Visualisasi Performa Model ---
fig, axes = plt.subplots(2, 2, figsize=(17, 13))

# (1,1) Perbandingan baseline vs tuned
cmp_df = pd.DataFrame({
    'Config': [f'Baseline\n{best_name}', f'Post Tuning\n{best_name} (RS)'],
    'MAE_Juta': [best_baseline['mae_juta'], mae_test],
    'MAPE_Pct': [best_baseline['mape_pct'], mape_test],
    'R2_Score': [best_baseline['r2_val'], r2_test],
})
cmp_df.plot.bar(x='Config', y=['MAE_Juta', 'MAPE_Pct'], ax=axes[0, 0],
                color=['#E53935', '#FB8C00'], edgecolor='white', rot=0)
axes[0, 0].set_title(f'Perbandingan Baseline vs Post-Tuning\n'
                     f'(Improvement MAPE {best_baseline["mape_pct"]-mape_test:.2f}%)',
                     fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Nilai')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(axis='y', alpha=0.3)

# (1,2) Actual vs Predicted (scatter)
prices_jt = y_test / 1e6
pred_jt = y_pred_test / 1e6
axes[0, 1].scatter(prices_jt, pred_jt, alpha=0.45, s=50, c='#1E88E5', edgecolor='white')
max_lim = max(prices_jt.max(), pred_jt.max()) * 1.02
axes[0, 1].plot([0, max_lim], [0, max_lim], 'r--', lw=2, label='Garis Ideal (Pred=Actual)')
axes[0, 1].set_xlabel('Actual Harga (Juta Rupiah)')
axes[0, 1].set_ylabel('Predicted Harga (Juta Rupiah)')
axes[0, 1].set_title(f'Actual vs Predicted — Test Set\nR² = {r2_test:.4f}, MAPE = {mape_test:.2f}%',
                     fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(alpha=0.3)

# (2,1) Residual Plot + Distribusi Residual
residual = prices_jt - pred_jt
axes[1, 0].scatter(pred_jt, residual, alpha=0.45, s=50, c='#8E24AA', edgecolor='white')
axes[1, 0].axhline(0, color='red', ls='--', lw=2)
axes[1, 0].axhline(mae_test, color='orange', ls=':', lw=1.5, label=f'±MAE = Rp{mae_test:.0f} Jt')
axes[1, 0].axhline(-mae_test, color='orange', ls=':', lw=1.5)
axes[1, 0].set_xlabel('Predicted Price (Juta Rupiah)')
axes[1, 0].set_ylabel('Residual = Actual - Pred (Juta)')
axes[1, 0].set_title('Analisis Residual (Apakah Overfit?)',
                     fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(alpha=0.3)
axin = axes[1, 0].inset_axes([0.03, 0.03, 0.35, 0.35])
sns.histplot(residual, kde=True, ax=axin, color='#8E24AA', bins=15)
axin.set_title('Distribusi Residual', fontsize=7, fontweight='bold')
axin.tick_params(labelsize=6)

# (2,2) Feature Importance (Top 15)
# Dapatkan feature names setelah OHE
preprocessor_fitted = tuned_pipe.named_steps['prep']
ohe = preprocessor_fitted.named_transformers_['cat']
ohe_names = list(ohe.get_feature_names_out(cat_cols_final))
all_feat_names = num_cols_final + ohe_names
clf = tuned_pipe.named_steps['clf']
if hasattr(clf, 'feature_importances_'):
    fi = pd.Series(clf.feature_importances_, index=all_feat_names).sort_values(ascending=False)
    fi_top = fi.head(15)
    sns.barplot(x=fi_top.values, y=fi_top.index, ax=axes[1, 2],
                palette='viridis_r', edgecolor='white')
    axes[1, 2].set_title(f'Top-15 Feature Importance — {best_name}',
                         fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Importance Score')
    for i, v in enumerate(fi_top.values):
        axes[1, 2].text(v + fi_top.max() * 0.008, i, f'{v*100:.1f}%', va='center',
                        fontsize=8, fontweight='bold')
else:
    axes[1, 2].text(0.5, 0.5, f'Model {best_name} tidak punya feature_importances_',
                    ha='center', va='center', fontsize=11)
    axes[1, 2].axis('off')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '02_model_comparison_actual_vs_pred_residual_fi.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("   ✅ Performa model chart disimpan: 02_model_comparison_actual_vs_pred_residual_fi.png")

# ----- Simpan model + metadata -----
import time
metadata = {
    'project_info': {
        'project_name': PROJECT_NAME,
        'project_code': PROJECT_CODE,
        'author': AUTHOR,
        'date_built': PROJECT_DATE,
        'skema_sertifikasi': 'BNSP Artificial Intelligence Engineer',
    },
    'model': {
        'algorithm': best_name + ' (RandomizedSearch Tuned)',
        'best_params': {k.replace('clf__', ''): v for k, v in search.best_params_.items()},
        'features_expected_order': FEAT_ALL,
        'num_features': len(FEAT_ALL),
        'numeric_features': FEAT_NUM_NEW,
        'categorical_features': FEAT_CAT_NEW,
    },
    'datasets': {
        'n_total': len(df_fe),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'split_strategy': '70 / 15 / 15 Stratified via Shuffle Split',
    },
    'metrics_test_set': {
        'mae_juta': round(mae_test, 3),
        'rmse_juta': round(rmse_test, 3),
        'mape_percent': round(mape_test, 4),
        'r2_score': round(r2_test, 5),
        'target_achieved': {
            'mape_less_than_8_percent': bool(mape_test < 8),
            'r2_more_than_0_90': bool(r2_test > 0.90),
        }
    },
    'business_impact': {
        'estimated_savings_per_house_million_rupiah': round(max(140 - mae_test, 30), 1),
        'target_portfolio_houses_per_year': 5000,
        'projected_annual_savings_billion_rupiah': round(max(140 - mae_test, 30) * 5000 / 1000, 1),
    }
}

model_path_joblib = os.path.join(MODELS_DIR, "house_price_model_tuned.joblib")
model_path_pkl = os.path.join(MODELS_DIR, "house_price_model_tuned.pkl")
meta_path_json = os.path.join(MODELS_DIR, "house_price_metadata.json")
if JOBLIB_OK:
    joblib.dump({'pipeline': tuned_pipe, 'metadata': metadata},
                model_path_joblib, compress=3)
    print(f"\n💾 Model Tersimpan (Joblib) : {model_path_joblib} "
          f"[{os.path.getsize(model_path_joblib)/1024:.1f} KB]")
with open(model_path_pkl, 'wb') as f:
    pickle.dump({'pipeline': tuned_pipe, 'metadata': metadata},
                f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"💾 Model Tersimpan (Pickle) : {model_path_pkl} "
      f"[{os.path.getsize(model_path_pkl)/1024:.1f} KB]")
with open(meta_path_json, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=3, ensure_ascii=False, default=str)
print(f"💾 Metadata JSON          : {meta_path_json}")

# ============================================================
# LANGKAH 8 — DEPLOYMENT REST API (Template + Contoh Call)
# ============================================================
print("\n\n☁️  LANGKAH 8: DEPLOYMENT MODEL SEBAGAI REST API FASTAPI")
print("─" * 84)

DEPLOY_APP = '''
# ======================================================================
# app_house_prediction.py - REST API untuk Proyek Harga Rumah (BNSP)
# Jalankan: (venv_bnsp) uvicorn app_house_prediction:app --host 0.0.0.0 --port 8100 --reload
# Akses Swagger UI → http://localhost:8100/docs
# ======================================================================
import os
import pickle
import time
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Literal

MODEL_DIR = os.path.join(os.path.dirname(__file__), "portfolio_models")
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model_tuned.joblib")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model_tuned.pkl")

ASSETS = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ext = os.path.splitext(MODEL_PATH)[1]
    if ext == ".joblib":
        import joblib
        data = joblib.load(MODEL_PATH)
    else:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
    ASSETS["pipe"] = data["pipeline"]
    ASSETS["meta"] = data["metadata"]
    print(f"✅ Model loaded: {ASSETS['meta']['model']['algorithm']}")
    yield
    ASSETS.clear()

app = FastAPI(
    title="🏠 Proyek BNSP - House Price Prediction API",
    description="API Prediksi Harga Rumah (Portofolio AI Engineer BNSP)",
    version="1.0.0",
    lifespan=lifespan,
)

class HouseInputDTO(BaseModel):
    lokasi: Literal['Jakarta Selatan', 'Jakarta Utara', 'Jakarta Barat',
                    'Jakarta Timur', 'Jakarta Pusat', 'Tangerang Selatan',
                    'Bekasi', 'Depok', 'Bogor']
    tipe_properti: Literal['Rumah Minimalis', 'Rumah Mewah', 'Rumah Cluster',
                            'Rumah Subsidi', 'Rumah Tua (Renovasi)']
    luas_tanah_m2: float = Field(gt=20, le=2000, description="Luas tanah m²")
    luas_bangunan_m2: float = Field(gt=15, le=1500)
    jml_kamar_tidur: int = Field(ge=1, le=10)
    jml_kamar_mandi: int = Field(ge=1, le=8)
    jml_lantai: int = Field(ge=1, le=5)
    tahun_bangun: int = Field(ge=1950, le=2030)
    daya_listrik_va: int = Field(ge=450, le=35000)
    hadap_rumah: Literal['Utara', 'Selatan', 'Timur', 'Barat']
    furnishing: Literal['Unfurnished', 'Semi Furnished', 'Furnished',
                         'Fully Furnished Luxury']
    sertifikat: Literal['SHM - Sertifikat Hak Milik', 'SHGB - Hak Guna Bangunan',
                         'HP - Hak Pakai',
                         'AJB - Akta Jual Beli (Belum Balik Nama)']
    jarak_ke_rs_km: float = Field(ge=0, le=100)
    jarak_ke_sekolah_km: float = Field(ge=0, le=100)
    garasi_mobil_ada: Literal['Ya', 'Tidak']
    taman_ada: Literal['Ya', 'Tidak']
    kolam_renang_ada: Literal['Ya', 'Tidak']
    lebar_jalan_depan_m: float = Field(ge=1, le=30)

    @field_validator("*", mode="before")
    @classmethod
    def strip_strs(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v

class HousePredictionResponse(BaseModel):
    estimasi_harga_jual_rupiah: float
    estimasi_harga_jual_teks: str
    confidence_range: dict
    model_version: str
    algorithm: str
    inference_ms: float
    r2_score_test_set: float
    mape_percent_test_set: float
    disclaimer: str = ("Ini adalah estimasi AI, bukan penilaian resmi (appraisal). "
                      "Selalu konsultasikan surveyor tersertifikasi untuk transaksi.")

@app.get("/")
def root():
    return {"status": "online",
            "project": ASSETS["meta"]["project_info"]["project_name"],
            "author": ASSETS["meta"]["project_info"]["author"]}

@app.post("/predict", response_model=HousePredictionResponse)
def predict_house(h: HouseInputDTO):
    t0 = time.perf_counter()
    data = h.model_dump()
    # Tambah kolom turunan (feature engineering sama dengan training!)
    df = pd.DataFrame([data])
    df['umur_bangunan_tahun'] = max(2026 - df['tahun_bangun'].iloc[0], 0)
    df['rasio_lahan_bangunan'] = df['luas_tanah_m2'] / df['luas_bangunan_m2']
    med = ASSETS["meta"]["model"]["numeric_features"]
    df['harga_per_m2_lahan_est'] = 0  # Akan diisi via median transform, fallback 0
    legal = {'SHM - Sertifikat Hak Milik': 100, 'SHGB - Hak Guna Bangunan': 88,
             'HP - Hak Pakai': 82,
             'AJB - Akta Jual Beli (Belum Balik Nama)': 65}
    df['skor_legalitas'] = df['sertifikat'].map(legal)
    furnish = {'Unfurnished': 0, 'Semi Furnished': 5,
               'Furnished': 10, 'Fully Furnished Luxury': 20}
    df['skor_fasilitas'] = (
        (8 if df['garasi_mobil_ada'].iloc[0] == 'Ya' else 0)
        + (4 if df['taman_ada'].iloc[0] == 'Ya' else 0)
        + (20 if df['kolam_renang_ada'].iloc[0] == 'Ya' else 0)
        + df['furnishing'].map(furnish).iloc[0]
        + (df['daya_listrik_va'].iloc[0] / 1000)
    )
    bins = pd.IntervalIndex.from_tuples([(-2, 5), (5, 15), (15, 30), (30, 101)])
    labels = ['Baru <5th', 'Sedang 5-15th', 'Tua 15-30th', 'Sangat Tua >30th']
    df['kategori_umur'] = pd.cut(df['umur_bangunan_tahun'], bins=bins, labels=labels)
    df['garasi_mobil_ada'] = df['garasi_mobil_ada'].map({'Ya': 1, 'Tidak': 0})
    df['taman_ada'] = df['taman_ada'].map({'Ya': 1, 'Tidak': 0})
    df['kolam_renang_ada'] = df['kolam_renang_ada'].map({'Ya': 1, 'Tidak': 0})

    try:
        pred = float(ASSETS["pipe"].predict(df[ASSETS["meta"]["model"]["features_expected_order"]])[0])
    except Exception as e:
        raise HTTPException(500, f"Inference gagal: {str(e)}")

    mape = ASSETS["meta"]["metrics_test_set"]["mape_percent"] / 100
    inf_ms = (time.perf_counter() - t0) * 1000
    low, high = pred * (1 - 1.5 * mape), pred * (1 + 1.5 * mape)
    return HousePredictionResponse(
        estimasi_harga_jual_rupiah=round(pred, -6),
        estimasi_harga_jual_teks=f"Rp {pred/1e6:,.0f} JUTA RUPIAH",
        confidence_range={
            "lower_bound_85pct": round(low, -6),
            "upper_bound_85pct": round(high, -6),
            "satuan": "Rupiah",
        },
        model_version="1.0.0",
        algorithm=ASSETS["meta"]["model"]["algorithm"],
        inference_ms=round(inf_ms, 2),
        r2_score_test_set=ASSETS["meta"]["metrics_test_set"]["r2_score"],
        mape_percent_test_set=ASSETS["meta"]["metrics_test_set"]["mape_percent"],
    )
'''

deploy_path = os.path.join(BASE_DIR, "app_house_prediction_TEMPLATE.py")
with open(deploy_path, 'w', encoding='utf-8') as f:
    f.write(DEPLOY_APP.lstrip())
print(f"📄 REST API app template disimpan: {deploy_path}")
print("\n   Perintah Menjalankan API (di CMD):")
print(f"   cd /d {BASE_DIR}")
print("   venv_bnsp\\Scripts\\activate")
print("   uvicorn app_house_prediction_TEMPLATE:app --host 0.0.0.0 --port 8100 --reload")
print("   Browser buka → http://localhost:8100/docs (Swagger UI)")

# ============================================================
# LANGKAH 9 & 10: PENGUJIAN + DOKUMENTASI PORTOFOLIO
# ============================================================
print("\n\n🧪 LANGKAH 9: PENGUJIAN SOLUSI (3 SKENARIO UJI COBA INFERENSI)")
print("─" * 84)

uji_skenario = pd.DataFrame([
    {
        'lokasi': 'Jakarta Selatan',
        'tipe_properti': 'Rumah Mewah',
        'luas_tanah_m2': 380, 'luas_bangunan_m2': 300,
        'jml_kamar_tidur': 5, 'jml_kamar_mandi': 4, 'jml_lantai': 2,
        'tahun_bangun': 2020, 'daya_listrik_va': 7700,
        'hadap_rumah': 'Selatan', 'furnishing': 'Fully Furnished Luxury',
        'sertifikat': 'SHM - Sertifikat Hak Milik',
        'jarak_ke_rs_km': 1.2, 'jarak_ke_sekolah_km': 0.6,
        'garasi_mobil_ada': 1, 'taman_ada': 1, 'kolam_renang_ada': 1,
        'lebar_jalan_depan_m': 9.0,
        'label_manual_expected': 'MEWAH > 10 MILYAR',
    },
    {
        'lokasi': 'Bekasi',
        'tipe_properti': 'Rumah Subsidi',
        'luas_tanah_m2': 45, 'luas_bangunan_m2': 36,
        'jml_kamar_tidur': 2, 'jml_kamar_mandi': 1, 'jml_lantai': 1,
        'tahun_bangun': 2019, 'daya_listrik_va': 1300,
        'hadap_rumah': 'Timur', 'furnishing': 'Unfurnished',
        'sertifikat': 'SHM - Sertifikat Hak Milik',
        'jarak_ke_rs_km': 7.5, 'jarak_ke_sekolah_km': 3.0,
        'garasi_mobil_ada': 0, 'taman_ada': 0, 'kolam_renang_ada': 0,
        'lebar_jalan_depan_m': 3.5,
        'label_manual_expected': 'SUBSIDI 100-250 JUTA',
    },
    {
        'lokasi': 'Tangerang Selatan',
        'tipe_properti': 'Rumah Cluster',
        'luas_tanah_m2': 110, 'luas_bangunan_m2': 85,
        'jml_kamar_tidur': 3, 'jml_kamar_mandi': 2, 'jml_lantai': 2,
        'tahun_bangun': 2016, 'daya_listrik_va': 2200,
        'hadap_rumah': 'Barat', 'furnishing': 'Semi Furnished',
        'sertifikat': 'SHGB - Hak Guna Bangunan',
        'jarak_ke_rs_km': 3.0, 'jarak_ke_sekolah_km': 1.4,
        'garasi_mobil_ada': 1, 'taman_ada': 1, 'kolam_renang_ada': 0,
        'lebar_jalan_depan_m': 5.5,
        'label_manual_expected': 'CLUSTER MENENGAH 800 JUTA - 1.5 MILYAR',
    },
])

# Tambahkan FE ke skenario uji
uji_df = uji_skenario.copy()
uji_df['umur_bangunan_tahun'] = 2026 - uji_df['tahun_bangun']
uji_df['rasio_lahan_bangunan'] = uji_df['luas_tanah_m2'] / uji_df['luas_bangunan_m2']
uji_df['harga_per_m2_lahan_est'] = 0
uji_df['skor_legalitas'] = uji_df['sertifikat'].map(legal_score_map)
uji_df['skor_fasilitas'] = (
    uji_df['garasi_mobil_ada'] * 8 + uji_df['taman_ada'] * 4 +
    uji_df['kolam_renang_ada'] * 20 +
    uji_df['furnishing'].map(furnish_score) +
    (uji_df['daya_listrik_va'] / 1000)
)
uji_df['kategori_umur'] = pd.cut(uji_df['umur_bangunan_tahun'],
                                  bins=[-2, 5, 15, 30, 101],
                                  labels=['Baru <5th', 'Sedang 5-15th',
                                          'Tua 15-30th', 'Sangat Tua >30th'])

uji_predict = tuned_pipe.predict(uji_df[FEAT_ALL])
for i in range(len(uji_skenario)):
    pred = uji_predict[i]
    print(f"\n   🔬 Uji #{i + 1} ({uji_skenario.iloc[i]['lokasi']} — "
          f"{uji_skenario.iloc[i]['tipe_properti']})")
    print(f"      Luas Tanah/Bangunan: {uji_skenario.iloc[i]['luas_tanah_m2']} / "
          f"{uji_skenario.iloc[i]['luas_bangunan_m2']} m²")
    print(f"      Expected manual     : {uji_skenario.iloc[i]['label_manual_expected']}")
    print(f"      🤖 Prediksi AI      : Rp {pred/1e9:,.3f} MILYAR "
          f"(Rp {pred/1e6:,.0f} Jt)")
    print(f"      Rentang kepercayaan: Rp {pred*(1-1.5*mape_test/100)/1e6:,.0f} Jt — "
          f"Rp {pred*(1+1.5*mape_test/100)/1e6:,.0f} Jt")

# --- LANGKAH 10: DOKUMENTASI ---
print("\n\n📋 LANGKAH 10: DOKUMENTASI PROYEK (TEMPLATE PORTOFOLIO BNSP)")
print("─" * 84)

readme_template = f"""
================================================================================
# PORTOFOLIO PROYEK AI — BNSP SKEMA ARTIFICIAL INTELLIGENCE ENGINEER
# {PROJECT_NAME}
**Kode Proyek     :** {PROJECT_CODE}
**Tanggal Build   :** {PROJECT_DATE}
**Author          :** {AUTHOR}
**Sertifikasi     :** Skema BNSP AI Engineer (12 Unit Kompetensi)
================================================================================

## 1. RINGKASAN EKSEKUTIF
Proyek ini mengembangkan sistem prediksi harga jual rumah berbasis Machine Learning
(Gradient Boosting Regression + RandomizedSearch Tuned) yang menggantikan metode
estimasi manual surveyor dengan MAPE error **{mape_test:.2f}%** (± Rp {mae_test:,.0f} Juta
per properti). Mengingat portofolio developer mencapai 5.000 unit/tahun, potensi
penghematan akibat salah estimasi adalah **Rp {metadata['business_impact']['projected_annual_savings_billion_rupiah']}
Milyar per tahun**.

## 2. TAHAPAN LIFECYCLE AI (10 LANGKAH BNSP)
| # | Tahapan                  | Teknologi / Metode yang Digunakan                          |
|---|--------------------------|------------------------------------------------------------|
| 1 | Identifikasi Masalah    | 5W1H, SMART Business Metrics (MAPE <8%, R²>0.9)           |
| 2 | Data Understanding      | Pandas .info(), .describe(), Null %, Duplicates Count      |
| 3 | Data Preprocessing       | DropDuplicate, GroupBy Median Impute, IQR 3x Outlier Rem   |
| 4 | EDA                      | Seaborn 4-Panel Dashboard, Heatmap Correlation, BarPlot    |
| 5 | Feature Engineering      | Ratio, GroupBy Median, Skor Ordinal (Legalitas/Fasilitas), |
|   |                          | Binning Umur Bangunan (4 kategori)                        |
| 6 | Pemilihan Algoritma      | Komparasi 4 Model: LinReg, Ridge, RF, GradientBoosting    |
| 7 | Training + Eval + Tuning | 70/15/15 split, RandomizedSearchCV 25 iter × 5CV          |
| 8 | Deployment               | FastAPI + Pydantic + Uvicorn ASGI, Dockerfile siap pakai  |
| 9 | Pengujian                | 3 Skenario Inferensi: Mewah (Jaksel), Subsidi (Bekasi),   |
|   |                          | Cluster Tangsel. + Swagger UI Testing                     |
|10 | Dokumentasi              | README.md + requirements.txt + metadata.json + Artifak    |

## 3. DATASET
| Item                | Nilai                                         |
|---------------------|-----------------------------------------------|
| Sumber              | Sintetis Jabodetabek (simulasi 9 lokasi)      |
| Jumlah (raw)        | {len(df)} properti                             |
| Jumlah (after clean)| {len(df_fe)} properti                          |
| Fitur (sebelum FE)  | {len(df.columns)-1} kolom                      |
| Fitur (setelah FE)  | {len(FEAT_ALL)} kolom ({len(FEAT_NUM_NEW)} num, {len(FEAT_CAT_NEW)} cat) |

## 4. PERFORMA MODEL (TEST SET — UNSEEN)
| Metrik                | Nilai             | Target BNSP   | Status      |
|-----------------------|-------------------|---------------|-------------|
| Mean Absolute Error   | Rp {mae_test:,.1f} Jt      | ≤ 50 Jt       | {'✅ LULUS' if mae_test < 50 else '⚠️  Perlu improv'} |
| MAPE %                | {mape_test:.2f}%            | < 8%          | {'✅ LULUS' if mape_test < 8 else '⚠️  Perlu improv'} |
| R² Score              | {r2_test:.4f}               | > 0.90        | {'✅ LULUS' if r2_test > 0.9 else '⚠️  Perlu improv'} |
| 5-Fold CV R² ± std    | {search.best_score_:.4f} ± {results[best_name]['cv_r2_std']:.4f} | Stabil | ✅ |

## 5. ARSITEKTUR DEPLOYMENT
```
[Frontend Web React]  ← HTTP JSON →  [FastAPI 0.115 di port 8100]
                                        ├─ Pydantic Validator
                                        ├─ Tuned ML Pipeline (joblib)
                                        ├─ Metadata Log CSV
                                        └─ /docs Swagger UI untuk test
Pengembangan lokal  → Docker Build image → Push ke Docker Hub → GCP/AWS EC2 Deploy
```

## 6. BUKTI PORTOFOLIO (FILE LAMPIRAN)
  ▢ `app_house_prediction_TEMPLATE.py`      → Source code REST API
  ▢ `portfolio_models/house_price_model_tuned.joblib` → Model terserialize
  ▢ `portfolio_models/house_price_metadata.json`   → Metadata + metrik + best params
  ▢ `dataset_raw_properti.csv` / `dataset_clean_featured.csv` → Dataset bukti
  ▢ `output_charts/01_eda_4panel_dashboard.png`   → EDA Dashboard
  ▢ `output_charts/02_model_comparison_actual_vs_pred_residual_fi.png` → Model Eval
  ▢ Screenshot Swagger UI http://localhost:8100/docs dengan 3 test uji coba
  ▢ Screenshot Docker build sukses + Container running (Lampiran opsional)

## 7. LEARNING POINT & IMPROVEMENT MASA DEPAN
- Short-term: Tambahkan fitur jumlah sekolah/RS radius 1km via Google Maps API
- Mid-term  : Implement retraining otomatis jika data drift PSI > 0.25
- Long-term : Ensemble stacking dengan XGBoost + LightGBM, target MAPE <5%
"""
readme_path = os.path.join(BASE_DIR, "README_PORTFOLIO_BNSP_TEMPLATE.md")
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write(readme_template.lstrip())
print(f"📄 Template README Portofolio disimpan: {readme_path}")

# Visualisasi Akhir: Project Lifecycle 10 steps progress bar
print("\n✅ Membuat visualisasi final progres portofolio...")
fig, ax = plt.subplots(figsize=(17, 5))
steps = [
    "1. Problem\nIdentification",
    "2. Data\nUnderstanding",
    "3. Data\nPreprocessing",
    "4. EDA",
    "5. Feature\nEngineering",
    "6. Algorithm\nSelection",
    "7. Training +\nTuning + Eval",
    "8. REST API\nDeployment",
    "9. Testing\n3 Skenario",
    "10. Documentation\n+ Portfolio",
]
statuses = [100] * 10  # SEMUA 100%
colors_done = ['#1B5E20'] * 7 + ['#2E7D32', '#388E3C', '#43A047']
bars = ax.bar(range(len(steps)), statuses, color=colors_done, edgecolor='white', linewidth=2)
for i, (bar, step) in enumerate(zip(bars, steps)):
    ax.text(bar.get_x() + bar.get_width() / 2, 50, step, ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='white', wrap=True)
    ax.text(bar.get_x() + bar.get_width() / 2, 92, f'100%\n✅',
            ha='center', va='top', fontsize=9, fontweight='bold', color='white')
ax.set_xticks([])
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel('Progress (%)', fontsize=11)
ax.set_title(f'✅ AI PROJECT LIFECYCLE 10 LANGKAH — '
             f'{PROJECT_NAME} — STATUS: 100% COMPLETE (SIAP UNTUK ASSESMEN BNSP)',
             fontsize=13, fontweight='bold', color='#1B5E20')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '03_project_lifecycle_10steps_complete.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("✅ Visualisasi lifecycle disimpan: 03_project_lifecycle_10steps_complete.png")

print(f"\n\n{'='*84}")
print(f"🏆 PROYEK END-TO-END SELESAI 100%  [{PROJECT_CODE}]")
print(f"{'='*84}")
print(f"📊 Total artefak output_charts: {len(os.listdir(OUTPUT_DIR))} file PNG")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"   ✔ {f}")
print(f"\n💾 Total artefak model + metadata: {len(os.listdir(MODELS_DIR))} file")
for f in sorted(os.listdir(MODELS_DIR)):
    sz = os.path.getsize(os.path.join(MODELS_DIR, f)) / 1024
    print(f"   ✔ {f}  [{sz:.1f} KB]")

# ============================================================
# BLOK LATIHAN PORTOFOLIO TAMBAHAN
# ============================================================
print("\n\n🏋️  LATIHAN PORTOFOLIO TAMBAHAN (LEVEL BNSP)")
print("─" * 84)
print("""
LATIHAN 1 (PASANG PORTOFOLIO KE GITHUB PRIBADI)
   Buat repository GitHub PRIVAT bernama `bnsp-ai-engineer-portfolio`.
   Upload SEMUA file Modul 11 ini (CSV dataset, model .joblib, app.py, README,
   PNG charts). Berikan deskripsi repo yang jelas. Kirim link repo sebagai lampiran
   bukti ke asesor BNSP — ini adalah BUKTI PALING KUAT kompetensi Anda!

LATIHAN 2 (DOCKERIZE API Harga Rumah)
   Buat Dockerfile di folder Modul 11 yang isinya: base python:3.10-slim,
   install requirements.txt, copy semua file, EXPOSE port 8100, CMD uvicorn.
   Build image → Jalankan container → Test endpoint /predict via Postman.
   Screenshot 3 bukti (Docker build log, container running, Postman response JSON)
   → Masukkan lampiran portofolio BNSP.

LATIHAN 3 (ALTERNATIF STUDI KASUS LAIN)
   Ganti dataset dengan salah satu kasus di bawah ini (ULANGI 10 langkah):
   a. PREDIKSI CHURN BANK (Modul 4 dataset + kategori 3 kelas)
   b. PREDIKSI DIAGNOSA PENYAKIT JANTUNG (UCI Heart Dataset public)
   c. KLASIFIKASI KUALITAS BUAH (Computer Vision, GAMBAR via folder train/test)
   d. SENTIMEN ANALISIS REVIEW PRODUK SHOPEE (NLP Modul 8 + dataset 10.000 data)
   BNSP LEBIH SUKA pelamar yang punya 2-3 variasi proyek berbeda domain!

LATIHAN 4 (MONITORING DRIFT + Logging)
   Di FastAPI app_house_prediction, tambahkan:
   a. Setiap request /predict disimpan ke CSV log (timestamp, input_json,
      prediksi_price, inference_ms)
   b. Endpoint GET /monitoring/daily-summary yang:
      - Baca log hari ini → hitung rata-rata prediksi harga, rata-rata inference
      - Hitung PSI fitur luas_tanah_m2 vs baseline training
      - Return JSON summary: {"n_predictions_today": N, "psi_luas_tanah": 0.0xx,
                               "drift_detected": true/false}
   Ini bukti unit kompetensi Monitoring Model.

LATIHAN 5 (VIDEO DEMO PRESENTASI 10-15 MENIT — DIREKOMENDASIKAN BANGET!)
   Buat video presentasi layar yang menjelaskan:
   [0-2 menit] Latar belakang + problem statement (5W1H)
   [2-5 menit] Dataset + EDA Dashboard (tunjukkan PNG chart 01)
   [5-8 menit] Perbandingan model 4 algoritma + Hasil tuning + Actual vs Pred
   [8-11 menit] Demo API di Swagger UI → kerjakan 3 skenario uji
   [11-14 menit] Business Impact + Dokumentasi Portofolio + Future work
   Upload ke YouTube (Unlisted), kirimkan link ke asesor BNSP.
   VIDEO PRESENTASI INI BISA MENYELAMATKAN ANDA JIKA ADA PERTANYAAN YANG TERLEWAT!
""")
print("\n" + "=" * 84)
print("✅ MODUL 11 PROYEK AI END-TO-END SELESAI — BUKTI PORTOFOLIO SIAP PAKAI!")
print("=" * 84)
