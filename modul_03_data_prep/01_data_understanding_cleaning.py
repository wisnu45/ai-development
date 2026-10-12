import os
import sys
import io
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

print("="*60)
print("MODUL 3: DATA PREPARATION & PREPROCESSING")
print("Bagian 1: Data Understanding & Data Cleaning")
print("="*60)

import numpy as np
import pandas as pd
from io import StringIO

print("""
┌─────────────────────────────────────────────────────────────────┐
│          TAHAPAN DATA PREPARATION (KRUSIAL DI AI!)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DATA UNDERSTANDING → Pahami struktur & karakteristik data  │
│  2. DATA CLEANING      → Bersihkan: null, duplikat, outlier    │
│  3. DATA TRANSFORMATION→ Ubah format: encoding, scaling,       │
│  4. FEATURE ENGINEERING→ Buat fitur baru dari yang ada         │
│  5. EDA                → Visual eksplorasi pola & insight      │
│  6. DATA SPLITTING     → Train / Validation / Test set         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")

print("\n>>> 1. MEMBUAT DATASET DENGAN MASALAH (untuk latihan cleaning)")
print("-" * 60)

data_kotor = """
ID,Nama,Umur,Gaji,Jenis_Kelamin,Pendidikan,Kota,Tanggal_Daftar,Pengalaman_Thn,Status_Nikah,Churn
1,Andi Pratama,28,8500000,L,S1,Jakarta,2022-01-15,3,Belum,0
2,Budi Santoso,32,12000000,L,S1,Bandung,2021/06/20,5,,0
3,citra dewi,25,7000000,P,D3,surabaya,,1,Belum,1
4,Dewi Anggraeni,,15000000,P,S2,Jakarta,2019-03-10,10,Kawin,0
5,Eko Wijaya,-5,9500000,L,S1,Bandung,2022-08-05,4,Belum,0
6,Fani Hidayat,40,20000000,L,S2,Jakarta,2015-12-01,,Kawin,0
7,Gita Permata,30,10500000,P,S1,Surabaya,2021-04-18,3,Belum,1
8,Andi Pratama,28,8500000,L,S1,Jakarta,2022-01-15,3,Belum,0
9,Hendra Gunawan,35,99999999,L,S3,Yogyakarta,2018-09-22,8,Kawin,0
10,Inge Lestari,29,,P,S1,Semarang,2023-02-14,2,Belum,1
11,Joko Susilo,200,11000000,L,S1,?,2022-11-30,4,Kawin,?
12,Kartika Sari,33,13500000,P,S2,Bandung,2020-07-08,6,
13,Lukman Hakim,38,,L,D4,Medan,2017-05-03,9,Kawin,0
14,Maya Putri,27,7800000,P,D3,surabaya,2023-01-20,,Belum,1
15,Nanda Putra,31,11500000,L,S1,Jakarta,2021-10-12,4,Kawin,0
"""

df = pd.read_csv(StringIO(data_kotor))
df["Tanggal_Daftar"] = df["Tanggal_Daftar"].astype(str)
df = df.replace({"": np.nan, "nan": np.nan, "?": np.nan, None: np.nan})
print(f"Shape awal: {df.shape}")
print(f"\nData mentah (15 baris pertama):")
print(df.to_string())

print("\n>>> 2. DATA UNDERSTANDING - Inspeksi Awal")
print("-" * 60)

print("\nInformasi kolom & null:")
import io as _info_io
buf = _info_io.StringIO()
df.info(buf=buf)
info_text = buf.getvalue()
print(info_text)

print(f"\nPersentase missing value per kolom:")
missing_pct = df.isnull().mean() * 100
missing_df = pd.DataFrame({
    "Jumlah_Null": df.isnull().sum(),
    "Persentase_%": missing_pct.round(2)
})
print(missing_df[missing_df["Jumlah_Null"] > 0])

print("\n>>> 3. DATA CLEANING - Tahapan 1: Hapus Duplikat")
print("-" * 60)

print(f"Jumlah baris duplikat (berdasarkan semua kolom): {df.duplicated().sum()}")
print(f"Jumlah duplikat berdasarkan ID: {df.duplicated(subset=['ID']).sum()}")
print(f"Jumlah duplikat berdasarkan Nama+Umur+Gaji: {df.duplicated(subset=['Nama', 'Umur', 'Gaji']).sum()}")

df_clean = df.drop_duplicates(subset=["Nama", "Umur", "Gaji"], keep="first").reset_index(drop=True)
print(f"Shape setelah hapus duplikat: {df_clean.shape}")

print("\n>>> 4. DATA CLEANING - Tahapan 2: Perbaiki Format Data")
print("-" * 60)

print("Sebelum perbaikan:")
print(f"  Nama: {df_clean['Nama'].tolist()[:5]}")
print(f"  Kota: {df_clean['Kota'].tolist()[:5]}")
print(f"  Jenis_Kelamin: {df_clean['Jenis_Kelamin'].unique()}")

df_clean["Nama"] = df_clean["Nama"].str.title()
df_clean["Kota"] = df_clean["Kota"].str.title()
df_clean["Jenis_Kelamin"] = df_clean["Jenis_Kelamin"].replace({
    "L": "Laki-laki", "P": "Perempuan", "l": "Laki-laki", "p": "Perempuan"
})
df_clean["Tanggal_Daftar"] = pd.to_datetime(df_clean["Tanggal_Daftar"], format="mixed", errors="coerce")

print("\nSetelah perbaikan:")
print(f"  Nama: {df_clean['Nama'].tolist()[:5]}")
print(f"  Kota: {df_clean['Kota'].tolist()[:5]}")
print(f"  Jenis_Kelamin: {df_clean['Jenis_Kelamin'].unique()}")
print(f"  Tanggal_Daftar dtype: {df_clean['Tanggal_Daftar'].dtype}")

print("\n>>> 5. DATA CLEANING - Tahapan 3: Handling Invalid Data")
print("-" * 60)

print("Nilai invalid terdeteksi:")
print(f"  Umur min: {df_clean['Umur'].min()}, max: {df_clean['Umur'].max()}")
print(f"  Gaji  max: {df_clean['Gaji'].max():,}")
print(f"  Pengalaman_Thn max: {df_clean['Pengalaman_Thn'].max()}")

umur_invalid = (df_clean["Umur"] < 17) | (df_clean["Umur"] > 65) | df_clean["Umur"].isnull()
gaji_invalid = (df_clean["Gaji"] > 50000000) | df_clean["Gaji"].isnull()

print(f"\nBaris Umur invalid: {umur_invalid.sum()}")
print(f"Baris Gaji invalid: {gaji_invalid.sum()}")

df_clean.loc[umur_invalid, "Umur"] = np.nan
df_clean.loc[gaji_invalid, "Gaji"] = np.nan

print(f"\nSetelah penandaan invalid → NaN:")
print(f"  Umur null: {df_clean['Umur'].isnull().sum()}")
print(f"  Gaji null: {df_clean['Gaji'].isnull().sum()}")

print("\n>>> 6. DATA CLEANING - Tahapan 4: Handling Missing Values")
print("-" * 60)

print("""
STRATEGI HANDLING MISSING VALUE:
  • DROP    → Jika <5% baris hilang atau kolom >50% null
  • IMPUTE  → Ganti dengan nilai yang masuk akal
     - Numerical: Mean / Median / Mode / Konstanta
     - Categorical: Mode / "Unknown" / Konstanta
  • FORWARD/BACKWARD FILL → Untuk time series
  • MODEL-BASED → KNN Imputer, MICE (lebih canggih)
""")

print("Sebelum impute:")
print(f"  Status_Nikah null: {df_clean['Status_Nikah'].isnull().sum()}")
print(f"  Pendidikan null: {df_clean['Pendidikan'].isnull().sum()}")

df_clean["Status_Nikah"] = df_clean["Status_Nikah"].fillna("Tidak_Diketahui")
df_clean["Kota"] = df_clean["Kota"].fillna(df_clean["Kota"].mode()[0])
df_clean["Pengalaman_Thn"] = df_clean["Pengalaman_Thn"].fillna(df_clean["Pengalaman_Thn"].median())
df_clean["Umur"] = df_clean.groupby("Pendidikan")["Umur"].transform(lambda x: x.fillna(x.median()))
df_clean["Umur"] = df_clean["Umur"].fillna(df_clean["Umur"].median())
df_clean["Gaji"] = df_clean.groupby(["Pendidikan", "Kota"])["Gaji"].transform(lambda x: x.fillna(x.median()))
df_clean["Gaji"] = df_clean["Gaji"].fillna(df_clean["Gaji"].median())
df_clean["Tanggal_Daftar"] = df_clean["Tanggal_Daftar"].fillna(df_clean["Tanggal_Daftar"].median())
df_clean["Churn"] = df_clean["Churn"].fillna(df_clean["Churn"].mode()[0]).astype(int)

print("\nSetelah impute:")
print(f"  Total null tersisa: {df_clean.isnull().sum().sum()}")
print(f"  Shape: {df_clean.shape}")
print("\nData bersih final:")
print(df_clean[["Nama", "Umur", "Gaji", "Status_Nikah", "Kota", "Pengalaman_Thn"]].to_string())

print("\n>>> 7. HANDLING OUTLIER - IQR Method & Z-Score")
print("-" * 60)

print("Sebelum outlier handling - Gaji:")
Q1 = df_clean["Gaji"].quantile(0.25)
Q3 = df_clean["Gaji"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"  Q1: {Q1:,} | Q3: {Q3:,} | IQR: {IQR:,}")
print(f"  Batas bawah: {lower_bound:,} | Batas atas: {upper_bound:,}")

outlier_mask = (df_clean["Gaji"] < lower_bound) | (df_clean["Gaji"] > upper_bound)
print(f"  Jumlah outlier Gaji: {outlier_mask.sum()}")

df_clean["Gaji_Capped"] = df_clean["Gaji"].clip(lower=lower_bound, upper=upper_bound)

print(f"\nZ-Score Method untuk Pengalaman_Thn:")
z = np.abs((df_clean["Pengalaman_Thn"] - df_clean["Pengalaman_Thn"].mean()) / df_clean["Pengalaman_Thn"].std())
print(f"  Z-Score > 3  : {(z > 3).sum()} baris")
print(f"  Z-Score max  : {z.max():.3f}")

print("\n>>> LATIHAN MANDIRI:")
print("-" * 60)
print("""
1. Drop kolom yang memiliki null >70% (jika ada)
2. Buat fungsi reusable bernama 'bersihkan_dataframe()'
   yang melakukan:
   - Drop duplikat
   - Title case kolom string
   - Replace placeholder (?, -, N/A) → NaN
   - Impute numerik dengan median
   - Impute kategorik dengan mode
3. Uji fungsi tersebut pada dataset baru dengan 30% null random
""")

print("\n✓ Bagian 1 Modul 3 Selesai: Data Understanding & Cleaning")
