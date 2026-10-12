import pandas as pd
import numpy as np
from io import StringIO

# --- Simulasi Data Mentah yang Kotor ---
# Perhatikan ada nilai kosong (,,), format tanggal beda, data aneh (-5, 200), dll.
data_kotor = """
ID,Nama,Umur,Gaji,Jenis_Kelamin,Pendidikan,Kota,Tanggal_Daftar,Pengalaman_Thn,Status_Nikah,Churn
1,Andi Pratama,28,8500000,L,S1,Jakarta,2022-01-15,3,Belum,0
2,Budi Santoso,32,12000000,L,S1,Bandung,2021/06/20,5,,0
4,Dewi Anggraeni,,15000000,P,S2,Jakarta,2019-03-10,10,Kawin,0
5,Eko Wijaya,-5,9500000,L,S1,Bandung,2022-08-05,4,Belum,0
8,Andi Pratama,28,8500000,L,S1,Jakarta,2022-01-15,3,Belum,0
11,Joko Susilo,200,11000000,L,S1,?,2022-11-30,4,Kawin,?
"""
# Membaca data string sebagai CSV
df = pd.read_csv(StringIO(data_kotor))

# Trik: Ubah string kosong atau tanda tanya '?' menjadi NaN (Not a Number) standar NumPy
df = df.replace({"": np.nan, "nan": np.nan, "?": np.nan, None: np.nan})

print("Data Mentah Awal:")
print(df[['Nama', 'Umur', 'Gaji', 'Kota', 'Status_Nikah']].to_string())

# --- TAHAP 1: Menghapus Duplikat ---
# Kita hapus jika Nama, Umur, dan Gaji sama persis. (Baris ID 1 dan 8)
df_clean = df.drop_duplicates(subset=["Nama", "Umur", "Gaji"], keep="first").reset_index(drop=True)
print(f"\n[Cleaning 1] Duplikat dihapus. Sisa baris: {len(df_clean)}")

# --- TAHAP 2: Perbaikan Format Data ---
# Standardisasi teks (Title Case) dan format tanggal
df_clean["Nama"] = df_clean["Nama"].str.title()
df_clean["Jenis_Kelamin"] = df_clean["Jenis_Kelamin"].replace({"L": "Laki-laki", "P": "Perempuan"})
# 'coerce' akan mengubah tanggal yang error menjadi NaT (Not a Time)
df_clean["Tanggal_Daftar"] = pd.to_datetime(df_clean["Tanggal_Daftar"], format="mixed", errors="coerce")
print("[Cleaning 2] Format teks dan tanggal distandardisasi.")

# --- TAHAP 3: Handling Data Invalid (Logic Check) ---
# Umur -5 atau 200 tahun tidak masuk akal. Kita tandai sebagai NaN.
umur_invalid = (df_clean["Umur"] < 17) | (df_clean["Umur"] > 65)
df_clean.loc[umur_invalid, "Umur"] = np.nan
print(f"[Cleaning 3] Data invalid (umur <17 atau >65) diubah jadi NaN.")

# --- TAHAP 4: Handling Missing Values (Imputasi) ---
# Strategi: Isi data numerik dengan MEDIAN (tahan outlier), kategorik dengan MODE (terbanyak).
df_clean["Umur"] = df_clean["Umur"].fillna(df_clean["Umur"].median())
df_clean["Status_Nikah"] = df_clean["Status_Nikah"].fillna("Tidak_Diketahui") # Kategori baru
print(f"[Cleaning 4] Missing value diimputasi (Umur pakai Median, Status pakai kategori baru).")

# --- TAHAP 5: Handling Outlier (Contoh: Metode Z-Score) ---
# Outlier adalah data yang jauh dari rata-rata.
z_score_gaji = np.abs((df_clean["Gaji"] - df_clean["Gaji"].mean()) / df_clean["Gaji"].std())
# Misal: Data dengan Z-Score > 2 dianggap outlier (di praktik nyata biasanya > 3)
print(f"\n[Cleaning 5] Deteksi Outlier Gaji (Z-Score > 2):")
print(df_clean[z_score_gaji > 2][['Nama', 'Gaji']])