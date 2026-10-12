import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

# Simulasi data bersih
data = pd.DataFrame({
    "Umur": [25, 35, 45, 22, 50],
    "Gaji": [5_000_000, 12_000_000, 18_000_000, 4_500_000, 25_000_000],
    "Pendidikan": ["S1", "S2", "S1", "D3", "S3"],
    "Kota": ["Jakarta", "Bandung", "Jakarta", "Surabaya", "Medan"],
    "Rating_Kinerja": ["Cukup", "Baik", "Sangat Baik", "Cukup", "Luar Biasa"]
})

print("Data Awal:")
print(data.head(2))

# --- 1. FEATURE ENGINEERING (Seni Menciptakan Fitur) ---
# Contoh: Kita tahu Gaji berhubungan dengan Umur. Mari buat rasio.
data["Rasio_Gaji_Umur"] = data["Gaji"] / data["Umur"]

# Contoh: Mengelompokkan (Binning) Umur menjadi kategori
bins = [0, 30, 45, 100]
labels = ["Muda", "Paruh Baya", "Senior"]
data["Kelompok_Umur"] = pd.cut(data["Umur"], bins=bins, labels=labels)

print("\nSetelah Feature Engineering (Fitur Baru):")
print(data[['Umur', 'Gaji', 'Rasio_Gaji_Umur', 'Kelompok_Umur']].head(3))


# --- 2. CATEGORICAL ENCODING ---
# a. Ordinal Encoding untuk data berurutan (Rating Kinerja)
ordinal_map = {"Cukup": 1, "Baik": 2, "Sangat Baik": 3, "Luar Biasa": 4}
data["Rating_Encoded"] = data["Rating_Kinerja"].map(ordinal_map)

# b. One-Hot Encoding untuk data nominal (Kota)
# CATATAN UNTUK BELAJAR:
#   • drop_first=False  → SEMUA kota punya kolom (Bandung, Jakarta, Medan, Surabaya) — MUDAH DIPAHAMI
#   • drop_first=True   → HAPUS 1 kolom (kota urutan alfabetis pertama = Bandung)
#                          Digunakan jika model = Linear / Logistic Regression (hindari Dummy Variable Trap / Multicollinearity)
#                          Untuk model berbasis Tree (RF/GBM/XGBoost): drop_first TIDAK USAH, biarkan saja semua kolom ada.
dummies_kota = pd.get_dummies(data["Kota"], prefix="Kota", drop_first=False, dtype=int)
data_final = pd.concat([data, dummies_kota], axis=1)

# Print daftar nama kolom Kota setelah One-Hot Encoding:
print(f"\nℹ️  Kolom hasil One-Hot Encoding (Kota): {list(dummies_kota.columns)}")
print("\nSetelah Encoding (Perhatikan kolom Kota menjadi angka 0/1):")
# Ambil 3 kolom pertama + Rating Encoded (agar tidak terlalu panjang):
list_kolom_kota = list(dummies_kota.columns)[:3]
print(data_final[['Kota'] + list_kolom_kota + ['Rating_Encoded']].head(3))


# --- 3. FEATURE SCALING ---
# Menggunakan StandardScaler (mengubah data jadi mean=0, std=1)
scaler = StandardScaler()
kolom_numerik = ["Umur", "Gaji", "Rasio_Gaji_Umur"]
data_final[kolom_numerik] = scaler.fit_transform(data_final[kolom_numerik])

print("\nSetelah Scaling (Perhatikan Umur dan Gaji sekarang berskala kecil):")
print(data_final[kolom_numerik].round(2).head(3))