import numpy as np
import pandas as pd

# --- 1. Konsep OOP untuk Dataset AI ---
class DatasetAI:
    """Contoh Class sederhana untuk membungkus data fitur dan label."""
    
    # Class attribute (dimiliki bersama oleh semua instance)
    total_dataset = 0

    def __init__(self, nama, fitur, label, sumber="internal"):
        self.nama = nama
        self.fitur = fitur # Biasanya NumPy array atau Pandas DataFrame
        self.label = label
        self.sumber = sumber
        DatasetAI.total_dataset += 1 # Increment counter setiap objek dibuat

    def info(self):
        """Menampilkan metadata dataset."""
        # Menggunakan getattr untuk menangani jika fitur tidak punya shape (misal list biasa)
        shape = getattr(self.fitur, 'shape', 'Unknown shape')
        print(f"📁 Dataset: {self.nama} | Shape: {shape} | Sumber: {self.sumber}")

    def split_data(self, test_size=0.2):
        """Simulasi fungsi train/test split."""
        n = len(self.fitur)
        n_test = int(n * test_size)
        print(f"✓ Splitting data: {n-n_test} training samples, {n_test} test samples.")
        # (Di sini biasanya implementasi slicing menggunakan index acak)
        # return X_train, X_test, y_train, y_test (disederhanakan)

# Membuat instance objek
fitur_dummy = np.random.rand(100, 5)
label_dummy = np.random.randint(0, 2, 100)

ds_proyek_A = DatasetAI("Data Transaksi Fraud", fitur_dummy, label_dummy)
ds_proyek_A.info()
ds_proyek_A.split_data(test_size=0.3)


# --- 2. Exception Handling (Menangani Eror) ---
# Kita buat custom exception untuk masalah kualitas data
class DataQualityError(Exception):
    pass

def validasi_kolom_wajib(df, kolom_wajib):
    """Memastikan DataFrame memiliki kolom yang dibutuhkan model."""
    missing = [c for c in kolom_wajib if c not in df.columns]
    if missing:
        # Raise error spesifik jika kolom hilang
        raise DataQualityError(f"Kolom wajib hilang: {missing}")
    print("✅ Validasi struktur data berhasil.")

# Data contoh yang tidak lengkap
df_rusak = pd.DataFrame({"Umur": [25, 30], "Gaji": [5000, 7000]})

print("\n--- Memulai Validasi Data ---")
try:
    # Kita mewajibkan kolom 'Email', tapi data tidak punya
    validasi_kolom_wajib(df_rusak, kolom_wajib=["Umur", "Gaji", "Email"])

except DataQualityError as e:
    # Tangkap eror spesifik kualitas data
    print(f"❌ PERINGATAN DATA: {e}")
    print("   -> Tindakan: Hubungi tim data engineer untuk perbaikan.")

except Exception as e:
    # Tangkap eror lain yang tidak terduga
    print(f"❌ Eror Sistem Tidak Terduga: {e}")

finally:
    # Blok ini selalu dijalankan, sukses atau gagal
    print("--- Proses Validasi Selesai ---")