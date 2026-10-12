import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# --- Simulasi Dataset Churn Nasabah ---
np.random.seed(42)
n = 1000
# Fitur
X_data = pd.DataFrame({
    "Umur": np.random.randint(20, 60, n),
    "Skor_Kredit": np.random.normal(650, 100, n),
    "Saldo": np.random.exponential(5000000, n)
})
# Target: Churn (1=Pindah, 0=Tetap). Kita buat agak tidak seimbang (imbalanced)
# Peluang churn lebih tinggi jika saldo rendah dan umur muda
peluang = (1 / (X_data["Saldo"] + 1e-9)) * 1e6 + (1 / X_data["Umur"]) * 10
y_target = (np.random.rand(n) < peluang.clip(0.1, 0.8)).astype(int)

print(f"Total Data: {len(X_data)}")
print(f"Distribusi Target Churn Awal:\n{y_target.value_counts(normalize=True)}")

# --- EDA Sederhana (Visualisasi Korelasi) ---
# Menggabungkan sementara untuk melihat korelasi
df_eda = X_data.copy()
df_eda['Churn'] = y_target
plt.figure(figsize=(6, 4))
sns.heatmap(df_eda.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korelasi Fitur dengan Target Churn")
plt.tight_layout()
plt.show() # Di notebook akan muncul plot


# --- DATA SPLITTING STRATEGY (Sangat Penting!) ---

# Langkah 1: Pisahkan Test Set (Kunci di lemari, jangan disentuh!)
# PENTING: Gunakan 'stratify=y' agar rasio Churn di data Test sama dengan data asli.
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_data, y_target, test_size=0.15, random_state=42, stratify=y_target
)

# Langkah 2: Pisahkan Train dan Validation dari sisa data
# Kita ambil ~15% dari total awal untuk validasi (0.15 / 0.85 ≈ 0.176)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, random_state=42, stratify=y_trainval
)

print("\n--- Hasil Splitting (Perhatikan proporsi stratify terjaga) ---")
print(f"Train Set : {len(y_train)} baris | Churn Rate: {y_train.mean():.3f}")
print(f"Val Set   : {len(y_val)} baris | Churn Rate: {y_val.mean():.3f}")
print(f"Test Set  : {len(y_test)} baris | Churn Rate: {y_test.mean():.3f}")




# --- Setup Awal & Preprocessing ---
# (Asumsikan data harga rumah sudah dimuat sebagai X dan y=Harga_Juta)
# Kita gunakan Pipeline preprocessing yang mirip dengan klasifikasi tadi.
X_pp_reg = preprocessor.fit_transform(X_reg) # X_reg adalah fitur rumah
y_reg = df_rumah["Harga_Juta"].values
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_pp_reg, y_reg, test_size=0.2, random_state=42)

# --- 1. Training Berbagai Model Regresi ---
reg_models = {
    "Linear Regression": LinearRegression(), # Garis lurus paling dasar
    "Ridge (L2)": Ridge(alpha=1.0), # Linear + Regularisasi (mencegah overfit)
    "Lasso (L1)": Lasso(alpha=1.0), # Linear + Seleksi Fitur otomatis
    "Random Forest Reg": RandomForestRegressor(n_estimators=100) # Versi regresi dari si hutan
}

print("Evaluasi Model Regresi (R² Score di Test Set):")
for name, model in reg_models.items():
    model.fit(X_train_reg, y_train_reg)
    r2_test = r2_score(y_test_reg, model.predict(X_test_reg))
    print(f"  {name:<25} | R² Test: {r2_test:.4f}")

# --- 2. Analisis Mendalam Model Terbaik ---
# Misal Random Forest Regressor adalah yang terbaik. Mari cek error-nya.
best_reg = reg_models["Random Forest Reg"]
y_pred_reg = best_reg.predict(X_test_reg)

mae = mean_absolute_error(y_test_reg, y_pred_reg)
mape = mean_absolute_percentage_error(y_test_reg, y_pred_reg)

print(f"\nAnalisis Error Random Forest:")
print(f"  Rata-rata Meleset (MAE) : Rp {mae:.0f} Juta")
print(f"  Persentase Meleset (MAPE): {mape:.2%} (Di bawah 10% dianggap sangat bagus di industri)")

# --- 3. Visualisasi: Prediksi vs Aktual ---
# Plot ini wajib dibuat untuk melihat seberapa dekat tebakan model dengan kenyataan.
plt.figure(figsize=(6, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5, color="blue")
# Garis merah adalah garis ideal di mana Prediksi = Aktual
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel("Harga Aktual (Juta Rp)")
plt.ylabel("Harga Prediksi (Juta Rp)")
plt.title("Seberapa Akurat Tebakan Model?")
plt.grid(True)
plt.show() # Di notebook akan muncul plot