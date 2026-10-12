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