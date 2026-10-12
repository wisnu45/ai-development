from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# --- Setup ---
# (Asumsikan data harga rumah Xr_train, Xr_test, yr_train, yr_test sudah siap dan scaled)

# --- 1. Arsitektur MLP untuk Regresi ---
# Bedanya hanya di layer output (linear/identity) dan loss function (MSE)
# scikit-learn menangani ini otomatis saat kita pakai MLPRegressor
mlp_reg = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32), # Arsitektur sedikit lebih besar
    activation="relu",
    solver="adam",
    alpha=0.001,
    max_iter=1000,
    early_stopping=True,
    random_state=42
)

# --- 2. Training ---
print("\nMemulai Training MLP Regresi...")
mlp_reg.fit(Xr_train, yr_train)

# --- 3. Evaluasi ---
yr_pred = mlp_reg.predict(Xr_test)
r2_test = r2_score(yr_test, yr_pred)
mae_test = mean_absolute_error(yr_test, yr_pred)

print(f"Hasil Evaluasi MLP Regresi Harga Rumah:")
print(f"  R² Score di Test Set : {r2_test:.4f}")
print(f"  Mean Absolute Error : {mae_test:,.0f} (Rata-rata meleset sekian Rupiah)")