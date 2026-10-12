print("="*60)
print("MODUL 4: MACHINE LEARNING ENGINEERING")
print("Bagian 2: Supervised Learning - REGRESSION")
print("="*60)

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, mean_absolute_percentage_error)

output_dir = "c:\\ai-engineer\\modul_04_machine_learning\\output_charts"
os.makedirs(output_dir, exist_ok=True)

sns.set_style("whitegrid")
np.random.seed(42)

print("""
┌─────────────────────────────────────────────────────────────────┐
│         SUPERVISED LEARNING: REGRESSION                        │
├─────────────────────────────────────────────────────────────────┤
│  Tujuan = Prediksi NILAI KONTINU (angka)                       │
│                                                                 │
│  METRIK EVALUASI REGRESI (PENTING BNSP!):                      │
│  MAE  = Mean Absolute Error      → Rata-rata |error|           │
│  MSE  = Mean Squared Error       → Hukum error BESAR           │
│  RMSE = Root MSE                  → Unit sama dengan target    │
│  R²   = Coefficient of Determ.  → 0.0 ~ 1.0 (1=sempurna)      │
│  MAPE = Mean Absolute % Error    → % (≤10% = bagus)            │
└─────────────────────────────────────────────────────────────────┘
""")

print("\n>>> 1. PERSIAPKAN DATASET - Prediksi Harga Rumah")
print("-" * 60)

n = 1500
df = pd.DataFrame({
    "LT": np.random.normal(150, 40, n).clip(60, 350).round(1),
    "LB": np.random.normal(100, 25, n).clip(40, 250).round(1),
    "Kamar_Tidur": np.random.choice([2, 3, 4, 5, 6], n, p=[0.1, 0.45, 0.30, 0.12, 0.03]).astype(int),
    "Kamar_Mandi": np.random.choice([1, 2, 3, 4], n, p=[0.25, 0.45, 0.22, 0.08]).astype(int),
    "Garasi": np.random.choice([0, 1, 2, 3], n, p=[0.15, 0.35, 0.40, 0.10]).astype(int),
    "Tahun_Bangun": np.random.randint(1985, 2025, n),
    "Lokasi": np.random.choice(["Pusat Kota", "Perbatasan", "Suburban", "Pedesaan"], n, p=[0.25, 0.25, 0.35, 0.15]),
    "Sertifikat": np.random.choice(["SHM", "HGB", "AJB"], n, p=[0.55, 0.30, 0.15]),
    "Jarak_Sekolah_km": np.random.uniform(0.1, 15, n).round(1),
    "Jarak_RS_km": np.random.uniform(0.5, 20, n).round(1),
    "SwimmingPool": np.random.choice([0, 1], n, p=[0.75, 0.25]),
})

lokasi_harga = {"Pusat Kota": 800, "Perbatasan": 550, "Suburban": 400, "Pedesaan": 250}
sertif_harga = {"SHM": 1.0, "HGB": 0.90, "AJB": 0.80}

y_raw = (
    df["LT"] * 3.5 + df["LB"] * 5.0 +
    df["Kamar_Tidur"] * 25 + df["Kamar_Mandi"] * 18 +
    df["Garasi"] * 20 +
    (2025 - df["Tahun_Bangun"]) * -2.5 +
    df["SwimmingPool"] * 80 +
    df["Jarak_Sekolah_km"] * -3 + df["Jarak_RS_km"] * -2
).astype(float)

harga_series = []
for i in range(n):
    loc = df.loc[i, "Lokasi"]
    ser = df.loc[i, "Sertifikat"]
    harga = (y_raw[i] + lokasi_harga[loc]) * sertif_harga[ser] + np.random.normal(0, 30)
    harga_series.append(max(200, min(5000, harga)))

df["Harga_Juta"] = np.array(harga_series).round(0)

print(f"Shape: {df.shape}")
print(f"Target Harga_Juta (juta Rp):")
desc = df["Harga_Juta"].describe().round(0)
print(f"  Count  : {int(desc['count'])}")
print(f"  Mean   : {int(desc['mean'])}")
print(f"  Std    : {int(desc['std'])}")
print(f"  Min    : {int(desc['min'])}")
print(f"  Max    : {int(desc['max'])}")

X = df.drop("Harga_Juta", axis=1)
y = df["Harga_Juta"].values

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"))])
preprocessor = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

X_pp = preprocessor.fit_transform(X)
print(f"Fitur setelah preprocess: {X_pp.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(X_pp, y, test_size=0.2, random_state=42)
print(f"Train: {len(y_train)} | Test: {len(y_test)}")

print("\n>>> 2. TRAINING BERBAGAI MODEL REGRESI")
print("-" * 60)

reg_models = {
    "Linear Regression": LinearRegression(),
    "Ridge (alpha=1)": Ridge(alpha=1.0, random_state=42),
    "Lasso (alpha=1)": Lasso(alpha=1.0, random_state=42, max_iter=10000),
    "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42, max_iter=10000),
    "KNN Reg (k=7)": KNeighborsRegressor(n_neighbors=7),
    "SVR (RBF, C=100)": SVR(kernel="rbf", C=100, gamma="scale"),
    "DT Reg (d=8)": DecisionTreeRegressor(max_depth=8, random_state=42),
    "RF Reg (n=100)": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "GB Reg (n=100)": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

res_reg = {}
for name, mdl in reg_models.items():
    mdl.fit(X_train, y_train)
    pred_tr = mdl.predict(X_train)
    pred_te = mdl.predict(X_test)
    res_reg[name] = {
        "Train_R2": r2_score(y_train, pred_tr),
        "Test_R2": r2_score(y_test, pred_te),
        "Test_MAE": mean_absolute_error(y_test, pred_te),
        "Test_RMSE": np.sqrt(mean_squared_error(y_test, pred_te)),
        "Test_MAPE_%": mean_absolute_percentage_error(y_test, pred_te)*100,
        "y_pred_test": pred_te,
    }
    print(f"  {name:<22} → R² Train: {res_reg[name]['Train_R2']:.4f} | R² Test: {res_reg[name]['Test_R2']:.4f}")

print("\n>>> 3. PERBANDINGAN REGRESSION MODELS (RANKING TEST R²)")
print("-" * 60)

cols = ["Train_R2", "Test_R2", "Test_MAE", "Test_RMSE", "Test_MAPE_%"]
df_reg = pd.DataFrame(res_reg).T[cols].astype(float).round(4).sort_values("Test_R2", ascending=False)
print(df_reg.to_string())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
df_reg_sorted = df_reg.sort_values("Test_R2", ascending=True)

colors_r2 = plt.cm.viridis(np.linspace(0.2, 0.9, len(df_reg_sorted)))
axes[0].barh(df_reg_sorted.index, df_reg_sorted["Test_R2"], color=colors_r2)
axes[0].axvline(0, color="k", linewidth=0.5)
axes[0].set_title("R² Score Test Set (LEBIH BESAR LEBIH BAIK)", fontweight="bold", fontsize=11)
for i, v in enumerate(df_reg_sorted["Test_R2"]):
    axes[0].text(max(v, 0) + 0.01, i, f"{v:.4f}", va="center", fontsize=8)

df_reg_mape = df_reg.sort_values("Test_MAPE_%", ascending=False)
colors_mape = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(df_reg_mape)))
axes[1].barh(df_reg_mape.index, df_reg_mape["Test_MAPE_%"], color=colors_mape)
axes[1].set_title("MAPE (%) Test Set (LEBIH KECIL LEBIH BAIK)", fontweight="bold", fontsize=11)
for i, v in enumerate(df_reg_mape["Test_MAPE_%"]):
    axes[1].text(v + 0.3, i, f"{v:.2f}%", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(f"{output_dir}\\04_regression_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart perbandingan regresi tersimpan")

print("\n>>> 4. ANALISIS RESIDUAL + PREDIKSI vs AKTUAL")
print("-" * 60)

best_reg_name = max(res_reg.items(), key=lambda kv: kv[1]["Test_R2"])[0]
y_pred_best = res_reg[best_reg_name]["y_pred_test"]
print(f"🏆 Model Regresi TERBAIK: {best_reg_name}")
residuals = y_test - y_pred_best

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes[0, 0].scatter(y_test, y_pred_best, alpha=0.6, s=35, c="#3b82f6", edgecolors="white")
mx_val = max(y_test.max(), y_pred_best.max())
mn_val = min(y_test.min(), y_pred_best.min())
axes[0, 0].plot([mn_val, mx_val], [mn_val, mx_val], "r--", linewidth=2, label="Ideal (y=x)")
axes[0, 0].set_xlabel("Harga Aktual (Juta Rp)")
axes[0, 0].set_ylabel("Harga Prediksi (Juta Rp)")
axes[0, 0].set_title(f"Actual vs Predicted - {best_reg_name}", fontweight="bold")
axes[0, 0].legend()

axes[0, 1].scatter(y_pred_best, residuals, alpha=0.6, s=35, c="#7c3aed", edgecolors="white")
axes[0, 1].axhline(0, color="red", linestyle="--")
axes[0, 1].set_xlabel("Nilai Prediksi")
axes[0, 1].set_ylabel("Residual (Aktual - Prediksi)")
axes[0, 1].set_title("Residual Plot (Homoscedastisitas?)", fontweight="bold")

sns.histplot(residuals, kde=True, ax=axes[1, 0], color="#f97316", bins=30)
axes[1, 0].axvline(residuals.mean(), color="red", linestyle="--", label=f"Mean: {residuals.mean():.1f}")
axes[1, 0].set_title("Distribusi Residual (ideal ~Normal)", fontweight="bold")
axes[1, 0].legend()

from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Q-Q Plot (Normalitas Residual)", fontweight="bold")
axes[1, 1].get_lines()[0].set_markersize(4)

plt.tight_layout()
plt.savefig(f"{output_dir}\\05_residual_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart analisis residual tersimpan")

print("\n>>> 5. REGULARISASI: RIDGE vs LASSO")
print("-" * 60)

alphas = np.logspace(-4, 5, 30)
ridge_coefs, lasso_coefs = [], []
for a in alphas:
    ridge_coefs.append(Ridge(alpha=a).fit(X_train, y_train).coef_)
    lasso_coefs.append(Lasso(alpha=a, max_iter=10000).fit(X_train, y_train).coef_)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].plot(alphas, np.array(ridge_coefs))
axes[0].set_xscale("log")
axes[0].set_xlabel("Alpha (Regularization Strength)")
axes[0].set_ylabel("Koefisien")
axes[0].set_title("Ridge: Shrinks semua koefisien MENDEKATI NOL", fontweight="bold", fontsize=11)
axes[0].axhline(0, color="k", linewidth=0.5)

axes[1].plot(alphas, np.array(lasso_coefs))
axes[1].set_xscale("log")
axes[1].set_xlabel("Alpha (Regularization Strength)")
axes[1].set_title("Lasso: MEMBUAT NOL koefisien (Feature Selection!)", fontweight="bold", fontsize=11)
axes[1].axhline(0, color="k", linewidth=0.5)

plt.tight_layout()
plt.savefig(f"{output_dir}\\06_ridge_lasso_regularization.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart regularisasi tersimpan")

print("""
┌─────────────────────────────────────────────────────────────────┐
│          KESIMPULAN REGRESI UNTUK BNSP:                        │
├─────────────────────────────────────────────────────────────────┤
│  R² > 0.80 → Bagus      R² > 0.95 → Sangat Bagus              │
│  MAPE <10% → Excellent  MAPE 10-20% → Good                     │
│  Residual: acak di seputar 0 (homoscedastis)                   │
│  Q-Q Plot: mendekati garis lurus → residual normal            │
│  Overfit: (R² Train - R² Test) > 0.15 → perlu regularisasi     │
└─────────────────────────────────────────────────────────────────┘
""")

print("""\n>>> LATIHAN:
1. Prediksi harga: LT=200, LB=150, Kamar=3, Tahun=2020, Pusat Kota, SHM
   Bandingkan prediksi 3 model terbaik!
2. Tambahkan PolynomialFeatures derajat=2 pada Linear Regression,
   apakah R² Test membaik atau malah overfit?
3. GB Reg: coba n_estimators=300, max_depth=5, bagus mana?
4. Hitung MAE, MSE, RMSE SECARA MANUAL tanpa sklearn.metrics
   untuk model Linear Regression!
""")

print("\n✓ Bagian 2 Modul 4 Selesai: REGRESSION")
