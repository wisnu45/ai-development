print("="*60)
print("MODUL 5: MODEL EVALUATION & OPTIMIZATION")
print("Bagian 2: Cross-Validation, Grid/Random Search, Hyperparameter Tuning")
print("="*60)

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, StratifiedKFold, KFold,
                                     cross_val_score, cross_validate,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from scipy.stats import randint, uniform

output_dir = "c:\\ai-engineer\\modul_05_evaluasi_optimasi\\output_charts"
os.makedirs(output_dir, exist_ok=True)

sns.set_style("whitegrid")
np.random.seed(42)

print("""
┌─────────────────────────────────────────────────────────────────┐
│       CROSS VALIDATION & HYPERPARAMETER TUNING                  │
├─────────────────────────────────────────────────────────────────┤
│  CROSS VALIDATION = Validasi model SECARA BERULANG              │
│                   tanpa boros split dataset                     │
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐     │
│  │  JENIS-JENIS CROSS VALIDATION:                        │     │
│  │                                                       │     │
│  │  1. K-FOLD CV        → Data dibagi K lipatan,        │     │
│  │                      → K-1 train, 1 test, ulang K x   │     │
│  │                      → Umum K=5 atau K=10            │     │
│  │                                                       │     │
│  │  2. STRATIFIED K-FOLD→ K-FOLD TAPI TIAP FOLD JAGA    │     │
│  │                      RASIO KELAS SAMA (CLASSIFICATION│     │
│  │                      ⚠ PALING UMUM DIGUNAKAN!)       │     │
│  │                                                       │     │
│  │  3. LEAVE-ONE-OUT (LOOCV) → K=N, sangan lambat       │     │
│  │  4. TIME SERIES SPLIT → Tidak di-shuffle (urut waktu)│     │
│  └───────────────────────────────────────────────────────┘     │
│                                                                 │
│  HYPERPARAMETER TUNING:                                         │
│  ┌───────────────────────────────────────────────────────┐     │
│  │  GRID SEARCH    → Cobai SEMUA kombinasi (exhaustive) │     │
│  │                → Akurat tapi LAMBAT & MAHAL          │     │
│  │  RANDOM SEARCH  → Sample random N kombinasi          │     │
│  │                → Cepat & hasil biasanya mirip        │     │
│  │  BAYESIAN OPT   → Optimal cerdas dengan surrogate    │     │
│  │                → Lebih optimal tapi implementasi     │     │
│  │                  rumit (Optuna/Hyperopt)             │     │
│  └───────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
""")

print("\n>>> 1. DATASET CHURN PELANGGAN (Classification Stratified)")
print("-" * 60)

n = 1500
np.random.seed(42)
df = pd.DataFrame({
    "Durasi_Langganan_Bln": np.random.randint(1, 72, n),
    "Tagihan_Bulanan": np.random.uniform(50, 200, n).round(2),
    "Total_Biaya": 0,
    "Support_Calls": np.random.poisson(2, n).astype(int),
    "Jml_Layanan": np.random.randint(1, 7, n),
    "Kontrak": np.random.choice(["Month-to-month", "1 year", "2 year"], n, p=[0.55, 0.25, 0.20]),
    "Internet": np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.3, 0.5, 0.2]),
    "Payment": np.random.choice(["Bank transfer", "Credit card", "Electronic check", "Mailed check"], n),
})
df["Total_Biaya"] = df["Durasi_Langganan_Bln"] * df["Tagihan_Bulanan"] + np.random.normal(0, 100, n)
kontrak_numerik = {"Month-to-month": 0, "1 year": 1, "2 year": 2}
peluang = (
    (1 - df["Kontrak"].map(kontrak_numerik) / 2) * 0.35 +
    df["Support_Calls"].clip(0, 10) / 10 * 0.25 +
    df["Tagihan_Bulanan"] / df["Tagihan_Bulanan"].max() * 0.20 +
    (df["Internet"] == "Fiber optic").astype(int) * 0.10
)
df["Churn"] = (np.random.rand(n) < peluang.clip(0.05, 0.95)).astype(int)

X = df.drop("Churn", axis=1)
y = df["Churn"].values
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"))])
preprocessor = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
X_pp = preprocessor.fit_transform(X)

print(f"Shape: {X_pp.shape}, Churn rate: {y.mean()*100:.1f}%")

print("\n>>> 2. K-FOLD vs STRATIFIED K-FOLD CROSS VALIDATION")
print("-" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

results_kf = cross_validate(base_model, X_pp, y, cv=kf, scoring=scoring_metrics, return_train_score=True)
results_skf = cross_validate(base_model, X_pp, y, cv=skf, scoring=scoring_metrics, return_train_score=True)

df_cv_compare = pd.DataFrame({
    "K-Fold (test)": {m[5:]: f"{results_kf[f'test_{m}'].mean():.4f} ± {results_kf[f'test_{m}'].std():.4f}"
                  for m in scoring_metrics},
    "Stratified K-Fold (test)": {m: f"{results_skf[f'test_{m}'].mean():.4f} ± {results_skf[f'test_{m}'].std():.4f}"
                             for m in scoring_metrics},
})
print("Perbandingan CV (mean ± std dari 5 fold):")
print(df_cv_compare.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fold_labels = [f"Fold {i+1}" for i in range(5)]
x = np.arange(5)
w = 0.35

axes[0].bar(x - w/2, results_kf["test_f1"], w, label="K-Fold F1", color="#3b82f6")
axes[0].bar(x + w/2, results_skf["test_f1"], w, label="Stratified K-Fold F1", color="#16a34a")
axes[0].axhline(results_kf["test_f1"].mean(), color="blue", linestyle=":", label=f"KFold μ={results_kf['test_f1'].mean():.3f}")
axes[0].axhline(results_skf["test_f1"].mean(), color="green", linestyle="--", label=f"Strat μ={results_skf['test_f1'].mean():.3f}")
axes[0].set_xticks(x); axes[0].set_xticklabels(fold_labels)
axes[0].set_title("F1 Score per Fold: K-Fold vs Stratified K-Fold", fontweight="bold")
axes[0].legend(fontsize=8)
axes[0].set_ylim(0.5, 0.9)

axes[1].bar(x - w/2, results_skf["test_accuracy"], w, label="Accuracy", color="#3b82f6")
axes[1].bar(x + w/2, results_skf["test_roc_auc"], w, label="ROC-AUC", color="#dc2626")
axes[1].axhline(results_skf["test_accuracy"].mean(), color="blue", linestyle=":")
axes[1].axhline(results_skf["test_roc_auc"].mean(), color="red", linestyle="--")
axes[1].set_xticks(x); axes[1].set_xticklabels(fold_labels)
axes[1].set_title("Stratified K-Fold (5 splits): Accuracy vs ROC-AUC", fontweight="bold")
axes[1].legend()
axes[1].set_ylim(0.6, 0.95)

plt.tight_layout()
plt.savefig(f"{output_dir}\\04_cross_validation_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart Cross Validation tersimpan")

print("\n>>> 3. GRID SEARCH CV - Random Forest (Exhaustive!)")
print("-" * 60)

X_train, X_test, y_train, y_test = train_test_split(X_pp, y, test_size=0.2, random_state=42, stratify=y)

param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
total_kombinasi = np.prod([len(v) for v in param_grid_rf.values()])
print(f"Parameter Grid RF: {param_grid_rf}")
print(f"TOTAL KOMBINASI: {total_kombinasi} x 5 CV = {total_kombinasi * 5} fit TRAINING!")

t0 = time.time()
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid_rf,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1",
    n_jobs=-1,
    verbose=0,
    refit=True,
)
grid_rf.fit(X_train, y_train)
t_grid = time.time() - t0

print(f"\n⏱ GridSearchCV waktu: {t_grid:.1f} detik")
print(f"🏆 Best params GridSearch: {grid_rf.best_params_}")
print(f"   Best CV F1 Score     : {grid_rf.best_score_:.4f}")
print(f"   Test Set F1 (final)  : {grid_rf.score(X_test, y_test):.4f}")

print("\n>>> 4. RANDOMIZED SEARCH CV - Random Forest + GB (Cepat!)")
print("-" * 60)

param_dist_rf = {
    "n_estimators": randint(50, 400),
    "max_depth": [None, 5, 10, 15, 20, 30],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
}

n_iter = 40
print(f"Random Search dengan {n_iter} iterasi ({n_iter} x 5 CV = {n_iter*5} fit)")

t0 = time.time()
rand_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_dist_rf,
    n_iter=n_iter,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1",
    n_jobs=-1,
    random_state=42,
    verbose=0,
    refit=True,
)
rand_rf.fit(X_train, y_train)
t_rand = time.time() - t0

print(f"\n⏱ RandomizedSearchCV waktu: {t_rand:.1f} detik")
print(f"🏆 Best params Random: {rand_rf.best_params_}")
print(f"   Best CV F1 Score    : {rand_rf.best_score_:.4f}")
print(f"   Test Set F1 (final) : {rand_rf.score(X_test, y_test):.4f}")
print(f"\n⚡ Perbandingan: {t_grid:.1f}s Grid vs {t_rand:.1f}s Random ({t_grid/t_rand:.1f}x lebih cepat!)")

print("\n>>> 5. HEATMAP HASIL TUNING + PERBANDINGAN SEBELUM vs SESUDAH TUNING")
print("-" * 60)

baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train)
models_compare = [
    ("Baseline (Default RF)", baseline),
    ("GridSearchCV Optimized", grid_rf.best_estimator_),
    ("RandomSearchCV Optimized", rand_rf.best_estimator_),
]

results_compare = []
for n_m, m in models_compare:
    yp = m.predict(X_test)
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
    results_compare.append({
        "Model": n_m,
        "Accuracy": accuracy_score(y_test, yp),
        "F1-Score": f1_score(y_test, yp),
        "ROC-AUC": roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]),
    })
df_res = pd.DataFrame(results_compare)
print(df_res.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

cv_results = pd.DataFrame(grid_rf.cv_results_)[["param_n_estimators", "param_max_depth", "mean_test_score"]]
cv_results_clean = cv_results[cv_results["param_max_depth"].isin([5, 10, 15, None])].copy()
cv_results_clean["param_max_depth"] = cv_results_clean["param_max_depth"].astype(str)
pivot = cv_results_clean.pivot_table(index="param_n_estimators", columns="param_max_depth", values="mean_test_score")
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis", ax=axes[0], linewidths=0.5)
axes[0].set_title("GridSearch: F1 Score vs n_estimators & max_depth", fontweight="bold")

x = np.arange(len(df_res))
w = 0.28
bars = []
for j, metric in enumerate(["Accuracy", "F1-Score", "ROC-AUC"]):
    bars.append(axes[1].bar(x + (j-1)*w, df_res[metric], w, label=metric))
axes[1].set_xticks(x)
axes[1].set_xticklabels([m[:20] for m in df_res["Model"]], rotation=15, ha="right", fontsize=8)
axes[1].set_title("SEBELUM vs SESUDAH HYPERPARAMETER TUNING", fontweight="bold")
axes[1].legend()
axes[1].set_ylim(0.6, 0.95)
for bs in bars:
    axes[1].bar_label(bs, fmt="%.3f", fontsize=7, padding=2)

plt.tight_layout()
plt.savefig(f"{output_dir}\\05_tuning_results_heatmap_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart tuning hasil tersimpan")

print("\n>>> 6. VALIDASI AKHIR - NESTED CROSS VALIDATION (JANGAN DATA LEAKAGE!)")
print("-" * 60)

print("""
⚠️  PENTING UNTUK BNSP:
    JANGAN lakukan Tuning + Evaluasi di DATA YANG SAMA!
    Ini menyebabkan OVERFITTING terhadap test set → hasil palsu!
    
    SOLUSI: NESTED CROSS VALIDATION (Double Loop CV)
    ┌────────────────────────────────────────────────────┐
    │ Outer Loop (CV)  → Evaluasi performa GENERALISASI │
    │   └── Inner Loop  → Grid/Random Hyperparameter Tun │
    └────────────────────────────────────────────────────┘
    Hasil Nested CV = estimasi performa YANG SEBENARNYA!
""")

outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

param_small = {"n_estimators": [100, 200], "max_depth": [None, 10]}

nested_scores = []
t0 = time.time()
for fold, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(X_pp, y), 1):
    X_tr_o, X_te_o = X_pp[train_outer_idx], X_pp[test_outer_idx]
    y_tr_o, y_te_o = y[train_outer_idx], y[test_outer_idx]

    grid_inner = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                              param_small, cv=inner_cv, scoring="f1", n_jobs=-1)
    grid_inner.fit(X_tr_o, y_tr_o)
    yp_o = grid_inner.predict(X_te_o)
    from sklearn.metrics import f1_score
    f1_o = f1_score(y_te_o, yp_o)
    nested_scores.append(f1_o)
    print(f"  Outer Fold {fold}: Best params={grid_inner.best_params_} → F1={f1_o:.4f}")

t_nested = time.time() - t0
print(f"\n⏱ Nested CV (3x3x2x2) waktu: {t_nested:.1f}s")
print(f"🏆 NESTED CV F1: {np.mean(nested_scores):.4f} ± {np.std(nested_scores):.4f}")
print("   ⚡ Ini adalah estimasi performa MODEL SEBENARNYA!")

print("""
┌─────────────────────────────────────────────────────────────────┐
│  RANGKUMAN TUNING BNSP:                                         │
├─────────────────────────────────────────────────────────────────┤
│  ✅ Gunakan STRATIFIED K-FOLD untuk CLASSIFICATION              │
│  ✅ RANDOM SEARCH > Grid Search jika banyak parameter            │
│  ✅ SCORING SESUAI KEBUTUHAN BISNIS (jangan selalu accuracy!)   │
│  ✅ NESTED CV untuk estimasi TRUE PERFORMANCE                   │
│  ✅ Simpan BEST PARAMS ke config file untuk reproducibility     │
│  ❌ JANGAN fit Scaler di FULL DATA sebelum split → LEAKAGE!     │
│     (PAKAI PIPELINE dengan CV = Aman)                           │
└─────────────────────────────────────────────────────────────────┘
""")

print("""\n>>> LATIHAN:
1. Bandingkan Grid vs Random Search dengan parameter space SAMA
   Ukur: waktu, best score, variasi hasil!
2. Coba Bayesian Optimization dengan library 'Optuna' (jika ada)
3. Untuk GradientBoostingClassifier: Tuning n_estimators, max_depth,
   learning_rate, subsample, colsample_bytree
4. Bandingkan hasil TUNED RF vs DEFAULT Gradient Boosting!
""")

print("\n✓ Bagian 2 Modul 5 Selesai: Cross-Val + Hyperparameter Tuning")
print("\n" + "="*60)
print("MODUL 5 SELESAI - Model Evaluation & Optimization")
print("="*60)
