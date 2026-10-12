print("="*60)
print("MODUL 4: MACHINE LEARNING ENGINEERING")
print("Bagian 1: Supervised Learning - CLASSIFICATION")
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
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

output_dir = "c:\\ai-engineer\\modul_04_machine_learning\\output_charts"
os.makedirs(output_dir, exist_ok=True)

sns.set_style("whitegrid")
np.random.seed(42)

print("""
┌─────────────────────────────────────────────────────────────────┐
│         SUPERVISED LEARNING: CLASSIFICATION                    │
├─────────────────────────────────────────────────────────────────┤
│  Tujuan = Prediksi LABEL KATEGORI (diskrit)                    │
│                                                                 │
│  ALGORITMA YANG UMUM DIPAKAI:                                   │
│  1. Logistic Regression  → Linear, Interpretabel, cepat         │
│  2. Naive Bayes          → Cepat, data kecil                    │
│  3. KNN (K-Nearest)     → Lazy, non-parametrik                 │
│  4. Decision Tree        → Interpretable, rawan overfit         │
│  5. SVM (RBF Kernel)    → High-dimensi, nonlinear               │
│  6. Random Forest       → Ensemble (bagging), tangguh           │
│  7. Gradient Boosting   → Ensemble boosting, akurasi tinggi     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")

print("\n>>> 1. PERSIAPAN DATASET - Customer Churn Prediction")
print("-" * 60)

n = 2000
np.random.seed(42)

df = pd.DataFrame({
    "Umur": np.random.randint(18, 65, n),
    "JK": np.random.choice(["L", "P"], n, p=[0.55, 0.45]),
    "Pendidikan": np.random.choice(["D3", "D4", "S1", "S2", "S3"], n, p=[0.12, 0.08, 0.55, 0.20, 0.05]),
    "Pengalaman": 0,
    "Gaji": 0,
    "Jml_Pinjaman": np.random.choice([0, 1, 2, 3, 4], n, p=[0.35, 0.35, 0.18, 0.08, 0.04]),
    "Skor_Kredit": np.random.normal(700, 80, n).clip(400, 900),
    "Kota": np.random.choice(["Jakarta", "Bandung", "Surabaya", "Yogyakarta", "Semarang"], n),
    "Status_Nikah": np.random.choice(["Belum", "Kawin", "Cerai"], n, p=[0.45, 0.45, 0.10]),
})

df["Pengalaman"] = (df["Umur"] - 22 + np.random.normal(0, 3, n)).clip(0, 40).astype(int)
gaji_base = {"S3": 25, "S2": 15, "S1": 9, "D4": 7.5, "D3": 6.5}
for p, b in gaji_base.items():
    mask = df["Pendidikan"] == p
    df.loc[mask, "Gaji"] = b + df.loc[mask, "Pengalaman"] * 0.7 + np.random.normal(0, 1.5, mask.sum())
    df.loc[mask, "Gaji"] = df.loc[mask, "Gaji"].clip(4, 40) * 1_000_000

gn = (df["Gaji"] - df["Gaji"].min()) / (df["Gaji"].max() - df["Gaji"].min())
kn = (df["Skor_Kredit"] - df["Skor_Kredit"].min()) / (df["Skor_Kredit"].max() - df["Skor_Kredit"].min())
un = (df["Umur"] - df["Umur"].min()) / (df["Umur"].max() - df["Umur"].min())
peluang = 0.15 + (1 - gn) * 0.35 + (1 - kn) * 0.30 + df["Jml_Pinjaman"] * 0.05 + un * 0.05
df["Churn"] = (np.random.rand(n) < peluang.clip(0.05, 0.9)).astype(int)

print(f"Shape dataset: {df.shape}")
print(f"Distribusi Churn:")
print(df["Churn"].value_counts().rename({0: "Tetap", 1: "Pindah"}))
print(f"Churn rate: {df['Churn'].mean() * 100:.1f}%")

print("\n>>> 2. PREPROCESSING PIPELINE + DATA SPLITTING")
print("-" * 60)

X = df.drop("Churn", axis=1)
y = df["Churn"].values

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"Fitur numerik ({len(num_cols)}): {num_cols}")
print(f"Fitur kategori ({len(cat_cols)}): {cat_cols}")

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"))])
preprocessor = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

X_pp = preprocessor.fit_transform(X)
feature_names = num_cols + preprocessor.named_transformers_["cat"].named_steps["oh"].get_feature_names_out(cat_cols).tolist()
print(f"Total fitur setelah preprocess: {X_pp.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(X_pp, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(y_train)}, Test: {len(y_test)}")
print(f"Train Churn rate: {y_train.mean()*100:.1f}% | Test: {y_test.mean()*100:.1f}%")

print("\n>>> 3. TRAINING 7 ALGORITMA KLASIFIKASI")
print("-" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Naive Bayes": GaussianNB(),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree (d=5)": DecisionTreeClassifier(max_depth=5, random_state=42),
    "SVM RBF": SVC(kernel="rbf", probability=True, random_state=42),
    "Random Forest (100)": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    acc_tr = accuracy_score(y_train, y_pred_train)
    acc_te = accuracy_score(y_test, y_pred_test)
    results[name] = {"Train_acc": acc_tr, "Test_acc": acc_te,
                   "y_pred": y_pred_test, "model": model}
    overfit_mark = "  <-- OVERFIT!" if (acc_tr - acc_te) > 0.08 else ""
    print(f"  {name:<28} → Train: {acc_tr:.4f} | Test: {acc_te:.4f}{overfit_mark}")

print("\n>>> 4. PERBANDINGAN MODEL (Model Selection)")
print("-" * 60)

df_compare = pd.DataFrame(results).T[["Train_acc", "Test_acc"]].astype(float).round(4)
df_compare["Accuracy_Selisih"] = (df_compare["Train_acc"] - df_compare["Test_acc"]).round(4)
print(df_compare.sort_values("Test_acc", ascending=False))

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(df_compare))
w = 0.35
bars1 = ax.bar(x - w/2, df_compare["Train_acc"], w, label="Train Accuracy", color="#3b82f6")
bars2 = ax.bar(x + w/2, df_compare["Test_acc"], w, label="Test Accuracy", color="#ef4444")
ax.set_xticks(x)
ax.set_xticklabels(df_compare.index, rotation=30, ha="right")
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("Accuracy")
ax.set_title("Perbandingan 7 Algoritma Klasifikasi - Churn Prediction", fontsize=13, fontweight="bold")
ax.legend()
ax.bar_label(bars1, fmt="%.3f", label_type="edge", padding=3, fontsize=8)
ax.bar_label(bars2, fmt="%.3f", label_type="edge", padding=3, fontsize=8)
plt.tight_layout()
plt.savefig(f"{output_dir}\\01_comparison_classifiers.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart perbandingan tersimpan")

print("\n>>> 5. ANALISIS MODEL TERBAIK + CLASSIFICATION REPORT")
print("-" * 60)

best_name = max(results.items(), key=lambda kv: kv[1]["Test_acc"])[0]
best_model = results[best_name]["model"]
y_pred_best = results[best_name]["y_pred"]
print(f"🏆 Model TERBAIK berdasarkan Test Accuracy: {best_name}")

print(f"\nClassification Report (Test Set):\n")
print(classification_report(y_test, y_pred_best, target_names=["Tetap (0)", "Pindah (1)"], digits=4))

print("\n>>> 6. FEATURE IMPORTANCE")
print("-" * 60)

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
else:
    rf_model = results["Random Forest (100)"]["model"]
    importances = rf_model.feature_importances_
    best_model_for_fi = rf_model
    best_name_fi = "Random Forest"

fi = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)
print("Top 10 Fitur Terpenting:")
print(fi.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
top_fi = fi.head(12)
sns.barplot(data=top_fi, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False, ax=ax)
ax.set_title(f"Feature Importance - {best_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{output_dir}\\02_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart feature importance tersimpan")

print("\n>>> 7. DECISION BOUNDARY VISUALISASI (PCA 2D)")
print("-" * 60)

from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_pp)
print(f"Explained variance (2 komponen): {pca.explained_variance_ratio_.sum()*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], c="#22c55e", alpha=0.5, s=20, label="Tetap")
axes[0].scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], c="#ef4444", alpha=0.5, s=20, label="Churn")
axes[0].set_title("Distribusi Data (PCA 2D)")
axes[0].legend()

Xtr_2d, Xte_2d, ytr, yte = train_test_split(X_2d, y, test_size=0.2, random_state=42, stratify=y)
viz_model = RandomForestClassifier(n_estimators=50, random_state=42).fit(Xtr_2d, ytr)
xx, yy = np.meshgrid(np.linspace(X_2d[:, 0].min()-1, X_2d[:, 0].max()+1, 150),
                     np.linspace(X_2d[:, 1].min()-1, X_2d[:, 1].max()+1, 150))
Z = viz_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[1].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlGn)
axes[1].scatter(Xte_2d[:, 0], Xte_2d[:, 1], c=yte, cmap=plt.cm.RdYlGn_r, s=25, edgecolors="k")
axes[1].set_title("Decision Boundary - Random Forest (PCA Space)")

plt.tight_layout()
plt.savefig(f"{output_dir}\\03_pca_decision_boundary.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart PCA Decision Boundary tersimpan")

print("""\n>>> LATIHAN:
1. Ganti n_neighbors KNN dari 3 sampai 50, amati akurasi
2. Tambahkan PolynomialFeatures derajat=2 ke pipeline
3. SVM kernel poly/sigmoid, bandingkan dengan rbf
4. Simpan model terbaik dengan joblib/pickle
""")

print("\n✓ Bagian 1 Modul 4 Selesai: CLASSIFICATION")
