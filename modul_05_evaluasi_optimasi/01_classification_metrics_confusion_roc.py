print("="*60)
print("MODUL 5: MODEL EVALUATION & OPTIMIZATION")
print("Bagian 1: Confusion Matrix, ROC-AUC, Classification Metric Deep-Dive")
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

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    classification_report, cohen_kappa_score, matthews_corrcoef
)

output_dir = "c:\\ai-engineer\\modul_05_evaluasi_optimasi\\output_charts"
os.makedirs(output_dir, exist_ok=True)

sns.set_style("whitegrid")
np.random.seed(42)

print("""
┌─────────────────────────────────────────────────────────────────┐
│       KLASIFIKASI METRIK EVALUASI (WAJIB PAHAM BNSP!)          │
├─────────────────────────────────────────────────────────────────┤
│  GAMBAR KONFUSI MATRIX (BINARY CLASSIFICATION):                │
│                                                                 │
│              ╔═════════════════╦═════════════════╗             │
│              ║  AKTUAL NEGATIF ║  AKTUAL POSITIF ║             │
│  ╔═══════════╬═════════════════╬═════════════════╣             │
│  ║ PREDIKSI  ║                 ║                 ║             │
│  ║ NEGATIF   ║  TRUE NEGATIVE  ║  FALSE NEGATIVE ║ ← TYPE II  │
│  ║ (TN=0,0)  ║  (TIDAK BERBAYAR║ (SAYA KIRA SEHAT║ ERROR!     │
│  ║           ║   & BENAR)      ║   TAPI SAKIT)   ║ PALING     │
│  ╠═══════════╬═════════════════╬═════════════════╣ BERBAHAYA!  │
│  ║ PREDIKSI  ║                 ║                 ║             │
│  ║ POSITIF   ║  FALSE POSITIVE ║  TRUE POSITIVE  ║             │
│  ║ (FP=1,0)  ║  (SAYA KIRA     ║  (SAYA KIRA     ║             │
│  ║  TYPE I   ║   SAKIT, TAPI   ║   SAKIT & BENAR)║             │
│  ║  ERROR    ║   SEHAT)        ║                 ║             │
│  ╚═══════════╩═════════════════╩═════════════════╝             │
│                                                                 │
│  RUMUS METRIK UTAMA:                                            │
│  1. Accuracy  = (TP+TN) / (TP+TN+FP+FN) → % benar TOTAL        │
│                  (⚠ TIDAK COCOK DATA IMBALANCED!)              │
│  2. Precision = TP / (TP+FP) → Sebagian yang diprediksi +      │
│                                 ternyata BENAR +               │
│  3. Recall    = TP / (TP+FN) → Sebagian BENAR +                │
│              (Sensitivity)    yang BERHASIL ketemu             │
│  4. F1-Score  = 2 * (P * R) / (P + R) → Harmonic Mean P & R   │
│  5. ROC-AUC   → Area Under Kurva ROC (0.5=acak, 1.0=sempurna)  │
│  6. Kappa     → Agreement corrected by chance (-1 s/d 1)       │
│  7. MCC       → Correlation (±1 = sempurna, 0 = acak)          │
└─────────────────────────────────────────────────────────────────┘
""")

print("\n>>> 1. PERSIAPKAN DATASET IMBALANCED (Simulasi Deteksi Penipuan)")
print("-" * 60)

n = 10000
np.random.seed(42)

df = pd.DataFrame({
    "Jumlah_Transaksi": np.concatenate([
        np.random.normal(500, 200, int(n*0.99)),
        np.random.normal(8000, 4000, int(n*0.01)),
    ]),
    "Jam_Transaksi": np.random.randint(0, 24, n).astype(float),
    "Jarak_Lokasi_km": np.concatenate([
        np.random.exponential(5, int(n*0.99)),
        np.random.exponential(50, int(n*0.01)),
    ]),
    "Jml_Beda_Kota_30Hari": np.concatenate([
        np.random.poisson(0.5, int(n*0.99)),
        np.random.poisson(3, int(n*0.01)),
    ]).astype(float),
    "Merchant_Risiko_Tinggi": np.concatenate([
        np.random.choice([0, 1], int(n*0.99), p=[0.9, 0.1]),
        np.random.choice([0, 1], int(n*0.01), p=[0.3, 0.7]),
    ]).astype(int),
})

df["Fraud"] = np.concatenate([np.zeros(int(n*0.99)), np.ones(int(n*0.01))]).astype(int)

sisa = n - len(df)
if sisa > 0:
    extra = pd.DataFrame({
        "Jumlah_Transaksi": np.random.normal(500, 200, sisa),
        "Jam_Transaksi": np.random.randint(0, 24, sisa).astype(float),
        "Jarak_Lokasi_km": np.random.exponential(5, sisa),
        "Jml_Beda_Kota_30Hari": np.random.poisson(0.5, sisa).astype(float),
        "Merchant_Risiko_Tinggi": np.random.choice([0, 1], sisa, p=[0.9, 0.1]).astype(int),
        "Fraud": np.zeros(sisa).astype(int),
    })
    df = pd.concat([df, extra], ignore_index=True)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Shape dataset: {df.shape}")
print(f"Distribusi Kelas (Fraud Detector - IMBALANCED!):")
fraud_dist = df["Fraud"].value_counts().sort_index()
for cls, cnt in fraud_dist.items():
    label = "FRAUD (1)" if cls == 1 else "SAH (0)"
    print(f"  Kelas {label}: {cnt:>5} transaksi ({cnt/len(df)*100:.2f}%)")
print(f"Imbalance Ratio: {max(fraud_dist) / min(fraud_dist):.0f} : 1 (SANGAT IMBALANCED!)")

X = df.drop("Fraud", axis=1)
y = df["Fraud"].values
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
preprocessor = ColumnTransformer([("num", num_pipe, num_cols)])
X_pp = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_pp, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\nTrain: {len(y_train)} (Fraud: {(y_train==1).sum()}) | Test: {len(y_test)} (Fraud: {(y_test==1).sum()})")

print("\n>>> 2. TRAINING + TUNJUKKAN LIMITASI ACCURACY DI DATA IMBALANCED")
print("-" * 60)

clf_models = {
    "Logistic Regression (imbalanced)": LogisticRegression(max_iter=3000, class_weight=None, random_state=42),
    "Logistic Regression (balanced class_weight)": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
    "Decision Tree (balanced)": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
    "Random Forest (balanced)": RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
}

eval_results = []
all_y_proba = {}
all_y_pred = {}

for name, model in clf_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    all_y_pred[name] = y_pred
    all_y_proba[name] = y_proba

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    kap = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")
    ap = average_precision_score(y_test, y_proba) if y_proba is not None else float("nan")
    dum = (y_test == 0).mean()

    eval_results.append({
        "Model": name,
        "Accuracy": acc,
        "Accuracy_Dummy_Mayoritas": dum,
        "Precision": prec,
        "Recall (Sensitivity)": rec,
        "F1-Score": f1,
        "ROC-AUC": roc,
        "PR-AUC (AvgPrec)": ap,
        "Cohen_Kappa": kap,
        "MCC": mcc,
    })

print("⚠ PERHATIKAN: Accuracy hampir sama dengan Dummy (acak, prediksi SEMUA 0)!")
print("↓↓↓  MAKA GUNAKAN F1, RECALL, ROC-AUC, KAPPA, MCC ↓↓↓\n")

cols_show = ["Model", "Accuracy", "Accuracy_Dummy_Mayoritas", "Precision", "Recall (Sensitivity)", "F1-Score", "ROC-AUC", "Cohen_Kappa", "MCC"]
df_eval = pd.DataFrame(eval_results).sort_values("MCC", ascending=False).set_index("Model")[cols_show[1:]]
df_eval.index.name = "Model"
pd.options.display.max_columns = 20
print(df_eval.round(4).to_string())

print("\n>>> 3. CONFUSION MATRIX VISUALISASI 2x2 (4 Model Terbaik)")
print("-" * 60)

top4 = sorted(eval_results, key=lambda x: x["MCC"], reverse=True)[:4]

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
for i, res in enumerate(top4):
    ax = axes[i//2][i%2]
    name = res["Model"]
    cm = confusion_matrix(y_test, all_y_pred[name])
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                xticklabels=["Pred Sah (0)", "Pred Fraud (1)"],
                yticklabels=["Aktual Sah (0)", "Aktual Fraud (1)"], cbar=False,
                linewidths=0.5, annot_kws={"fontsize": 14, "fontweight": "bold"})
    tn, fp, fn, tp = cm.ravel()
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec = tp/(tp+fn) if (tp+fn) > 0 else 0
    ax.set_title(f"{name[:30]}\nPrecision={prec:.3f}, Recall={rec:.3f}\nFP={fp:,} (Type I), FN={fn:,} (Type II!)", fontweight="bold", fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("True Label", fontsize=9)
fig.suptitle("Confusion Matrix 4 Model Terbaik - Deteksi Penipuan IMBALANCED DATA", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{output_dir}\\01_confusion_matrix_imbalanced.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart Confusion Matrix tersimpan")

print("\n>>> 4. ROC CURVE + AUC (Receiver Operating Characteristic)")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

colors = ["#3b82f6", "#16a34a", "#dc2626", "#7c3aed", "#f59e0b"]
for i, (name, y_prob) in enumerate(all_y_proba.items()):
    if y_prob is None:
        continue
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    lab = f"{name[:25]} (AUC={roc_auc:.4f})"
    axes[0].plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=lab)

axes[0].plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier (AUC=0.5000)")
axes[0].set_xlim([0.0, 1.0]); axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)")
axes[0].set_title("ROC CURVE - Lebih ke KIRI ATAS = LEBIH BAIK", fontweight="bold")
axes[0].legend(loc="lower right", fontsize=8)
axes[0].grid(True, alpha=0.3)

for i, (name, y_prob) in enumerate(all_y_proba.items()):
    if y_prob is None:
        continue
    prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    baseline = (y_test == 1).mean()
    lab = f"{name[:25]} (AP={ap:.4f})"
    axes[1].plot(rec_c, prec_c, color=colors[i % len(colors)], lw=2, label=lab)

axes[1].axhline(baseline, color="k", linestyle=":", lw=2, label=f"Baseline Random (AP={baseline:.4f})")
axes[1].set_xlim([0.0, 1.0]); axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel("Recall (TPR)")
axes[1].set_ylabel("Precision")
axes[1].set_title("PRECISION-RECALL CURVE (PENTING DATA IMBALANCED!)", fontweight="bold")
axes[1].legend(loc="lower left", fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}\\02_roc_auc_pr_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart ROC & PR Curve tersimpan")

print("\n>>> 5. TRADE-OFF PRECISION vs RECALL - THRESHOLD TUNING")
print("-" * 60)

best_name_for_tune = top4[0]["Model"]
y_prob_best = all_y_proba[best_name_for_tune]
print(f"Threshold tuning untuk model: {best_name_for_tune}")

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
tune_results = []
for t in thresholds:
    y_pred_t = (y_prob_best >= t).astype(int)
    tune_results.append({
        "Threshold": t,
        "TP": ((y_test == 1) & (y_pred_t == 1)).sum(),
        "FP": ((y_test == 0) & (y_pred_t == 1)).sum(),
        "FN": ((y_test == 1) & (y_pred_t == 0)).sum(),
        "TN": ((y_test == 0) & (y_pred_t == 0)).sum(),
        "Precision": precision_score(y_test, y_pred_t, zero_division=0),
        "Recall": recall_score(y_test, y_pred_t, zero_division=0),
        "F1": f1_score(y_test, y_pred_t, zero_division=0),
        "Accuracy": accuracy_score(y_test, y_pred_t),
    })

df_tune = pd.DataFrame(tune_results)
print(df_tune.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_tune["Threshold"], df_tune["Precision"], "o-", label="Precision", lw=2, color="#3b82f6", markersize=8)
ax.plot(df_tune["Threshold"], df_tune["Recall"], "s-", label="Recall", lw=2, color="#dc2626", markersize=8)
ax.plot(df_tune["Threshold"], df_tune["F1"], "D-", label="F1-Score", lw=2, color="#16a34a", markersize=8)
ax.axvline(0.5, color="gray", linestyle=":", label="Default 0.5")
ax.axvline(df_tune.iloc[df_tune["F1"].argmax()]["Threshold"],
           color="purple", linestyle="--", linewidth=2,
           label=f"Best F1 @ Threshold = {df_tune.iloc[df_tune['F1'].argmax()]['Threshold']}")
ax.set_xlabel("Classification Threshold (batas probabilitas)")
ax.set_ylabel("Score")
ax.set_title("PRECISION vs RECALL vs F1 vs THRESHOLD - Pilih sesuai kebutuhan bisnis!", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}\\03_threshold_tuning_precision_recall.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart Threshold Tuning tersimpan")

print("""
┌─────────────────────────────────────────────────────────────────┐
│    KESIMPULAN: PILIH METRIK BERDASARKAN KEBUTUHAN BISNIS        │
├─────────────────────────────────────────────────────────────────┤
│  KASUS FRAUD DETECTION → UTAMAKAN RECALL (FN harus minim!)      │
│    (Lebih baik salah blokir 100 transaksi daripada 1 penipuan  │
│     lolos)                                                      │
│  KASUS SPAM DETECTION → UTAMAKAN PRECISION (FP minim)          │
│    (Email pekerjaan tidak boleh masuk spam!                     │
│  KASUS SAKIT KANKER   → UTAMAKAN SENSITIVITY (RECALL)           │
│  KASUS CREDIT SCORING → UTAMAKAN ROC-AUC + F1                  │
│  BALANCED DATA       → Accuracy sudah OK                       │
│  IMBALANCED DATA      → F1 / MCC / KAPPA / ROC-AUC / PR-AUC     │
└─────────────────────────────────────────────────────────────────┘
""")

print("""\n>>> LATIHAN:
1. Untuk kasus Deteksi Peneyalahgunaan Narkoba (drug test),
   metrik mana yang UTAMA: Precision atau Recall? Jelaskan!
2. Buat klasifikasi MULTI-CLASS (3 kelas: Sah / Low Risk Fraud / High Risk Fraud)
   Hitung micro vs macro vs weighted F1-score!
3. Hitung manual Confusion Matrix dari prediksi model di atas
   dan bandingkan dengan sklearn.metrics.confusion_matrix!
""")

print("\n✓ Bagian 1 Modul 5 Selesai: Classification Metrics")
