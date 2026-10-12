print("="*60)
print("MODUL 6: DEEP LEARNING")
print("Bagian 1: Artificial Neural Network (ANN) + MLP + Konsep Backprop")
print("="*60)

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, classification_report

output_dir = "c:\\ai-engineer\\modul_06_deep_learning\\output_charts"
os.makedirs(output_dir, exist_ok=True)

sns.set_style("whitegrid")
np.random.seed(42)

print("""
┌─────────────────────────────────────────────────────────────────┐
│   ARTIFICIAL NEURAL NETWORK (ANN) - DARI PERCEPTRON HINGGA MLP  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  NEURON BIOLOGIS       →       NEURON BUATAN (PERCEPTRON)       │
│  ──────────────────          ─────────────────────────────     │
│  Dendrit (input)       →      Features x1, x2, ..., xn          │
│  Bobot sinaptik        →      Weights w1, w2, ..., wn + bias    │
│  Soma (cell body)     →      Summation: z = Σ(wixi) + b        │
│  Axon hillock          →      ACTIVATION FUNCTION f(z)          │
│  Axon (output)         →      Output y_pred = f(z)             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ACTIVATION FUNCTIONS (WAJIB PAHAM DI BNSP!):           │   │
│  │                                                          │   │
│  │  SIGMOID  : σ(z) = 1/(1+e^-z)      → [0, 1]            │   │
│  │             Use Case: Output BINARY classification       │   │
│  │             Problem: Vanishing gradient di 0 & 1         │   │
│  │                                                          │   │
│  │  TANH     : tanh(z)                   → [-1, 1]            │   │
│  │             Use Case: Hidden layer, centred di 0          │   │
│  │             Problem: Masih ada vanishing gradien         │   │
│  │                                                          │   │
│  │  ReLU     : max(0, z)                 → [0, +∞)          │   │
│  │             Use Case: Hidden layer DEFAULT SEKARANG!     │   │
│  │             Kelebihan: Cepat, TIDAK vanishing (z>0)    │   │
│  │             Problem: Dead ReLU (z≤0 selalu → gradien 0) │   │
│  │                                                          │   │
│  │  Leaky ReLU: max(αz, z) (α=0.01)  → Perbaiki Dead ReLU │   │
│  │  Softmax : exp(zi)/Σexp(zj)       → Jumlah=1 (MULTICLASS)│   │
│  │  Linear  : z (identity)            → Output REGRESI      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  MULTILAYER PERCEPTRON (MLP) = DEEP LEARNING TIDAK TERLALU DALAM│
│                                                                 │
│  Input Layer → [HIDDEN LAYER 1] → [HIDDEN LAYER 2] → Output    │
│  (features)    (activation fn)     (activation fn)   (Softmax/ │
│                                                   Sigmoid/Linear│
│                                                                 │
│  TRAINING NEURAL NET = BACKPROPAGATION + OPTIMIZER              │
│  = Forward pass  → Hitung prediksi dari input → output          │
│  = Hitung LOSS    → Perbedaan prediksi vs ground truth          │
│  = Backward pass → Hitung gradien LOSS terhadap SETIAP WEIGHT   │
│                    (dari OUTPUT ke INPUT via CHAIN RULE!)       │
│  = Update WEIGHTS → w_new = w_old - lr * ∂Loss/∂w              │
│                                                                 │
│  LOSS FUNCTION:                                                 │
│  • Binary Cross-Entropy  → Klasifikasi BINARY                  │
│  • Categorical CE       → Klasifikasi MULTI-CLASS (one-hot)     │
│  • MSE / MAE            → Regression                           │
│                                                                 │
│  OPTIMIZER POPULER:                                             │
│  • SGD        → Stochastic Gradient Descent (klasik, butuh lama)│
│  • SGD+Momentum → + Momentum (akselerasi arah sama)           │
│  • Adam       → Adaptive Momentum (DEFAULT BAIK SEMUA KASUS!)   │
│  • RMSprop    → Adaptive lr per parameter                      │
└─────────────────────────────────────────────────────────────────┘
""")

input("TEKAN ENTER UNTUK MULAI PRAKTIK MLP...")

print("\n>>> 1. IMPLEMENTASI XOR GATE - BUKTI BUTUH HIDDEN LAYER!")
print("-" * 60)

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

print("XOR Truth Table:")
for i in range(4):
    print(f"  x1={X_xor[i,0]}, x2={X_xor[i,1]} → y={y_xor[i]}")

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=42).fit(X_xor, y_xor)
print(f"\n❌ Logistic Regression (LINEAR!) di XOR: Acc={accuracy_score(y_xor, lr.predict(X_xor))*100:.0f}%")
print("   TIDAK BISA memisahkan XOR (non-linear!) tanpa hidden layer")

mlp_simple = MLPClassifier(
    hidden_layer_sizes=(4,), activation="tanh",
    solver="lbfgs", max_iter=2000, random_state=42
).fit(X_xor, y_xor)
print(f"✅ MLP (hidden=4 neuron, tanh) di XOR: Acc={accuracy_score(y_xor, mlp_simple.predict(X_xor))*100:.0f}%")
print("   BISA memisahkan karena HIDDEN LAYER belajar fitur non-linear!")

print("\n>>> 2. VISUALISASI DECISION BOUNDARY BERDASARKAN AKTIVASI FN")
print("-" * 60)

from sklearn.datasets import make_moons, make_circles
X_m, y_m = make_moons(n_samples=300, noise=0.2, random_state=42)

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
fig.suptitle("PENGARUH HIDDEN LAYER & ACTIVATION FN terhadap Decision Boundary", fontsize=13, fontweight="bold")

konfigs = [
    ("(1) LogReg (Linear, no hidden)", LogisticRegression()),
    ("(2) MLP hidden=(3) tanh", MLPClassifier((3,), "tanh", solver="adam", max_iter=2000, random_state=42)),
    ("(3) MLP hidden=(10,10) ReLU", MLPClassifier((10,10), "relu", solver="adam", max_iter=2000, random_state=42)),
    ("(4) MLP hidden=(20,15,10) ReLU", MLPClassifier((20,15,10), "relu", solver="adam", max_iter=2000, random_state=42)),
]

for i, (title, model) in enumerate(konfigs):
    ax = axes[i//2][i%2]
    model.fit(X_m, y_m)
    acc = accuracy_score(y_m, model.predict(X_m))
    h = 0.02
    x_min, x_max = X_m[:, 0].min() - 0.5, X_m[:, 0].max() + 0.5
    y_min, y_max = X_m[:, 1].min() - 0.5, X_m[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.35, cmap=plt.cm.coolwarm)
    ax.scatter(X_m[:, 0], X_m[:, 1], c=y_m, cmap=plt.cm.coolwarm, s=40, edgecolors="k")
    ax.set_title(f"{title}\nAccuracy = {acc:.3f}", fontweight="bold", fontsize=10)
plt.tight_layout()
plt.savefig(f"{output_dir}\\01_mlp_activation_decision_boundary.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart Decision Boundary MLP tersimpan")
print("   → Semakin DALAM hidden layer, decision boundary SEMAKIN KOMPLEKS!")

print("\n>>> 3. PRAKTIK: MLP CLASSIFIER untuk Customer Churn (nyata!)")
print("-" * 60)

n = 2000
np.random.seed(42)
df = pd.DataFrame({
    "Tenure": np.random.randint(1, 72, n),
    "MonthlyCharges": np.random.uniform(30, 130, n).round(2),
    "SupportTickets": np.random.poisson(2, n).astype(int),
    "ProductCount": np.random.randint(1, 8, n),
    "Contract": np.random.choice(["Month-to-month", "1yr", "2yr"], n, p=[0.55, 0.25, 0.20]),
    "Internet": np.random.choice(["DSL", "Fiber", "No"], n, p=[0.3, 0.5, 0.2]),
    "Paperless": np.random.choice([0, 1], n),
    "Senior": np.random.choice([0, 1], n, p=[0.85, 0.15]),
})
df["TotalCharges"] = df["Tenure"] * df["MonthlyCharges"] + np.random.normal(0, 100, n)
kontrak_num = {"Month-to-month": 0, "1yr": 1, "2yr": 2}
peluang = (
    (1 - df["Contract"].map(kontrak_num)/2) * 0.35 +
    df["SupportTickets"].clip(0, 10)/10 * 0.25 +
    df["MonthlyCharges"]/df["MonthlyCharges"].max() * 0.20 +
    (df["Internet"] == "Fiber").astype(int) * 0.10
)
df["Churn"] = (np.random.rand(n) < peluang.clip(0.05, 0.95)).astype(int)

X = df.drop("Churn", axis=1)
y = df["Churn"].values
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"))])
pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
X_pp = pre.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_pp, y, test_size=0.25, random_state=42, stratify=y)

print(f"Shape Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Churn rate: Train={y_train.mean()*100:.1f}% | Test={y_test.mean()*100:.1f}%")

mlp_configs = [
    ("MLP Tiny: (16,) ReLU", {"hidden": (16,), "act": "relu", "alpha": 0.0001}),
    ("MLP Small: (32,16) ReLU", {"hidden": (32, 16), "act": "relu", "alpha": 0.0001}),
    ("MLP Medium: (64,32,16) ReLU", {"hidden": (64, 32, 16), "act": "relu", "alpha": 0.0001}),
    ("MLP Regularized: (32,16) + α=0.01", {"hidden": (32, 16), "act": "relu", "alpha": 0.01}),
    ("MLP Large: (128,64,32)", {"hidden": (128, 64, 32), "act": "relu", "alpha": 0.0001}),
]

hasil_mlp = []
training_curves = {}
for name, cfg in mlp_configs:
    t0 = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=cfg["hidden"],
        activation=cfg["act"],
        solver="adam",
        alpha=cfg["alpha"],
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=42,
        verbose=0,
    )
    mlp.fit(X_train, y_train)
    t = time.time() - t0
    ypt = mlp.predict(X_train)
    ype = mlp.predict(X_test)
    acc_tr, acc_te = accuracy_score(y_train, ypt), accuracy_score(y_test, ype)
    f1_te = f1_score(y_test, ype)
    hasil_mlp.append({"Model": name, "Train_Acc": acc_tr, "Test_Acc": acc_te,
                     "Test_F1": f1_te, "Waktu_s": t, "Epochs": mlp.n_iter_,
                     "Overfit_Gap": acc_tr - acc_te})
    training_curves[name] = {"loss": mlp.loss_curve_, "val_score": mlp.validation_scores_}
    print(f"  {name:<36} → Train={acc_tr:.4f} | Test={acc_te:.4f} | F1={f1_te:.4f} | Epochs={mlp.n_iter_} | {t:.1f}s")

print("\nPerbandingan MLP architectures:")
df_mlp = pd.DataFrame(hasil_mlp).set_index("Model")
print(df_mlp.round(4).to_string())

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for name, curves in training_curves.items():
    epochs = np.arange(1, len(curves["loss"]) + 1)
    axes[0].plot(epochs, curves["loss"], lw=2, label=f"{name[:22]}")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Training Loss (Cross-Entropy)")
axes[0].set_title("TRAINING LOSS CURVE per Arsitektur MLP", fontweight="bold")
axes[0].legend(fontsize=7, loc="upper right")
axes[0].grid(True, alpha=0.3)

x = np.arange(len(hasil_mlp))
w = 0.28
axes[1].bar(x - w, [h["Train_Acc"] for h in hasil_mlp], w, label="Train Accuracy", color="#3b82f6")
bars = axes[1].bar(x, [h["Test_Acc"] for h in hasil_mlp], w, label="Test Accuracy", color="#16a34a")
axes[1].bar(x + w, [h["Test_F1"] for h in hasil_mlp], w, label="Test F1", color="#f59e0b")
axes[1].set_xticks(x)
axes[1].set_xticklabels([h["Model"][:20] for h in hasil_mlp], rotation=35, ha="right", fontsize=7)
axes[1].set_title("PERBANDINGAN PERFORMA 5 ARSITEKTUR MLP", fontweight="bold")
axes[1].legend(fontsize=8)
axes[1].set_ylim(0.6, 1.0)
axes[1].bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

plt.tight_layout()
plt.savefig(f"{output_dir}\\02_mlp_architecture_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart perbandingan MLP tersimpan")

print("\n>>> 4. MLP REGRESSION - Prediksi Harga Rumah + LOSS MSE")
print("-" * 60)

n = 1200
np.random.seed(42)
df_r = pd.DataFrame({
    "LT": np.random.normal(150, 40, n).clip(60, 350),
    "LB": np.random.normal(100, 25, n).clip(40, 250),
    "KT": np.random.choice([2, 3, 4, 5, 6], n, p=[0.1, 0.45, 0.30, 0.12, 0.03]).astype(int),
    "KM": np.random.choice([1, 2, 3, 4], n, p=[0.25, 0.45, 0.22, 0.08]).astype(int),
    "Tahun": np.random.randint(1985, 2025, n),
    "Lokasi": np.random.choice(["Pusat", "Pinggir", "Suburb", "Desa"], n),
    "Garasi": np.random.choice([0, 1, 2, 3], n, p=[0.15, 0.35, 0.40, 0.10]).astype(int),
})
lokasi_prem = {"Pusat": 800, "Pinggir": 550, "Suburb": 400, "Desa": 250}
y_raw = df_r["LT"]*3.5 + df_r["LB"]*5 + df_r["KT"]*25 + df_r["KM"]*18 + df_r["Garasi"]*20 + (2025-df_r["Tahun"])*-2.5
harga_series = []
for i in range(n):
    harga = (y_raw[i] + lokasi_prem[df_r.loc[i, "Lokasi"]]) + np.random.normal(0, 30)
    harga_series.append(max(200, min(5000, harga)))
df_r["Harga"] = np.array(harga_series)

X_r = df_r.drop("Harga", axis=1)
y_r = df_r["Harga"].values
nr_cols = X_r.select_dtypes(include=[np.number]).columns.tolist()
cr_cols = X_r.select_dtypes(exclude=[np.number]).columns.tolist()
pre_r = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), nr_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"))]), cr_cols)
])
Xr_pp = pre_r.fit_transform(X_r)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr_pp, y_r, test_size=0.25, random_state=42)

mlp_reg = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    solver="adam",
    alpha=0.001,
    batch_size=32,
    learning_rate_init=0.001,
    max_iter=800,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=42,
)
t0 = time.time()
mlp_reg.fit(Xr_train, yr_train)
t_r = time.time() - t0
yr_pred = mlp_reg.predict(Xr_test)

print(f"MLP Regression (64,32,16) ReLU - selesai dalam {t_r:.1f}s, {mlp_reg.n_iter_} epochs")
print(f"  Train R²: {mlp_reg.score(Xr_train, yr_train):.4f} | Test R²: {r2_score(yr_test, yr_pred):.4f}")
print(f"  Test MAE: {mean_squared_error(yr_test, yr_pred, squared=False):.0f} juta | RMSE: {mean_squared_error(yr_test, yr_pred, squared=False):.0f} juta")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].plot(mlp_reg.loss_curve_, lw=2, color="#dc2626")
axes[0].set_title(f"MLP Regression Training Loss (MSE)\nn_iter={mlp_reg.n_iter_} epoch", fontweight="bold")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(yr_test, yr_pred, alpha=0.65, s=45, c="#3b82f6", edgecolors="white")
lims = [min(yr_test.min(), yr_pred.min()), max(yr_test.max(), yr_pred.max())]
axes[1].plot(lims, lims, "r--", lw=2, label="Ideal (y=x)")
axes[1].set_xlabel("Harga Aktual (juta Rp)")
axes[1].set_ylabel("Harga Prediksi MLP (juta Rp)")
axes[1].set_title(f"Actual vs Predicted: MLP Regression\nR²={r2_score(yr_test, yr_pred):.4f}", fontweight="bold")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{output_dir}\\03_mlp_regression_loss_actual_pred.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart MLP Regression tersimpan")

print("""
┌─────────────────────────────────────────────────────────────────┐
│       FINE-TUNING HYPERPARAMETER MLP (BNSP IMPORTANT!)         │
├─────────────────────────────────────────────────────────────────┤
│  ⚠ OVERFITTING di Neural Network → SOLUSI:                      │
│  1. Regularization L2 (alpha di MLPClassifier / weight decay    │
│  2. DROPOUT layer (matikan sebagian neuron saat training)       │
│  3. EARLY STOPPING (stop jika val loss membaik 10-20 iterasi)   │
│  4. Batch Normalization → Stabilisasi distribusi input per layer│
│  5. Kurangi ukuran arsitektur (hidden layer / neuron)           │
│  6. AUGMENTASI DATA (tambah data training)                       │
│                                                                 │
│  ⚠ UNDERFITTING di Neural Network → SOLUSI:                     │
│  1. Tambah depth/width (hidden layer & neuron)                  │
│  2. Kurangi regularization strength (alpha kecil)               │
│  3. Lebih banyak epoch training                                  │
│  4. Feature engineering & tambah fitur informatif               │
│  5. Learning rate terlalu besar? Kurangi lr_init                │
└─────────────────────────────────────────────────────────────────┘
""")

print("""\n>>> LATIHAN:
1. MLP dengan aktivasi LOGISTIC (sigmoid) vs ReLU: mana yang
   konvergen lebih cepat? Bandingkan n_iter_ dan loss curve!
2. Coba parameter solver='sgd' dengan learning_rate='adaptive'
   Bandingkan kecepatan & akurasi dengan Adam!
3. Batch size: coba 8, 32, 128, 256. Bagaimana pengaruhnya
   terhadap kestabilan loss curve & kecepatan training?
4. Tambah hidden layer sampai model OVERFIT, lalu gunakan
   alpha=0.1 dan early_stopping=True, apakah overfit hilang?
""")

print("\n✓ Bagian 1 Modul 6 Selesai: ANN/MLP Konsep + Praktik")
