import os, sys, io
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Import Library dengan Error Handling yang Jelas ---
try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    print("❌ ERROR: numpy / pandas belum terinstall!")
    print("   Install: python -m pip install numpy pandas")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    print("⚠️  WARNING: matplotlib tidak terinstall, skip save chart visualisasi.")
    plt = None

# Scikit-learn Imports (SEMUA yang dibutuhkan untuk Pipeline + 4 Model + Evaluasi)
try:
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        roc_auc_score, roc_curve
    )
except ImportError as e:
    print("=" * 70)
    print("❌ ERROR: scikit-learn BELUM TERINSTALL di virtual environment Anda!")
    print(f"   Detail error: {type(e).__name__}: {e}")
    print("")
    print("   CARA MEMPERBAIKI (jalankan di CMD dengan venv_bnsp aktif):")
    print("     cd /d c:\\ai-engineer")
    print("     venv_bnsp\\Scripts\\activate.bat")
    print("     python -m pip install scikit-learn==1.4.2 -U")
    print("=" * 70)
    sys.exit(1)

np.random.seed(42)

print("=" * 70)
print("MODUL 04: MACHINE LEARNING (Latihan Custom User)")
print("💡 Topik: Churn Prediction — Pelanggan Tetap vs Pindah (Binar Klasifikasi)")
print("   Pipeline + ColumnTransformer (Preprocessing Rapi & Reproducible)")
print("   Model: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting")
print("=" * 70)
print(f"✅ Output visualisasi disimpan ke: {OUTPUT_DIR}\n")

# ============================================================
# LANGKAH 1: GENERATE DATASET SINTETIS CHURN YANG REALISTIS (2000 DATA)
# Catatan: X dan y TIDAK ADA di file asli user → kita generate SESUAI daftar kolom user!
# num_cols = ['Umur','Gaji','Skor_Kredit','Jml_Pinjaman']
# cat_cols = ['JK','Pendidikan','Kota','Status_Nikah']
# ============================================================
N = 2000
print("=" * 70)
print("🔧 LANGKAH 1: Generate Dataset Simulasi Churn (n=2000)")
print("=" * 70)

data_raw = {
    # ---- NUMERIK ----
    "Umur":           np.random.normal(38, 10, N).clip(18, 70).round(0).astype(int),
    "Gaji":           np.random.normal(12_000_000, 4_000_000, N).clip(4_500_000, 30_000_000).round(-3),
    "Skor_Kredit":    np.random.normal(680, 75, N).clip(300, 850).round(0).astype(int),
    "Jml_Pinjaman":   np.random.choice([0, 1, 2, 3, 4, 5], N, p=[0.18, 0.30, 0.27, 0.15, 0.07, 0.03]),
    # ---- KATEGORIK ----
    "JK":             np.random.choice(["L", "P"], N, p=[0.52, 0.48]),
    "Pendidikan":     np.random.choice(["SMA", "D3", "S1", "S2", "S3"], N, p=[0.20, 0.18, 0.47, 0.12, 0.03]),
    "Kota":           np.random.choice(["Jakarta","Bandung","Surabaya","Medan","Makassar","Yogyakarta"], N),
    "Status_Nikah":   np.random.choice(["Belum","Kawin","Cerai"], N, p=[0.45, 0.48, 0.07]),
}

# Hitung probabilitas churn (formula logika bisnis agar dataset realistis):
def hitung_churn_prob(row):
    p = 0.10  # Base rate 10%
    # Faktor risiko NAIK p (pelanggan MAU PINDAH)
    if row["Skor_Kredit"] < 520:                         p += 0.28
    elif row["Skor_Kredit"] < 600:                       p += 0.14
    if row["Gaji"] < 7_000_000:                          p += 0.18
    if row["Jml_Pinjaman"] >= 3:                         p += row["Jml_Pinjaman"] * 0.05
    if row["Umur"] <= 25:                                p += 0.10
    if row["Pendidikan"] == "SMA":                       p += 0.08
    if row["Kota"] in ["Makassar", "Medan"]:             p += 0.05
    # Faktor pelindung TURUN p (pelanggan KECIL kemungkinan pindah)
    if row["Pendidikan"] == "S2" or row["Pendidikan"]=="S3":  p -= 0.10
    if row["Status_Nikah"] == "Kawin":                        p -= 0.05
    if row["Gaji"] >= 20_000_000:                             p -= 0.08
    return float(np.clip(p, 0.01, 0.97))

df = pd.DataFrame(data_raw)

# --- Tambahkan MISSING VALUES sengaja (uji fitur SimpleImputer user) ---
kolom_null = {
    "Gaji": int(N * 0.05),
    "Skor_Kredit": int(N * 0.07),
    "Umur": int(N * 0.02),
    "Pendidikan": int(N * 0.03),
    "Kota": int(N * 0.04),
}
for col, jml in kolom_null.items():
    idx_null = np.random.choice(N, jml, replace=False)
    df.loc[idx_null, col] = np.nan

# Generate target berdasarkan probabilitas (pindahkan ke list & loop aman)
probs = []
for i in range(N):
    row = df.iloc[i]
    # Isi sementara nan untuk hitung prob (tidak ubah dataframe asli)
    r_filled = {
        "Umur":        row["Umur"]        if pd.notna(row["Umur"])        else 38,
        "Gaji":        row["Gaji"]        if pd.notna(row["Gaji"])        else 12_000_000,
        "Skor_Kredit": row["Skor_Kredit"] if pd.notna(row["Skor_Kredit"]) else 680,
        "Jml_Pinjaman":row["Jml_Pinjaman"],
        "Pendidikan":  row["Pendidikan"]  if pd.notna(row["Pendidikan"])  else "S1",
        "Kota":        row["Kota"]        if pd.notna(row["Kota"])        else "Jakarta",
        "Status_Nikah":row["Status_Nikah"],
        "JK":          row["JK"],
    }
    probs.append(hitung_churn_prob(r_filled))
df["Churn"] = (np.random.rand(N) < np.array(probs)).astype(int)

print(f"   Shape dataset: {df.shape[0]} baris × {df.shape[1]} kolom")
print(f"   Missing values sengaja ditambahkan untuk uji SimpleImputer:")
for col, jml in kolom_null.items():
    print(f"     - {col:<15s}: {jml} baris null ({jml/N*100:.1f}%)")
print(f"   Target distribusi (Churn):")
print(f"     → 0 = Tetap  : {(df['Churn']==0).sum():>5d} ({(df['Churn']==0).mean():.1%})")
print(f"     → 1 = Pindah : {(df['Churn']==1).sum():>5d} ({(df['Churn']==1).mean():.1%})")
print()

# PISAHKAN X (Fitur) dan y (Target)
X = df.drop(columns=["Churn"])
y = df["Churn"].copy()
num_cols = ["Umur", "Gaji", "Skor_Kredit", "Jml_Pinjaman"]
cat_cols = ["JK", "Pendidikan", "Kota", "Status_Nikah"]

# ============================================================
# LANGKAH 2: PIPELINE PREPROCESSING USER (DIPERTAHANKAN 100%!)
# ============================================================
print("=" * 70)
print("🔧 LANGKAH 2: Preprocessing Pipeline & ColumnTransformer")
print("=" * 70)

# --- (100% KODE ASLI USER, TIDAK DIUBAH LOGIKA!) ---
# Numerik: Median Imputer → Standard Scaler
num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())])
# Kategorik: Mode Imputer → One-Hot (drop_first=True + sparse_output=False)
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"))])

preprocessor = ColumnTransformer([("num", num_pipe, num_cols),
                                  ("cat", cat_pipe, cat_cols)])

# TERAPKAN KE DATA SELURUH (nanti di split — untuk feature importance mapping nama kolom)
X_pp_all = preprocessor.fit_transform(X)
# Dapatkan NAMA FITUR SETELAH ONE-HOT ENCODING (untuk feature importance nanti)
feature_names = list(preprocessor.get_feature_names_out())
print(f"   ✅ Jumlah fitur sebelum preprocessing: {X.shape[1]} (4 numerik + 4 kategorik)")
print(f"   ✅ Jumlah fitur SETELAH One-Hot Encoding: {len(feature_names)}")
print(f"      → 4 Numerik (prefix num__)")
print(f"      → Sisa = One-Hot Encoded kategorik (jumlah kategori - 1 karena drop first)")
print()

# ============================================================
# LANGKAH 3: SPLIT DATA 80:20 (Stratify=y SESUAI SARAN USER!)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
# Terapkan preprocessor (fit HANYA dari training data! -> transform ke train+test)
X_train_pp = preprocessor.fit_transform(X_train)
X_test_pp  = preprocessor.transform(X_test)
# Update feature names (sudah fit ulang di train, nama sama):
feature_names = list(preprocessor.get_feature_names_out())

print(f"   ✅ Train Split: X_train = {X_train_pp.shape} | y_train = {y_train.shape}")
print(f"   ✅ Test  Split: X_test  = {X_test_pp.shape } | y_test  = {y_test.shape}")
print(f"   ✅ Stratify Berhasil: Churn Test={y_test.mean():.1%} (sama dengan train≈{y_train.mean():.1%})")
print()

# ============================================================
# LANGKAH 4: TRAINING 4 MODEL (100% MODEL MILIK USER DIPERTAHANKAN!)
# ============================================================
print("=" * 70)
print("🚀 LANGKAH 4: Training & Perbandingan 4 Model Classifier")
print("=" * 70)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree (d=5)": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest (100)": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
}

results = {}
model_metrics = []
best_name = None
best_acc  = -1.0

for name, model in models.items():
    print(f"   Training {name:<25s}…", end=" ", flush=True)
    try:
        model.fit(X_train_pp, y_train)
    except Exception as e:
        print(f"❌ ERROR TRAIN: {e}")
        continue
    acc_tr = accuracy_score(y_train, model.predict(X_train_pp))
    acc_te = accuracy_score(y_test,  model.predict(X_test_pp))
    try:
        auc_te = roc_auc_score(y_test, model.predict_proba(X_test_pp)[:, 1])
    except Exception:
        auc_te = float("nan")
    of_flag = "  ⚠️  OVERFIT >8%" if (acc_tr - acc_te) > 0.08 else ""
    print(f"Train={acc_tr:.2%} | Test={acc_te:.2%} | AUC={auc_te:.3f}{of_flag}")
    results[name] = model
    model_metrics.append({
        "Model": name,
        "Train_Acc": acc_tr,
        "Test_Acc": acc_te,
        "AUC": auc_te,
        "Overfit_Gap": (acc_tr - acc_te),
    })
    if acc_te > best_acc:
        best_acc = acc_te
        best_name = name

# Pilih MODEL TERBAIK SECARA DINAMIS (bukan hardcode RF seperti file asli!)
print(f"\n   🎉 Model TERBAIK berdasarkan Test Accuracy:")
print(f"      → {best_name} (Test Acc = {best_acc:.2%})")
best_model = results[best_name]
best_df = pd.DataFrame(model_metrics).sort_values("Test_Acc", ascending=False).reset_index(drop=True)
print("\n📊 Tabel Perbandingan 4 Model:")
print(best_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print()

# ============================================================
# LANGKAH 5: EVALUASI MODEL TERBAIK (Classification Report dll)
# ============================================================
print("=" * 70)
print(f"📊 LANGKAH 5: Evaluasi Detail Model Terbaik — {best_name}")
print("=" * 70)
y_pred_best = best_model.predict(X_test_pp)
try:
    y_prob_best = best_model.predict_proba(X_test_pp)[:, 1]
except Exception:
    y_prob_best = None

print("\n📋 Classification Report (kelas 0='Tetap', kelas 1='Pindah'):")
print(classification_report(y_test, y_pred_best, target_names=["Tetap (0)", "Pindah (1)"], digits=3))

cm = confusion_matrix(y_test, y_pred_best)
print(f"🔢 Confusion Matrix:")
print(f"                 Prediksi TETAP   Prediksi PINDAH")
print(f"   Aktual TETAP :        {cm[0][0]:>4d}              {cm[0][1]:>4d}")
print(f"   Aktual PINDAH:        {cm[1][0]:>4d}              {cm[1][1]:>4d}")
tn, fp, fn, tp = cm.ravel()
print(f"\n   🧮 Ringkasan bisnis:")
print(f"     → True  Positive (Pindah tertangkap benar) : {tp} orang")
print(f"     → False Negative (Pindah TIDAK tertangkap) : {fn} orang  ← Kerugian terbesar bisnis!")
print(f"     → Precision (Churn) : {tp/(tp+fp) if tp+fp>0 else float('nan'):.2%}")
print(f"     → Recall    (Churn) : {tp/(tp+fn) if tp+fn>0 else float('nan'):.2%} ← Yang paling penting di kasus churn!")
print()

# ============================================================
# LANGKAH 6: FEATURE IMPORTANCE (BENARAN dihitung, BUKAN hardcode!)
# File asli user line 66-70: importances dihitung tapi print di-HARDCODE.
# Kita HITUNG BENERAN + URUTKAN + SIMPAN CHART!
# ============================================================
print("=" * 70)
print("🎯 LANGKAH 6: Feature Importance (Fitur Paling Mempengaruhi Churn)")
print("=" * 70)

HAS_IMPORTANCE = hasattr(best_model, "feature_importances_")
if HAS_IMPORTANCE:
    importances = best_model.feature_importances_
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    print(f"   ✅ {best_name} mendukung feature_importances_. Berikut Top 10:")
    print(df_imp.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    top3 = df_imp.head(3)["Feature"].tolist()
    # ---- GANTI LINE 70 USER YANG HARDCODE! ----
    # Line 70 ASLI user: print("1. Gaji\n2. Skor_Kredit\n3. Umur") (HARDCODE palsu!)
    # KITA GANTI MENJADI DINAMIS SESUAI NILAI IMPORTANCE SESUNGGUHNYA:
    print("\n   ✅ TOP 3 FITUR PALING PENGARUH (Hasil Perhitungan OTOMATIS, BUKAN hardcode):")
    for i, (idx, row) in enumerate(df_imp.head(3).iterrows(), start=1):
        nama_clean = row["Feature"].replace("num__", "").replace("cat__", "")
        print(f"      {i}. {nama_clean:<25s} | Importance score: {row['Importance']*100:5.1f}%")
    # ---- VISUALISASI BAR CHART ----
    if plt is not None:
        plt.figure(figsize=(12, 8))
        df_plot = df_imp.head(10).iloc[::-1]  # reverse agar tertinggi di atas
        bars = plt.barh(df_plot["Feature"], df_plot["Importance"], color="#2563eb", alpha=0.85)
        for bar, val in zip(bars, df_plot["Importance"]):
            plt.text(bar.get_width() + 0.003, bar.get_y()+bar.get_height()/2,
                     f"{val*100:.1f}%", va="center", fontsize=10, color="#1e3a8a", fontweight="bold")
        plt.title(f"Top 10 Feature Importance — Best Model: {best_name}", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Skor Importance (semakin besar semakin berpengaruh)")
        plt.ylabel("Nama Fitur (hasil preprocessor ColumnTransformer)")
        plt.grid(True, axis="x", alpha=0.25)
        plt.tight_layout()
        fp_bar = os.path.join(OUTPUT_DIR, "M04_CUSTOM_01_top10_feature_importance.png")
        plt.savefig(fp_bar, dpi=150)
        plt.close()
        print(f"\n   📊 Chart Feature Importance disimpan: {fp_bar}")
else:
    print(f"   ℹ️  {best_name} tidak punya feature_importances_ (contoh: Logistic Regression).")
    if hasattr(best_model, "coef_"):
        coef = best_model.coef_[0]
        df_coef = pd.DataFrame({"Feature": feature_names, "Coef": coef})
        df_coef["Abs"] = df_coef["Coef"].abs()
        df_coef = df_coef.sort_values("Abs", ascending=False).reset_index(drop=True)
        print("   Kita pakai koefisien absolute (|coef|) dari model linear:")
        print(df_coef[["Feature","Coef"]].head(10).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
        print("\n   Top 3 fitur (dari |coef| tertinggi):")
        for i, (_, r) in enumerate(df_coef.head(3).iterrows(), 1):
            tanda = "PIC (risiko naik)" if r["Coef"] > 0 else "PROT (risiko turun)"
            print(f"      {i}. {r['Feature']:<25s} | coef={r['Coef']:+.3f} → {tanda}")

# ---- ROC CURVE (jika ada predict_proba) ----
if plt is not None and y_prob_best is not None:
    fpr, tpr, _ = roc_curve(y_test, y_prob_best)
    auc_val = roc_auc_score(y_test, y_prob_best)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color="#dc2626", linewidth=2.5, label=f"ROC Best Model (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="#64748b", linestyle="--", linewidth=1.5, label="Random Guess (AUC=0.500)")
    plt.title(f"ROC Curve — Best Model: {best_name}", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall/Sensitivity)")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    fp_roc = os.path.join(OUTPUT_DIR, "M04_CUSTOM_02_roc_curve_best_model.png")
    plt.savefig(fp_roc, dpi=150)
    plt.close()
    print(f"   📊 Chart ROC Curve disimpan: {fp_roc}")
print()

# ============================================================
# LANGKAH 7: PREDIKSI CUSTOMER BARU (INTERAKTIF!)
# Ini sesuai line 70 user "print 1.Gaji 2.Skor_Kredit 3.Umur" -> Kita upgrade jadi interaktif!
# ============================================================
print("=" * 70)
print("🎯 LANGKAH 7: PREDIKSI DATA CUSTOMER BARU (UJI MODEL SECARA MANUAL)")
print("=" * 70)
print("""
   Kita akan memprediksi apakah seorang customer BARU akan CHURN atau TETAP.
   Anda bisa pilih mode:
     A) 3 Contoh customer siap prediksi (AUTO TEST 3 skenario)
     B) Input manual via terminal (interaktif, pilih 3 fitur kunci sesuai line 70 file asli)
""")

def prediksi_customer(data_dict, model, preprocessor, label_model):
    df_new = pd.DataFrame([data_dict])
    # Pastikan urut kolom SAMA dengan training (penting untuk ColumnTransformer!)
    for c in num_cols + cat_cols:
        if c not in df_new.columns:
            df_new[c] = np.nan
    df_new = df_new[num_cols + cat_cols]
    X_new_pp = preprocessor.transform(df_new)
    pred = int(model.predict(X_new_pp)[0])
    prob = float(model.predict_proba(X_new_pp)[0, 1]) if hasattr(model, "predict_proba") else float("nan")
    label = "🔴 PINDAH (Churn=1) — Risiko pelanggan pergi!" if pred == 1 else "🟢 TETAP (Churn=0) — Pelanggan loyal"
    print(f"\n   ✅ Hasil Prediksi {label_model}:")
    print(f"      → Kelas prediksi : {label}")
    print(f"      → Probabilitas pindah (Churn=1): {prob*100:.1f}%")
    print(f"      → Nilai fitur input: {data_dict}")
    return pred, prob

print("   ▶️  Menjalankan AUTO TEST 3 SKENARIO (mode A):")
print("-" * 70)
# Skenario A1: Pelanggan BERISIKO TINGGI (seharusnya PINDAH)
skenario_risiko = {
    "Umur": 23, "Gaji": 5_200_000, "Skor_Kredit": 480, "Jml_Pinjaman": 4,
    "JK": "L", "Pendidikan": "SMA", "Kota": "Medan", "Status_Nikah": "Belum",
}
print("\n📘 Skenario 1: Pelanggan BERISIKO (Gaji <7juta, Skor_Kredit<520, Jml_Pinjaman=4, SMA):")
prediksi_customer(skenario_risiko, best_model, preprocessor, best_name)

# Skenario A2: Pelanggan AMAN (seharusnya TETAP)
skenario_aman = {
    "Umur": 42, "Gaji": 24_500_000, "Skor_Kredit": 790, "Jml_Pinjaman": 0,
    "JK": "P", "Pendidikan": "S2", "Kota": "Jakarta", "Status_Nikah": "Kawin",
}
print("\n📗 Skenario 2: Pelanggan LOYAL (Gaji >20juta, Skor_Kredit tinggi, S2 Kawin tanpa pinjaman):")
prediksi_customer(skenario_aman, best_model, preprocessor, best_name)

# Skenario A3: Pelanggan MODERAT (abu-abu)
skenario_mod = {
    "Umur": 34, "Gaji": 13_000_000, "Skor_Kredit": 650, "Jml_Pinjaman": 2,
    "JK": "L", "Pendidikan": "S1", "Kota": "Yogyakarta", "Status_Nikah": "Belum",
}
print("\n📙 Skenario 3: Pelanggan MODERAT (nilai rata-rata semua fitur):")
prediksi_customer(skenario_mod, best_model, preprocessor, best_name)
print()

# ---- MODE B: INPUT MANUAL (sesuai line 70 file asli user "1. Gaji 2. Skor_Kredit 3. Umur") ----
MODE_INPUT_MANUAL = True
if MODE_INPUT_MANUAL:
    try:
        print("-" * 70)
        print("   🖐️  MODE INPUT MANUAL (tekan Ctrl+C untuk skip / enter default):")
        print("   (Masukkan nilai untuk 3 fitur kunci sesuai line 70 file asli Anda):")
        def baca_input(prompt, default, cast=float):
            try:
                raw = input(f"   {prompt} (default={default}): ").strip()
                if raw == "": return default
                return cast(raw)
            except (EOFError, KeyboardInterrupt):
                print("\n   ↩️  Skip input manual (EOF/Ctrl+C).")
                raise
            except Exception as e:
                print(f"   ⚠️  Input invalid, pakai default. Error: {e}")
                return default
        try:
            inp_umur = baca_input("Masukkan Umur customer (18-70)", 35, int)
            inp_gaji = baca_input("Masukkan Gaji customer (rupiah)", 12_000_000, float)
            inp_skor = baca_input("Masukkan Skor_Kredit customer (300-850)", 680, int)
            inp_pinj = baca_input("Masukkan Jml_Pinjaman (0-5)", 1, int)
            manual = {
                "Umur": int(np.clip(inp_umur, 18, 70)),
                "Gaji": float(np.clip(inp_gaji, 4_500_000, 30_000_000)),
                "Skor_Kredit": int(np.clip(inp_skor, 300, 850)),
                "Jml_Pinjaman": int(np.clip(inp_pinj, 0, 5)),
                "JK": "L", "Pendidikan": "S1", "Kota": "Jakarta", "Status_Nikah": "Belum",
            }
            print(f"\n   ✅ Anda memasukkan: Umur={manual['Umur']}, Gaji={manual['Gaji']:,.0f}, Skor_Kredit={manual['Skor_Kredit']}, Pinjaman={manual['Jml_Pinjaman']}")
            print("   (Fitur lain set default: JK=L, Pendidikan=S1, Kota=Jakarta, Status=Belum)")
            prediksi_customer(manual, best_model, preprocessor, f"{best_name} (input manual)")
        except (EOFError, KeyboardInterrupt):
            pass
    except Exception as e:
        print(f"   ℹ️  Input manual di-skip (tidak interaktif di batch mode): {type(e).__name__}.")
        MODE_INPUT_MANUAL = False

# ============================================================
# LANGKAH 8: LATIHAN PRAKTIK MANDIRI BNSP (5 soal)
# ============================================================
print("\n" + "=" * 70)
print("📝 LANGKAH 8: LATIHAN PRAKTIK MANDIRI UNTUK PORTOFOLIO BNSP")
print("=" * 70)
LATIHAN = [
    "1. Ubah dataset sintetis: Tambahkan fitur 'Tenure_Months' & 'MonthlyCharges'. Hitung peningkatan Test Accuracy & AUC model Gradient Boosting.",
    "2. Tambahkan SMOTE dari imblearn untuk menangani data imbalance (jika churn <20%). Bandingkan Recall Churn sebelum vs sesudah SMOTE.",
    "3. Simpan model terbaik best_model & preprocessor menggunakan joblib/pickle ke file 'saved_models/churn_custom_model_v1.pkl'. Buat script predict baru yang memuat file tersebut.",
    "4. Ganti strategi imputasi numerik: 'median' → 'KNNImputer(n_neighbors=5)'. Bandingkan 4 model apakah Test Accuracy meningkat.",
    "5. Buat FastAPI 2 endpoint sederhana: POST /predict (ambil JSON 8 field input → return pred+prob), GET /model/info (nama model, best_acc, list fitur). Deploy uji coba di port 8200.",
]
for soal in LATIHAN:
    print(f"   ☑️  {soal}")
print()

print("=" * 70)
print("✅ SCRIPT SELESAI! (modul_04_machine_learning/churn-prediction.py)")
print("=" * 70)
print(f"""
   Ringkasan Output yang dihasilkan:
   • Dataset sintetis churn 2000 baris siap olah
   • 4 model classifier terlatih & terbandingkan (best model: {best_name})
   • Confusion Matrix + Recall / Precision + Classification Report
   • Feature Importance TOP 3-10 (DINAMIS, bukan hardcode!) + 1 chart PNG
   • ROC Curve + AUC score + 1 chart PNG
   • AUTO TEST 3 Skenario customer (Risiko / Aman / Moderat)
   • MODE INPUT MANUAL: Umur + Gaji + Skor_Kredit + Pinjaman (line 70 asli Anda!) → Hasil prediksi
   • 5 Latihan Praktik Mandiri BNSP untuk portofolio
""")
sys.exit(0)
