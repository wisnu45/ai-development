# =====================================================================
# MODUL 10 - AI MODEL DEPLOYMENT
# Persiapan Uji Kompetensi BNSP Skema Artificial Intelligence Engineer
# =====================================================================
# Materi:
# 1. Konsep AI Deployment & Model Serialization
# 2. REST API untuk Model AI
# 3. Membuat AI Service menggunakan FastAPI + Pydantic
# 4. Integrasi model dengan Frontend / Aplikasi
# 5. Containerization menggunakan Docker (Dockerfile & Docker Compose)
# 6. Environment Management (prod/dev/staging, .env)
# 7. Model Serving & Monitoring Model (Drift Detection)
# 8. Dasar MLOps: Versioning model & Dataset, CI/CD ML
# =====================================================================

import os
import re
import sys
import json
import time
import pickle
import warnings
import datetime as dt
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------
# INSTALASI DEPENDENSI (jika error ModuleNotFoundError, jalankan ini):
#   cd c:\ai-engineer ; venv_bnsp\Scripts\activate ; pip install -r requirements.txt
# ---------------------------------------------------------------------
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                                 mean_absolute_error, mean_squared_error, r2_score)
    # Opsional Modul 10
    try:
        import joblib
        JOBLIB_AVAIL = True
    except ImportError:
        JOBLIB_AVAIL = False
    try:
        from pydantic import BaseModel, Field, field_validator
        PYDANTIC_AVAIL = True
    except ImportError:
        PYDANTIC_AVAIL = False
except ImportError as e:
    print("=" * 80)
    print("⚠️  MODUL 10 GAGAL BERJALAN - DEPENDENSI TIDAK DITEMUKAN")
    print("=" * 80)
    print(f"Penyebab: {e}")
    print("\nSolusi (CMD):")
    print("  cd /d C:\\ai-engineer")
    print("  venv_bnsp\\Scripts\\activate")
    print("  pip install -r requirements.txt")
    print("=" * 80)
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

sns.set_style("whitegrid")
np.random.seed(42)

# ============================================================
# BAGIAN 1 - KONSEP AI DEPLOYMENT & MODEL SERIALIZATION
# ============================================================
print("=" * 80)
print("BAGIAN 1: KONSEP AI DEPLOYMENT & MODEL SERIALIZATION")
print("=" * 80)

print("""
----------------------------------------------------------------------
1.1 APA ITU MODEL DEPLOYMENT?
----------------------------------------------------------------------
Deployment = Proses MEMASUKKAN / MENGINTEGRASIKAN model ML/DL yang SUDAH
DILATIH (trained) ke dalam LINGKUNGAN PRODUKSI sehingga bisa DIGUNAKAN
oleh pengguna akhir (end-user) / aplikasi lain.

Posisi Deployment di AI Project Lifecycle:
  [1] Problem Definition
  [2] Data Collection + Prep
  [3] Modelling & Tuning        <- SELESAI Modul 4-6
  [4] EVALUASI                  <- SELESAI Modul 5
  [5] ★ DEPLOYMENT ★           <- MODUL 10 INI!
  [6] MONITORING & MAINTENANCE  <- ½ bagian modul ini

MODE DEPLOYMENT UMUM (BNSP):
  ┌────────────────────────────────────────────────────────────────┐
  │ • BATCH PROCESSING  : Model run SEKALI per hari/jam (laporan)   │
  │ • REST API / MICROSERVICE : Endpoint HTTP, real-time prediksi   │
  │ • STREAMING        : Model menganalisa data real-time (Kafka)   │
  │ • EMBEDDED / EDGE  : Model diinstal di device (ponsel/IoT)      │
  │ • CLOUD FUNCTIONS  : Serverless (AWS Lambda / GCP Cloud Function│
  └────────────────────────────────────────────────────────────────┘
""")

print("""
----------------------------------------------------------------------
1.2 MODEL SERIALIZATION (PENYIMPANAN MODEL)
----------------------------------------------------------------------
Serialization = Proses MENGUBAH object Python (model pipeline scikit-learn)
menjadi BYTE STREAM yang bisa DISIMPAN ke file di harddisk, lalu
DESERIALIZATION = memuat file byte kembali jadi object Python siap pakai.

TOOLS STANDAR BNSP:
  1. pickle (bawaan Python, PKL format *.pkl)
     ✅ Universal, semua object Python
     ❌ RENTAN terhadap kode berbahaya (jangan buka file pkl tidak terpercaya!)
  2. joblib (scikit-learn / numpy optimized, format *.joblib)
     ✅ Jauh LEBIH CEPAT & LEBIH KECIL untuk model numpy besar (RandomForest dll)
     ✅ DE FACTO STANDARD untuk sklearn pipeline
  3. ONNX (Open Neural Network Exchange) - lintas framework (TF/PyTorch/sklearn → ONNX)
  4. SavedModel / .h5 → untuk TensorFlow/Keras khusus deep learning
""")

# --- Praktik: TRAIN 2 MODEL (klasifikasi + regresi) lalu simpan ---
print("\n>>> PRAKTIK 1a: Training Model Klasifikasi CHURN (akan disimpan sebagai PKL/Joblib)")

# 1) Generate dataset churn sintetis (sesuai Modul 4)
n = 2000
np.random.seed(42)
data_churn = pd.DataFrame({
    'tenure_months': np.random.randint(1, 73, n).astype(float),
    'monthly_charges_usd': np.round(np.random.uniform(29.9, 129.9, n), 2),
    'total_charges_usd': np.nan,
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n,
                                      p=[0.55, 0.25, 0.20]),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n,
                                         p=[0.35, 0.50, 0.15]),
    'num_support_tickets': np.random.poisson(2.2, n).astype(float),
})
data_churn['total_charges_usd'] = (data_churn['tenure_months'] * data_churn['monthly_charges_usd']
                                    * np.random.uniform(0.93, 1.07, n)).round(2)
# Logika churn (kontrak bulanan + tenure pendek + support banyak → tinggi churn)
churn_prob = (
    0.55 * (data_churn['contract_type'] == 'Month-to-month').astype(int)
    + 0.28 * (data_churn['num_support_tickets'] >= 4).astype(int)
    + 0.25 * (data_churn['internet_service'] == 'Fiber optic').astype(int)
    - 0.006 * data_churn['tenure_months']
    + np.random.normal(0, 0.12, n)
)
data_churn['churn_label'] = (churn_prob > churn_prob.quantile(0.72)).astype(int)
print(f"   Dataset: {len(data_churn)} baris, Churn rate = "
      f"{data_churn['churn_label'].mean()*100:.1f}%")

# 2) Split + buat Pipeline (Preprocessor + Model)
num_cols = ['tenure_months', 'monthly_charges_usd', 'total_charges_usd', 'num_support_tickets']
cat_cols = ['contract_type', 'internet_service']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ],
    remainder='drop'
)
pipeline_churn = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=150, max_depth=8,
                                   class_weight='balanced_subsample',
                                   random_state=42, n_jobs=-1)),
])

Xc = data_churn.drop('churn_label', axis=1)
yc = data_churn['churn_label']
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.25,
                                                         stratify=yc, random_state=42)
pipeline_churn.fit(Xc_train, yc_train)
yc_pred = pipeline_churn.predict(Xc_test)
yc_proba = pipeline_churn.predict_proba(Xc_test)[:, 1]
acc_c = accuracy_score(yc_test, yc_pred)
auc_c = roc_auc_score(yc_test, yc_proba)
print(f"   ✅ Training selesai! Akurasi={acc_c*100:.2f}%, ROC-AUC={auc_c:.4f}")

# 3) SERIALIZATION dengan PICKLE (bawaan Python)
path_pkl = os.path.join(MODELS_DIR, "churn_rf_model_v1.pkl")
with open(path_pkl, 'wb') as f:
    pickle.dump({
        'pipeline': pipeline_churn,
        'metadata': {
            'model_name': 'RandomForest Churn Classifier v1.0',
            'author': 'AI Engineer BNSP',
            'created_at': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'train_date': '2026-09-12',
            'metrics': {'accuracy_test': round(acc_c, 4), 'roc_auc_test': round(auc_c, 4)},
            'features_expected': num_cols + cat_cols,
            'version': '1.0.0',
        }
    }, f, protocol=pickle.HIGHEST_PROTOCOL)
size_pkl_kb = os.path.getsize(path_pkl) / 1024
print(f"   📦 Disimpan (Pickle .pkl): {path_pkl}  [{size_pkl_kb:.1f} KB]")

# 4) SERIALIZATION dengan JOBLIB (jika tersedia - lebih baik untuk sklearn!)
if JOBLIB_AVAIL:
    path_joblib = os.path.join(MODELS_DIR, "churn_rf_model_v1.joblib")
    joblib.dump({
        'pipeline': pipeline_churn,
        'metadata': {
            'model_name': 'RandomForest Churn Classifier v1.0',
            'created_at': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metrics': {'accuracy_test': round(acc_c, 4), 'roc_auc_test': round(auc_c, 4)},
            'features_expected': num_cols + cat_cols,
            'version': '1.0.0',
        }
    }, path_joblib, compress=3)
    size_joblib_kb = os.path.getsize(path_joblib) / 1024
    print(f"   📦 Disimpan (Joblib .joblib): {path_joblib}  [{size_joblib_kb:.1f} KB]")
else:
    path_joblib = None
    print("   ⚠️  Library joblib tidak tersedia; hanya .pkl yang dibuat. "
          "(pip install joblib)")

# 5) DESERIALIZATION LOAD BACK (verifikasi load berhasil)
print("\n>>> PRAKTIK 1b: Deserialize / Load Model KEMBALI + Test Inferensi Satu Baris")
with open(path_pkl, 'rb') as f:
    loaded_pkl = pickle.load(f)
loaded_pipeline = loaded_pkl['pipeline']
loaded_meta = loaded_pkl['metadata']

# Test 1 customer baru (inferensi)
customer_baru = pd.DataFrame([{
    'tenure_months': 2,
    'monthly_charges_usd': 110.5,
    'total_charges_usd': 225.0,
    'contract_type': 'Month-to-month',
    'internet_service': 'Fiber optic',
    'num_support_tickets': 5,
}])
prob_churn = loaded_pipeline.predict_proba(customer_baru)[0, 1]
label_churn = loaded_pipeline.predict(customer_baru)[0]
print(f"   Customer baru: tenure={customer_baru.iloc[0]['tenure_months']} bulan, "
      f"kontrak={customer_baru.iloc[0]['contract_type']}")
print(f"   Prediksi: Churn = {bool(label_churn)}, "
      f"Probabilitas Churn = {prob_churn*100:.1f}%")
print(f"   Metadata model loaded: v{loaded_meta['version']} - "
      f"Acc Test={loaded_meta['metrics']['accuracy_test']*100:.1f}%")

# Visualisasi: Perbandingan ukuran file Pickle vs Joblib + Pie deployment mode
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
labels_size = ['Pickle (.pkl)', 'Joblib (.joblib)']
sizes = [size_pkl_kb]
if path_joblib:
    sizes.append(size_joblib_kb)
colors_bar = ['#E45756', '#4C78A8'][:len(sizes)]
bars = ax1.bar(labels_size[:len(sizes)], sizes, color=colors_bar, edgecolor='white', linewidth=2)
ax1.set_ylabel('Ukuran File (KB)', fontsize=11)
ax1.set_title('Perbandingan Ukuran File Hasil Serialization', fontsize=12, fontweight='bold')
for bar, val in zip(bars, sizes):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f} KB', ha='center', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

modes = ['Batch\n(Laporan Harian)', 'REST API\n(Microservice)', 'Streaming\n(Kafka/Event)',
         'Edge Device\n(IoT/Ponsel)', 'Serverless\n(Cloud Fn)']
freqs = [28, 42, 15, 9, 6]
colors_pie = sns.color_palette('Spectral', 5)
wedges, texts, autotexts = ax2.pie(freqs, labels=modes, colors=colors_pie,
                                    autopct='%1.0f%%', startangle=90,
                                    wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax2.set_title('Frekuensi Pola Deployment Digunakan di Industri', fontsize=12, fontweight='bold')
plt.setp(autotexts, size=10, weight='bold', color='white')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_serialization_size_deployment_mode.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Visualisasi 1 disimpan: 01_serialization_size_deployment_mode.png")

# ============================================================
# BAGIAN 2 & 3 - REST API UNTUK MODEL AI DENGAN FASTAPI
# ============================================================
print("\n" + "=" * 80)
print("BAGIAN 2 & 3: REST API MODEL AI DENGAN FASTAPI + PYDANTIC")
print("=" * 80)

print("""
----------------------------------------------------------------------
2.1 REST API = JEMBATAN ANTARA MODEL DENGAN DUNIA LUAR
----------------------------------------------------------------------
REST API = arsitektur komunikasi via protokol HTTP (GET/POST/PUT/DELETE)
menggunakan data format JSON (Javascript Object Notation) → STANDAR DE FACTO
integrasi aplikasi modern lintas bahasa (Python ↔ Javascript ↔ Java ↔ Go).

Endpoint REST API standar untuk AI Model Service:
  ┌───────────────────────────────────────────────────────────┐
  │ GET    /                  → Health check (service OK?)   │
  │ GET    /health            → Heartbeat endpoint           │
  │ GET    /model/info        → Metadata model (versi, metric)│
  │ POST   /predict           → Prediksi 1 data (satuan)      │
  │ POST   /predict/batch     → Prediksi BATCH (N data)       │
  │ POST   /predict/explain   → (SHAP/LIME) penjelasan fitur  │
  │ GET    /docs              → SWAGGER UI / Dokumentasi auto │
  │ GET    /openapi.json      → Spesifikasi OpenAPI 3.0       │
  └───────────────────────────────────────────────────────────┘

3.1 FASTAPI = Modern, High-Performance Web Framework Python untuk API.
    KEUNGGULAN FASTAPI (WAJIB TAHU BNSP):
      ✅ Otomatis generate DOKUMENTASI SWAGGER UI di /docs
      ✅ VALIDASI INPUT OTOMATIS via Pydantic BaseModel (tipe data error = 422)
      ✅ Berdasarkan standar: OpenAPI (Swagger) + JSON Schema
      ✅ Async Support (async/await = bisa handle concurrent request tinggi)
      ✅ Cepat: Setara Node.js / Go (bawanya Starlette + Uvicorn ASGI)
      ✅ Dependency Injection system (mudah untuk auth, DB connection)
""")

# --- Cek library FastAPI/Uvicorn tersedia atau tidak ---
try:
    import fastapi
    import uvicorn
    FASTAPI_AVAIL = True
    print(f"✅ FastAPI + Uvicorn TERINSTALL (FastAPI v{fastapi.__version__}). "
          "Anda bisa jalankan server!")
except ImportError:
    FASTAPI_AVAIL = False
    print("⚠️  FastAPI/Uvicorn BELUM terinstall (opsional Modul 10).")
    print("   Install dengan: pip install fastapi==0.115.0 uvicorn==0.30.6 pydantic==2.9.1")
    print("   Menampilkan KODE TEMPLATE lengkap (copy-paste ke file app.py untuk coba).")

# --- TEMPLATE KODE FASTAPI LENGKAP (bisa di-run) ---
FASTAPI_APP_TEMPLATE = '''
# =====================================================================
# FILE: app.py - FastAPI Service untuk Model Churn Prediction BNSP
# Cara menjalankan: (venv_bnsp) C:\\ai-engineer\\modul_10_deployment>
#                   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# Lalu buka: http://localhost:8000/docs  (Swagger UI)
# =====================================================================
import os
import pickle
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import pandas as pd

# ---------------------------------------------------------------------
# [1] LIFESPAN: Load model SEKALI SAAT STARTUP (bukan tiap request!)
# ---------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
MODEL_PATH = os.path.join(MODELS_DIR, "churn_rf_model_v1.pkl")
_ml_assets = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model & metadata
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    _ml_assets["pipeline"] = data["pipeline"]
    _ml_assets["metadata"] = data["metadata"]
    print(f"✅ Model loaded v{_ml_assets['metadata']['version']}")
    yield
    # Shutdown: cleanup (tutup koneksi DB dll)
    _ml_assets.clear()
    print("👋 Service shutdown: model unloaded.")

app = FastAPI(
    title="BNSP AI Engineer - Churn Prediction Service",
    description="API REST untuk prediksi customer churn telco, "
                "sesuai Modul 10 BNSP Skema AI Engineer",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------
# [2] PYDANTIC SCHEMA = Validasi Input OTOMATIS
# ---------------------------------------------------------------------
class CustomerInputDTO(BaseModel):
    """DTO (Data Transfer Object) untuk 1 customer prediksi churn."""
    tenure_months: int = Field(ge=0, le=120, description="Masa berlangganan (bulan)")
    monthly_charges_usd: float = Field(gt=0, le=1000, description="Tagihan bulanan USD")
    total_charges_usd: float = Field(gt=0, le=1_000_000, description="Total tagihan")
    contract_type: str = Field(
        pattern="^(Month-to-month|One year|Two year)$",
        description="Jenis kontrak: Month-to-month / One year / Two year"
    )
    internet_service: str = Field(
        pattern="^(DSL|Fiber optic|No)$",
        description="Layanan internet: DSL / Fiber optic / No"
    )
    num_support_tickets: int = Field(ge=0, le=50, description="Jumlah tiket support")

    @field_validator("total_charges_usd")
    @classmethod
    def total_ge_monthly(cls, v, info):
        if "monthly_charges_usd" in info.data and v < info.data["monthly_charges_usd"]:
            raise ValueError("Total charges tidak boleh lebih kecil dari monthly charges")
        return v

class PredictionResponseDTO(BaseModel):
    customer_id: str = "CUST-AUTO-" + "00001"
    churn_prediction: bool
    churn_probability_pct: float = Field(ge=0, le=100)
    risk_level: str
    model_version: str
    inference_timestamp: str

# ---------------------------------------------------------------------
# [3] ENDPOINT UTAMA
# ---------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {"status": "online",
            "service": "BNSP-Churn-Predictor-v1",
            "timestamp": pd.Timestamp.now().isoformat()}

@app.get("/health", tags=["Health"])
def health_check():
    return {"health": "healthy",
            "model_loaded": "pipeline" in _ml_assets}

@app.get("/model/info", tags=["Model"])
def get_model_info():
    return _ml_assets.get("metadata", {})

@app.post("/predict", tags=["Prediction"], response_model=PredictionResponseDTO)
def predict_churn(cust: CustomerInputDTO):
    # Step 1: Ubah input DTO → DataFrame (sesuai format training pipeline!)
    df = pd.DataFrame([cust.model_dump()])
    # Step 2: Inferensi (predict_proba!)
    try:
        proba = float(_ml_assets["pipeline"].predict_proba(df)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Inference error: {str(e)}")
    label = bool(proba >= 0.5)
    # Tentukan level risiko bisnis
    if proba >= 0.75:
        risk = "TINGGI - Retensi segera!"
    elif proba >= 0.4:
        risk = "SEDANG - Proaktif follow-up"
    else:
        risk = "RENDAH - Aman"
    return PredictionResponseDTO(
        churn_prediction=label,
        churn_probability_pct=round(proba * 100, 2),
        risk_level=risk,
        model_version=_ml_assets["metadata"]["version"],
        inference_timestamp=pd.Timestamp.now().isoformat(),
    )
'''

# Simpan template app.py ke folder modul 10
path_app_py = os.path.join(os.path.dirname(__file__), "fastapi_churn_app_TEMPLATE.py")
with open(path_app_py, 'w', encoding='utf-8') as f:
    f.write(FASTAPI_APP_TEMPLATE.lstrip())
print(f"\n📄 Template FastAPI disimpan ke file: {path_app_py}")
print("\n>>> Contoh request JSON untuk endpoint POST /predict (bisa dicoba di /docs Swagger):")
sample_req = {
    "tenure_months": 2,
    "monthly_charges_usd": 110.5,
    "total_charges_usd": 225.0,
    "contract_type": "Month-to-month",
    "internet_service": "Fiber optic",
    "num_support_tickets": 5,
}
print("   Request Body JSON:")
print(f"   {json.dumps(sample_req, indent=6)}")
print("\n>>> Contoh Response JSON yang dihasilkan API:")
sample_resp = {
    "customer_id": "CUST-AUTO-00001",
    "churn_prediction": True,
    "churn_probability_pct": 82.3,
    "risk_level": "TINGGI - Retensi segera!",
    "model_version": "1.0.0",
    "inference_timestamp": "2026-09-12T21:30:00",
}
print(f"   {json.dumps(sample_resp, indent=6)}")

# ============================================================
# BAGIAN 4 & 5: INTEGRASI DENGAN APLIKASI + DOCKER CONTAINER
# ============================================================
print("\n" + "=" * 80)
print("BAGIAN 4 & 5: INTEGRASI APLIKASI + DOCKER CONTAINERIZATION")
print("=" * 80)

print("""
----------------------------------------------------------------------
4.1 PATTERN INTEGRASI MODEL DENGAN APLIKASI LAIN
----------------------------------------------------------------------
Pattern Panggilan dari Frontend / Backend Lain:
  ┌──────────────┐    HTTP POST /predict (JSON)   ┌──────────────┐
  │ Frontend Web  │ ─────────────────────────────→ │ FastAPI AI   │
  │ (React/Vue)   │ ←───────────────────────────── │ Service      │
  │ Mobile App    │    Response JSON               │ (Modul 10)   │
  └──────────────┘                                  └──────────────┘
         ↑                                                ↑
         └─── Auth JWT / API KEY ────────────────────────┘
  ┌──────────────┐     Internal gRPC / REST             ┌──────────────┐
  │ Backend Java  │ ─────────────────────────────→      │  FastAPI AI  │
  │ / Go / Node   │ ←─────────────────────────────      │  Service     │
  └──────────────┘                                       └──────────────┘

----------------------------------------------------------------------
5.1 DOCKER = PACKAGING APLIKASI (BESERTA ENVIRONMENTNYA)
----------------------------------------------------------------------
Masalah klasik deployment: "Di laptop SAYA jalan, tapi di SERVER ERROR!"
Docker menyelesaikan dengan: 👉 IMAGE = snapshot OS + Python + Semua dependensi + kode app
Container = Instance yang berjalan dari Image.

Langkah Dockerisasi:
  (a) Buat requirements.txt (KITA SUDAH PUNYA!)
  (b) Buat Dockerfile (resep build image)
  (c) Build image: docker build -t churn-service:v1 .
  (d) Jalankan container: docker run -d -p 8000:8000 churn-service:v1
  (e) Push ke registry: docker push <registry>/churn-service:v1 → Deploy ke server
""")

# --- Buat file TEMPLATE Dockerfile + .dockerignore + docker-compose ---
dockerfile_content = '''# ============================================================
# DOCKERFILE BNSP - Churn Prediction AI Service (FastAPI)
# ============================================================
# [1] BASE IMAGE: Python 3.10 Slim (ringan, secure, cocok untuk production)
FROM python:3.10-slim-bookworm AS base

# [2] ENVIRONMENT production (hindari .pyc + buffer log)
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1 \\
    APP_HOME=/app

# [3] SET WORKDIR
WORKDIR ${APP_HOME}

# [4] Install system dependencies (curl untuk healthcheck, build-essential untuk numpy)
RUN apt-get update && \\
    apt-get install -y --no-install-recommends curl build-essential && \\
    rm -rf /var/lib/apt/lists/*

# [5] Copy requirements DULU (Docker layer caching!) - install dulu
COPY requirements.txt .
RUN pip install --upgrade pip==24.0 && \\
    pip install --no-cache-dir -r requirements.txt

# [6] Copy SELURUH source code (termasuk folder saved_models/)
COPY . .

# [7] (Opsional) Buat user non-root untuk keamanan production
RUN useradd -m appuser && chown -R appuser:appuser ${APP_HOME}
USER appuser

# [8] EXPOSE port yang dipakai FastAPI (uvicorn 8000)
EXPOSE 8000

# [9] HEALTHCHECK = Docker monitor apakah service sehat (restart otomatis jika mati)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# [10] DEFAULT COMMAND jalankan uvicorn server (production: 1 worker per CPU core)
CMD ["uvicorn", "fastapi_churn_app_TEMPLATE:app", \\
     "--host", "0.0.0.0", \\
     "--port", "8000", \\
     "--workers", "2"]
'''

dockerignore_content = '''
# =====================
# .dockerignore - JANGAN MASUKKAN FILE INI KE DALAM IMAGE
# =====================
# Folder virtual environment
venv_bnsp/
__pycache__/
*.pyc
*.pyo
.venv/
.env

# Git
.git/
.gitignore

# Output chart (besar & tidak dibutuhkan di service)
output_charts/
sample_images/

# Notebook temp
.ipynb_checkpoints/
*.ipynb

# Log & cache
*.log
.cache/
.pytest_cache/

# Dokumentasi (jika besar, opsional dikeluarkan)
README.md
docs/
'''

compose_content = '''# ============================================================
# docker-compose.yml - Menjalankan BERSAMA: Service AI + Monitoring + Redis Cache
# Jalankan: docker compose up --build -d
# ============================================================
services:

  # 1) AI Service UTAMA (FastAPI Churn Predictor)
  churn-api:
    build: .
    container_name: bnsp-churn-api-v1
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - ENV_NAME=production
      - LOG_LEVEL=info
      # (Opsional) simpan secret dari file .env, JANGAN hardcode!
      # - API_KEY=${API_KEY}
    volumes:
      # Mount folder models agar bisa update model tanpa rebuild image
      - ./saved_models:/app/saved_models:ro
      # Mount folder logs untuk persistensi
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
    networks:
      - ai-net
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 2G

  # 2) (Opsional) REDIS Cache - simpan hasil prediksi 5 menit, hindari hit model ulang
  redis-cache:
    image: redis:7-alpine
    container_name: bnsp-redis-cache
    restart: unless-stopped
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
    networks:
      - ai-net

networks:
  ai-net:
    driver: bridge

volumes:
  logs:
'''

for fname, content in [
    ("Dockerfile", dockerfile_content),
    (".dockerignore", dockerignore_content),
    ("docker-compose.yml", compose_content),
]:
    path = os.path.join(os.path.dirname(__file__), fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content.lstrip())
    print(f"📄 {fname} template ditulis ke: {path}")

# --- Environment Management (.env file) ---
dotenv_content = '''# ============================================================
# .env - SEMUA VARIABEL RAHASIA & ENVIRONMENT (JANGAN DI COMMIT KE GIT!)
# ============================================================
# Application
ENV_NAME=production
LOG_LEVEL=INFO
APP_PORT=8000

# Security (Gunakan API KEY untuk proteksi endpoint dari publik)
API_KEY=BNSP-SuperSecretKey-AIEngineer-2026-ChangeThisInProd!
JWT_SECRET=ganti-dengan-random-string-panjang-min-32-char-xxx

# Database (jika model butuh simpan log prediksi)
DATABASE_URL=postgresql://bnsp_user:passwordRahasia@localhost:5432/ai_mlops_db

# Provider LLM (jika service pakai LLM Modul 9)
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
# ANTHROPIC_API_KEY=sk-ant-xxxxxx

# MLOps
MLFLOW_TRACKING_URI=http://localhost:5000
'''
dotenv_path = os.path.join(os.path.dirname(__file__), ".env_TEMPLATE")
with open(dotenv_path, 'w', encoding='utf-8') as f:
    f.write(dotenv_content.lstrip())
print(f"📄 .env template (copy jadi .env dan isi sendiri): {dotenv_path}")

# ============================================================
# BAGIAN 6 & 7: ENV MANAGEMENT + MODEL SERVING & MONITORING DRIFT
# ============================================================
print("\n" + "=" * 80)
print("BAGIAN 6 & 7: ENV MANAGEMENT + MODEL MONITORING (DATA DRIFT DETECTION)")
print("=" * 80)

print("""
----------------------------------------------------------------------
6.1 ENVIRONMENT MANAGEMENT: 4 TAHAP SDLC
----------------------------------------------------------------------
  1. DEVELOPMENT (DEV)  : Developer laptop - debug, logs verbose, sample data kecil
  2. STAGING / TEST     : Mirror production environment - UAT internal
  3. PRE-PRODUCTION     : Mirror 100% prod - Final performance test
  4. PRODUCTION (PROD)  : Pengguna nyata - High avail / Scaling / Monitoring ketat

Prinsip: DRIFT ENVIRONMENT = Pembunuh deployment! Pastikan:
  - Python version SAMA di semua stage (3.10.9)
  - Semua dependencies LOCK VERSI (==x.y.z BUKAN >=)
  - Gunakan pip freeze → requirements.txt.lock untuk production
""")

# --- Bagian 7: Data Drift Simulation + Visualisasi ---
print("""
----------------------------------------------------------------------
7.1 APA ITU DATA DRIFT? (MUSUH UTAMA MODEL DI PRODUKSI!)
----------------------------------------------------------------------
Data Drift = PERUBAHAN DISTRIBUSI fitur input ATAU target output dari
waktu training vs waktu produksi → AKURASI MODEL TURUN drastis
dari waktu ke waktu (Model Decay).

JENIS-JENIS DRIFT WAJIB BNSP:
  A. DATA DRIFT / COVARIATE SHIFT : Distribusi FITUR INPUT berubah
     (misal: tagihan bulanan = $50-100 saat training, tapi di PROD naik
      jadi $150-250 karena inflasi — model tidak terbiasa!)
  B. CONCEPT DRIFT : HUBUNGAN antara X → y berubah
     (misal: Dulu "kontrak bulanan" → churn tinggi. Sekarang promo besar,
      kontrak bulanan tidak lagi menyebabkan churn → konsep BERUBAH!)
  C. LABEL DRIFT  : Distribusi label (y) target berubah drastis

STRATEGI MONITORING BNSP:
  ✅ Hitung statistik tiap minggu (mean/std/missing%) vs baseline training
  ✅ Statistical Test: KS-Test (kontinyu) / Chi-Square (kategorik)
  ✅ PSI (Population Stability Index) → STANDARD INDUSTRI untuk credit scoring
  ✅ Retrain model periodik (misal tiap 3 bulan) ATAU trigger-based jika PSI > 0.25
""")

# 👉 PRAKTIK: Simulasi Data Drift pada fitur tenure_months & monthly_charges
print("\n>>> PRAKTIK: Simulasi Data Drift (Baseline Training vs Produksi 6 Bulan Kemudian)")

# Baseline (Training data)
baseline = data_churn.copy()
# Production data = drift dibuat sengaja:
# - monthly_charges naik 25% (inflasi)
# - tenure lebih pendek (banyak customer baru promo)
# - contract type murni Month-to-month semua
production = data_churn.copy()
production['monthly_charges_usd'] = (production['monthly_charges_usd'] * 1.25
                                      * np.random.uniform(0.95, 1.10, n)).round(2)
production['total_charges_usd'] = (production['tenure_months'] * production['monthly_charges_usd']
                                    * np.random.uniform(0.93, 1.07, n)).round(2)
production['tenure_months'] = np.clip(production['tenure_months'] * 0.55
                                       + np.random.normal(-2, 3, n), 0, 72).astype(int)
production['contract_type'] = np.random.choice(['Month-to-month', 'One year', 'Two year'],
                                                n, p=[0.88, 0.08, 0.04])

# Hitung perbandingan statistik
drift_compare = pd.DataFrame({
    'Fitur': ['tenure_months (mean)', 'monthly_charges_usd (mean)',
              'Month-to-month %', 'Two year %'],
    'Training (Baseline)': [
        f"{baseline['tenure_months'].mean():.1f}",
        f"${baseline['monthly_charges_usd'].mean():.2f}",
        f"{(baseline['contract_type']=='Month-to-month').mean()*100:.1f}%",
        f"{(baseline['contract_type']=='Two year').mean()*100:.1f}%",
    ],
    'Production (Now)': [
        f"{production['tenure_months'].mean():.1f}",
        f"${production['monthly_charges_usd'].mean():.2f}",
        f"{(production['contract_type']=='Month-to-month').mean()*100:.1f}%",
        f"{(production['contract_type']=='Two year').mean()*100:.1f}%",
    ],
})
print(drift_compare.to_string(index=False))

# Simulasi akurasi model TURUN karena drift
yc_test_prod = (churn_prob * 0.75
                + 0.4 * (production['contract_type'] == 'Month-to-month').astype(int)
                - 0.01 * production['tenure_months']
                + np.random.normal(0, 0.1, n))
yc_test_prod = (yc_test_prod > yc_test_prod.quantile(0.70)).astype(int)
X_prod = production.drop('churn_label', axis=1, errors='ignore')
yp_proba = loaded_pipeline.predict_proba(X_prod)[:, 1]
yp_pred = (yp_proba >= 0.5).astype(int)
acc_prod = accuracy_score(yc_test_prod, yp_pred)
auc_prod = roc_auc_score(yc_test_prod, yp_proba)
print(f"\n📉 Penurunan performa karena DRIFT:")
print(f"   Akurasi  (Training → Prod Drift): {acc_c*100:.1f}% → {acc_prod*100:.1f}% "
      f"(↓ {(acc_c - acc_prod)*100:.1f}pp)")
print(f"   ROC-AUC  (Training → Prod Drift): {auc_c:.3f} → {auc_prod:.3f} "
      f"(↓ {auc_c - auc_prod:.3f})")

# --- Hitung PSI sederhana untuk 2 fitur ---
def hitung_psi(expected_series, actual_series, bins=10):
    """Population Stability Index sederhana (kontinyu)."""
    exp = np.asarray(expected_series.dropna())
    act = np.asarray(actual_series.dropna())
    breaks = np.quantile(exp, np.linspace(0, 1, bins + 1))
    breaks = np.unique(breaks)
    exp_counts, _ = np.histogram(exp, bins=breaks)
    act_counts, _ = np.histogram(act, bins=breaks)
    exp_pct = np.clip(exp_counts / len(exp), 1e-6, 1 - 1e-6)
    act_pct = np.clip(act_counts / len(act), 1e-6, 1 - 1e-6)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

psi_monthly = hitung_psi(baseline['monthly_charges_usd'], production['monthly_charges_usd'])
psi_tenure = hitung_psi(baseline['tenure_months'], production['tenure_months'])
print(f"\n📊 PSI (Population Stability Index - standard industri):")
print(f"   monthly_charges_usd PSI = {psi_monthly:.4f} "
      f"({'⚠️ DRIFT PARAH (>0.25)' if psi_monthly>0.25 else 'Aman'})")
print(f"   tenure_months        PSI = {psi_tenure:.4f} "
      f"({'⚠️ DRIFT PARAH (>0.25)' if psi_tenure>0.25 else 'Aman'})")

# --- Visualisasi 2 Drift + Akurasi Over Time (4 panel) ---
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# [1,1] Distribusi Monthly Charges: Baseline vs Prod
sns.histplot(baseline['monthly_charges_usd'], bins=25, ax=axes[0, 0],
             kde=True, color='#4C78A8', alpha=0.6, label='Training (Baseline)')
sns.histplot(production['monthly_charges_usd'], bins=25, ax=axes[0, 0],
             kde=True, color='#E45756', alpha=0.6, label='Produksi (Drift +25%)')
axes[0, 0].axvline(baseline['monthly_charges_usd'].mean(), color='#4C78A8',
                   ls='--', lw=2, label='Mean Train')
axes[0, 0].axvline(production['monthly_charges_usd'].mean(), color='#E45756',
                   ls='--', lw=2, label='Mean Prod')
axes[0, 0].set_title(f'Distribusi Monthly Charges — PSI = {psi_monthly:.3f} (⚠️ DRIFT)',
                     fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('Monthly Charges USD', fontsize=10)
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(alpha=0.3)

# [1,2] Distribusi Tenure
sns.histplot(baseline['tenure_months'], bins=20, ax=axes[0, 1],
             kde=True, color='#4C78A8', alpha=0.6, label='Training (Baseline)')
sns.histplot(production['tenure_months'], bins=20, ax=axes[0, 1],
             kde=True, color='#E45756', alpha=0.6, label='Produksi (Pelanggan Baru)')
axes[0, 1].axvline(baseline['tenure_months'].mean(), color='#4C78A8', ls='--', lw=2)
axes[0, 1].axvline(production['tenure_months'].mean(), color='#E45756', ls='--', lw=2)
axes[0, 1].set_title(f'Distribusi Tenure (Bulan) — PSI = {psi_tenure:.3f} (⚠️ DRIFT)',
                     fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Tenure (bulan)', fontsize=10)
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(alpha=0.3)

# [2,1] Komposisi Kontrak (Stacked bar)
kontrak_comp = pd.DataFrame({
    'Training': baseline['contract_type'].value_counts(normalize=True)[
        ['Month-to-month', 'One year', 'Two year']].values * 100,
    'Production': production['contract_type'].value_counts(normalize=True).reindex(
        ['Month-to-month', 'One year', 'Two year'], fill_value=0).values * 100,
}, index=['Month-to-month', 'One year', 'Two year'])
kontrak_comp.T.plot(kind='bar', stacked=True, ax=axes[1, 0],
                    color=['#E45756', '#F58518', '#54A24B'], edgecolor='white', linewidth=1)
axes[1, 0].set_title('Komposisi Jenis Kontrak — Label Drift!',
                     fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Persentase (%)', fontsize=10)
axes[1, 0].legend(title='Contract Type', fontsize=9)
for c in axes[1, 0].containers:
    axes[1, 0].bar_label(c, fmt='%.1f%%', label_type='center', color='white', fontsize=9, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# [2,2] Performance Degradation Over Time (bulanan)
waktu = ['M-0\n(Training)', 'M+1', 'M+2', 'M+3', 'M+4', 'M+5', 'M+6\n(Saat ini)']
acc_over_time = np.linspace(acc_c * 100, acc_prod * 100, 7) + np.random.normal(0, 0.5, 7)
auc_over_time = np.linspace(auc_c, auc_prod, 7) + np.random.normal(0, 0.01, 7)

ax_acc = axes[1, 1]
line1, = ax_acc.plot(waktu, acc_over_time, color='#4C78A8', marker='o', ms=9, lw=2.5, label='Accuracy (%)')
ax_acc.set_ylabel('Accuracy (%)', color='#4C78A8', fontsize=10)
ax_acc.tick_params(axis='y', labelcolor='#4C78A8')
ax_acc.axhline(80, color='red', ls='--', lw=1.5, label='Threshold Retrain (80%)')
ax_acc.fill_between(range(len(waktu)), 0, 80, alpha=0.08, color='red')

ax_auc = ax_acc.twinx()
line2, = ax_auc.plot(waktu, auc_over_time, color='#E45756', marker='s', ms=9, lw=2.5, label='ROC-AUC')
ax_auc.set_ylabel('ROC-AUC Score', color='#E45756', fontsize=10)
ax_auc.tick_params(axis='y', labelcolor='#E45756')

axes[1, 1].set_title('📉 Degradasi Performa Model Akibat Drift (7 Bulan)\n'
                     'Trigger RETRAIN ketika akurasi < 80% (threshold)',
                     fontsize=11, fontweight='bold')
lines = [line1, line2] + [axes[1, 1].lines[-1]]
labels = [l.get_label() for l in lines]
ax_acc.legend(lines, labels, loc='lower left', fontsize=9)
ax_acc.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_data_drift_performance_degradation_4panel.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Visualisasi 2 disimpan: 02_data_drift_performance_degradation_4panel.png")

# ============================================================
# BAGIAN 8: DASAR MLOPS - VERSIONING & CI/CD ML
# ============================================================
print("\n" + "=" * 80)
print("BAGIAN 8: DASAR MLOPS - VERSIONING MODEL, DATASET & CI/CD ML")
print("=" * 80)

print("""
----------------------------------------------------------------------
8.1 APA ITU MLOps? Gabungan ML + DEVOPS
----------------------------------------------------------------------
MLOps = Praktik MENGGABUNGKAN lifecycle Machine Learning dengan
DevOps software engineering, agar model ML bisa:
  ✅ DIPERBAIKI dengan cepat (Iterate)
  ✅ DILACAK jejaknya (Auditable - penting untuk BNSP Portofolio!)
  ✅ DIDEPLOY secara OTOMATIS (CI/CD Pipeline)
  ✅ TERMONITOR dan DIRETRAIN otomatis ketika drift

3 PILLAR UTAMA MLOps (WAJIB TAHU BNSP):
  ① DATA VERSIONING : Dataset harus bisa di-rollback! (Tools: DVC / Git LFS / Delta Lake)
  ② MODEL VERSIONING: Tiap training menghasilkan artifact versi unik + metric
     (Tools: MLflow / Kubeflow / Weights & Biases / Comet ML)
  ③ CI/CD + CONTINUOUS TRAINING (CT):
     - CI = Test code + Test data schema
     - CD = Deploy model otomatis jika metric > threshold
     - CT = Retrain otomatis tiap minggu / trigger drift

----------------------------------------------------------------------
8.2 TOOLING MLOPS STANDARD INDUSTRI (BNSP REFERENSI):
  ┌──────────────────────────────────────────────────────────────┐
  │ Experiment Tracker : MLflow (OPEN SOURCE PALING POPULER!),    │
  │                       Weights & Biases, Neptune.ai            │
  │ Feature Store      : Feast, Tecton                            │
  │ Orchestrator       : Airflow, Kubeflow Pipelines, Prefect     │
  │ Model Registry     : MLflow Registry, SageMaker Model Registry│
  │ Serving            : Seldon Core, KServe, BentoML, FastAPI    │
  │ Monitoring         : Evidently AI, Arize AI, WhyLabs          │
  │ Data Versioning    : DVC (Data Version Control), LakeFS       │
  └──────────────────────────────────────────────────────────────┘
""")

# --- Simulasi MLflow-style Experiment + Model Registry ---
print("\n>>> PRAKTIK Simulasi Model Registry (3 versi model + perbandingan)")
registry = pd.DataFrame([
    {
        'model_version': 'v0.9.0-beta',
        'stage': 'Archived',
        'created_at': '2026-06-10',
        'algorithm': 'Logistic Regression',
        'roc_auc_test': 0.7921,
        'accuracy_test': 0.7780,
        'training_rows': 1500,
        'owner': 'Trainee Awal',
        'reason_change': 'Baseline model pertama; underfit.'
    },
    {
        'model_version': 'v1.0.0',
        'stage': 'Production',
        'created_at': '2026-09-12',
        'algorithm': 'RandomForest (sklearn)',
        'roc_auc_test': round(auc_c, 4),
        'accuracy_test': round(acc_c, 4),
        'training_rows': len(Xc_train),
        'owner': 'AI Engineer BNSP',
        'reason_change': 'Improve AUC +9.4pp, pakai Pipeline + Preprocessor'
    },
    {
        'model_version': 'v1.1.0-rc1',
        'stage': 'Staging',
        'created_at': '2026-09-15',
        'algorithm': 'XGBoost + Tuned',
        'roc_auc_test': 0.9143,
        'accuracy_test': 0.8825,
        'training_rows': 2000,
        'owner': 'Senior AI Eng',
        'reason_change': 'Tuning + Tambah fitur Promo; AUC ↑ 3.4pp'
    },
])
registry.set_index('model_version', inplace=True)
print(registry.to_string())
print("\n✅ Snapshot registry tersimpan (contoh format MLflow).")

# --- Visualisasi: MLOps Cycle Diagram + Model Registry Bar ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Kiri: Diagram MLOps Loop (circular 7 langkah)
mlops_steps = ['Data\nCollection', 'Data Prep\n(ETL)', 'Model\nTraining',
               'Model\nEvaluation', 'Model\nRegistry', 'Deployment\n(CD)',
               'Monitoring\n& Retrain Trigger']
n_step = len(mlops_steps)
angles = np.linspace(0, 2 * np.pi, n_step, endpoint=False)
radius = 5
x = radius * np.cos(angles)
y = radius * np.sin(angles)
# Draw arrows
for i in range(n_step):
    j = (i + 1) % n_step
    ax1.annotate('', xy=(x[j] * 0.85, y[j] * 0.85),
                 xytext=(x[i] * 0.92, y[i] * 0.92),
                 arrowprops=dict(arrowstyle='->', lw=3, color='#2171B5'))
for i, step in enumerate(mlops_steps):
    ax1.scatter(x[i], y[i], s=2500, zorder=3,
                color=sns.color_palette('Blues_r', n_step)[i],
                edgecolors='white', linewidths=3)
    ax1.text(x[i] * 1.22, y[i] * 1.22, f'S{i+1}\n{step}',
             ha='center', va='center', fontsize=9, fontweight='bold')
ax1.text(0, 0, "MLOps\nFeedback\nLoop", ha='center', va='center',
         fontsize=18, fontweight='bold', color='#08306B',
         bbox=dict(facecolor='#C6DBEF', alpha=0.7, boxstyle='round,pad=0.5'))
ax1.set_xlim(-8, 8)
ax1.set_ylim(-8, 8)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title("🔄 MLOps Continuous Loop (7 Tahapan BNSP)", fontsize=13, fontweight='bold')

# Kanan: Bar perbandingan 3 versi model di registry
versions = ['v0.9-beta', 'v1.0-PROD', 'v1.1-RC1']
accs = [77.8, acc_c * 100, 88.25]
aucs = [79.21, auc_c * 100, 91.43]
xx = np.arange(len(versions))
width = 0.35
bars1 = ax2.bar(xx - width / 2, accs, width, color='#4C78A8', label='Accuracy %', edgecolor='white')
bars2 = ax2.bar(xx + width / 2, aucs, width, color='#F58518', label='ROC-AUC %', edgecolor='white')
ax2.set_xticks(xx)
ax2.set_xticklabels(versions, fontsize=11, fontweight='bold')
ax2.set_ylabel('Persentase (%)', fontsize=11)
ax2.set_ylim(60, 100)
ax2.set_title('📦 MODEL REGISTRY: 3 Versi Model Churn\n(Experiment History BNSP)',
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(axis='y', alpha=0.3)
for bars in [bars1, bars2]:
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                 f'{bar.get_height():.1f}', ha='center', fontsize=9, fontweight='bold')
# Tag stage
for i, stage in enumerate(['Archived', 'Production ←', 'Staging']):
    ax2.text(xx[i], 97, stage, ha='center', fontsize=9, color='#555',
             style='italic', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_mlops_cycle_model_registry_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("✅ Visualisasi 3 disimpan: 03_mlops_cycle_model_registry_comparison.png")

# ============================================================
# RINGKASAN AKHIR MODUL 10
# ============================================================
print("\n" + "=" * 80)
print("📋 RINGKASAN AKHIR MODUL 10 - AI MODEL DEPLOYMENT & MLOPS")
print("=" * 80)
print("""
+==================================================================+
| 8 ARTEFAK DEPLOYMENT YANG WAJIB ADA DI PORTOFOLIO BNSP:          |
+==================================================================+
| [1] File model terserialize (.joblib / .pkl) + metadata.json     |
| [2] Source code FastAPI + Pydantic BaseModel (app.py)            |
| [3] Dockerfile + .dockerignore (Production-ready)                |
| [4] docker-compose.yml (Service + Redis/DB sidecar)              |
| [5] .env template + .gitignore (JANGAN commit secret!)           |
| [6] Script Data Drift detector + Grafana dashboard config        |
| [7] MLflow experiment log + Model Registry snapshot (3+ versi)   |
| [8] Test script: 10 test case REST API (pytest)                  |
+==================================================================+
""")
print(f"Total file saved_models/ : {len(os.listdir(MODELS_DIR))} file (pickle + joblib)")
for f in sorted(os.listdir(MODELS_DIR)):
    size_kb = os.path.getsize(os.path.join(MODELS_DIR, f)) / 1024
    print(f"  📦 {f} [{size_kb:.1f} KB]")

print(f"\nTotal visualisasi output_charts/ : {len(os.listdir(OUTPUT_DIR))} file PNG")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  ✔ {f}")

# ============================================================
# BLOK LATIHAN PRAKTIK - PORTOFOLIO BNSP
# ============================================================
print("\n" + "=" * 80)
print("🏋️  LATIHAN PRAKTIK MANDIRI (MATERI PORTOFOLIO BNSP)")
print("=" * 80)

print("""
LATIHAN 1 (Serialization Wajib)
   Latih model REGRESI HARGA RUMAH (pakai Modul 4 data yang sama).
   Buat Pipeline (preprocessor + GradientBoostingRegressor).
   Simpan dengan TIGA format: (a) pickle .pkl, (b) joblib .joblib,
   (c) JSON metadata yang berisi MAE, RMSE, R², fitur list, tanggal train,
   nama author, versi model. Upload ke folder saved_models/.

LATIHAN 2 (FastAPI Production)
   Copy file [fastapi_churn_app_TEMPLATE.py] di folder Modul 10 menjadi
   app.py. Edit file tersebut:
   a) Tambahkan API KEY validation via Dependency Injection (header X-API-KEY)
   b) Tambahkan endpoint POST /predict/batch (list N customer sekaligus)
   c) Tambahkan logging: Setiap request dicatat ke CSV logs/predictions_history.csv
      (kolom: timestamp, customer_id, input_json, prediction, probability, latency_ms)
   Jalankan dengan: uvicorn app:app --reload, test via http://localhost:8000/docs.

LATIHAN 3 (Docker Containerization Wajib Portofolio)
   Install Docker Desktop di laptop Anda (gratis).
   Jalankan perintah di FOLDER Modul 10:
      docker build -t bnsp-churn:v1 .
      docker run -p 8000:8000 bnsp-churn:v1
   Test akses http://localhost:8000/health (pastikan sehat).
   Screenshot (a) Build log sukses, (b) Container di Docker Desktop,
   (c) Swagger UI di browser → BUKTI PORTOFOLIO KUAT!

LATIHAN 4 (Drift Detection & Retraining Policy)
   Buatlah fungsi Python:
     def check_drift_retrain_decision(baseline_df, prod_df, thresholds_psi=0.25,
                                     threshold_acc_drop_pct=10):
   Yang mengembalikan tuple (drift_report_dict, should_retrain_bool).
   Kriteria retrain:
   - Jika ada 2+ fitur PSI > 0.25 → retrain
   - ATAU jika accuracy production turun > 10% dari baseline → retrain
   - Selain itu → TIDAK retrain (lakukan monitoring lanjutan)
   Test fungsi dengan 3 skenario: (a) Tanpa drift, (b) Drift 1 fitur,
   (c) Drift banyak fitur + akurasi turun 15%.

LATIHAN 5 (Mini MLOps Pipeline dengan MLflow - Opsional Level Atas)
   Install MLflow: pip install mlflow==2.15.1
   Buat script train_churn_mlflow.py yang:
     a. Set experiment name = "BNSP-Churn-Experiment"
     b. Log params (n_estimators, max_depth, class_weight)
     c. Log metrics (accuracy, ROC-AUC, F1, Recall)
     d. Log model (sklearn flavor) + register ke Model Registry
     e. Log artifacts (confusion matrix PNG, classification report TXT)
   Jalankan 3 kali experiment dengan parameter berbeda.
   Buka UI MLflow: mlflow ui → port 5000 → Screenshot perbandingan runs.
   Ini adalah BUKTI TERBAIK untuk unit MLOps BNSP!
""")

print("\n" + "=" * 80)
print("✅ MODUL 10 AI MODEL DEPLOYMENT SELESAI!")
print("   8 sub-topik BNSP + 3 artefak Docker + 3 visualisasi chart + 5 latihan.")
print("=" * 80)
