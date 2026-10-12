# =====================================================================
# FILE: app.py - FastAPI Service untuk Model Churn Prediction BNSP
# Cara menjalankan: (venv_bnsp) C:\ai-engineer\modul_10_deployment>
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
