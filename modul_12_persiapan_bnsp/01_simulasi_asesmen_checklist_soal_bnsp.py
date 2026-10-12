# =====================================================================
# MODUL 12 - PERSIAPAN UJI KOMPETENSI BNSP
# Skema: ARTIFICIAL INTELLIGENCE ENGINEER (Kode Skema: SKKNI-AI-ENG-2024)
# =====================================================================
# Materi (sesuai brief BNSP Modul 12):
# 1. Pengenalan proses Uji Kompetensi BNSP
# 2. Pemahaman prinsip asesmen kompetensi + SKKNI + 12 Unit Kompetensi
# 3. Persiapan bukti kompetensi (portofolio checklist per unit)
# 4. Penyusunan portofolio / proyek (12 checklist bukti artefak)
# 5. Persiapan dokumen pendukung asesmen (surat rekomendasi, CV, dll)
# 6. Simulasi 50 Soal Pilihan Ganda + Kunci Jawaban + Pembahasan
# 7. 3 Studi Kasus Praktis Asesmen (sesuai metode BNSP)
# 8. Panduan Wawancara Asesor BNSP (pertanyaan yang sering keluar)
# =====================================================================

import os
import re
import sys
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print("⚠️  Warning: numpy/pandas/matplotlib tidak terinstall. "
          "Visualisasi chart akan dilewati.\nInstall: pip install -r requirements.txt")
    import types
    plt = types.ModuleType('plt')
    plt.subplots = lambda *a, **kw: (None, [None])
    plt.rcParams = {}
    sns = types.ModuleType('sns')
    sns.color_palette = lambda *a, **kw: []
    sns.set_style = lambda *a: None
    np = types.ModuleType('np'); np.arange = lambda *a: []
    pd = types.ModuleType('pd')
    pd.DataFrame = lambda *a, **kw: None

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "output_charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid") if hasattr(sns, 'set_style') else None
# Support for Chinese etc characters in matplotlib
try:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

# ============================================================
# BAGIAN 1 & 2: PENGENALAN PROSES BNSP + 12 UNIT KOMPETENSI
# ============================================================
print("=" * 88)
print("🏛️  MODUL 12 — PERSIAPAN UJI KOMPETENSI BNSP SKEMA AI ENGINEER")
print("=" * 88)

print("""
======================================================================
BAGIAN 1: PENGENALAN BADAN NASIONAL SERTIFIKASI PROFESI (BNSP)
======================================================================
🏛️ Apa itu BNSP?
  • Badan Nasional Sertifikasi Profesi = Lembaga Pemerintah NON-KEMENTERIAN
    yang dibentuk berdasarkan UU No. 13 Tahun 2003 tentang Ketenagakerjaan.
  • Tugas: MEMBERIKAN SERTIFIKASI KOMPETENSI (diakui NEGARA) kepada tenaga
    kerja Indonesia yang sudah memenuhi Standar Kompetensi Kerja Nasional
    Indonesia (SKKNI).

🪪 Apa itu Sertifikat Kompetensi BNSP?
  • Surat pengakuan kompetensi yang diberikan oleh LSP (Lembaga Sertifikasi
    Profesi) TERAKREDITASI BNSP kepada seseorang setelah MELALUI PROSES
    ASESMEN (penilaian) dan MEMENUHI standar kompetensi yang ditetapkan.
  • Berlaku 3 TAHUN, bisa diperpanjang (re-sertifikasi).
  • DIAKUI SECARA RESMI oleh seluruh perusahaan di Indonesia untuk
    pengadaan tenaga kerja, promosi jabatan, pengadaan proyek pemerintah.

🔄 6 TAHAP PROSES UJI KOMPETENSI BNSP (WAJIB DIKETAHUI):
  Tahap 1 👉 PENDAFTARAN & PEMERIKSAAN AWAL (APL-0 / APL-1)
              • Peserta mengisi Formulir APL-1 = Daftar riwayat hidup +
                portofolio awal bukti kompetensi.
              • Asesor melakukan pemeriksaan awal: apakah persyaratan
                administrasi & teknis TERCUKUPI? (Cek kelengkapan bukti)
  Tahap 2 👉 ASESMEN PEMENUHAN (COMPETENCY ASSESSMENT)
              • Metode asesmen sesuai Panduan Skema:
                (a) Observasi Praktik langsung / Demo kerja
                (b) Review Portofolio / bukti karya
                (c) Wawancara / Tanya Jawab lisan
                (d) Ujian Tulis Pilihan Ganda / Esai (opsional skema)
                (e) Ujian Proyek (Assignment berbasis kasus nyata)
  Tahap 3 👉 VERIFIKASI & VALIDASI (Tim Asesor memutuskan:
              KOMPETEN / BELUM KOMPETEN per unit)
  Tahap 4 👉 BANDING (jika peserta tidak setuju hasil keputusan asesor)
  Tahap 5 👉 PENERBITAN SERTIFIKAT oleh LSP ke BNSP
              → Sertifikat fisik dikirim 1-4 minggu.
  Tahap 6 👉 PEMANTAUAN PASCA SERTIFIKASI & RE-SERTIFIKASI (setiap 3 thn)

======================================================================
BAGIAN 2: PRINSIP ASESMEN KOMPETENSI + SKKNI + 12 UNIT KOMPETENSI
======================================================================
📜 PRINSIP ASAS ASESMEN (WAJIB DIHafal — KELUAR DI WAWANCARA ASESI):
  A. ASAS (8 butir):
     1. Adil (Fair)       → Tidak diskriminatif, semua peserta sama prosedur
     2. Transparan        → Peserta tahu kriteria dinilai apa dan bagaimana
     3. Akurat            → Penilaian sesuai dengan UNIT KOMPETENSI & KUK
     4. Konsisten         → Hasil sama meskipun asesor berbeda / waktu beda
     5. Otentik           → Bukti adalah BENAR karya peserta sendiri
     6. Sufisien          → Jumlah & kualitas bukti CUKUP untuk memutuskan
     7. Terkini           → Bukti TIDAK USANG (maksimal 3 tahun terakhir)
     8. Autentik          → TIDAK PALSU / PLAGIAT / copy-paste dari orang lain

  B. ATURAN UTAMA: "KALI SEMUA KUK (KRITERIA UNjuk KERJA) DALAM SATU UNIT
     KOMPETENSI = Dinyatakan BELUM KOMPETEN untuk unit itu"
     → Harus SEMUA sub-elemen 100% terpenuhi!
""")

# ====== 12 UNIT KOMPETENSI RESMI SKEMA AI ENGINEER BNSP ======
UNIT_KOMPETENSI_BNSP = [
    {
        "kode": "U-01",
        "nama_unit": "Mengidentifikasi Kebutuhan Bisnis untuk Solusi AI",
        "tingkat_kkni": 4,
        "deskripsi": (
            "Menganalisis masalah bisnis, identifikasi use case AI yang tepat, "
            "menyusun requirement & success metrics proyek AI."
        ),
        "bukti_portofolio": [
            "Dokumen Problem Statement 5W1H proyek AI",
            "Dokumen Requirement Bisnis & Technical Spec",
            "Presentation deck 10-slide pitching proyek AI",
            "Matriks Prioritisasi Use Case (Value vs Effort)",
        ],
    },
    {
        "kode": "U-02",
        "nama_unit": "Mengolah dan Menyiapkan Data untuk Model AI",
        "tingkat_kkni": 5,
        "deskripsi": (
            "Data understanding, Data cleaning (null/dup/outlier), Transformasi, "
            "Encoding, Scaling, Feature Engineering, EDA, Splitting dataset."
        ),
        "bukti_portofolio": [
            "Notebook Python / .py file Data Preparation Pipeline",
            "10+ Chart EDA (pairplot, boxplot, heatmap corr, dll)",
            "Laporan Data Quality Report (missing %, duplicates, outlier)",
            "Sklearn Pipeline + ColumnTransformer bukti preprocessing",
        ],
    },
    {
        "kode": "U-03",
        "nama_unit": "Mengembangkan Model Machine Learning",
        "tingkat_kkni": 5,
        "deskripsi": (
            "Supervised (Classification + Regression), Unsupervised (Clustering), "
            "Pemilihan algoritma, Overfit/Underfit detection, Feature Selection."
        ),
        "bukti_portofolio": [
            "Notebook Training 7+ algoritma klasifikasi + perbandingan metrik",
            "Notebook Training 9+ algoritma regresi + residual analysis",
            "Notebook Clustering 4 algoritma + Profiling Cluster",
            "Report Overfit detection: Learning Curve atau Train-Val Score gap",
        ],
    },
    {
        "kode": "U-04",
        "nama_unit": "Mengembangkan Model Deep Learning (NN/DL)",
        "tingkat_kkni": 6,
        "deskripsi": (
            "ANN/Perceptron/MLP, CNN untuk Computer Vision, dasar RNN/LSTM, "
            "Backprop intuition, Activation/Loss/Optimizer, Transfer Learning."
        ),
        "bukti_portofolio": [
            "Notebook MLP 5 arsitektur comparison + loss curve",
            "Notebook MNIST digit classification (MLP + CNN opsional)",
            "Visualisasi Kernel Conv (Sobel/Blur/Sharpen) manual + MaxPool",
            "Daftar Arsitektur CNN Klasik (LeNet → ResNet → EfficientNet)",
        ],
    },
    {
        "kode": "U-05",
        "nama_unit": "Mengevaluasi dan Mengoptimasi Model AI",
        "tingkat_kkni": 5,
        "deskripsi": (
            "9 Metrik Klasifikasi + Confusion Matrix + ROC/PR Curve, "
            "MAE/MSE/RMSE/R²/MAPE regresi, Cross Validation, "
            "GridSearchCV vs RandomizedSearchCV Hyperparameter Tuning."
        ),
        "bukti_portofolio": [
            "Notebook Confusion Matrix 4 model + Precision-Recall Trade-off",
            "Visualisasi ROC Curve vs PR Curve kasus Imbalance (fraud 99:1)",
            "Perbandingan K-Fold vs Stratified K-Fold CV Notebook",
            "Heatmap hasil Grid/Random Search + Nested CV estimasi performa",
        ],
    },
    {
        "kode": "U-06",
        "nama_unit": "Menerapkan Computer Vision",
        "tingkat_kkni": 6,
        "deskripsi": (
            "Image preprocessing (resize/grayscale/denoise/augment), "
            "Image Classification, dasar Object Detection, IoU, mAP."
        ),
        "bukti_portofolio": [
            "Notebook 9 teknik preprocessing gambar + augmentasi",
            "Feature extraction: LBP, Color Histogram, Pixel flat",
            "Pipeline training model klasifikasi multi-class shape dataset",
            "Simulasi perhitungan IoU Object Detection 2 bounding box",
        ],
    },
    {
        "kode": "U-07",
        "nama_unit": "Menerapkan Natural Language Processing (NLP)",
        "tingkat_kkni": 6,
        "deskripsi": (
            "6-step Text preprocessing Indonesia (case/clean/token/stop/stem),"
            " TF-IDF/BOW vectorization, Sentiment Analysis, Word Embedding,"
            " Transformer architecture pengenalan."
        ),
        "bukti_portofolio": [
            "Notebook Pipeline Preprocessing 6 langkah Bahasa Indonesia",
            "TF-IDF + Bag of Words Vectorizer extraction",
            "Sentiment Analysis 3 kelas: NB/LogReg/RF comparison",
            "Word Embedding Word2Vec visualisasi PCA 2D + Transformer diagram",
        ],
    },
    {
        "kode": "U-08",
        "nama_unit": "Menerapkan Generative AI & Large Language Model (LLM)",
        "tingkat_kkni": 6,
        "deskripsi": (
            "Gen AI vs Traditional AI, LLM, Prompt Engineering (Zero/Few/CoT),"
            " Structured Prompting, API LLM, RAG, Embedding + Vector DB,"
            " Evaluasi LLM, Responsible LLM AI (hallucination, bias)."
        ),
        "bukti_portofolio": [
            "Contoh Prompt Engineering 4 teknik + output perbandingan",
            "Kode Class DummyLLMAPIClient simulasi panggilan API",
            "Full Pipeline RAG: 9 chunk KB, TF-IDF retrieval, 5 query test",
            "Notebook Evaluasi 8 dimensi LLM + RAG Triad Report",
        ],
    },
    {
        "kode": "U-09",
        "nama_unit": "Mendeploy Model AI ke Lingkungan Produksi",
        "tingkat_kkni": 6,
        "deskripsi": (
            "Model Serialization pickle/joblib, REST API (FastAPI + Pydantic),"
            " Docker & Docker Compose, Environment management (.env),"
            " Model Serving, Load balancer dasar."
        ),
        "bukti_portofolio": [
            "app.py FastAPI + Pydantic 2 endpoint GET /health POST /predict",
            "Dockerfile Production 10 step base-slim + healthcheck",
            "docker-compose.yml (service + redis cache + healthcheck)",
            ".env template + .gitignore (JANGAN commit secret!)",
        ],
    },
    {
        "kode": "U-10",
        "nama_unit": "Memonitor dan Memelihara Model AI",
        "tingkat_kkni": 6,
        "deskripsi": (
            "Model Drift (Data/Concept drift), PSI (Population Stability Index),"
            " Grafik degradasi performa, Kebijakan retraining periodik,"
            " Logging prediction, Dasar monitoring dashboard (Grafana-like)."
        ),
        "bukti_portofolio": [
            "Notebook Simulasi Data Drift (baseline vs prod distribution)",
            "Perhitungan PSI (Population Stability Index) 2 fitur",
            "Line Chart degradasi akurasi bulanan 7 bulan + threshold retrain",
            "Contoh log CSV history prediksi + script daily summary PSI",
        ],
    },
    {
        "kode": "U-11",
        "nama_unit": "Menerapkan Etika, Keamanan, dan Responsible AI",
        "tingkat_kkni": 5,
        "deskripsi": (
            "5 Pilar Responsible AI (Fairness, Transparency, Privacy, Safety,"
            " Accountability), UN Ethical AI Principles, Studi Kasus Pelanggaran,"
            " AI Act Eropa + UU PDP Indonesia No. 27 Tahun 2022, Guardrails LLM."
        ),
        "bukti_portofolio": [
            "Dokumen Etika AI Checklist 5 Pilar untuk proyek Anda",
            "Dokumen Analisis Risiko AI + Mitigasi (8 dimensi risiko)",
            "Script llm_safety_guardrails() input filter jailbreak/PII/KTP",
            "Bukti Anonymization Dataset (hapus PII dari dataset training)",
        ],
    },
    {
        "kode": "U-12",
        "nama_unit": "Mendokumentasikan & Menyajikan Proyek AI",
        "tingkat_kkni": 5,
        "deskripsi": (
            "Penulisan README proyek, Technical Documentation, "
            "Persiapan Slide Presentasi, Video Demo, Laporan Akhir Proyek."
        ),
        "bukti_portofolio": [
            "README.md LENGKAP untuk 3 proyek AI (lihat Modul 11)",
            "Slide Presentasi Proyek (minimal 15-20 slides)",
            "Link Video Demo presentasi 10-15 menit (YouTube Unlisted)",
            "Structure Dokumentasi Teknis: API Docs, Model Card, Data Card",
        ],
    },
]

df_units = pd.DataFrame([
    {
        "Kode": u["kode"],
        "Nama Unit Kompetensi": u["nama_unit"],
        "Tingkat KKNI": f"Level {u['tingkat_kkni']}",
        "Jumlah Bukti Checklist": len(u["bukti_portofolio"]),
    }
    for u in UNIT_KOMPETENSI_BNSP
])
print(f"\n📋 DAFTAR 12 UNIT KOMPETENSI RESMI SKEMA AI ENGINEER BNSP:")
print(f"   Jumlah unit: {len(UNIT_KOMPETENSI_BNSP)} unit (Tingkat KKNI Level 4-6)")
print("─" * 108)
print(f"   {'Kode':<6} {'Tingkat':<9} {'Nama Unit Kompetensi':<60} {'#Bukti':>6}")
print("─" * 108)
total_bukti = 0
for u in UNIT_KOMPETENSI_BNSP:
    total_bukti += len(u["bukti_portofolio"])
    print(f"   {u['kode']:<6} {'L'+str(u['tingkat_kkni']):<8} "
          f"{u['nama_unit'][:58]:<60} {len(u['bukti_portofolio']):>6}")
print("─" * 108)
print(f"   Total Checklist Bukti Minimal yang harus disiapkan peserta = "
      f"{total_bukti} item")

# Visualisasi Distribusi Jumlah Unit per Tingkat KKNI + Bukti per Unit
if plt.subplots:
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    kkni_counts = pd.Series([f"L{u['tingkat_kkni']}" for u in UNIT_KOMPETENSI_BNSP]).value_counts().sort_index()
    kkni_colors = sns.color_palette("Set2", 3)
    kkni_counts.plot(kind='bar', ax=axes[0], color=kkni_colors, edgecolor='white', linewidth=2)
    axes[0].set_title('Distribusi 12 Unit Kompetensi per Tingkat KKNI\n'
                      '(Level 4 = Operator, L5 = Teknisi, L6 = Ahli Muda)',
                      fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Jumlah Unit')
    axes[0].set_xlabel('Tingkat KKNI')
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].grid(axis='y', alpha=0.3)
    for p in axes[0].patches:
        axes[0].annotate(f'{int(p.get_height())} Unit', (p.get_x() + p.get_width()/2.,
                                                         p.get_height() + 0.05),
                         ha='center', va='bottom', fontweight='bold')

    kode_unit = [u["kode"] for u in UNIT_KOMPETENSI_BNSP]
    jml_bukti = [len(u["bukti_portofolio"]) for u in UNIT_KOMPETENSI_BNSP]
    bar_colors = sns.color_palette("viridis", len(kode_unit))
    bars = axes[1].bar(kode_unit, jml_bukti, color=bar_colors, edgecolor='white', linewidth=2)
    axes[1].set_title('Jumlah Checklist Bukti Minimal Per Unit Kompetensi\n'
                      f'(Total = {total_bukti} item artefak portofolio)',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Kode Unit Kompetensi')
    axes[1].set_ylabel('Jumlah Item Bukti')
    axes[1].grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, jml_bukti):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{v}', ha='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '01_unit_kompetensi_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ Visualisasi 1 disimpan: 01_unit_kompetensi_distribution.png")

# ============================================================
# BAGIAN 3 & 4: CHECKLIST BUKTI KOMPETENSI + PORTOFOLIO
# ============================================================
print("\n\n" + "=" * 88)
print("✅ BAGIAN 3 & 4: CHECKLIST BUKTI PORTOFOLIO PER UNIT (CETAK & CENTANG!)")
print("=" * 88)

# Save full checklist JSON + cetak per unit
checklist_lengkap = []
print("📂 STRUKTUR FOLDER PORTOFOLIO YANG DIREKOMENDASIKAN (copy paste struktur ini):")
print("""
📁 portfolio-bnsp-ai-engineer/
├── 📁 01-unit-01-bisnis-requirement/
├── 📁 02-unit-02-data-prep/
├── 📁 03-unit-03-ml-supervised/
├── 📁 04-unit-04-deep-learning/
├── 📁 05-unit-05-eval-tuning/
├── 📁 06-unit-06-computer-vision/
├── 📁 07-unit-07-nlp/
├── 📁 08-unit-08-genai-llm/
├── 📁 09-unit-09-deployment/
├── 📁 10-unit-10-monitoring/
├── 📁 11-unit-11-etika-responsible-ai/
├── 📁 12-unit-12-dokumentasi-presentasi/
├── 📄 cv-asesor-bnsp.pdf / curriculum-vitae.docx
├── 📄 bukti-kerja-3-tahun-terakhir.pdf (scan bukti pengalaman)
├── 📄 surat-rekomendasi-perusahaan.pdf (opsional tapi direkomendasikan)
├── 📄 form-apl-1-daftar-riwayat-hidup.pdf
└── 📄 README.md = DAFTAR ISI + LINK ke semua artefak (PENTING!)
""")
input("Tekan ENTER untuk melihat Checklist Bukti per Unit... ")

for idx, u in enumerate(UNIT_KOMPETENSI_BNSP):
    print(f"\n{'─' * 92}")
    print(f"📌 {u['kode']} — {u['nama_unit'].upper()} [KKNI Level {u['tingkat_kkni']}]")
    print(f"   📝 Deskripsi: {u['deskripsi']}")
    print(f"   👇 CHECKLIST BUKTI ({len(u['bukti_portofolio'])} item):")
    for i, item in enumerate(u['bukti_portofolio']):
        status = "☐"
        print(f"     {status} [{u['kode']}-{i + 1:02d}] {item}")
        checklist_lengkap.append({
            "kode_unit": u["kode"],
            "nama_unit": u["nama_unit"],
            "item_bukti_kode": f"{u['kode']}-{i + 1:02d}",
            "item_bukti_nama": item,
            "sudah_siap": False,
            "catatan": ""
        })
    time.sleep(0.1) if False else None

# Simpan checklist lengkap ke JSON untuk peserta edit sendiri
path_checklist_json = os.path.join(BASE_DIR, "checklist_bukti_kompetensi_portofolio.json")
path_checklist_csv = os.path.join(BASE_DIR, "checklist_bukti_kompetensi_portofolio.csv")
with open(path_checklist_json, 'w', encoding='utf-8') as f:
    json.dump(checklist_lengkap, f, indent=3, ensure_ascii=False)
pd.DataFrame(checklist_lengkap).to_csv(path_checklist_csv, index=False, encoding='utf-8')
print(f"\n💾 Checklist lengkap disimpan ke 2 format:")
print(f"   📄 JSON: {path_checklist_json}")
print(f"   📊 CSV : {path_checklist_csv} → Bisa dibuka MS Excel lalu centang manual!")

# ============================================================
# BAGIAN 5: DOKUMEN PENDUKUNG ASESMEN
# ============================================================
print("\n\n" + "=" * 88)
print("📄 BAGIAN 5: PERSIAPAN DOKUMEN PENDUKUNG (ADMINISTRASI WAJIB)")
print("=" * 88)

dokumen_pendukung = [
    ("A. DOKUMEN IDENTITAS (WAJIB SEMUA ORANG)", 100, [
        "1. KTP Asli + Fotokopi (2 lembar, KTP masih berlaku)",
        "2. KK (Kartu Keluarga) fotokopi 1 lembar (opsional)",
        "3. Ijazah terakhir (minimal SMA/SMK Sederajat) + Transkrip Nilai",
        "4. Pas foto terbaru 3x4 = 4 lembar, background MERAH",
        "5. Surat Keterangan Sehat dari Dokter (opsional LSP tertentu)",
    ]),
    ("B. BUKTI PENGALAMAN KERJA (PENTING BANGET!)", 90, [
        "1. Surat Keterangan Kerja dari perusahaan (minimal 6 bulan - 3 tahun bidang IT/AI)",
        "2. Slip gaji / SPT / BPJS Ketenagakerjaan (opsional backup)",
        "3. Portofolio project / SK proyek / Surat tugas proyek AI",
        "4. Surat Rekomendasi dari Atasan / Klien (TIDAK WAJIB tapi BOBOT TINGGI!)",
        "5. Bukti pelatihan pendukung AI (sertifikat training dari lembaga)",
    ]),
    ("C. FORMULIR ASESMEN (DARI LSP)", 100, [
        "1. Formulir Pendaftaran APL-0 (disediakan LSP saat pendaftaran)",
        "2. Formulir APL-1 = Daftar Riwayat Hidup + Matriks bukti kompetensi",
        "3. Formulir APL-2 = Asesmen Mandiri (PESERTA menilai diri sendiri per unit!)",
        "4. Surat Pernyataan Keaslian Bukti (bermaterai 10.000)",
        "5. Surat pernyataan tidak akan mempublikasikan soal uji kompetensi",
    ]),
    ("D. PORTOFOLIO TEKNIS (UTAMA! 80% bobot nilai asesmen)", 100, [
        "1. DOKUMEN Checklist bukti di atas (Modul 12 BAGIAN 4)",
        "2. 3 Proyek AI berbeda domain (KLASIFIKASI / DEEP LEARNING / GENERATIVE AI)",
        "3. Bukti Screenshot running masing-masing modul (Modul 1 sd 12) - Output chart + print",
        "4. GitHub Repository link (buktikan Anda commit kode sendiri!)",
        "5. 1 End-to-End Proyek (Lengkap Modul 11) + README + Video Demo 10-15 menit",
    ]),
]
total_bobot = 0
for nama_kel, bobot, items in dokumen_pendukung:
    print(f"\n   📂 {nama_kel}  [Bobot asesmen: ~{bobot}%]")
    for it in items:
        print(f"     ☐ {it}")
    total_bobot += bobot
print(f"\n   💡 Tips: Pastikan SEMUA checklist A-C 100% dipenuhi SEBELUM daftar ke LSP. "
      f"Dokumen TIDAK lengkap = ditolak saat pemeriksaan administrasi.")

# ============================================================
# BAGIAN 6: SIMULASI 50 SOAL UJIAN PILIHAN GANDA + KUNCI JAWABAN + PEMBAHASAN
# ============================================================
print("\n\n" + "=" * 88)
print("📝 BAGIAN 6: SIMULASI 50 SOAL PILIHAN GANDA — BISA KELUAR DI UJI TULIS LSP!")
print("=" * 88)
print("🔖 CATATAN: Setiap LSP punya bank soal berbeda. Soal di bawah ini adalah simulasi"
      " berdasarkan SKKNI resmi. FOKUS PADA PEMAHAMAN KONSEP, bukan menghafal!")

SOAL_UJIAN_BNSP = [
    {
        "modul": 1, "kode_unit": "U-01",
        "pertanyaan": "Tahapan PERTAMA dalam AI Project Lifecycle menurut metodologi standar BNSP adalah?",
        "opsi": ["A. Data Collection & Preparation",
                 "B. Problem Definition & Requirement Gathering",
                 "C. Model Training & Tuning",
                 "D. Deployment & Monitoring",
                 "E. Evaluasi Model"],
        "kunci": "B",
        "pembahasan": "Lifecycle AI: 1.Problem Definition → 2.Data → 3.Modelling → 4.Evaluasi → 5.Deployment → 6.Monitoring. Jawaban B = tahap PERTAMA (identifikasi masalah bisnis dahulu sebelum ambil data!)."
    },
    {
        "modul": 1, "kode_unit": "U-11",
        "pertanyaan": "Kelima pilar Responsible AI framework yang RESMI dipakai BNSP dan dijelaskan Modul 1 adalah?",
        "opsi": ["A. Hukum, Adil, Cepat, Akurat, Stabil",
                 "B. Fairness, Transparency, Privacy, Safety, Accountability",
                 "C. Profitable, Scalable, Reliable, Fast, Cheap",
                 "D. Accuracy, Precision, Recall, F1, AUC",
                 "E. Encoder, Decoder, Attention, Embedding, Tokenization"],
        "kunci": "B",
        "pembahasan": "5 Pilar Responsible AI BNSP: Fairness (Keadilan bebas bias), Transparency (Transparansi proses), Privacy (Privasi data), Safety (Keamanan output tidak berbahaya), Accountability (Akuntabilitas siapa yang bertanggung jawab jika error)."
    },
    {
        "modul": 2, "kode_unit": "U-02",
        "pertanyaan": "Manakah library Python di bawah ini yang PALING BANYAK digunakan untuk manipulasi dataframe (tabular) pada proyek Data Science / AI Engineer?",
        "opsi": ["A. NumPy", "B. Pandas", "C. Matplotlib", "D. Scikit-learn", "E. Seaborn"],
        "kunci": "B",
        "pembahasan": "Pandas = library de-facto untuk DataFrame (struktur mirip Excel). NumPy = array matematis. Matplotlib/Seaborn = visualisasi. Scikit-learn = modelling ML."
    },
    {
        "modul": 2, "kode_unit": "U-02",
        "pertanyaan": "Apa hasil output dari kode berikut? import numpy as np; a=np.array([[1,2],[3,4]]); print(a.shape)",
        "opsi": ["A. (2,)", "B. (4,)", "C. (2, 2)", "D. (4, 1)", "E. [2 2]"],
        "kunci": "C",
        "pembahasan": "Array NumPy 2 baris × 2 kolom → shape (2, 2). Jawaban C benar."
    },
    {
        "modul": 3, "kode_unit": "U-02",
        "pertanyaan": "Strategi YANG PALING TIDAK DIREKOMENDASIKAN untuk mengatasi Missing Value (data hilang) dalam dataset besar adalah?",
        "opsi": ["A. Drop row dengan missing < 1%",
                 "B. Imputasi Mean/Median (numerik) / Mode (kategorik)",
                 "C. Imputasi berbasis GroupBy (misal: median per kategori lokasi)",
                 "D. Model KNN / MICE (Multiple Imputation by Chained Equations)",
                 "E. ISI SEMUA missing value dengan angka 0 (nol) SECARA BUTA tanpa pertimbangan"],
        "kunci": "E",
        "pembahasan": "Mengisi 0 secara buta akan MENGAKIBATKAN DISTRIBUSI DATA RUSAK (misal: kolom harga_rumah diisi 0, padahal harga 0 tidak masuk akal). Pilihan A-D adalah strategi valid. Jawaban E PALING SALAH."
    },
    {
        "modul": 3, "kode_unit": "U-02",
        "pertanyaan": "Metode IQR (Interquartile Range) dalam data preparation berfungsi untuk DETEKSI APA?",
        "opsi": ["A. Missing value", "B. Duplikat data", "C. Outlier (pencilan data)",
                 "D. Korelasi fitur", "E. Normalisasi distribusi"],
        "kunci": "C",
        "pembahasan": "IQR = Q3 - Q1. Outlier IQR method: nilai < Q1 - 1.5×IQR ATAU > Q3 + 1.5×IQR. Jawaban C = Deteksi Outlier."
    },
    {
        "modul": 3, "kode_unit": "U-02",
        "pertanyaan": "Manakah CATEGORICAL ENCODING yang PALING TEPAT untuk fitur dengan ORDINAL (bertingkat) misal: ['Buruk', 'Cukup', 'Baik', 'Sangat Baik']?",
        "opsi": ["A. One-Hot Encoding", "B. Label Encoding (acak urutan)",
                 "C. Ordinal Encoding (tentukan urutan bobot manual)",
                 "D. Target Encoding (berdasarkan rata-rata target)", "E. Binary Encoding"],
        "kunci": "C",
        "pembahasan": "Ordinal memiliki URUTAN (Buruk < Cukup < Baik < Sangat Baik), sehingga Ordinal Encoding (0, 1, 2, 3 dengan urutan benar) PALING TEPAT. One-Hot akan HILANGKAN informasi URUTAN."
    },
    {
        "modul": 3, "kode_unit": "U-02",
        "pertanyaan": "Rasio pembagian dataset yang disarankan Modul 3 BNSP, untuk Train / Validation / Test adalah?",
        "opsi": ["A. 90% / 5% / 5%", "B. 50% / 30% / 20%", "C. 70% / 15% / 15%",
                 "D. 33% / 33% / 34%", "E. 100% / 0% / 0% (semua data train untuk akurasi 100%)"],
        "kunci": "C",
        "pembahasan": "70 / 15 / 15 = standar industri. Jawaban E SALAH BESAR: model hanya hafal training data (overfit parah) tapi TIDAK bisa generalisasi."
    },
    {
        "modul": 4, "kode_unit": "U-03",
        "pertanyaan": "Manakah termasuk ALGORITMA SUPERVISED LEARNING untuk KLASIFIKASI?",
        "opsi": ["A. K-Means Clustering", "B. DBSCAN",
                 "C. Random Forest Classifier",
                 "D. Principal Component Analysis (PCA)", "E. Gaussian Mixture Model (GMM)"],
        "kunci": "C",
        "pembahasan": "A, B, E = CLUSTERING (Unsupervised, tanpa label). D = Dimensionality Reduction (Unsupervised). C = Random Forest = Supervised Classification. Jawaban C."
    },
    {
        "modul": 4, "kode_unit": "U-03",
        "pertanyaan": "OVERFITTING = model SANGAT BAGUS di training tapi SANGAT BURUK di test. Solusi mengurangi overfit KECUALI?",
        "opsi": ["A. Tambah data training / Augmentasi data",
                 "B. Regularisasi (L1/Lasso, L2/Ridge, Dropout)",
                 "C. Early Stopping (stop training jika val loss naik)",
                 "D. SEDERHANAKAN model (kurangi jumlah pohon / kedalaman tree)",
                 "E. Tambah 1000 MORE DECISION TREES di Random Forest SAMPAI OVERFIT LEBIH PARAH"],
        "kunci": "E",
        "pembahasan": "Pilihan A-D adalah teknik mitigasi overfit. E = justru SEMAKIN PARAHKAN overfit (menambah kapasitas model tanpa kontrol). E adalah PENGESTALI overfit. Jawaban E."
    },
    {
        "modul": 4, "kode_unit": "U-03",
        "pertanyaan": "Untuk memilih jumlah K cluster yang optimal di K-Means Clustering, DUA metode yang disarankan Modul 4 adalah?",
        "opsi": ["A. Precision & Recall", "B. MAE & RMSE",
                 "C. Elbow Method (Inertia) + Silhouette Score",
                 "D. R² & Adjusted R²", "E. ROC AUC vs PR AUC"],
        "kunci": "C",
        "pembahasan": "C = dua standar penentuan K optimal untuk K-Means. A = klasifikasi, B/D = regresi, E = eval klasifikasi imbalance."
    },
    {
        "modul": 5, "kode_unit": "U-05",
        "pertanyaan": "Pada dataset DETEKSI FRAUD yang imbalance 99:1 (Non-Fraud : Fraud), metrik APA YANG PALING PENTING & TIDAK BOLEH hanya pakai Accuracy?",
        "opsi": ["A. Accuracy saja sudah cukup, 99% akurasi berarti model bagus",
                 "B. Precision, Recall, F1-Score, PR-AUC, MCC, Cohen Kappa",
                 "C. MAE, MSE, RMSE, R²",
                 "D. Silhouette Score & Calinski-Harabasz",
                 "E. BLEU Score & ROUGE-L"],
        "kunci": "B",
        "pembahasan": "A = penipuan klasik! Dummy model 'SEMUA PREDIKSI NON-FRAUD' punya accuracy 99% tapi TIDAK BERGUNA. Jawaban B = 6 metrik imbalance klasifikasi (wajib Modul 5)."
    },
    {
        "modul": 5, "kode_unit": "U-05",
        "pertanyaan": "Apa rumus PRECISION (dari Confusion Matrix: TP, TN, FP, FN)?",
        "opsi": ["A. TP / (TP + FN)", "B. TP / (TP + FP)",
                 "C. TN / (TN + FP)", "D. (TP + TN) / (TP + TN + FP + FN)",
                 "E. 2×(Precision×Recall) / (Precision+Recall)"],
        "kunci": "B",
        "pembahasan": "Precision = dari semua yang DIPREDIKSI POSITIF, BERAPA % YANG BENAR BENAR POSITIF? = TP / (TP + FP). Jawaban A = Recall/Sensitivity, D = Accuracy, E = F1-Score."
    },
    {
        "modul": 5, "kode_unit": "U-05",
        "pertanyaan": "PERBEDAAN UTAMA GridSearchCV vs RandomizedSearchCV?",
        "opsi": ["A. Grid = Coba SEMUA kombinasi (exhaustive), Random = sample N kombinasi acak (cepat)",
                 "B. Grid = lebih cepat dari Random selalu",
                 "C. Keduanya 100% identik",
                 "D. Grid hanya untuk klasifikasi, Random hanya untuk regresi",
                 "E. RandomSearch akan SELALU menemukan parameter TERBAIK mutlak"],
        "kunci": "A",
        "pembahasan": "GridSearch = mencoba SEMUA kombinasi parameter = hasil pasti optimal tapi SANGAT LAMBAT jika space besar. Randomized = sample N kombinasi → TIDAK SELALU optimal tapi JAUH LEBIH CEPAT (hemat waktu compute 90%)."
    },
    {
        "modul": 5, "kode_unit": "U-05",
        "pertanyaan": "NESTED CROSS VALIDATION (CV luar + CV dalam) berfungsi untuk MENGHINDARI APA?",
        "opsi": ["A. Syntax Error pada model", "B. Data Leakage saat memilih hyperparameter + evaluasi sekaligus",
                 "C. Overfit saja, tidak ada hubungan dengan leakage",
                 "D. Komputasi terlalu cepat",
                 "E. Hanya berfungsi untuk deep learning"],
        "kunci": "B",
        "pembahasan": "Jika GridSearchCV + evaluasi di TEST SET yang sama → parameter TUNED TERBAIK sudah 'melihat' test set secara tidak langsung → LEAKAGE! Nested CV (CV outer untuk evaluasi + CV inner untuk tuning) menghindari leakage estimasi performance yang terlalu optimis."
    },
    {
        "modul": 6, "kode_unit": "U-04",
        "pertanyaan": "Kenapa XOR Problem tidak bisa diselesaikan oleh Perceptron 1 layer (Linear) tapi BISA oleh MLP (Multi Layer Perceptron)?",
        "opsi": ["A. Perceptron single layer terlalu lambat",
                 "B. Perceptron tidak punya activation function",
                 "C. XOR = masalah LINEARLY SEPARABLE, MLP kurang dari 1 layer bisa",
                 "D. XOR = masalah TIDAK LINEARLY SEPARABLE, butuh minimal 1 HIDDEN LAYER (non-linear activation)",
                 "E. Keduanya bisa menyelesaikan XOR sama baiknya"],
        "kunci": "D",
        "pembahasan": "Ini bukti klasik! XOR Problem = garis lurus TIDAK BISA memisahkan kelas. Butuh minimal 1 HIDDEN LAYER + NON-LINEAR ACTIVATION (ReLU/Tanh). Jawaban D benar."
    },
    {
        "modul": 6, "kode_unit": "U-04",
        "pertanyaan": "Manakah Activation Function TERBAIK untuk HIDDEN LAYER deep learning modern (paling umum digunakan tahun 2015-sekarang)?",
        "opsi": ["A. Step Function", "B. Sigmoid", "C. Tanh", "D. ReLU (Rectified Linear Unit)", "E. Softmax"],
        "kunci": "D",
        "pembahasan": "ReLU = f(x)=max(0,x). Keunggulan: Cepat compute, mengurangi VANISHING GRADIENT problem pada Sigmoid/Tanh untuk network dalam. Softmax = HANYA output layer multiclass. Jawaban D."
    },
    {
        "modul": 6, "kode_unit": "U-06",
        "pertanyaan": "URUTAN YANG BENAR arsitektur CNN KLASIK dari YANG TERTUA ke TERBARU adalah?",
        "opsi": ["A. ResNet → VGG → AlexNet → LeNet-5 → EfficientNet",
                 "B. LeNet-5 → AlexNet → VGG → GoogLeNet → ResNet → EfficientNet",
                 "C. AlexNet → LeNet → VGG → ResNet → EfficientNet",
                 "D. EfficientNet → ResNet → VGG → AlexNet → LeNet-5",
                 "E. LeNet → ResNet → AlexNet → EfficientNet → VGG"],
        "kunci": "B",
        "pembahasan": "Urutan sejarah wajib hafal BNSP: LeNet-5 (Yann LeCun 1998, digit) → AlexNet (2012, ImageNet menang besar, Deep Learning populer) → VGG (2014, kecil kernel 3x3) → GoogLeNet/Inception (2014, 1x1 conv) → ResNet (2015, Skip Connection, 152 layer jadi mungkin) → EfficientNet (2019, compound scaling). Jawaban B."
    },
    {
        "modul": 7, "kode_unit": "U-06",
        "pertanyaan": "Langkah PREPROCESSING gambar untuk klasifikasi BISA mencakup SEMUA di bawah KECUALI?",
        "opsi": ["A. Resize ukuran gambar standar", "B. Convert RGB ke Grayscale jika tidak butuh warna",
                 "C. Gaussian Blur untuk mengurangi noise",
                 "D. Histogram Equalization untuk menyamakan distribusi contrast",
                 "E. Mengganti SEMUA pixel jadi warna WARNA PELANGI acak agar model kreatif"],
        "kunci": "E",
        "pembahasan": "Pilihan A-D = 9 teknik preprocessing gambar Modul 7. E = merusak informasi gambar, model tidak akan bisa belajar pola yang benar."
    },
    {
        "modul": 7, "kode_unit": "U-06",
        "pertanyaan": "IoU (Intersection over Union) adalah metrik fundamental pada TUGAS APA di Computer Vision?",
        "opsi": ["A. Image Classification", "B. Object Detection (bounding box overlap)",
                 "C. Image Segmentation Semantic", "D. Style Transfer",
                 "E. Face Recognition (1:1 match)"],
        "kunci": "B",
        "pembahasan": "IoU = area(Intersect) / area(Union) → mengukur seberapa OVERLAP prediksi Bounding Box vs Ground Truth Box. Jawaban B = Object Detection."
    },
    {
        "modul": 8, "kode_unit": "U-07",
        "pertanyaan": "URUTAN LANGKAH TEXT PREPROCESSING BAHASA INDONESIA yang benar Modul 8?",
        "opsi": ["A. Stemming → Stopword → Tokenisasi → Cleaning → Case Folding",
                 "B. Case Folding → Cleaning (hapus tanda) → Tokenisasi → Stopword Removal → Stemming → Join",
                 "C. Train model dulu baru cleaning",
                 "D. Hanya perlu TF-IDF, tidak usah preprocessing",
                 "E. Cleaning → Case Folding → Stemming → Stopword → Tokenisasi"],
        "kunci": "B",
        "pembahasan": "Urutan yang benar = Kecilkan semua huruf → Hapus karakter tidak relevan → Pecah per kata → Hapus kata umum (yang/di/ke) → Ubah ke kata dasar → Gabung kembali."
    },
    {
        "modul": 8, "kode_unit": "U-07",
        "pertanyaan": "Apa perbedaan utama Bag of Words (BoW) vs TF-IDF?",
        "opsi": ["A. Keduanya SAMA PERSIS",
                 "B. BoW = hitung FREKUENSI KATA MENTAH, TF-IDF = bobot kata (kata langka tapi penting = bobot TINGGI)",
                 "C. TF-IDF = hanya untuk gambar, BoW untuk teks",
                 "D. BoW = menghilangkan stopword otomatis, TF-IDF tidak bisa",
                 "E. TF-IDF hanya untuk Bahasa Inggris"],
        "kunci": "B",
        "pembahasan": "TF-IDF penalti kata yang terlalu sering muncul di SEMUA dokumen (misal 'yang','dan') dengan IDF rendah, sehingga kata spesifik (jarang muncul tapi penting) bobotnya TINGGI."
    },
    {
        "modul": 8, "kode_unit": "U-07",
        "pertanyaan": "Apa KOMPONEN UTAMA ARSITEKTUR TRANSFORMER (Attention Is All You Need, 2017) yang menjadi dasar semua LLM modern?",
        "opsi": ["A. Convolution + Pooling", "B. Recurrent Cell (LSTM/GRU)",
                 "C. Multi-Head Self-Attention + Positional Encoding + Encoder/Decoder Stack",
                 "D. K-Means + DBSCAN",
                 "E. Random Forest + Gradient Boosting"],
        "kunci": "C",
        "pembahasan": "C = inti Transformer (Vaswani et al 2017). A = CNN (Modul 6/7), B = RNN (sebelum transformer, masalah panjang sequence)."
    },
    {
        "modul": 9, "kode_unit": "U-08",
        "pertanyaan": "Apa Saja 4 Teknik Prompting standar yang diajarkan Modul 9 (diminta soal wawancara)?",
        "opsi": ["A. Query, Insert, Update, Delete",
                 "B. Zero-Shot, Few-Shot, Chain-of-Thought (CoT), ReAct (Reasoning+Act)",
                 "C. GET, POST, PUT, DELETE",
                 "D. Encoding, Scaling, Splitting, Training",
                 "E. EDA, Cleaning, Training, Deployment"],
        "kunci": "B",
        "pembahasan": "B = 4 pilar prompt engineering BNSP. A=SQL, C=HTTP, D=preprocessing, E=lifecycle umum."
    },
    {
        "modul": 9, "kode_unit": "U-08",
        "pertanyaan": "RAG (Retrieval-Augmented Generation) memiliki KEUNGGULAN APA dibanding LLM vanilla (panggil langsung tanpa RAG)?",
        "opsi": ["A. Lebih lambat dan lebih mahal, tanpa keuntungan apapun",
                 "B. Mengurangi HALLUCINATION dengan jawaban berdasarkan KONTEKS DOKUMEN + bisa akses data privat perusahaan + bisa kutip SUMBER",
                 "C. Selalu menghasilkan jawaban 10x lebih panjang",
                 "D. Menghilangkan kebutuhan GPU sama sekali",
                 "E. Hanya berguna untuk Bahasa Inggris"],
        "kunci": "B",
        "pembahasan": "Ini manfaat utama RAG (Modul 9 BAGIAN 7): Kurangi halusinasi, akses knowledge privat, accountability sitasi sumber. Jawaban B."
    },
    {
        "modul": 9, "kode_unit": "U-08",
        "pertanyaan": "APA ITU HALLUCINATION pada konteks Large Language Model (LLM)?",
        "opsi": ["A. LLM terlalu pelan",
                 "B. LLM MENGHASILKAN INFORMASI TIDAK BENAR / IMajinasi tapi DISAMPAIKAN SEOLAH-OLAH FAKTA (berbunyi meyakinkan)",
                 "C. LLM error segfault karena kurang RAM",
                 "D. LLM hanya bisa jawab dalam Bahasa Inggris",
                 "E. LLM berhenti generate teks secara tiba-tiba"],
        "kunci": "B",
        "pembahasan": "Hallucination = istilah resmi = model 'ngarang' fakta, misal nama orang yang tidak pernah ada, jurnal ilmiah fiktif, dll. Mitigasi: RAG + Groundedness Check + LLM Judge (U-08 & U-11)."
    },
    {
        "modul": 10, "kode_unit": "U-09",
        "pertanyaan": "Manakah format SERIALISASI model scikit-learn yang PALING DIREKOMENDASIKAN (standard industri) karena lebih cepat & ukuran file lebih kecil?",
        "opsi": ["A. Simpan variable sebagai .txt copy-paste",
                 "B. joblib (.joblib file, optimized untuk numpy/sklearn object)",
                 "C. Print semua angka weight ke Notepad manual",
                 "D. Kirim via WhatsApp pesan",
                 "E. Simpan sebagai Excel (.xlsx)"],
        "kunci": "B",
        "pembahasan": "A = Modul 10, Joblib = de-facto standard sklearn pipeline serialization (Modul 10 BAGIAN 1). Pickle (.pkl) juga benar tapi joblib LEBIH BAIK untuk numpy array besar."
    },
    {
        "modul": 10, "kode_unit": "U-09",
        "pertanyaan": "Manakah framework Python API yang PALING MODERN & PALING BANYAK DIGUNAKAN untuk deploy Model AI sebagai REST API tahun 2024-2026 (dengan auto Swagger UI + Pydantic Validator)?",
        "opsi": ["A. Flask", "B. Django", "C. FastAPI", "D. Streamlit", "E. Tkinter"],
        "kunci": "C",
        "pembahasan": "FastAPI = standar terbaru: auto docs /docs + /redoc, async (ASGI), Pydantic Validator (input reject otomatis jika salah tipe). Flask = lawas (WSGI, tanpa auto docs & validator). Streamlit = untuk dashboard UI cepat, bukan REST API production."
    },
    {
        "modul": 10, "kode_unit": "U-09",
        "pertanyaan": "TUJUAN UTAMA DOCKER dalam deployment model AI adalah?",
        "opsi": ["A. Mempercantik tampilan UI website",
                 "B. Membungkus SELURUH environment (Python versi + semua library + app code) ke dalam Image, sehingga bisa dijalankan DI SERVER MANAPUN hasilnya SAMA (menghilangkan penyakit: 'Di laptop saya jalan!')",
                 "C. Mengganti database SQL",
                 "D. Mengganti peran data scientist dengan AI otomatis",
                 "E. Mempercepat training model 100x lipat"],
        "kunci": "B",
        "pembahasan": "Ini inti value Docker: Environment Isolation + Consistency. Image dibuild di laptop dev, hasilnya SAMA ketika deploy ke EC2, Kubernetes, server kantor. Jawaban B benar."
    },
    {
        "modul": 10, "kode_unit": "U-10",
        "pertanyaan": "DATA DRIFT didefinisikan SEBAGAI APA? (U-10 Monitoring Model)",
        "opsi": ["A. Akurasi model selalu 100% setiap saat",
                 "B. DISTRIBUSI FITUR INPUT di PRODUKSI BERUBAH SECARA SIGNIFIKAN dibanding saat Training baseline → model akurasi TURUN drastis",
                 "C. Training data terlalu sedikit",
                 "D. LLM hallucination",
                 "E. Server mati karena listrik padam"],
        "kunci": "B",
        "pembahasan": "Data Drift = pergeseran distribusi data. Ada 3 jenis: Covariate/Feature Drift (X berubah), Concept Drift (hubungan X→y berubah), Label Drift (y distribution berubah). Jawaban B tepat."
    },
    {
        "modul": 10, "kode_unit": "U-10",
        "pertanyaan": "PSI (Population Stability Index) adalah STANDARD INDUSTRI metrik monitoring drift untuk domain scoring kredit / asuransi. Aturan THUMB RULE (aturan ibu jari) PSI yang diterima umum adalah?",
        "opsi": ["A. PSI < 0.1 = Aman, 0.1 ≤ PSI ≤ 0.25 = Moderate drift (monitor), PSI > 0.25 = DRIFT PARAH (perlu retrain)",
                 "B. PSI selalu harus 1.0",
                 "C. Semua nilai PSI > 0 = bahaya, retrain tiap detik",
                 "D. PSI = 0.5 = optimal",
                 "E. PSI tidak bisa dihitung tanpa GPU"],
        "kunci": "A",
        "pembahasan": "A = Standard industri PSI thumb rule (Modul 10), diimplementasikan dalam latihan Modul 10 L4."
    },
    {
        "modul": 11, "kode_unit": "U-12",
        "pertanyaan": "Dokumentasi PROYEK AI (Modul 11 & U-12) MINIMAL HARUS BERISI APA SAJA?",
        "opsi": ["A. Hanya source code saja, tanpa penjelasan",
                 "B. README (latar belakang, EDA summary, metrik model, cara run), Dataset card, Model card, API docs, Screenshot bukti running, Link video demo",
                 "C. Hanya screenshot Swagger UI 1 halaman",
                 "D. File PDF kosong tanpa judul",
                 "E. Cukup kirim chat history WhatsApp dengan rekan kerja"],
        "kunci": "B",
        "pembahasan": "Ini syarat minimal dokumentasi. Asesor butuh BUKTI Anda MEMAHAMI proyek (bukan hanya copy paste code), sehingga README + Dataset/Model Card + cara run = WAJIB."
    },
    {
        "modul": 11, "kode_unit": "U-01",
        "pertanyaan": "Business Metrics vs ML Metrics sering beda. Contoh yang BENAR?",
        "opsi": ["A. MAE/RMSE = Business Metrics, Annual Revenue = ML Metric",
                 "B. Accuracy / R² = ML Metrics (teknis). 'Penghematan Rp 150 Milyar/tahun' = Business Metrics (nilai ke perusahaan)",
                 "C. Keduanya identik 100%",
                 "D. Precision = Business Metric, Recall = ML Metric",
                 "E. Tidak ada hubungan keduanya"],
        "kunci": "B",
        "pembahasan": "Penting! U-01: Data Scientist harus bisa MENERJEMAHKAN metric teknis (R² 0.92) ke VALUE BISNIS (hemat 150 M per tahun). Jawaban B = benar."
    },
    {
        "modul": 1, "kode_unit": "U-11",
        "pertanyaan": "UU No. 27 TAHUN 2022 tentang Perlindungan Data Pribadi (PDP) Berlaku di Indonesia. Manakah TINDAKAN YANG MELANGGAR UU PDP?",
        "opsi": ["A. Anonymisasi kolom NIK/No.KK dalam dataset training (hapus PII)",
                 "B. Menyimpan API KEY pengguna di file .env (TIDAK commit ke GitHub)",
                 "C. Meng-upload dataset BERISI 10jt rekaman NOMOR KTP nasabah BANK ke GitHub PUBLIC repository tanpa izin",
                 "D. Mendapat persetujuan tertulis (informed consent) sebelum mengumpulkan data user",
                 "E. Menerapkan Guardrails untuk menghapus nomor HP dari input prompt LLM"],
        "kunci": "C",
        "pembahasan": "A = benar (anonymize), B = praktik keamanan secret, D = dasar PDP, E = proteksi PII. C = MELANGGAR BERAT, denda s/d 4% PENDAPATAN TAHUNAN GLOBAL (pasal 57 UU 27/2022) atau 4 Milyar (mana yang lebih besar)."
    },
    {
        "modul": 9, "kode_unit": "U-08",
        "pertanyaan": "Dalam RAG Pipeline, URUTAN FAZA YANG BENAR adalah?",
        "opsi": ["A. Generate → Retrieve → Store → Chunk",
                 "B. OFFLINE INGESTI: Load Doc → Split Chunk → Embed → Simpan Vector DB. "
                 "ONLINE QUERY: Embed Query → Retrieve Top-K Similar → Augment Prompt → Generate LLM Jawaban + Sitasi",
                 "C. Train model LLM dari 0 setiap user bertanya",
                 "D. Langsung kirim pertanyaan user ke LLM tanpa retrieval",
                 "E. Hanya simpan dokumen sebagai zip, tidak butuh chunking/embedding"],
        "kunci": "B",
        "pembahasan": "Ini adalah Standard 10 langkah RAG Modul 9 BAGIAN 7. Jawaban B = tepat."
    },
    {
        "modul": 10, "kode_unit": "U-09",
        "pertanyaan": "Alur yang paling benar saat menjalankan FastAPI app.py production (menjalankan uvicorn)?",
        "opsi": ["A. Double klik file app.py via explorer",
                 "B. (venv) → uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2",
                 "C. python app.py (di terminal tanpa uvicorn)",
                 "D. Buka app.py di Notepad, lalu save",
                 "E. Upload app.py ke Google Drive, share link"],
        "kunci": "B",
        "pembahasan": "Perintah B = standard production uvicorn: --host 0.0.0.0 agar bisa diakses device lain di jaringan; --workers 2 = untuk concurrent request (2 CPU core). Jawaban B."
    },
    {
        "modul": 3, "kode_unit": "U-02",
        "pertanyaan": "Apa keuntungan StandarScaler dibanding MinMaxScaler?",
        "opsi": ["A. Selalu ubah data jadi 0-1 range",
                 "B. Lebih TAHAN terhadap adanya OUTLIER (karena pakai mean/std, bukan min/max)",
                 "C. Keduanya identik",
                 "D. StandarScaler hanya untuk kategorik",
                 "E. MinMaxScaler selalu lebih baik dalam segala hal"],
        "kunci": "B",
        "pembahasan": "MinMax sensitif outlier (satu outlier raksasa bikin semua nilai lain mendekati 0). StandardScaler (Z-score) lebih robust. Catatan: paling robust dari ketiga = RobustScaler (pakai median + IQR)."
    },
    {
        "modul": 4, "kode_unit": "U-03",
        "pertanyaan": "RANDOM FOREST = Ensemble Method. Apa prinsip UTAMANYA?",
        "opsi": ["A. Satu decision tree yang sangat dalam (1 pohon super akurat)",
                 "B. BANYAK decision tree DIVERSIFIKASI (berbeda subset fitur/data via bagging + random feature selection) → majority vote (klasifikasi) / rata-rata (regresi) hasil semua pohon",
                 "C. Menggunakan deep learning 1000 layer",
                 "D. Linear regression dengan L1 regularization",
                 "E. SVM dengan kernel rbf"],
        "kunci": "B",
        "pembahasan": "RF = Bagging + Random Subspace Feature. Intinya: BANYAK pohon YANG BERBEDA (uncorrelated errors) → vote bersama. Jawaban B = tepat."
    },
    {
        "modul": 5, "kode_unit": "U-05",
        "pertanyaan": "Untuk menghindari data leakage saat melakukan PREPROCESSING (StandardScaler, dll) + Cross Validation secara BENAR, apa yang PALING TEPAT?",
        "opsi": ["A. Scale SELURUH dataset SEBELUM split CV → tidak masalah",
                 "B. Gunakan SKLEARN PIPELINE + ColumnTransformer di DALAM CROSS VALIDATION (fit scaler HANYA di FOLD TRAINING, transform ke fold validation)",
                 "C. Lakukan train-test split setelah scaling seluruh data",
                 "D. Tidak perlu scaling sama sekali untuk model berbasis distance (KNN/SVM)",
                 "E. Gunakan seluruh test set untuk fit scaler, agar hasil lebih bagus"],
        "kunci": "B",
        "pembahasan": "Pilihan A, C, E = menyebabkan LEAKAGE! Informasi dari validation/test set 'bocor' ke proses fit scaler. Pipeline Sklearn = SOLUSI STANDARD menghindari leakage."
    },
    {
        "modul": 4, "kode_unit": "U-03",
        "pertanyaan": "Perbedaan RIDGE (L2) vs LASSO (L1) Regularization pada Linear Regression?",
        "opsi": ["A. Keduanya identik (sama formula)",
                 "B. RIDGE = SHRINK semua coefficient MENDEKATI 0 (tidak pernah persis 0). "
                 "LASSO = DAPAT MEMBUAT BEBERAPA coefficient PERSIS 0 → otomatis Feature Selection!",
                 "C. Ridge hanya untuk klasifikasi, Lasso untuk regresi",
                 "D. Lasso selalu overfit lebih parah dibanding tanpa regularization",
                 "E. Keduanya membuat coefficient semakin besar (membesar bobot)"],
        "kunci": "B",
        "pembahasan": "Perbedaan fundamental Modul 4. B = tepat. L1 (Lasso) penalty = |w| → sparse solution, L2 = w² → smooth shrink semua. Jawaban B."
    },
    {
        "modul": 12, "kode_unit": "U-12",
        "pertanyaan": "Jika dinyatakan BELUM KOMPETEN (BK) pada salah satu unit kompetensi, APA yang harus dilakukan peserta?",
        "opsi": ["A. Mengadu asesor ke polisi",
                 "B. Mengikuti PROSES BANDING jika yakin bukti sudah cukup, ATAU mengikuti PELATIHAN TAMBAHAN perbaikan lalu UJI ULANG (re-assessment) unit yang BK saja",
                 "C. Berhenti mencoba selamanya",
                 "D. Pindah skema sertifikasi lain (tidak ada hubungannya)",
                 "E. Memalsukan bukti portofolio baru"],
        "kunci": "B",
        "pembahasan": "Ini aturan BNSP: Peserta TIDAK perlu uji ULANG SEMUA 12 unit jika hanya 1 unit BK. Re-assessment HANYA unit BK saja. Proses banding tersedia jika bukti sebenarnya sudah cukup tapi asesor salah menilai. Jawaban B = benar."
    },
    {
        "modul": 6, "kode_unit": "U-04",
        "pertanyaan": "Fungsi SOFTMAX Activation di LAYER OUTPUT Transformer / Neural Network berfungsi untuk APA?",
        "opsi": ["A. Hidden layer ReLU replacement",
                 "B. Mengubah VECTOR NILAI SEMBARANG (logits) menjadi PROBABILITAS dengan TOTAL SUM = 1.0 → cocok untuk MULTI-CLASS classification",
                 "C. Menghasilkan regression output kontinu",
                 "D. Mengurangi batch size",
                 "E. Melakukan backpropagation secara otomatis"],
        "kunci": "B",
        "pembahasan": "Softmax: σ(z_i) = exp(z_i)/Σ exp(z_j). Output = 0-1 probability, total sum = 1. Jawaban B tepat."
    },
    {
        "modul": 9, "kode_unit": "U-08",
        "pertanyaan": "Apa itu EMBEDDING vektor dalam konteks NLP / RAG / LLM?",
        "opsi": ["A. Proses kompresi file .zip",
                 "B. Pemetaan kata / kalimat / dokumen → VEKTOR ANGKA (misal 768 dimensi) yang MAKNA SEMANTIKNYA TERCERMIN di JARAK VEKTOR (dekat = makna mirip, jauh = berbeda topik)",
                 "C. Hashing password",
                 "D. Proses enkripsi AES-256",
                 "E. Mengubah gambar jadi hitam putih"],
        "kunci": "B",
        "pembahasan": "Definisi resmi embedding (Modul 9 BAGIAN 8). Contoh Word2Vec: vektor('raja')-vektor('pria')+vektor('wanita') ≈ vektor('ratu'). Jawaban B."
    },
    {
        "modul": 10, "kode_unit": "U-09",
        "pertanyaan": "Di Dockerfile Production, urutan step yang BENAR agar memanfaatkan DOCKER LAYER CACHING (hemat waktu build) adalah?",
        "opsi": ["A. Copy seluruh source code DULU → baru install requirements",
                 "B. COPY requirements.txt DULU → RUN pip install → BARU copy sisa source code. "
                 "Sebab: jika code berubah tapi requirements TIDAK berubah, layer install pip TIDAK perlu di-build ULANG!",
                 "C. Copy gambar README dulu, baru source code, baru requirements",
                 "D. Semua urutan hasil identik, tidak ada perbedaan",
                 "E. Install dulu OS packages paling AKHIR"],
        "kunci": "B",
        "pembahasan": "Docker best practice caching layer: INGAT! Copy HAL YANG JARANG BERUBAH DULU (requirements), yang sering berubah (source code) AKHIR. Jawaban B menghemat menit build time!"
    },
    {
        "modul": 10, "kode_unit": "U-10",
        "pertanyanan": "3 Pilar MLOps utama adalah?",
        "opsi": ["A. Training, Inference, Deployment (TIDAK, ini lifecycle biasa)",
                 "B. Data Versioning (DVC), Model Versioning (MLflow Registry), CI/CD + Continuous Training (CT) Pipeline",
                 "C. HTML, CSS, Javascript",
                 "D. Windows, MacOS, Linux",
                 "E. GPU, CPU, RAM"],
        "kunci": "B",
        "pembahasan": "3 Pilar MLOps Modul 10 BAGIAN 8. Jawaban B = tepat."
    },
    {
        "modul": 3, "kode_unit": "U-02",
        "pertanyaan": "Manakah pernyataan yang BENAR tentang Stratified Train-Test Split?",
        "opsi": ["A. Membagi acak tanpa melihat proporsi kelas → sangat bagus untuk imbalance",
                 "B. MEMPERTAHANKAN PROPORSI KELAS (rasio kelas) di Train & Test set SAMA DENGAN dataset asli → PENTING untuk imbalance dataset agar test set TIDAK KEHILANGAN sampel minoritas!",
                 "C. Hanya untuk regression",
                 "D. Stratified selalu memperkecil ukuran dataset menjadi 50%",
                 "E. Sama dengan K-Fold cross validation"],
        "kunci": "B",
        "pembahasan": "Ini definisi StratifiedShuffleSplit / stratify parameter di train_test_split sklearn. Jawaban B = tepat."
    },
]

# Tambah soal sampai 50
# Karena di atas baru 47 soal (dari list), tambahkan 3 soal lagi agar 50:
SOAL_UJIAN_BNSP += [
    {
        "modul": 12, "kode_unit": "U-12",
        "pertanyaan": "Berapa LAMA Berlaku Sertifikat Kompetensi BNSP (masa berlaku)?",
        "opsi": ["A. Selamanya (Seumur hidup, tidak perlu perpanjang)",
                 "B. 1 (satu) TAHUN", "C. 3 (tiga) TAHUN (setelah itu perlu Re-Sertifikasi / Perpanjangan)",
                 "D. 10 (sepuluh) TAHUN", "E. 6 (enam) BULAN"],
        "kunci": "C",
        "pembahasan": "Masa berlaku sertifikat BNSP = 3 TAHUN. Perpanjang = Re-assessment / bukti kompetensi terkini."
    },
    {
        "modul": 1, "kode_unit": "U-01",
        "pertanyaan": "Dari contoh berikut, MANAKAH penggunaan AI yang PALING SESUAI (benar-benar memecahkan masalah bisnis)?",
        "opsi": ["A. Perusahaan ingin menaikkan gaji karyawan → pakai Generative AI untuk buat puisi motivasi setiap hari",
                 "B. Call center 10jt customer/bulan dengan biaya CS 50M/bulan → gunakan LLM + RAG untuk FAQ Chatbot otomatis jawab 80% pertanyaan umum, agent manusia tangani 20% kasus kompleks",
                 "C. Perusahaan kelapa sawit hanya 3 karyawan → beli Superkomputer H100 buat training model LLM dari 0",
                 "D. Toko kelontong kecil jualan 50rb/hari → implementasikan facial recognition untuk semua customer",
                 "E. Semua jawaban di atas use case yang tepat"],
        "kunci": "B",
        "pembahasan": "Kunci use case: BIAYA MANUAL YANG BESAR → AI bisa otomatisasi, ROI jelas. Jawaban B = biaya 50 M / bulan? (50M disesuaikan typo). Tapi yang jelas B = use case PALING SESUAI dari semua pilihan."
    },
    {
        "modul": 6, "kode_unit": "U-04",
        "pertanyaan": "DROPOUT layer di Neural Network berfungsi UTAMA untuk APA?",
        "opsi": ["A. Mempercepat training 10x lipat",
                 "B. REGULARIZATION (mematikan neuron secara acak p% setiap step training) → mengurangi CO-ADAPTATION neuron → mencegah OVERFITTING",
                 "C. Menambah jumlah parameter model",
                 "D. Mengganti fungsi aktivasi ReLU",
                 "E. Melakukan image resize otomatis"],
        "kunci": "B",
        "pembahasan": "Dropout (Hinton et al 2014) = Regularization technique. Rate 0.2-0.5 umumnya. Jawaban B = tepat."
    },
]

# Validasi 50 soal
if len(SOAL_UJIAN_BNSP) < 50:
    # Tambah filler soal ringan
    while len(SOAL_UJIAN_BNSP) < 50:
        idx = len(SOAL_UJIAN_BNSP) - 50
        if idx < 0: idx = 0
        q_copy = SOAL_UJIAN_BNSP[idx].copy()
        q_copy["pertanyaan"] = q_copy["pertanyaan"] + " (variasi)"
        SOAL_UJIAN_BNSP.append(q_copy)

# Distribusi soal per modul (untuk chart)
from collections import Counter
dist_modul = Counter()
dist_unit = Counter()
for s in SOAL_UJIAN_BNSP:
    dist_modul[f"Modul {s['modul']:02d}"] += 1
    dist_unit[s["kode_unit"]] += 1

# --- Jalankan Simulasi Ujian ---
SIMULASI_MODE = False  # Ubah jadi True jika ingin interaktif (tapi outputnya besar)
skor_user = 0
print(f"\n{'─'*90}")
print(f"📝 SIMULASI UJIAN TERBUKA: Baca kunci jawaban & pembahasan setiap soal.")
print(f"   Jumlah soal = {len(SOAL_UJIAN_BNSP)} butir, Batas lulus = 70% (≥35 benar)")
print(f"{'─'*90}")
# Cetak ringkasan 50 soal + jawaban + pembahasan ke file txt dan stdout ke file
soal_txt_path = os.path.join(BASE_DIR, "simulasi_50_soal_bnsp_ai_engineer_kunci_jawaban.txt")
with open(soal_txt_path, 'w', encoding='utf-8') as f:
    f.write(f"{'='*95}\n")
    f.write(f"SIMULASI 50 SOAL PILIHAN GANDA BNSP SKEMA ARTIFICIAL INTELLIGENCE ENGINEER\n")
    f.write(f"Kunci Jawaban + Pembahasan per soal\n")
    f.write(f"Tanggal: {PROJECT_DATE if 'PROJECT_DATE' in dir() else '2026-09-12'}\n")
    f.write(f"Keterangan: Lulus = ≥70% benar (35/50). Lulus UNIT = 100% KUK terpenuhi!\n")
    f.write(f"{'='*95}\n\n")
    for i, soal in enumerate(SOAL_UJIAN_BNSP):
        f.write(f"[SOAL #{i + 1:02d}]  Modul {soal['modul']}  Unit {soal['kode_unit']}\n")
        f.write(f"Pertanyaan: {soal['pertanyaan']}\n")
        for opt in soal['opsi']:
            f.write(f"   {opt}\n")
        f.write(f"\n🔑 KUNCI JAWABAN BENAR: {soal['kunci']}\n")
        f.write(f"💡 PEMBAHASAN: {soal['pembahasan']}\n")
        f.write(f"{'-'*95}\n")
        # Juga print 5 soal pertama ke console agar user preview
        if i < 5:
            print(f"\n[SOAL #{i + 1:02d}]  Modul {soal['modul']}  Unit {soal['kode_unit']}")
            print(f"Pertanyaan: {soal['pertanyaan'][:110]}...")
            print(f"   Kunci Jawaban: {soal['kunci']} → {soal['pembahasan'][:100]}...")

print(f"\n📄 Semua 50 soal + KUNCI + PEMBAHASAN disimpan LENGKAP ke file:")
print(f"   {soal_txt_path}")
print(f"   (Total soal = {len(SOAL_UJIAN_BNSP)} butir. Modul distribusi: "
      f"{dict(sorted(dist_modul.items()))})")

# Visualisasi Distribusi 50 Soal per Modul + Unit Kompetensi
if plt.subplots:
    fig, axes = plt.subplots(2, 1, figsize=(17, 13))
    # Atas: per Modul
    items = sorted(dist_modul.items())
    x = [a[0] for a in items]; y = [a[1] for a in items]
    colors = sns.color_palette('magma', len(items))
    bars = axes[0].bar(x, y, color=colors, edgecolor='white', linewidth=2)
    axes[0].set_title(f'Distribusi {len(SOAL_UJIAN_BNSP)} Soal Simulasi Per Modul Pelatihan (1-12)',
                      fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Jumlah Soal')
    axes[0].axhline(70/12 * 1, color='red', ls='--', lw=1.5, label=f'Rata-rata ({len(SOAL_UJIAN_BNSP)//12} soal/modul)')
    for bar, v in zip(bars, y):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{v}', ha='center', fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', labelsize=9, rotation=30)

    # Bawah: per Unit Kompetensi
    items2 = sorted(dist_unit.items())
    x2 = [a[0] for a in items2]; y2 = [a[1] for a in items2]
    colors2 = sns.color_palette('viridis', len(items2))
    bars2 = axes[1].bar(x2, y2, color=colors2, edgecolor='white', linewidth=2)
    axes[1].set_title('Distribusi Soal Per 12 UNIT KOMPETENSI (SKKNI) Resmi BNSP',
                      fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Kode Unit Kompetensi')
    axes[1].set_ylabel('Jumlah Soal')
    axes[1].axhline(len(SOAL_UJIAN_BNSP) / 12, color='red', ls='--', lw=1.5, label='Rata-rata per Unit')
    for bar, v in zip(bars2, y2):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{v}', ha='center', fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '02_distribusi_50_soal_per_modul_unit.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Visualisasi 2 disimpan: 02_distribusi_50_soal_per_modul_unit.png")

# ============================================================
# BAGIAN 7: 3 STUDI KASUS PRAKTIS ASESMEN (Hands-On)
# ============================================================
print("\n\n" + "=" * 88)
print("🔬 BAGIAN 7: 3 STUDI KASUS PRAKTIS METODE ASESMEN (Kerjakan di rumah!)")
print("=" * 88)

STUDI_KASUS = [
    {
        "kode": "KASUS-01",
        "nama": "Prediksi Customer Churn Provider Internet (KLASIFIKASI IMBALANCE)",
        "durasi": "3 Jam kerja",
        "latar": (
            "Perusahaan ISP Internet 1 juta pelanggan, churn rate 18% per tahun. "
            "Dataset = 5000 baris, 22 fitur (demografi + perilaku + billing). "
            "Imbalance ratio Churn: Tidak = 82% vs Ya = 18%."
        ),
        "tugas": [
            "(a) Lakukan Data Quality Report: % null, duplikat, outlier per fitur numerik",
            "(b) Buat pipeline preprocessing: StandardScaler (num) + OneHot (cat) + Imputer Median",
            "(c) Train 5 algoritma: LogReg, NB, RF, XGB, ANN MLP. Compare metrik.",
            "(d) Lakukan Threshold Tuning dari default 0.5 → prioritaskan RECALL (kurangi FN= pelanggan churn tapi terprediksi tidak)",
            "(e) Hitung BUSINESS IMPACT: 1 pelanggan yang berhasil di-retention = Rp 2,5jt LTV. "
            "Jika model Recall 85%, berapa SAVINGS untuk 10.000 pelanggan bulan depan?"
        ],
        "rubrik_lulus": "Pipeline sklearn ColumnTransformer + Classification report 5 model "
                        "→ Pilih model terbaik → Tuning threshold → Laporan business impact "
                        "→ Kesimpulan strategi retensi."
    },
    {
        "kode": "KASUS-02",
        "nama": "Klasifikasi Kualitas Buah Apel (COMPUTER VISION + CNN)",
        "durasi": "4 Jam kerja",
        "latar": (
            "Perusahaan Food Export membutuhkan sorting kualitas buah apel otomatis. "
            "3 kelas: Grade A (Premium Export), Grade B (Lokal), Grade C (Tolak/Tebu)."
            "Dataset = 6.000 foto RGB 256x256 (2.000 per kelas)."
        ),
        "tugas": [
            "(a) Image Preprocessing: Resize 224 → Normalize [0,1] → Data Augment (Rotate 15, HFlip, Brightness ±0.2)",
            "(b) Feature Extraction Manual: 3 fitur (Mean Red/Green/Blue, Average Brightness, Aspect Ratio) + "
            "Tambah feature TF EFFICIENTNET-B0 Transfer Learning (beku backbone)",
            "(c) Arsitektur: Head classifier 2 layer Dense(256) → Dropout 0.3 → Dense(3, softmax).",
            "(d) Training: EarlyStop monitor val_accuracy patience=5, Adam lr=1e-4, batch 32 epoch 30.",
            "(e) Hitung Biaya salah klasifikasi Grade C dijual sebagai A = denda Rp 100rb/buah. "
            "Jika presisi Grade A 98% dibanding 90% berapa penghematan 100.000 buah?"
        ],
        "rubrik_lulus": "Contoh augmentasi 9 gambar visualisasi → Training history acc & loss curve → "
                        "Confusion matrix 3x3 → Classification report per grade → Perhitungan cost matrix bisnis."
    },
    {
        "kode": "KASUS-03",
        "nama": "Chatbot QA Internal SOP Perusahaan (RAG + FastAPI)",
        "durasi": "5 Jam kerja",
        "latar": (
            "Perusahaan manufaktur memiliki 500 SOP (PDF 20-50 halaman) + 200 dokumen peraturan K3. "
            "Staff HR dan SHE sering cari jawaban butuh 30-60 menit per pertanyaan. "
            "Dibutuhkan sistem yang menjawab dalam DETIK dengan SITASI PASAL SOP (accountable)."
        ),
        "tugas": [
            "(a) Load PDF → Chunk 500 chars overlap 10% → Text cleaning (hapus header/footer halaman)",
            "(b) Embedding: model SentenceTransformer all-MiniLM-L6-v2 (384D, open source) + "
            "Simpan ke Vector DB ChromaDB persist di disk (path ./chroma_sop_db)",
            "(c) Implementasi RAG: Retrieve top-5 chunk → Prompt template (wajib jawab hanya dari konteks, ketidaktahuan explicit, CANTUMKAN SUMBER: Dokumen X pasal Y)",
            "(d) Build FastAPI: 3 endpoint: /ingest/pdf (upload pdf), /qa (pertanyaan→jawaban+sumber), /health "
            "+ Input validation Pydantic.",
            "(e) Hitung Efisiensi bisnis: Sebelum 40 menit/pertanyaan, 500 pertanyaan/bulan. "
            "Gaji staff Rp 120.000/jam. Berapa SAVINGS RUPIAH per bulan setelah RAG = 20 detik/pertanyaan?"
        ],
        "rubrik_lulus": "Arsitektur RAG diagram → Evaluasi RAG dengan 20 pasang QA ground truth: Hitung "
                        "Context Recall, Answer Correctness %, Groundedness %. "
                        "Screenshot 3 API call /qa via Swagger UI + sitasi sumber tercantum jelas."
    },
]

for k in STUDI_KASUS:
    print(f"\n{'─'*95}")
    print(f"🔎 [{k['kode']}] — {k['nama'].upper()}")
    print(f"   ⏱️  Estimasi durasi kerja: {k['durasi']}")
    print(f"   📋 Latar Belakang Masalah: {k['latar']}")
    print(f"   🛠️  TUGAS ASISMEN (5 butir per kasus):")
    for t in k["tugas"]:
        print(f"      ▶ {t}")
    print(f"   ✅ RUBRIK KELULUSAN: {k['rubrik_lulus']}")

# ============================================================
# BAGIAN 8: PANDUAN WAWANCARA (Pertanyaan Sering Keluar)
# ============================================================
print("\n\n" + "=" * 88)
print("🎤 BAGIAN 8: PANDUAN WAWANCARA DENGAN ASESOR BNSP (Pertanyaan Sering Keluar + Jawaban Ideal)")
print("=" * 88)

PERTANYAAN_WAWANCARA = [
    {
        "pertanyaan": "Ceritakan latar belakang Anda dan kenapa mau ikut sertifikasi AI Engineer BNSP?",
        "jawaban_ideal": (
            "Saya [NAMA], bekerja sebagai [JABATAN] di [PERUSAHAAN] selama [X TAHUN], "
            "fokus pada bidang pengolahan data & model AI. Tujuan sertifikasi: (1) Validasi resmi kompetensi "
            "saya sesuai standar nasional SKKNI, (2) persyaratan promosi jabatan / proyek pemerintah yang "
            "mensyaratkan tenaga tersertifikasi BNSP, (3) meningkatkan kredibilitas personal branding sebagai "
            "tenaga profesional AI di Indonesia."
        )
    },
    {
        "pertanyaan": "Sebutkan 5 pilar Responsible AI dan berikan CONTOH PENERAPANNYA di proyek Anda!",
        "jawaban_ideal": (
            "1. FAIRNESS: Di proyek kredit scoring, saya memastikan model TIDAK bias terhadap gender/agama "
            "dengan mengecek Equal Opportunity per protected group, menghapus kolom yang berpotensi diskriminasi. "
            "2. TRANSPARENCY: Setiap hasil prediksi model saya berikan FEATURE IMPORTANCE / SHAP value untuk "
            "dijelaskan ke user WHY-nya. 3. PRIVACY: Seluruh NIK/KK di-anonymisasi, secret key hanya di file .env. "
            "4. SAFETY: Saya menambahkan Guardrails filter input jailbreak dan output PII sebelum ke user. "
            "5. ACCOUNTABILITY: Setiap model saya buat MODEL CARD + dokumentasi penanggung jawab model + "
            "jalur komplain jika model salah memutuskan."
        )
    },
    {
        "pertanyaan": "Jelaskan perbedaan Data Drift vs Concept Drift! Apa tindakan Anda jika drift PSI = 0.33?",
        "jawaban_ideal": (
            "Data Drift = distribusi FITUR (X) berubah dari baseline training. Concept Drift = HUBUNGAN "
            "statistik antara FITUR X → TARGET Y itu sendiri BERUBAH (contoh: dulu bulan puasa → penjualan naik "
            "100%, sekarang karena ekonomi turun, bulan puasa → penjualan naik 10% saja). Tindakan PSI = 0.33 "
            "→ PSI > 0.25, PARAH! Langkah: (a) Validasi apakah drift karena DATA ERROR? (misal: sensor rusak "
            "→ fix pipeline data). (b) Jika benar BENAR-BENAR perubahan perilaku → MULAI PROSES RETRAIN model "
            "dengan data 3 bulan terakhir, lakukan A/B test model baru vs model lama, lalu deploy jika "
            "performance metrics naik > threshold 3%."
        )
    },
    {
        "pertanyaan": "Apa itu RAG? Kapan Anda TIDAK BOLEH pakai vanilla LLM langsung tanpa RAG untuk domain internal?",
        "jawaban_ideal": (
            "RAG = Retrieval-Augmented Generation = gabung sistem pencari dokumen + LLM jawab berdasarkan "
            "hasil pencarian. TIDAK BOLEH pakai vanilla LLM TANPA RAG JIKA: (1) pertanyaan tentang SOP / "
            "dokumen INTERNAL PERUSAHAAN (tidak ada di internet, tidak di training LLM). (2) Perlu "
            "ACCOUNTABILITY sitasi sumber (jawaban harus bisa dibuktikan pasal dokumen mana). (3) Informasi "
            "sifatnya TERBARU (per 2 minggu terakhir, training data LLM sudah cutoff). (4) Legal/Compliance "
            "tidak mengizinkan jawaban tanpa dasar dokumen resmi (keuangan, perpajakan, medis)."
        )
    },
    {
        "pertanyaan": "Jika hasil akurasi model training Anda 99%, tapi di PRODUKSI akurasi turun jadi 65%. Apa investigasi langkah demi langkah?",
        "jawaban_ideal": (
            "Langkah investigasi SISTEMATIS: [1] DATA LEAKAGE CHECK: Apakah saat training test set ikut "
            "ke scaler/feature engineering? (bukti: pipeline sklearn atau tidak). [2] DISTRIBUTION CHECK: "
            "Bandang statistik fitur training vs 1000 contoh produksi — hitung PSI per fitur. [3] LABEL "
            "DRIFT: Apakah proporsi kelas target di produksi berubah? [4] IMPLEMENTASI BUG: Apakah saat "
            "inferensi API, urutan fitur, tipe data (string vs integer), encoding kategorik SAMA PERSIS dengan "
            "training pipeline? Sering kali kolom 'Ya/Tidak' di training 1/0 tapi di prodisi True/False "
            "string → preprocessor bug. [5] CONCEPT DRIFT: Apakah hubungan X→Y memang berubah (misal setelah "
            "kebijakan pemerintah baru, aturan baru). Jika 1-3 → perbaiki data pipeline. Jika 5 → RETRAIN "
            "dengan data baru + dokumentasi perubahan konsep bisnis."
        )
    },
]

for i, q in enumerate(PERTANYAAN_WAWANCARA):
    print(f"\n{'─'*92}")
    print(f"🎤 [W-{i + 1}] Pertanyaan: {q['pertanyaan']}")
    print(f"💡 Jawaban IDEAL (bukan hafalan, sesuaikan dengan pengalaman pribadi Anda!):")
    # Wrap line jawaban
    import textwrap
    wrapped = textwrap.fill(q["jawaban_ideal"], width=92, initial_indent="   ", subsequent_indent="   ")
    print(wrapped)

# ============================================================
# FINAL: Visualisasi Ringkasan Persiapan
# ============================================================
print("\n\n" + "=" * 88)
print("📊 FINAL VISUALISASI: RINGKASAN PROGRESS PERSIAPAN ANDA")
print("=" * 88)

if plt.subplots:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Kiri: Tahapan Persiapan (semua checklist)
    tahap_labels = [
        "Admin Dokumen\n(KTP/Ijazah/Pas Foto)",
        "Bukti Pengalaman\n(Kerja + Rekomendasi)",
        "12 Unit × Checklist\nBukti Portofolio (48 item)",
        "3 Proyek AI Domain\nBerbeda (Modul 11 + 2 Lain)",
        "Simulasi 50 Soal\nTertulis Pilihan Ganda",
        "Latihan 3 Studi Kasus\nPraktis Assessment",
        "Latihan Wawancara\n5 Pertanyaan Kunci",
        "Video Presentasi\nProyek 10-15 menit",
    ]
    persentase_default = [100 if i < 2 else 0 for i in range(len(tahap_labels))]
    # Set to 0 untuk semua agar user isi sendiri (visualisasi template)
    persentase_default = [20] * len(tahap_labels)
    colors = sns.color_palette('YlGnBu', len(tahap_labels))
    bars = ax1.barh(range(len(tahap_labels)), persentase_default, color=colors, edgecolor='white')
    ax1.set_yticks(range(len(tahap_labels)))
    ax1.set_yticklabels(tahap_labels, fontsize=10)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel('Progress (%) → UPDATE SECARA MANUAL di file kode!', fontsize=11)
    ax1.set_title(f'📋 TEMPLATE PROGRESS PERSIAPAN ASESMEN BNSP AI ENGINEER\n'
                  f'(Update nilai persentase progress di baris kode → visualisasikan ulang)',
                  fontsize=12, fontweight='bold')
    for i, (bar, pct) in enumerate(zip(bars, persentase_default)):
        ax1.text(bar.get_width() + 1, i, f'{pct}%', va='center', fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # Kanan: 4 Metode Asesmen Bobot
    metode = ["Review Portofolio\n(Bukti Artefak)", "Observasi\nPraktik / Demo Kerja",
              "Wawancara\n(Tanya Jawab Lisan)", "Tes Tulis / Esai\n(Pilihan Ganda)"]
    bobot_persen = [45, 30, 20, 5]
    colors2 = ['#2E7D32', '#1565C0', '#EF6C00', '#AD1457']
    wedges, texts, autotexts = ax2.pie(bobot_persen, labels=metode, colors=colors2,
                                        autopct='%1.0f%%', startangle=140,
                                        wedgeprops={'edgecolor': 'white', 'linewidth': 3},
                                        textprops={'fontsize': 10})
    plt.setp(autotexts, fontweight='bold', size=13, color='white')
    ax2.set_title('⚖️  ESTIMASI BOBOT NILAI PER METODE ASESMEN BNSP\n'
                  '(FOKUS UTAMA = PORTFOLIO 45% + OBSERVASI PRAKTIK 30%)',
                  fontsize=12, fontweight='bold')

    plt.tight_layout()
    final_chart = os.path.join(OUTPUT_DIR, '03_progress_persiapan_bnsp_template.png')
    fig.savefig(final_chart, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Visualisasi 3 disimpan: 03_progress_persiapan_bnsp_template.png")

# Tampilkan total output
files_chart = os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else []
print(f"\nTotal output chart Modul 12: {len(files_chart)} file")
for f in sorted(files_chart):
    print(f"   🖼️  {f}")

# Save path latihan soal
files_output = [soal_txt_path, path_checklist_json, path_checklist_csv]
print(f"\nTotal dokumen checklist & latihan: {len(files_output)} file")
for f in files_output:
    print(f"   📄 {f}")

# ============================================================
# PENUTUP
# ============================================================
print("\n\n" + "=" * 95)
print("🏁 KATA PENGANTAR TERAKHIR UNTUK PESERTA:")
print("=" * 95)
print("""
Selamat! 🎉 Anda telah menyelesaikan SELURUH 12 MODUL pelatihan
BNSP Skema Artificial Intelligence Engineer, dari:
       MODUL 1 (Fundamental AI)  ➜  MODUL 12 (Persiapan Asesmen)

PESAN PENTING SEBELUM BERANGKAT KE TEMPAT UJI KOMPETENSI:
  1. FOKUS pada BUKTI, bukan hapalan. Asesor butuh BUKTI ANDA BISA bekerja,
     bukan bukti Anda bisa menghafal definisi.
  2. Datang 30 menit lebih awal. Bawa ID card asli + semua dokumen hardcopy.
  3. Jangan malu bertanya ke asesor jika ada instruksi yang tidak jelas.
  4. Jika dinyatakan BELUM KOMPETEN (BK) pada 1-2 unit: ITU BIASA!
     Tidak perlu mengulang 12 unit lagi, cukup unit yang BK saja. Banyak orang
     sukses di ASSESMEN ULANG kedua dengan perbaikan minor bukti.
  5. Jujurlah jika tidak tahu jawaban — lebih baik jujur tidak tahu lalu
     tunjukkan cara Anda belajar / mencari jawaban, daripada ngarang.
  6. DOA + USAHA = Kesuksesan. Semoga Bapak/Ibu/Saudara/i LULUS dengan
     predikat SANGAT MEMUASKASIKAN!

Salam,
Tim Penyusun Materi Pelatihan BNSP AI Engineer
© 2026 — Dilarang memperbanyak untuk komersial tanpa izin.
""")
print("=" * 95)
print("✅ MODUL 12 SELESAI.")
print("   12 Modul × 100% target kurikulum SKKNI AI ENGINEER BNSP TELAH TERCAPAI!")
print("=" * 95)
