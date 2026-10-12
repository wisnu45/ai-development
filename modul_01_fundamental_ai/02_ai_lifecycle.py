print("="*60)
print("MODUL 1: FUNDAMENTAL ARTIFICIAL INTELLIGENCE")
print("Bagian 2: AI Lifecycle & Problem Solving")
print("="*60)

print("""
┌─────────────────────────────────────────────────────────────────┐
│                    AI PROJECT LIFECYCLE                        │
│                   (Siklus Hidup Proyek AI)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ╔══════════════════╗                                           │
│  ║  1. PROBLEM      ║ ←──── "Apa masalah yang diselesaikan?"   │
│  ║  DEFINITION      ║       - Apakah cocok pakai AI?           │
│  ╚══════╤═══════════╝       - Business objective jelas?        │
│         ▼                                                     │
│  ╔══════╧═══════════╗                                           │
│  ║  2. DATA         ║ ←──── "Data apa yang dibutuhkan?"       │
│  ║  COLLECTION      ║       - Sumber data?                     │
│  ╚══════╤═══════════╝       - Kualitas & kuantitas?            │
│         ▼                                                     │
│  ╔══════╧═══════════╗                                           │
│  ║  3. DATA         ║ ←──── "Data siap dipakai?"              │
│  ║  PREPROCESSING   ║       - Cleaning, encoding, scaling      │
│  ╚══════╤═══════════╝       - EDA (Exploratory Data Analysis)  │
│         ▼                                                     │
│  ╔══════╧═══════════╗                                           │
│  ║  4. MODELING     ║ ←──── "Algoritma apa yang tepat?"       │
│  ║                  ║       - Training, validasi               │
│  ╚══════╤═══════════╝       - Overfitting? Underfitting?       │
│         ▼                                                     │
│  ╔══════╧═══════════╗                                           │
│  ║  5. EVALUATION   ║ ←──── "Seberapa bagus modelnya?"        │
│  ║                  ║       - Metrik evaluasi (Acc, F1, AUC)   │
│  ╚══════╤═══════════╝       - A/B testing?                      │
│         ▼                                                     │
│  ╔══════╧═══════════╗                                           │
│  ║  6. DEPLOYMENT   ║ ←──── "Model dipakai di produksi"       │
│  ║                  ║       - API / Microservice               │
│  ╚══════╤═══════════╝       - Container (Docker)               │
│         ▼                                                     │
│  ╔══════╧═══════════╗                                           │
│  ║  7. MONITORING   ║ ←──── "Model masih akurat?"             │
│  ║  & MAINTENANCE   ║       - Data drift / Concept drift       │
│  ╚══════════════════╝       - Retraining berkala               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")

input("\nTekan Enter untuk melanjutkan ke studi kasus...")

print("""
================================================================
STUDI KASUS: IDENTIFIKASI KEBUTUHAN BISNIS → AI SOLUTION
================================================================

KASUS 1: TOKO ONLINE MAJU JAYA
├─ Masalah: Tingkat pengembalian barang (return) tinggi 25%
├─ Alasan: Ukuran baju tidak sesuai
│
├─ Problem Definition:
│  └─ Bisnis: Kurangi return rate minimal 10%
│  └─ AI: Prediksi ukuran baju yang cocok untuk customer
│
├─ Data Yang Dibutuhkan:
│  ├─ Data customer (tinggi, berat, jenis kelamin)
│  ├─ Data transaksi pembelian & return historis
│  └─ Data ukuran baju setiap merek
│
├─ Jenis Pendekatan AI:
│  └─ Supervised Learning → Classification (S/M/L/XL)
│
└─ Success Metric:
   └─ Akurasi prediksi > 85%, Return rate turun menjadi < 15%

────────────────────────────────────────────────────────────────

KASUS 2: BANK MAKMUR SEJAHTERA
├─ Masalah: Banyak kredit macet (non-performing loan)
├─ Kebutuhan: Deteksi dini calon debitur yang berisiko
│
├─ Problem Definition:
│  └─ Prediksi probabilitas default calon debitur
│
├─ Data Yang Dibutuhkan:
│  ├─ Data nasabah (umur, penghasilan, pekerjaan)
│  ├─ Riwayat pembayaran pinjaman
│  └─ Data skor kredit
│
├─ Jenis Pendekatan AI:
│  └─ Supervised Learning → Binary Classification
│     (High Risk / Low Risk)
│
└─ Success Metric:
   └─ Recall > 90% (minim false negative)

================================================================
LATIHAN (Diskusi Mandiri):
1. Sebuah rumah sakit ingin memprediksi pasien yang berisiko
   diabetes tipe 2. Tentukan Problem Definition, Data, Jenis AI,
   dan Success Metric!

2. Sebuah platform e-learning ingin memberikan rekomendasi
   kursus yang relevan kepada user. Jenis pendekatan AI
   apa yang cocok?

3. Sebuah perusahaan ingin mengotomatiskan klasifikasi
   tiket customer service ke dalam kategori: Billing,
   Technical, Account. Solusi AI apa?
================================================================
""")

print("\n✓ Bagian 2 Selesai: AI Lifecycle")
