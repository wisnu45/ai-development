print("="*60)
print("MODUL 1: FUNDAMENTAL ARTIFICIAL INTELLIGENCE")
print("Bagian 3: Etika, Keamanan, dan Responsible AI")
print("="*60)

print("""
┌─────────────────────────────────────────────────────────────────┐
│                    RESPONSIBLE AI FRAMEWORK                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. FAIRNESS (KEADILAN)                                        │
│     ├─ Model tidak diskriminasi berdasarkan: gender, suku,     │
│     │  agama, usia, disabilitas, status ekonomi                │
│     ├─ Bias dalam training data harus diidentifikasi           │
│     ├─ Contoh: Model rekrutmen yang lebih suka laki-laki       │
│     │          karena data training historis didominasi pria   │
│     └─ Mitigasi: Auditing data, fairness metric (equalized     │
│        odds, demographic parity)                               │
│                                                                 │
│  2. TRANSPARENCY & EXPLAINABILITY (KETERBUKAAN)                │
│     ├─ "Black Box" harus bisa dijelaskan ke stakeholder        │
│     ├─ User berhak tahu:                                       │
│     │  ├─ Mengapa saya ditolak pinjaman?                       │
│     │  ├─ Mengapa iklan ini muncul untuk saya?                 │
│     └─ Tools: SHAP, LIME, Feature Importance                   │
│                                                                 │
│  3. PRIVACY & DATA PROTECTION (PRIVASI)                        │
│     ├─ Kepatuhan: UU PDP (Perlindungan Data Pribadi) No.27/2022│
│     ├─ GDPR, CCPA untuk luar negeri                            │
│     ├─ Prinsip: Data Minimization, Purpose Limitation         │
│     ├─ Anonymization / Pseudonymization                       │
│     └─ Federated Learning (model datang ke data, bukan         │
│        sebaliknya)                                             │
│                                                                 │
│  4. SAFETY & ROBUSTNESS (KEAMANAN & KETAHANAN)                 │
│     ├─ Model tahan terhadap adversarial attack                 │
│     ├─ Tidak menghasilkan output berbahaya                     │
│     ├─ Fallback mechanism jika model gagal                     │
│     └─ Monitoring performa di produksi                         │
│                                                                 │
│  5. ACCOUNTABILITY (PERTANGGUNGJAWABAN)                        │
│     ├─ Siapa yang bertanggung jawab jika model salah?          │
│     ├─ Developer? Data Scientist? Perusahaan?                  │
│     ├─ Governance structure & human-in-the-loop                │
│     └─ Audit trail (jejak keputusan model)                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")

input("\nTekan Enter untuk melanjutkan...")

print("""
================================================================
CONTOH KASUS PELANGGARAN ETIKA AI
================================================================

KASUS 1: BIAS REKRUTMEN AMAZON (2018)
├─ Masalah: Model AI penilai resume menurunkan skor CV wanita
│  karena data training 10 tahun terakhir didominasi pria
├─ Dampak: Diskriminasi gender
└─ Solusi: Proyek dihentikan total, model tidak pernah deploy

────────────────────────────────────────────────────────────────

KASUS 2: FACIAL RECOGNITION BIAS
├─ Masalah: Akurasi deteksi wajah orang kulit gelap 30% lebih
│  rendah dibanding kulit putih (MIT Gender Shades Study)
├─ Dampak: Salah tangkap, ketidakadilan hukum
└─ Solusi: Diversitas data training, testing lintas demografi

────────────────────────────────────────────────────────────────

KASUS 3: CAMBRIDGE ANALYTICA + FACEBOOK (2018)
├─ Masalah: Data 87 juta user diambil tanpa izin untuk
│  microtargeting iklan politik
├─ Dampak: Pelanggaran privasi skala besar, manipulasi opini
└─ Solusi: Denda GDPR milyaran dollar, regulasi ketat

================================================================
PRINSIP-PRINSIP UN ETIKAL AI (AI4People / UNESCO)
================================================================

1.  Hak Asasi Manusia, Non-Diskriminasi, Keadilan
2.  Keselamatan, Keamanan, dan Kesejahteraan
3.  Transparansi dan Keterbukaan
4.  Partisipasi dan Inklusivitas
5.  Tanggung Jawab dan Akuntabilitas
6.  Perlindungan Data dan Privasi
7.  Keberlanjutan (Sustainability)

================================================================
PANDUAN PRAKTIS UNTUK DEVELOPER:
================================================================

✅ SEBELUM TRAINING:
   • Periksa bias dalam dataset
   • Pastikan consent untuk penggunaan data
   • Identifikasi risiko potensial model

✅ SAAT TRAINING:
   • Gunakan fairness metric
   • Dokumentasikan semua asumsi
   • Hati-hati dengan proxy variable
     (contoh: kode pos → proxy etnis/sosial)

✅ SAAT DEPLOYMENT:
   • Berikan penjelasan keputusan model
   • Sediakan channel untuk komplain
   • Tetapkan batas penggunaan model
   • Human oversight untuk keputusan kritis (hukum, medis)

================================================================
LATIHAN:
Anda membuat model AI penilai kredit untuk bank. Model ini
menolak pengajuan kredit 80% orang dari daerah X, padahal
secara historis tingkat default daerah X sama dengan daerah lain.

1. Apa masalah etika di sini?
2. Apa kemungkinan penyebabnya?
3. Bagaimana solusinya?
================================================================
""")

print("\n✓ Bagian 3 Selesai: Etika & Responsible AI")
print("\n" + "="*60)
print("MODUL 1 SELESAI - Fundamental Artificial Intelligence")
print("="*60)
