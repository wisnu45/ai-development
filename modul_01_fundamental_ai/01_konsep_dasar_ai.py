print("="*60)
print("MODUL 1: FUNDAMENTAL ARTIFICIAL INTELLIGENCE")
print("Bagian 1: Konsep Dasar AI, ML, DL, dan Generative AI")
print("="*60)

print("""
┌─────────────────────────────────────────────────────────────┐
│                   Cara Kerja AI                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Artificial Intelligence (AI)                              │
│   ├─ Konsep: Sistem yang meniru kecerdasan manusia         │
│   ├─ Cakupan: SELURUH teknologi cerdas                     │
│   │                                                         │
│   ├─── Machine Learning (ML)                                │
│   │    ├─ Konsep: AI yang belajar dari data tanpa          │
│   │    │         pemrograman eksplisit                     │
│   │    ├─ Tipe: Supervised, Unsupervised, Reinforcement    │
│   │    │                                                   │
│   │    └─── Deep Learning (DL)                             │
│   │         ├─ Konsep: ML dengan Neural Network berlapis   │
│   │         ├─ Arsitektur: CNN, RNN, Transformer           │
│   │         │                                              │
│   │         └─── Generative AI                             │
│   │              ├─ Konsep: DL yang menghasilkan konten    │
│   │              │         baru (teks, gambar, audio)      │
│   │              └─ Contoh: LLM (GPT, BERT), DALL-E, SD   │
│   │                                                         │
│   └─ AI Tradisional (Rule-based, Expert System)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
""")

input("\nTekan Enter untuk melanjutkan...")

print("""
================================================================
PERBANDINGAN: TRADITIONAL PROGRAMMING vs MACHINE LEARNING
================================================================

Traditional Programming:                    Machine Learning:
┌─────────────────────┐                     ┌─────────────────────┐
│  Data + Rules       │                     │  Data + Output      │
│         │           │                     │         │           │
│         ▼           │                     │         ▼           │
│      Output         │                     │      Rules / Model  │
└─────────────────────┘                     └─────────────────────┘

Contoh: Deteksi Email SPAM
- Traditional: Buat aturan manual (jika ada kata "menang hadiah",
  jika pengirim tidak dikenal, dll.)
- ML          : Beri 10.000 email berlabel (spam/ham),
                model BELAJAR pola sendiri
================================================================
""")

input("\nTekan Enter untuk melanjutkan...")

print("""
================================================================
JENIS-JENIS PENDEKATAN AI
================================================================

1. RULE-BASED AI
   └─ Sistem pakar dengan aturan if-then eksplisit
   └─ Contoh: Sistem diagnosa penyakit sederhana

2. MACHINE LEARNING
   ├─ SUPERVISED LEARNING (Dengan label)
   │  ├─ Classification (Kategori): Spam/Ham, Kanker Ganas/Jinak
   │  └─ Regression (Nilai kontinu): Harga rumah, Prediksi curah hujan
   │
   ├─ UNSUPERVISED LEARNING (Tanpa label)
   │  ├─ Clustering: Segmentasi pelanggan
   │  └─ Dimensionality Reduction: PCA, t-SNE
   │
   └─ REINFORCEMENT LEARNING
      └─ Agent belajar melalui reward/hukum dari environment
      └─ Contoh: AI bermain chess, self-driving car

3. DEEP LEARNING
   ├─ CNN (Convolutional Neural Network)  → Gambar (Computer Vision)
   ├─ RNN/LSTM                             → Teks & Sequential (NLP)
   ├─ Transformer                          → LLM, Generative AI
   └─ GAN (Generative Adversarial Network) → Generate gambar

4. GENERATIVE AI
   └─ Menghasilkan konten baru yang realistis
   └─ Text Generation, Image Generation, Audio Synthesis
================================================================
""")

print("\n✓ Bagian 1 Selesai: Konsep Dasar AI")
