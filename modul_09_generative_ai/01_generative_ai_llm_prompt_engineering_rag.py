# =====================================================================
# MODUL 9 - GENERATIVE AI & LARGE LANGUAGE MODEL (LLM)
# Persiapan Uji Kompetensi BNSP Skema Artificial Intelligence Engineer
# =====================================================================
# Materi:
# 1. Konsep Generative AI vs Traditional AI
# 2. Large Language Model (LLM)
# 3. Konsep Transformer & kaitan dengan LLM
# 4. Prompt Engineering + Teknik-teknik prompting
# 5. Structured Prompting & Context Management
# 6. Pemanfaatan API model AI
# 7. Retrieval-Augmented Generation (RAG)
# 8. Embedding & Vector Database
# 9. Membangun aplikasi berbasis LLM (Chatbot)
# 10. Evaluasi output & Responsible AI untuk LLM
# =====================================================================

import os
import re
import json
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------
# INSTALASI DEPENDENSI (jika error ModuleNotFoundError, jalankan ini):
#   cd c:\ai-engineer
#   python -m venv venv_bnsp
#   venv_bnsp\Scripts\activate        (CMD)  atau  .\venv_bnsp\Scripts\Activate.ps1 (PS)
#   pip install -r requirements.txt
# ---------------------------------------------------------------------
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
except ImportError as e:
    print("=" * 80)
    print("⚠️  MODUL TIDAK BISA BERJALAN - DEPENDENSI TIDAK DITEMUKAN")
    print("=" * 80)
    print(f"Penyebab: {e}")
    print("\nSolusi: Jalankan perintah BERIKUT di CMD/PowerShell (LUAR TRAE):")
    print("  cd /d C:\\ai-engineer")
    print("  venv_bnsp\\Scripts\\activate")
    print("  pip install -r requirements.txt")
    print("\nSetelah install selesai, jalankan script ini LAGI.")
    print("=" * 80)
    import sys
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output_charts')
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# BAGIAN 1 - KONSEP GENERATIVE AI vs TRADITIONAL AI
# ============================================================
print("=" * 80)
print("BAGIAN 1: KONSEP GENERATIVE AI vs TRADITIONAL AI")
print("=" * 80)

print("""
----------------------------------------------------------------------
1.1 APA ITU GENERATIVE AI?
----------------------------------------------------------------------
Generative AI adalah cabang AI yang mempelajari pola dari data training 
(lisan, tulisan, gambar, audio, video) lalu MENGHASILKAN KONTEN BARU 
yang serupa dengan data asli, bukan hanya memprediksi label/regresi 
seperti AI tradisional.

Contoh output Generative AI:
  • Teks: Esai, puisi, kode program, ringkasan dokumen (GPT, Claude)
  • Gambar: Ilustrasi, desain produk, foto sintetis (DALL-E, Midjourney, Stable Diffusion)
  • Audio: Musik baru, voice cloning (Suno, ElevenLabs)
  • Video: Video animasi dari teks (Runway, Sora)
  • Kode: Generator fungsi Python dari deskripsi (GitHub Copilot)
""")

# --- Tabel Perbandingan Traditional AI vs Generative AI ---
df_compare = pd.DataFrame({
    'Dimensi': [
        'Tujuan utama',
        'Output',
        'Contoh tugas',
        'Model terkenal',
        'Fase training',
        'Evaluasi kualitas',
        'Risiko utama',
        'Pendekatan data',
    ],
    'Traditional AI': [
        'Klasifikasi / Prediksi',
        'Label kelas, angka regresi, cluster',
        'Churn prediction, deteksi fraud, klasifikasi gambar',
        'Random Forest, XGBoost, SVM, MLP klasifikasi',
        'Supervised (labeled) / Unsupervised',
        'Accuracy, F1, MAE, R², Silhouette (obyektif, terukur)',
        'Bias data, underfit/overfit',
        'Butuh dataset berlabel (supervised)',
    ],
    'Generative AI': [
        'Membuat / Menghasilkan konten BARU',
        'Teks panjang, gambar, audio, video, kode',
        'Chatbot, ringkasan PDF, desain poster, pembuatan cerita',
        'GPT-4o, Claude, LLaMA, Stable Diffusion, DALL-E',
        'Self-supervised / Semi-supervised (data luas tanpa label)',
        'Kualitas subjektif + metrik manusia (BLEU, ROUGE, Human Eval)',
        'Hallucination, hak cipta, deepfake, bias output',
        'Corpora data besar tanpa label (internet, buku, gambar)',
    ],
})
print(">>> Tabel Perbandingan Traditional AI vs Generative AI:")
print(df_compare.to_string(index=False))

# --- Diagram Taksonomi AI visual ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Kiri: Traditional AI Flow
categories = ['Data Input\n(Labeled Dataset)', 'Feature\nEngineering', 'Model\nTraining\n(RF/XGB/SVM)', 'Prediksi\nLabel/Nilai', 'Evaluasi\nMetric Obyektif']
colors_trad = ['#4C78A8', '#72B7B2', '#54A24B', '#E45756', '#F58518']
y_pos = np.arange(len(categories))
bars = axes[0].barh(y_pos, [5, 4, 6, 3, 4], color=colors_trad, edgecolor='white', linewidth=2)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(categories, fontsize=10, fontweight='bold')
axes[0].set_xlabel('Tingkat kompleksitas proses', fontsize=11)
axes[0].set_title('Alur Traditional AI (Predictive)', fontsize=13, fontweight='bold', color='#1F4E79')
axes[0].grid(axis='x', alpha=0.3)
for i, bar in enumerate(bars):
    axes[0].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f'Step {i+1}', va='center', fontsize=9, color='black')
axes[0].invert_yaxis()

# Kanan: Generative AI Flow
categories_gen = ['Corpora Data Besar\n(Tanpa Label)', 'Pre-training\nSelf-Supervised', 'Fine-tuning /\nRLHF Alignment', 'Inference\nPrompt -> Generate', 'Evaluasi Kualitas\nSubjektif + Human']
colors_gen = ['#9C755F', '#B79A76', '#BCBD22', '#FF9DA6', '#E7BAF2']
bars2 = axes[1].barh(np.arange(len(categories_gen)), [6, 7, 5, 4, 5], color=colors_gen, edgecolor='white', linewidth=2)
axes[1].set_yticks(np.arange(len(categories_gen)))
axes[1].set_yticklabels(categories_gen, fontsize=10, fontweight='bold')
axes[1].set_xlabel('Tingkat kompleksitas proses', fontsize=11)
axes[1].set_title('Alur Generative AI (Creative)', fontsize=13, fontweight='bold', color='#7A271F')
axes[1].grid(axis='x', alpha=0.3)
for i, bar in enumerate(bars2):
    axes[1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f'Step {i+1}', va='center', fontsize=9, color='black')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_traditional_vs_generative_flow.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Visualisasi perbandingan alur disimpan: 01_traditional_vs_generative_flow.png")

# ============================================================
# BAGIAN 2 - LARGE LANGUAGE MODEL (LLM)
# ============================================================
print("\n" + "=" * 80)
print("BAGIAN 2: LARGE LANGUAGE MODEL (LLM)")
print("=" * 80)

print("""
----------------------------------------------------------------------
2.1 APA ITU LLM?
----------------------------------------------------------------------
Large Language Model = Model deep learning (berbasis ARSITEKTUR TRANSFORMER,
biasanya Decoder-only / Encoder-Decoder) yang dilatih pada DATA TEKS SANGAT
BESAR (triliunan token dari internet, buku, kode) dengan tujuan memprediksi
TOKEN BERIKUTNYA dalam sebuah urutan (next-token prediction).

Ukuran model LLM diukur dengan:
  • Jumlah PARAMETER (Miliar → Triliunan): GPT-3 = 175B, LLaMA 3 = 70B, GPT-4o = ?
  • Jumlah TOKEN TRAINING: 300B → 13T token
  • Konteks Window (panjang input maksimal): 4K → 128K → 1M token

Mengapa disebut "Large"?
  • Skala parameter >> model NLP klasik (BERT-base = 110M parameter SAJA)
  • Membutuhkan GPU A100/H100 (biaya training jutaan - milyaran rupiah)
  • Muncul "Emergent Abilities" (kemampuan tak terduga) pada skala besar:
    - Reasoning / penalaran matematis
    - Kemampuan multi-bahasa
    - In-context learning (belajar dari contoh di prompt saja)
    - Kemampuan koding
""")

print("""
----------------------------------------------------------------------
2.2 FAMILIARISASI DENGAN TOKEN
----------------------------------------------------------------------
LLM tidak memproses KATA, melainkan TOKEN = sub-unit teks hasil encoding
dengan tokenizer (mis. BPE - Byte Pair Encoding).

Contoh tokenisasi sederhana simulasi:
""")

# --- Simulasi Tokenisasi Sederhana ---
def tokenize_sederhana(teks):
    """Simulasi tokenizer BPE sederhana untuk ilustrasi (bukan implementasi asli)."""
    tokens = re.findall(r'[A-Za-z]+|\d+|[^\sA-Za-z0-9]', teks)
    token_list = []
    for tok in tokens:
        if len(tok) <= 4:
            token_list.append(tok)
        else:
            token_list.append(tok[:len(tok)//2])
            token_list.append(tok[len(tok)//2:])
    return token_list

contoh_teks = "Saya ingin mempelajari Artificial Intelligence Engineer untuk sertifikasi BNSP pada tahun 2026!"
tokens = tokenize_sederhana(contoh_teks)
print(f"  Teks asli      : {contoh_teks}")
print(f"  Panjang kata   : {len(contoh_teks.split())} kata")
print(f"  Token (simulasi): {tokens}")
print(f"  Jumlah token   : {len(tokens)} token")
print(f"  Estimasi biaya : {len(tokens) * 0.000005:.7f} USD (misal $5 per 1jt token input)")

# --- Visualisasi Perkembangan Skala LLM ---
llm_data = pd.DataFrame({
    'Tahun': [2017, 2018, 2019, 2020, 2022, 2023, 2024, 2025],
    'Model': ['Transformer', 'GPT-2', 'T5/GPT-2 XL', 'GPT-3', 'PaLM/GPT-3.5', 'LLaMA 2/GPT-4', 'GPT-4o/LLaMA 3', 'Claude 3.5/LLaMA 4'],
    'Parameter_Miliar': [0.213, 1.5, 11, 175, 540, 1760, 4000, 8000],
    'Token_Triliun': [0.004, 0.04, 0.75, 0.5, 0.78, 2.0, 13.0, 20.0],
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

bars = ax1.bar(llm_data['Model'], llm_data['Parameter_Miliar'], color=sns.color_palette('rocket', 8))
ax1.set_yscale('log')
ax1.set_xticklabels(llm_data['Model'], rotation=35, ha='right', fontsize=9)
ax1.set_ylabel('Jumlah Parameter (Miliar, log scale)', fontsize=11)
ax1.set_title('Perkembangan Skala Parameter LLM (2017-2025)', fontsize=12, fontweight='bold')
for bar, val in zip(bars, llm_data['Parameter_Miliar']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, f'{val}B',
             ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

bars2 = ax2.bar(llm_data['Model'], llm_data['Token_Triliun'], color=sns.color_palette('mako', 8))
ax2.set_yscale('log')
ax2.set_xticklabels(llm_data['Model'], rotation=35, ha='right', fontsize=9)
ax2.set_ylabel('Jumlah Token Training (Triliun, log scale)', fontsize=11)
ax2.set_title('Perkembangan Volume Data Training LLM', fontsize=12, fontweight='bold')
for bar, val in zip(bars2, llm_data['Token_Triliun']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, f'{val}T',
             ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_llm_scale_growth.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Visualisasi skala LLM disimpan: 02_llm_scale_growth.png")

# ============================================================
# BAGIAN 3 - KONSEP TRANSFORMER (RECAP) & KETERKAITAN DENGAN LLM
# ============================================================
print("\n" + "=" * 80)
print("BAGIAN 3: KONSEP TRANSFORMER (RECAP) & KETERKAITAN DENGAN LLM")
print("=" * 80)

print("""
----------------------------------------------------------------------
3.1 TRANSFORMER ARCHITECTURE (Vaswani et al., 2017 - "Attention is All You Need")
----------------------------------------------------------------------
ARSITEKTUR DASAR YANG MENJADI DASAR SEMUA LLM MODERN!

Komponen inti Transformer (2 blok besar):
  ┌───────────────────────────────────────────────────────────┐
  │ ENCODER STACK × N (kiri)     │  DECODER STACK × N (kanan) │
  │ • Input Embedding             │  • Output Embedding        │
  │ • Positional Encoding         │  • Positional Encoding     │
  │ • Multi-Head Self-Attention   │  • Masked Multi-Head Self- │
  │ • LayerNorm + Residual        │     Attention              │
  │ • Feed Forward Network        │  • Cross-Attention         │
  │                               │  • LayerNorm + Residual    │
  │                               │  • Feed Forward Network    │
  │ Output: Konteks vektor tiap  │  Output: Prob distribusi   │
  │         token input           │       token selanjutnya    │
  └───────────────────────────────────────────────────────────┘

  • Attention: "Perhatian" model terhadap token relevan di urutan lain
  • Multi-Head: Beberapa "sudut pandang attention" berbeda paralel
  • Positional Encoding: Memberi informasi urutan token (karena model = non-sequential)
  • Residual + LayerNorm: Stabilitas training pada kedalaman besar

VARIAN ARSITEKTUR UNTUK LLM:
  • Decoder-Only (paling umum untuk LLM chat): GPT series, LLaMA, Mistral, Claude
      - Hanya pakai blok Decoder → prediksi next-token → unggul text generation
  • Encoder-Only (pemahaman/klasifikasi): BERT, RoBERTa, DistilBERT
      - Hanya blok Encoder → unggul representasi embedding, klasifikasi teks, NER
  • Encoder-Decoder (terjemahan/ringkasan): T5, BART, FLAN-T5, NLLB
      - Keduanya → unggul translation, summarization panjang
""")

# --- Diagram Komponen Transformer visual dengan heatmap Attention Simulasi ---
np.random.seed(42)
n_tokens = 8
tokens_demo = ['<SOS>', 'Saya', 'sedang', 'mempelajari', 'Transformers', 'untuk', 'LLM', '.']

# Simulasi bobot attention (normalisasi baris = probability)
attention_matrix = np.random.dirichlet(np.ones(n_tokens) * 2, size=n_tokens)
# Perkuat perhatian token ke dirinya sendiri + kata penting
for i in range(n_tokens):
    attention_matrix[i, i] += 0.3
attention_matrix[4, 3] += 0.4
attention_matrix[6, 4] += 0.5
attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(attention_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=tokens_demo, yticklabels=tokens_demo, ax=ax,
            cbar_kws={'label': 'Bobot Attention'})
ax.set_title('Simulasi Multi-Head Self-Attention Matrix\n(Lebih gelap = perhatian lebih kuat)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Token yang DIPERHATIKAN (Key/Value)', fontsize=11)
ax.set_ylabel('Token YANG SEDANG MEMPERHATIKAN (Query)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_attention_heatmap_simulation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Visualisasi attention matrix disimpan: 03_attention_heatmap_simulation.png")

# ============================================================
# BAGIAN 4 - PROMPT ENGINEERING + TEKNIK PROMPTING
# ============================================================
print("\n" + "=" * 80)
print("BAGIAN 4: PROMPT ENGINEERING + TEKNIK PROMPTING")
print("=" * 80)

print("""
----------------------------------------------------------------------
4.1 APA ITU PROMPT ENGINEERING?
----------------------------------------------------------------------
Prompt = Input yang diberikan user ke LLM untuk menghasilkan output.
Prompt Engineering = Seni & ilmu merancang prompt agar output LLM:
  ✅ Akurat & relevan
  ✅ Sesuai format yang diinginkan
  ✅ Tidak hallucinate
  ✅ Konsisten & sesuai batasan yang diberikan

ANATOMI PROMPT YANG BAIK (ROLE-CONTEXT-INSTRUCTION-CONSTRAINTS-EXAMPLES):
  1. ROLE        : Tentukan peran LLM (mis. "Kamu adalah AI Engineer BNSP")
  2. CONTEXT     : Latar belakang / informasi yang relevan
  3. INSTRUCTION : Tugas jelas yang harus dikerjakan
  4. CONSTRAINTS : Batasan (bahasa, panjang, format, nada)
  5. EXAMPLES    : Contoh input-output jika dibutuhkan (few-shot)
""")

# --- Template Prompt Lengkap ---
template_prompt = """
ROLE       : Kamu adalah Asisten Data Scientist BNSP yang ahli dalam analisis data 
             dan penjelasan konsep statistik dalam Bahasa Indonesia.
CONTEXT    : Saya adalah peserta pelatihan AI Engineer tingkat pemula yang baru 
             mempelajari Confusion Matrix. Saya tidak paham perbedaan Precision 
             dan Recall pada kasus medis (deteksi kanker payudara).
INSTRUCTION: Jelaskan perbedaan Precision dan Recall:
             1. Definisi formal dengan rumus (gunakan TP, FP, TN, FN)
             2. Intuitif analogi medis (pasien kanker)
             3. Mana yang lebih PENTING untuk kasus ini dan MENGAPA?
CONSTRAINTS:
             - Gunakan Bahasa Indonesia yang mudah dipahami
             - Berikan contoh angka konkret (misal 1000 pasien)
             - Maksimal 300 kata
             - Gunakan bullet point agar jelas
EXAMPLES   : (tidak dibutuhkan untuk penjelasan konsep)
""".strip()
print(">>> Contoh TEMPLATE PROMPT LENGKAP (RCICE framework):")
print(template_prompt)

print("""
----------------------------------------------------------------------
4.2 TEKNIK-TEKNIK PROMPTING STANDAR
----------------------------------------------------------------------
""")

# ===== TEKNIK 1: ZERO-SHOT PROMPTING =====
print("\n📌 TEKNIK 1: ZERO-SHOT PROMPTING (TANPA CONTOH)")
zero_shot = """
Klasifikasikan teks review berikut ke dalam label: Positif, Netral, Negatif.
Review: "Pelayanannya cukup standar, makanannya biasa saja harganya mahal."
Jawaban:
""".strip()
print(f"Prompt zero-shot:\n{zero_shot}\n  → Output yang diharapkan: Negatif")

# ===== TEKNIK 2: FEW-SHOT PROMPTING (DENGAN CONTOH) =====
print("\n📌 TEKNIK 2: FEW-SHOT PROMPTING (DENGAN N CONTOH)")
few_shot = """
Tugas: Balik urutan KATA dari kalimat input.

Contoh 1:
Input : "Saya suka makan nasi goreng"
Output: "goreng nasi makan suka Saya"

Contoh 2:
Input : "Machine Learning sangat menarik"
Output: "menarik sangat Learning Machine"

Sekarang kerjakan:
Input : "Saya lulus uji kompetensi BNSP bulan depan"
Output:
""".strip()
print(few_shot)
print("  → Output yang diharapkan: 'depan bulan BNSP kompetensi uji lulus Saya'")

# ===== TEKNIK 3: CHAIN-OF-THOUGHT (CoT) =====
print("\n📌 TEKNIK 3: CHAIN-OF-THOUGHT (CoT) PROMPTING - PENALARAN LANGKAH DEMI LANGKAH")
cot_prompt = """
Sebuah toko menjual 3 jenis barang:
  • A : Harga 15.000, terjual 120 unit
  • B : Harga 25.000, terjual 80 unit  
  • C : Harga 40.000, terjual 45 unit
Pajak = 11% dari total pendapatan KOTOR.
Berapa pendapatan BERSIH toko tersebut setelah dipotong pajak?

Jawablah DENGAN MENUNJUKKAN LANGKAH PENYELESAIAN LANGKAH DEMI LANGKAH,
bukan hanya angka akhir.
""".strip()
print(cot_prompt)

# Simulasi output CoT dari LLM
print("\n  ✅ Output LLM (Chain-of-Thought yang benar):")
langkah = [
    "Langkah 1: Hitung pendapatan per barang",
    "  A = 15.000 × 120 = 1.800.000",
    "  B = 25.000 × 80  = 2.000.000",
    "  C = 40.000 × 45  = 1.800.000",
    "Langkah 2: Total pendapatan KOTOR",
    "  = 1.800.000 + 2.000.000 + 1.800.000 = 5.600.000",
    "Langkah 3: Hitung pajak 11%",
    "  Pajak = 5.600.000 × 0.11 = 616.000",
    "Langkah 4: Pendapatan BERSIH",
    "  = 5.600.000 - 616.000 = Rp4.984.000",
    "Jawaban AKHIR: Rp4.984.000",
]
for l in langkah:
    print(f"    {l}")

# ===== TEKNIK 4: ReAct (Reasoning + Acting) =====
print("\n📌 TEKNIK 4: ReAct (REASONING + ACTING) - LLM BISA MENGGUNAKAN TOOL")
print("""
  Prinsip ReAct = Siklus: THINK → ACT → OBSERVE → THINK → ACT → ... → ANSWER
  
  Contoh alur ReAct untuk pertanyaan: "Berapa tinggi gunung tertinggi di ASEAN?"
  
  [THOUGHT] Saya perlu mencari data tinggi gunung tertinggi di ASEAN.
            Saya belum yakin ada di training data, jadi harus cari di Wikipedia.
  [ACTION]  SEARCH("gunung tertinggi ASEAN tinggi mdpl")
  [OBSERVE] "Hasil pencarian: Gunung Hkakabo Razi di Myanmar, 5.881 mdpl.
             Puncak tertinggi kedua: Puncak Jaya (Carstensz Pyramid) Indonesia, 4.884 mdpl."
  [THOUGHT] Data menunjukkan Hkakabo Razi 5.881 mdpl adalah tertinggi.
            Perlu cross-validasi agar tidak salah.
  [ACTION]  SEARCH("Hkakabo Razi ketinggian resmi BGF Myanmar")
  [OBSERVE] "Badan Geologi Myanmar mengonfirmasi Hkakabo Razi = 5.881 ±3 mdpl."
  [ANSWER]  "GUNUNG TERTINGGI di ASEAN adalah GUNUNG HKAKABO RAZI di Myanmar,
             dengan ketinggian resmi 5.881 mdpl di atas permukaan laut."
""")

# --- Visualisasi perbandingan 4 teknik prompting ---
teknik = ['Zero-Shot', 'Few-Shot\n(n=3)', 'Chain-of-Thought', 'ReAct\n(With Tools)']
akurasi_simulasi = [62, 78, 89, 95]
biaya_token = [1.0, 3.5, 2.8, 12.0]
kecepatan_detik = [0.5, 1.8, 2.2, 8.5]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = sns.color_palette('viridis', 4)

axes[0].bar(teknik, akurasi_simulasi, color=colors, edgecolor='white', linewidth=2)
axes[0].set_ylim(0, 110)
axes[0].set_ylabel('Akurasi rata-rata (%)', fontsize=10)
axes[0].set_title('Akurasi Task Reasoning', fontsize=12, fontweight='bold')
for i, v in enumerate(akurasi_simulasi):
    axes[0].text(i, v + 2, f'{v}%', ha='center', fontweight='bold')

axes[1].bar(teknik, biaya_token, color=colors, edgecolor='white', linewidth=2)
axes[1].set_ylabel('Biaya relatif (token = 1x)', fontsize=10)
axes[1].set_title('Biaya Token (Semakin besar = mahal)', fontsize=12, fontweight='bold')
for i, v in enumerate(biaya_token):
    axes[1].text(i, v + 0.3, f'{v}x', ha='center', fontweight='bold')

axes[2].bar(teknik, kecepatan_detik, color=colors, edgecolor='white', linewidth=2)
axes[2].set_ylabel('Waktu respons (detik)', fontsize=10)
axes[2].set_title('Latency / Kecepatan', fontsize=12, fontweight='bold')
for i, v in enumerate(kecepatan_detik):
    axes[2].text(i, v + 0.2, f'{v}s', ha='center', fontweight='bold')

plt.suptitle('TRADE-OFF 4 TEKNIK PROMPTING UTAMA', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_prompting_techniques_tradeoff.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Visualisasi tradeoff prompting disimpan: 04_prompting_techniques_tradeoff.png")

# ============================================================
# BAGIAN 5 - STRUCTURED PROMPTING & CONTEXT MANAGEMENT
# ============================================================
print("\n" + "=" * 80)
print("BAGIAN 5: STRUCTURED PROMPTING & CONTEXT MANAGEMENT")
print("=" * 80)

print("""
----------------------------------------------------------------------
5.1 STRUCTURED PROMPTING = MEMAKSA OUTPUT FORMAT TETAP
----------------------------------------------------------------------
Masalah: LLM bisa output format berbeda tiap kali dipanggil → susah di-parse
         oleh kode downstream.

Solusi: STRUCTURED PROMPTING → Tentukan SKEMA OUTPUT dengan jelas
        (JSON schema, XML, YAML, Markdown tabel) sehingga output selalu
        konsisten dan bisa diparse otomatis tanpa regex rumit.

  • Alat bantu modern: OpenAI response_format={"type":"json_object"},
    LangChain PydanticOutputParser, Microsoft Guidance, Outlines.
""")

# --- Contoh Prompt Structured JSON ---
structured_prompt_example = """
ROLE: Kamu adalah asisten rekrutmen yang mengekstrak informasi dari CV.
TUGAS: Ekstrak informasi dari CV di bawah ini, OUTPUT HANYA FORMAT JSON
       dengan skema di bawah, TIDAK BOLEH ada teks lain selain JSON valid!

SKEMA JSON YANG WAJIB DIIKUTI:
{
  "nama_lengkap": string,
  "email": string,
  "nomor_telepon": string,
  "pendidikan_terakhir": {
    "jenjang": "SMA|D3|S1|S2|S3",
    "nama_universitas": string,
    "tahun_lulus": integer
  },
  "pengalaman_kerja_tahun": integer,
  "skill_teknis": array of string (maksimal 10),
  "salary_expectation_juta": integer | null
}

=== ISI CV ===
Nama: Budi Santoso
Email: budi.santoso@example.com
No HP: +62 812-3456-7890
Pendidikan: S1 Teknik Informatika Universitas Indonesia, lulus 2020
Pengalaman: 4 tahun sebagai Data Scientist di 2 perusahaan
Skill: Python, SQL, Scikit-learn, TensorFlow, Pandas, Tableau, Spark, 
       Docker, REST API, GCP
Harapan Gaji: Rp 18.000.000 per bulan
=== AKHIR CV ===
""".strip()
print(">>> Contoh STRUCTURED PROMPTING (paksa JSON Schema):")
print("-" * 60)
print(structured_prompt_example)

# Simulasi output JSON valid dari LLM
output_json_llm = {
    "nama_lengkap": "Budi Santoso",
    "email": "budi.santoso@example.com",
    "nomor_telepon": "+62 812-3456-7890",
    "pendidikan_terakhir": {
        "jenjang": "S1",
        "nama_universitas": "Universitas Indonesia",
        "tahun_lulus": 2020,
    },
    "pengalaman_kerja_tahun": 4,
    "skill_teknis": ["Python", "SQL", "Scikit-learn", "TensorFlow", "Pandas",
                     "Tableau", "Spark", "Docker", "REST API", "GCP"],
    "salary_expectation_juta": 18,
}
print("\n>>> ✅ Output LLM yang valid (bisa langsung json.loads):")
print(json.dumps(output_json_llm, indent=2, ensure_ascii=False))

# Validasi parsing JSON
try:
    parsed = json.loads(json.dumps(output_json_llm))
    print(f"\n✅ JSON parsing BERHASIL: Nama={parsed['nama_lengkap']}, "
          f"Skill count={len(parsed['skill_teknis'])}")
except json.JSONDecodeError as e:
    print(f"❌ JSON tidak valid: {e}")

print("""
----------------------------------------------------------------------
5.2 CONTEXT MANAGEMENT = PENGELOLAAN KONTEKS PANJANG
----------------------------------------------------------------------
Masalah: LLM punya CONTEXT WINDOW terbatas (misal 4K → 128K token).
         Data besar (PDF 1000 halaman, database 10.000 record) TIDAK BISA
         dimasukkan SEMUA ke prompt sekaligus.

TEKNIK CONTEXT MANAGEMENT UTAMA:
  1. CHUNKING: Potong dokumen menjadi potongan kecil (chunk = 200-1000 token)
               dengan overlap (10-20% token) agar konteks tidak terpotong.
  2. RANKING / RETRIEVAL: Hanya ambil TOP-N chunk PALING RELEVAN dengan query
                          (pakai embedding similarity = RAG).
  3. CONTEXT COMPRESSION: Ringkas chunk terlebih dahulu sebelum dimasukkan ke
                          prompt (misal LLM lain merangkum → hemat token).
  4. SUMMARIZATION CHAIN: Dokumen panjang → ringkas per bagian → gabung
                          (map-reduce pattern).
  5. SLIDING WINDOW: Untuk chat, simpan hanya N pesan terakhir + ringkasan
                     percakapan awal.
""")

# --- Demo Konsep Chunking dengan Overlap ---
def split_chunks(teks_panjang, chunk_size=50, overlap=10):
    """Chunking sederhana per karakter dengan overlap (simulasi chunk token)."""
    chunks = []
    start = 0
    while start < len(teks_panjang):
        end = min(start + chunk_size, len(teks_panjang))
        chunks.append(teks_panjang[start:end])
        if end == len(teks_panjang):
            break
        start = end - overlap
    return chunks

dokumen_panjang = (
    "Artificial Intelligence Engineer adalah profesi yang bertanggung jawab "
    "merancang, membangun, menguji, dan mengevaluasi solusi berbasis AI. "
    "Kompetensi utama meliputi pengolahan data, pengembangan model ML/DL, "
    "NLP, computer vision, Generative AI, deployment model ke dalam sistem, "
    "serta memastikan aspek etika dan responsible AI dalam setiap implementasi. "
    "Untuk sertifikasi BNSP, kandidat harus menunjukkan bukti portofolio proyek "
    "AI end-to-end yang mencakup seluruh lifecycle dari identifikasi masalah "
    "hingga dokumentasi dan monitoring model pasca-deployment."
)

chunks = split_chunks(dokumen_panjang, chunk_size=120, overlap=25)
print(f"\n>>> Demo Chunking: Panjang dokumen = {len(dokumen_panjang)} karakter")
print(f"    Jumlah chunk (size=120, overlap=25) = {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"    Chunk {i+1} [{len(c)} char]: ...{c[:50]}...")

# ============================================================
# BAGIAN 6 - PEMANFAATAN API MODEL AI (DUMMY SIMULATION)
# ============================================================
print("\n" + "=" * 80)
print("BAGIAN 6: PEMANFAATAN API MODEL AI")
print("=" * 80)

print("""
----------------------------------------------------------------------
6.1 POLA UMUM PEMANGGILAN API LLM
----------------------------------------------------------------------
Setiap provider LLM menyediakan REST API dengan pola umum:

  ┌──────────┐   POST /v1/chat/completions    ┌──────────────┐
  │ Aplikasi │ ─────────────────────────────→ │ LLM Provider │
  │   Kita   │   Header: Authorization: Bearer│  (OpenAI,    │
  │          │   Body: {model, messages,      │  Anthropic,  │
  │          │          temperature, ...}     │  Groq, dll)  │
  └──────────┘ ←───────────────────────────── └──────────────┘
                  Response: {choices: [{message: {content}}], usage: {...}}

KOMPONEN REQUEST PENTING:
  • model       : Nama model (mis. "gpt-4o-mini", "claude-3-sonnet-20240229")
  • messages    : Daftar pesan {role: "system|user|assistant", content: "..."}
  • temperature : "Kreativitas" model (0 = deterministik, 2 = random)
  • max_tokens  : Batas panjang output
  • top_p       : Nucleus sampling (0.1 = hanya token teratas 10% probability)
""")

# --- Try/except library opsional OpenAI ---
try:
    import openai
    print(">>> ✅ Library 'openai' TERINSTALL. Anda bisa memanggil API langsung.")
except ImportError:
    print(">>> ⚠️  Library 'openai' TIDAK terinstall. "
          "Install dengan: pip install openai==1.30.0")
    print("    Menjalankan SIMULASI pola request/response API (offline)...")
    openai = None

# --- Dummy Class LLMAPIClient simulasi panggilan API ---
class DummyLLMAPIClient:
    """Simulasi offline LLM API Client untuk pembelajaran tanpa API Key."""

    def __init__(self, model_name="dummy-bnsp-llm-v1"):
        self.model_name = model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        print(f"    [DummyLLM] Client siap, model: {self.model_name}")

    def chat_completion(self, messages, temperature=0.7, max_tokens=500):
        """Simulasi panggilan /v1/chat/completions."""
        # Hitung token simulasi
        all_text = " ".join([m['content'] for m in messages])
        in_tokens = len(all_text.split()) * 1.3
        self.total_input_tokens += in_tokens

        # Logika dummy response berdasarkan pesan terakhir user
        last_user_msg = [m for m in messages if m['role'] == 'user'][-1]['content'].lower()

        if 'hai' in last_user_msg or 'halo' in last_user_msg:
            response_content = "Halo! Saya DummyAI asisten pelatihan BNSP. Ada yang bisa saya bantu?"
        elif 'siapa namamu' in last_user_msg or 'siapa kamu' in last_user_msg:
            response_content = ("Nama saya DummyBNSP-LLM v1, model simulasi untuk pelatihan "
                                "Modul 9 Generative AI. Saya dibuat offline agar Anda bisa "
                                "belajar tanpa API Key.")
        elif 'ringkas' in last_user_msg:
            response_content = ("[Ringkasan Simulasi] Dokumen tersebut membahas tentang pentingnya "
                                "penerapan etika AI (Responsible AI) di perusahaan, dengan fokus "
                                "pada 5 pilar utama: Fairness, Transparency, Privacy, Safety, dan "
                                "Accountability.")
        elif 'klasifikasi' in last_user_msg or 'sentimen' in last_user_msg:
            response_content = json.dumps({
                "label_sentimen": "POSITIF",
                "confidence": 0.94,
                "kata_kunci_pendukung": ["cepat", "original", "ramah", "puas"]
            }, ensure_ascii=False)
        else:
            response_content = (f"[Jawaban Dummy] Pertanyaan Anda: '{last_user_msg[:80]}...' "
                                f"(diproses dengan temperature={temperature}). "
                                "Untuk jawaban nyata, gunakan API Key provider LLM resmi.")

        out_tokens = len(response_content.split()) * 1.2
        self.total_output_tokens += out_tokens

        return {
            "id": "chatcmpl-dummy-" + str(np.random.randint(10000, 99999)),
            "model": self.model_name,
            "usage": {
                "prompt_tokens": int(in_tokens),
                "completion_tokens": int(out_tokens),
                "total_tokens": int(in_tokens + out_tokens),
            },
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response_content,
                    },
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
        }

# --- Demo Panggilan API Simulasi ---
client = DummyLLMAPIClient()

print("\n>>> Demo 1: Percakapan sederhana (System + User)")
messages1 = [
    {"role": "system", "content": "Kamu adalah asisten pelatihan BNSP AI Engineer."},
    {"role": "user", "content": "Halo, siapa namamu?"},
]
resp1 = client.chat_completion(messages1, temperature=0.2, max_tokens=150)
print(f"  🤖 Assistant: {resp1['choices'][0]['message']['content']}")
print(f"  📊 Penggunaan token: Input={resp1['usage']['prompt_tokens']}, "
      f"Output={resp1['usage']['completion_tokens']}, "
      f"Total={resp1['usage']['total_tokens']}")

print("\n>>> Demo 2: Task Klasifikasi Sentimen (Structured Output)")
messages2 = [
    {"role": "system", "content": "Kamu classifier sentimen. Output HANYA JSON!"},
    {"role": "user", "content": "Klasifikasikan sentimen review: 'Pengirimannya super cepat, "
                                  "barangnya original, pelayanan ramah, saya sangat puas!'"}
]
resp2 = client.chat_completion(messages2, temperature=0.0, max_tokens=100)
print(f"  🤖 Assistant JSON: {resp2['choices'][0]['message']['content']}")
try:
    sentimen_data = json.loads(resp2['choices'][0]['message']['content'])
    print(f"  ✅ Parse JSON sukses → Label: {sentimen_data['label_sentimen']}, "
          f"Confidence: {sentimen_data['confidence']*100:.1f}%")
except Exception as e:
    print(f"  ⚠️  Bukan JSON, error: {e}")

print(f"\n  📈 Total kumulatif token client: "
      f"In={int(client.total_input_tokens)}, Out={int(client.total_output_tokens)}")

# ============================================================
# BAGIAN 7 - RETRIEVAL-AUGMENTED GENERATION (RAG)
# ============================================================
print("\n" + "=" * 80)
print("BAGIAN 7: RETRIEVAL-AUGMENTED GENERATION (RAG)")
print("=" * 80)

print("""
----------------------------------------------------------------------
7.1 APA ITU RAG & MENGAPA PENTING?
----------------------------------------------------------------------
RAG (Lewis et al., 2020) = Arsitektur yang menggabungkan LLM dengan
SISTEM RETRIEVAL (pencari informasi) dari KNOWLEDGE BASE privat,
sehingga LLM bisa:
  ✅ Jawab pertanyaan dari data PRIVAT perusahaan (tidak ada di training data)
  ✅ Kurangi HALLUCINATION (jawaban didukung sumber dokumen konkret)
  ✅ Update informasi TERBARU tanpa retrain model LLM (hemat milyaran rupiah!)
  ✅ Kutip SUMBER referensi (transparent & accountable)

----------------------------------------------------------------------
7.2 ALUR RAG LENGKAP (NAIVE RAG - STANDARD)
----------------------------------------------------------------------
FASE A: INGESTI (Offline / Pre-processing)
  1. LOAD          : Muat dokumen (PDF, DOCX, HTML, DB records, Notion, Slack)
  2. SPLIT/CHUNK   : Potong dokumen menjadi chunk kecil (500 token, 10% overlap)
  3. EMBED         : Ubah tiap chunk menjadi VECTOR EMBEDDING (1536D / 768D / ...)
  4. STORE         : Simpan vector + metadata (sumber halaman, judul) ke VECTOR DATABASE

FASE B: QUERY (Online / Runtime)
  5. QUERY EMBED   : Ubah pertanyaan user menjadi vector embedding
  6. RETRIEVE      : Cari TOP-N chunk PALING MIRIP (cosine similarity tertinggi)
                     → = KONTEKS RELEVAN
  7. AUGMENT PROMPT: Gabungkan SYSTEM_PROMPT + KONTEKS + PERTANYAAN USER
  8. GENERATE      : Kirim prompt gabungan ke LLM → JAWABAN TERDASAR SUMBER
  9. CITE SOURCES  : Tampilkan link/sumber halaman chunk yang dipakai (evidence!)
""")

# --- FASE A: IMPLEMENTASI SIMULASI RAG OFFLINE DENGAN TF-IDF + COSINE SIM ---
print("\n>>> [FASE A: INGESTI - MEMBANGUN KNOWLEDGE BASE RAG SIMULASI]")

# Knowledge Base = 9 dokumen tentang BNSP AI Engineer (chunk simulasi)
knowledge_base_chunks = [
    {
        "id": 1,
        "sumber": "Modul 1 - Konsep Dasar AI.pdf, Hal. 5",
        "teks": (
            "Skema Sertifikasi BNSP Artificial Intelligence Engineer adalah pengakuan "
            "profesi dari Badan Nasional Sertifikasi Profesi Republik Indonesia untuk "
            "individu yang kompeten merancang, mengembangkan, dan mengimplementasikan "
            "solusi berbasis AI. Kode skema: SKKNI-AI-ENGINEER-2024 dengan total 12 "
            "unit kompetensi utama."
        ),
    },
    {
        "id": 2,
        "sumber": "Modul 3 - Data Preparation.pdf, Hal. 12",
        "teks": (
            "Dalam persiapan data untuk model AI, tahapan CRISP-DM yang diadaptasi BNSP "
            "mencakup: (1) Data Understanding, (2) Data Cleaning (null, duplikat, outlier), "
            "(3) Data Transformation (encoding, scaling), (4) Feature Engineering, "
            "(5) Splitting menjadi train 70%, validation 15%, dan test set 15%."
        ),
    },
    {
        "id": 3,
        "sumber": "Modul 4 - ML Engineering.pdf, Hal. 22",
        "teks": (
            "Untuk kasus klasifikasi dengan data imbalance (rasio 99:1) seperti deteksi "
            "fraud, Anda TIDAK BOLEH hanya menggunakan akurasi. Metrik yang wajib dipakai: "
            "Precision, Recall, F1-Score, PR-AUC, Matthews Correlation Coefficient (MCC), "
            "dan Cohen Kappa Score. Threshold tuning sangat diperlukan."
        ),
    },
    {
        "id": 4,
        "sumber": "Modul 5 - Evaluasi Model.pdf, Hal. 8",
        "teks": (
            "Cross Validation adalah teknik membagi dataset menjadi K lipatan untuk "
            "menghindari leakage estimasi performa. Untuk data imbalance, gunakan "
            "Stratified K-Fold, bukan K-Fold biasa. Untuk hyperparameter tuning yang "
            "akurat, disarankan Nested Cross Validation (CV luar + CV dalam)."
        ),
    },
    {
        "id": 5,
        "sumber": "Modul 6 - Deep Learning.pdf, Hal. 17",
        "teks": (
            "Convolutional Neural Network (CNN) terdiri dari layer: Conv2D + Activation, "
            "Batch Normalization, Max Pooling, Dropout, lalu Flatten dan Dense di bagian "
            "akhir. Arsitektur CNN klasik yang wajib diketahui urutannya: LeNet-5 (1998), "
            "AlexNet (2012), VGGNet (2014), GoogLeNet/Inception (2014), ResNet (2015), "
            "EfficientNet (2019)."
        ),
    },
    {
        "id": 6,
        "sumber": "Modul 8 - NLP.pdf, Hal. 14",
        "teks": (
            "Preprocessing teks Bahasa Indonesia memiliki 6 langkah standar BNSP: "
            "(1) Case Folding, (2) Cleaning (hapus tanda baca/angka tidak relevan), "
            "(3) Tokenization, (4) Stopword Removal (pakai NLTK bahasa Indonesia), "
            "(5) Stemming (Sastrawi atau Porter untuk Bahasa Indonesia), "
            "(6) Joining kembali menjadi clean string."
        ),
    },
    {
        "id": 7,
        "sumber": "Modul 9 - Generative AI.pdf, Hal. 25",
        "teks": (
            "RAG (Retrieval-Augmented Generation) adalah teknik terbaik untuk mengurangi "
            "hallucination pada chatbot internal perusahaan. Komponen RAG minimal: "
            "Dokumen Sumber, Chunker, Embedding Model, Vector Database, Retriever, "
            "Prompt Builder, LLM Generator, dan Citation Tracker."
        ),
    },
    {
        "id": 8,
        "sumber": "Panduan Portofolio BNSP.pdf, Hal. 3",
        "teks": (
            "Bukti portofolio BNSP untuk AI Engineer minimal harus berisi 3 proyek berbeda: "
            "(1) Proyek klasifikasi/regresi (ML tradisional), (2) Proyek deep learning "
            "(CV atau NLP), (3) Proyek Generative AI / RAG. Setiap proyek wajib memiliki "
            "README lengkap: latar belakang, EDA, preprocessing, modelling, evaluasi, "
            "deployment plan, dan bukti screenshot output."
        ),
    },
    {
        "id": 9,
        "sumber": "Modul 10 - Deployment.pdf, Hal. 11",
        "teks": (
            "Langkah deployment model AI dengan FastAPI: (1) Simpan model dengan joblib/pickle, "
            "(2) Buat endpoint POST /predict dengan Pydantic BaseModel untuk validasi input, "
            "(3) Muat model sekali saat startup event, (4) Buat DTO request/response, "
            "(5) Tambahkan /docs Swagger UI, (6) Dockerize dengan base image python:3.10-slim, "
            "(7) Upload ke container registry, deploy ke server cloud atau on-premise."
        ),
    },
]
print(f"    Jumlah chunk knowledge base: {len(knowledge_base_chunks)} dokumen")

# Step 2-4: Vectorize dengan TF-IDF (fallback embedding sederhana tanpa library eksternal)
corpus_texts = [c['teks'] for c in knowledge_base_chunks]
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=None)
tfidf_matrix = vectorizer.fit_transform(corpus_texts)
print(f"    TF-IDF Matrix shape: {tfidf_matrix.shape} (dokumen × unique terms)")

# --- FASE B: IMPLEMENTASI RAG RETRIEVAL + GENERATION ---
print("\n>>> [FASE B: QUERY RAG LIVE - 5 Pertanyaan Uji Simulasi]")

def rag_retrieve_topk(pertanyaan, top_k=3):
    """Retrieve top-k chunk termirip dengan cosine similarity TF-IDF."""
    vec_question = vectorizer.transform([pertanyaan])
    similarities = cosine_similarity(vec_question, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            **knowledge_base_chunks[idx],
            'similarity_score': float(similarities[idx]),
        })
    return results

def rag_build_prompt(pertanyaan, konteks_chunks):
    """Bangun prompt RAG: System + Context + Question."""
    konteks_joined = "\n\n".join([
        f"[SUMBER: {c['sumber']}]\n{c['teks']}" for c in konteks_chunks
    ])
    return f"""
--- SYSTEM PROMPT RAG ---
Anda adalah Asisten QA BNSP AI Engineer yang SANGAT HATI-HATI.
PERATURAN WAJIB:
1. Jawab HANYA berdasarkan KONTEKS DI BAWAH INI SAJA.
2. Jika jawaban TIDAK ADA di KONTEKS, katakan: 
   "Maaf, informasi tersebut tidak tersedia di knowledge base."
3. Selalu KUTIP SUMBER referensi di akhir jawaban dengan format: [Sumber: <nama file> Hal. X]
4. Gunakan Bahasa Indonesia yang jelas dan terstruktur.
--- AKHIR SYSTEM PROMPT ---

--- KONTEKS (DARI RETRIEVAL) ---
{konteks_joined}
--- AKHIR KONTEKS ---

--- PERTANYAAN USER ---
{pertanyaan}
--- AKHIR PERTANYAAN ---

Jawaban:
""".strip()

def rag_generate_jawaban_dummy(prompt, retrieved_chunks):
    """Simulasi LLM menjawab berdasarkan konteks RAG (offline)."""
    # Dummy generator: rangkum pertanyaan + sebutkan sumber + top score
    if len(retrieved_chunks) == 0 or retrieved_chunks[0]['similarity_score'] < 0.05:
        return "Maaf, informasi tersebut tidak tersedia di knowledge base."
    top = retrieved_chunks[0]
    jawaban_template = {
        "Apa itu skema BNSP AI Engineer?": (
            "Skema Sertifikasi BNSP Artificial Intelligence Engineer adalah pengakuan "
            "profesi resmi untuk individu yang kompeten merancang, mengembangkan, dan "
            "mengimplementasikan solusi berbasis AI, dengan 12 unit kompetensi utama."
        ),
        "Metrik apa untuk data imbalance?": (
            "Untuk kasus klasifikasi dengan data imbalance seperti deteksi fraud, "
            "metrik yang WAJIB digunakan (bukan hanya akurasi) adalah: Precision, "
            "Recall, F1-Score, PR-AUC, Matthews Correlation Coefficient (MCC), "
            "dan Cohen Kappa Score. Perlu juga threshold tuning."
        ),
        "Langkah deployment FastAPI?": (
            "Langkah deployment model AI dengan FastAPI: (1) Simpan model (joblib/pickle), "
            "(2) Buat endpoint POST /predict + Pydantic BaseModel, (3) Load model saat startup, "
            "(4) DTO request/response, (5) Swagger UI /docs, (6) Dockerize, "
            "(7) Deploy ke server/cloud."
        ),
        "Apa komponen RAG?": (
            "RAG (Retrieval-Augmented Generation) adalah teknik terbaik untuk mengurangi "
            "hallucination chatbot internal perusahaan. Komponen RAG minimal: "
            "Dokumen Sumber, Chunker, Embedding Model, Vector Database, Retriever, "
            "Prompt Builder, LLM Generator, dan Citation Tracker."
        ),
        "Bagaimana preprocessing NLP Indonesia?": (
            "Preprocessing teks Bahasa Indonesia memiliki 6 langkah standar: "
            "(1) Case Folding, (2) Cleaning, (3) Tokenization, (4) Stopword Removal, "
            "(5) Stemming (Sastrawi/Porter ID), (6) Joining kembali."
        ),
    }
    # Coba match pertanyaan yang diketahui
    for kw, jawaban in jawaban_template.items():
        if any(k.lower() in top['teks'].lower() for k in kw.split()[:3]):
            # Fallback general jawaban + sumber
            jawaban_akhir = f"{jawaban}\n\n[Sumber: {top['sumber']} (skor mirip: {top['similarity_score']:.3f})]"
            # Tambah sumber kedua jika score cukup tinggi
            if len(retrieved_chunks) > 1 and retrieved_chunks[1]['similarity_score'] > top['similarity_score'] * 0.6:
                jawaban_akhir += f"\n[Sumber pendukung: {retrieved_chunks[1]['sumber']}]"
            return jawaban_akhir
    # Default: Jawab dengan cuplikan top chunk
    cuplikan = top['teks'][:280] + ("..." if len(top['teks']) > 280 else "")
    return (f"[RAG Jawaban]\nBerdasarkan sumber yang ada:\n\n{cuplikan}\n\n"
            f"[Sumber: {top['sumber']}] [Skor relevansi: {top['similarity_score']:.3f}]")

# --- Jalankan 5 Query RAG ---
pertanyaan_uji = [
    "Apa itu skema sertifikasi BNSP untuk AI Engineer?",
    "Saya punya data imbalance 99:1 deteksi fraud, metrik apa yang harus dipakai?",
    "Bagaimana urutan langkah deployment model AI pakai FastAPI?",
    "Sebutkan komponen utama arsitektur RAG dan manfaatnya!",
    "Bagaimana 6 langkah preprocessing teks untuk Bahasa Indonesia?",
]

for i, q in enumerate(pertanyaan_uji):
    print(f"\n{'─'*60}")
    print(f"🔍 Query {i+1}: {q}")
    retrieved = rag_retrieve_topk(q, top_k=3)
    print(f"   Top-{len(retrieved)} retrieval:")
    for r in retrieved:
        print(f"     • [{r['similarity_score']:.3f}] {r['sumber']}")
    prompt_rag = rag_build_prompt(q, retrieved)
    jawaban = rag_generate_jawaban_dummy(prompt_rag, retrieved)
    print(f"   💡 Jawaban RAG LLM:\n      {jawaban.replace(chr(10), chr(10) + '      ')}")

# --- Visualisasi Diagram Arsitektur RAG (Flow) ---
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Judul
ax.text(50, 97, 'ARSITEKTUR RETRIEVAL-AUGMENTED GENERATION (RAG)',
        ha='center', va='center', fontsize=16, fontweight='bold', color='#1F4E79')

# Fase A - Offline Ingesti
ax.text(25, 90, '📍 FASE A: INGESTI (OFFLINE)', ha='center', fontsize=12,
        fontweight='bold', color='#006D2C')
boxes_a = [
    (10, 80, 20, 8, '1. LOAD\nDokumen PDF/DB', '#C7E9C0'),
    (35, 80, 20, 8, '2. SPLIT\nChunking 500T', '#A1D99B'),
    (60, 80, 20, 8, '3. EMBED\nVector 768D', '#74C476'),
    (85, 80, 12, 8, '4. STORE\nVector DB', '#31A354'),
]
for x, y, w, h, label, color in boxes_a:
    rect = plt.Rectangle((x-w/2, y-h/2), w, h, facecolor=color,
                         edgecolor='black', linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=9,
            fontweight='bold', zorder=3)

# Panah fase A
for i in range(3):
    x_start = [20, 45, 70][i]
    ax.annotate('', xy=(x_start + 15, 80), xytext=(x_start, 80),
                arrowprops=dict(arrowstyle='->', lw=2, color='#006D2C'), zorder=1)

# Fase B - Online Query
ax.text(50, 68, '📍 FASE B: QUERY (ONLINE / RUNTIME)', ha='center', fontsize=12,
        fontweight='bold', color='#7A271F')

boxes_b_left = [
    (15, 58, 20, 8, '5. USER\nQUERY', '#FDD0A2'),
    (15, 46, 20, 8, '6. QUERY\nEMBEDDING', '#FDAE6B'),
]
boxes_b_right = [
    (75, 46, 25, 8, '7. RETRIEVE TOP-K\nCosine Similarity', '#FD8D3C'),
    (60, 34, 35, 8, '8. AUGMENT PROMPT\n(System + Context + Question)', '#F16913'),
    (60, 22, 20, 8, '9. LLM\nGENERATE', '#D94801'),
    (60, 10, 20, 8, '10. JAWABAN\nDENGAN SITASI', '#8C2D04'),
]
for x, y, w, h, label, color in boxes_b_left + boxes_b_right:
    rect = plt.Rectangle((x-w/2, y-h/2), w, h, facecolor=color,
                         edgecolor='black', linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=9,
            fontweight='bold', color='white' if color in ['#F16913','#D94801','#8C2D04'] else 'black', zorder=3)

# Panah antar step B
panah_b = [
    (15, 54, 15, 50),      # query → query embed
    (15, 42, 70, 46),      # query embed → retrieve (cross)
    (75, 42, 68, 34),      # retrieve → prompt build
    (60, 30, 60, 26),      # prompt → llm
    (60, 18, 60, 14),      # llm → jawaban
]
for x1, y1, x2, y2 in panah_b:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=2, color='#7A271F'), zorder=1)

# Vector DB connection
ax.annotate('', xy=(70, 50), xytext=(91, 76),
            arrowprops=dict(arrowstyle='<->', lw=2, color='#006D2C', linestyle='--'), zorder=1)
ax.text(82, 63, 'Vector\nLookup', ha='center', fontsize=8, color='#006D2C',
        fontweight='bold', style='italic')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_rag_architecture_diagram.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Visualisasi arsitektur RAG disimpan: 05_rag_architecture_diagram.png")

# ============================================================
# BAGIAN 8 - EMBEDDING & VECTOR DATABASE
# ============================================================
print("\n" + "=" * 80)
print("BAGIAN 8: EMBEDDING & VECTOR DATABASE")
print("=" * 80)

print("""
----------------------------------------------------------------------
8.1 APA ITU EMBEDDING?
----------------------------------------------------------------------
Embedding = Proses memetakan DATA TINGKAT TINGGI (teks, gambar, audio)
menjadi VEKTOR ANGKA dengan dimensi tetap (misal 256D, 768D, 1536D, 3072D),
dimana:
  ✅ Vektor dekat  → makna serupa
  ✅ Vektor jauh   → makna berbeda
  ✅ Operasi aritmatika vektor bermakna (contoh terkenal Word2Vec:
     king - man + woman ≈ queen)

PROVIDER EMBEDDING MODEL POPULER:
  • OpenAI     : text-embedding-3-small (1536D), text-embedding-ada-002 (1536D)
  • Google     : text-embedding-004, gecko-embedding
  • Open Source: all-MiniLM-L6-v2 (384D, cepat!), BERT-base (768D), LLaMA-Embed
  • Cohere     : embed-english-v3.0 (1024D)
""")

# --- Demo Simulasi Embedding + Visualisasi 2D (PCA) ---
print("\n>>> Demo Simulasi Embedding 10 kalimat → Cosine Similarity Matrix")

kalimat_embedding_demo = [
    "Cara training model Random Forest klasifikasi",
    "Tuning hyperparameter Random Forest dengan Grid Search",
    "Decision Tree algoritma dasar dari Random Forest",
    "Cara preprocessing gambar untuk CNN klasifikasi digit MNIST",
    "Training CNN dengan TensorFlow Keras dataset MNIST",
    "Arsitektur LeNet-5 AlexNet VGG ResNet untuk Computer Vision",
    "Text preprocessing Bahasa Indonesia case folding stopword",
    "TF-IDF vectorizer untuk ekstraksi fitur teks sentimen",
    "Transformer attention mechanism untuk LLM generative",
    "Prompt engineering Zero-shot Few-shot Chain-of-Thought",
]

# Simulasi vektor embedding 20D per kalimat (dimensi reduksi untuk demo)
np.random.seed(99)
simulasi_embeddings = []
# Cluster 1 (RF/ML) - vektor basis serupa
base1 = np.random.randn(20) * 3
for _ in range(3):
    simulasi_embeddings.append(base1 + np.random.randn(20) * 1.0)
# Cluster 2 (CV/CNN)
base2 = np.random.randn(20) * 3
for _ in range(3):
    simulasi_embeddings.append(base2 + np.random.randn(20) * 1.0)
# Cluster 3 (NLP)
base3 = np.random.randn(20) * 3
for _ in range(4):
    simulasi_embeddings.append(base3 + np.random.randn(20) * 1.0)
simulasi_embeddings = np.array(simulasi_embeddings)
print(f"    Shape embedding matrix: {simulasi_embeddings.shape} (10 teks × 20 dimensi)")

# Hitung Cosine Similarity matrix (pakai alias dari import atas baris 29)
cos_matrix = cosine_similarity(simulasi_embeddings)

# Visualisasi 2 panel: Heatmap similarity + Scatter 2D
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: Heatmap Cosine Sim
short_labels = [f"T{i+1}: {t[:27]}.." for i, t in enumerate(kalimat_embedding_demo)]
sns.heatmap(cos_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0.5,
            xticklabels=short_labels, yticklabels=short_labels, ax=ax1,
            cbar_kws={'label': 'Cosine Similarity'})
ax1.set_title('Cosine Similarity Matrix (Simulasi Embedding)\nWarna merah = sangat mirip, biru = berbeda topik',
              fontsize=11, fontweight='bold')
ax1.tick_params(axis='both', labelsize=7)

# Panel 2: PCA 2D Scatter Plot
pca = PCA(n_components=2, random_state=42)
emb_2d = pca.fit_transform(simulasi_embeddings)

cluster_colors = ['#E45756']*3 + ['#4C78A8']*3 + ['#54A24B']*4
cluster_labels_text = ['ML (Random Forest)']*3 + ['CV (CNN)']*3 + ['NLP/GenAI']*4

for i in range(len(emb_2d)):
    ax2.scatter(emb_2d[i, 0], emb_2d[i, 1], s=220, c=cluster_colors[i],
                edgecolors='black', linewidths=2, zorder=3)
    ax2.annotate(f"T{i+1}", (emb_2d[i, 0], emb_2d[i, 1]),
                 ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Legend manual
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='ML (Random Forest)',
               markerfacecolor='#E45756', markersize=14, markeredgecolor='black'),
    plt.Line2D([0], [0], marker='o', color='w', label='CV (CNN/Computer Vision)',
               markerfacecolor='#4C78A8', markersize=14, markeredgecolor='black'),
    plt.Line2D([0], [0], marker='o', color='w', label='NLP / Generative AI',
               markerfacecolor='#54A24B', markersize=14, markeredgecolor='black'),
]
ax2.legend(handles=legend_elements, loc='upper left', fontsize=10)
ax2.set_xlabel(f'Principal Component 1 (var explained: {pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
ax2.set_ylabel(f'Principal Component 2 (var explained: {pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
ax2.set_title('Visualisasi Embedding 2D via PCA\n(Teks serupa → berdekatan dalam ruang vektor)',
              fontsize=11, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_embedding_similarity_pca_visual.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Visualisasi embedding disimpan: 06_embedding_similarity_pca_visual.png")

print("""
----------------------------------------------------------------------
8.2 VECTOR DATABASE
----------------------------------------------------------------------
Vector Database = Database khusus yang di-OPTIMALKAN untuk:
  • MENYIMPAN vektor embedding (jutaan → milyaran vektor)
  • MENCARI tetangga terdekat (Top-K Nearest Neighbors) DENGAN CEPAT
    → bukan O(n) linear search, tapi menggunakan INDEXING khusus:
      • HNSW (Hierarchical Navigable Small World)  → paling umum
      • IVF (Inverted File) + Flat / PQ
      • FAISS Library (Facebook) + banyak library lain

VECTOR DB POPULER:
  ┌──────────────┬────────────────────────────────────────────┐
  │ Nama         │ Tipe & Karakteristik                        │
  ├──────────────┼────────────────────────────────────────────┤
  │ ChromaDB     │ Open-source, lokal ringan, Python-native   │
  │ FAISS        │ Library Meta, super cepat, butuh wrapper   │
  │ Pinecone     │ Managed SaaS, scalable, tanpa infra        │
  │ Weaviate     │ Open-source + managed, GraphQL/REST API    │
  │ pgvector     │ Extension PostgreSQL → pake DB biasa!      │
  │ Qdrant       │ Open-source Rust, cepat, production-grade  │
  │ Milvus/Zilliz│ Enterprise open-source, skala milyaran     │
  └──────────────┴────────────────────────────────────────────┘

Try-import library vector DB:
""")

for lib_name in ['chromadb', 'faiss', 'qdrant_client']:
    try:
        __import__(lib_name)
        print(f"  ✅ Library {lib_name} TERINSTALL")
    except ImportError:
        print(f"  ⚠️  Library {lib_name} TIDAK terinstall (opsional untuk Modul 9)")

# ============================================================
# BAGIAN 9 - MEMBANGUN APLIKASI BERBASIS LLM (CHATBOT RAG)
# ============================================================
print("\n" + "=" * 80)
print("BAGIAN 9: MEMBANGUN APLIKASI BERBASIS LLM (CHATBOT RAG SEDERHANA)")
print("=" * 80)

print("""
----------------------------------------------------------------------
9.1 STRUKTUR APLIKASI CHATBOT RAG PRODUKSI (REFERENSI ARSITEKTUR)
----------------------------------------------------------------------
Tumpukan teknologi minimum untuk Chatbot QA Internal Perusahaan:

  [LAYER UI]        : Streamlit / Gradio / Next.js + React (Frontend chat)
  [LAYER API]       : FastAPI / Flask / LangServe (Backend endpoints)
  [LAYER ORKESTRASI]: LangChain / LlamaIndex (RAG pipeline, tool calling)
  [LAYER LLM]       : Provider OpenAI / Anthropic / Groq / Local LLM (Ollama)
  [LAYER EMBEDDING] : OpenAI text-embedding / Sentence-Transformers (open source)
  [LAYER VECTOR DB] : ChromaDB / Qdrant / pgvector / Pinecone
  [LAYER SUMBER]    : Loader (PyPDF2, Unstructured, S3, Google Drive, Notion, Slack)
  [LAYER LOGGING]   : LangSmith / Langfuse / W&B Prompts (trace & evaluasi)
  [LAYER AUTH]      : OAuth2 / JWT / SSO (keamanan data perusahaan)
""")

# --- Implementasi Chatbot Sederhana Class (Offline Demo) ---
print("\n>>> Implementasi Demo Class SimpleChatbotRAG (menggabungkan semua modul di atas)")

class SimpleChatbotRAG:
    """Chatbot RAG sederhana untuk pelatihan (offline, tanpa API key)."""

    MEMORY_MAX_LEN = 10  # Simpan 5 Q + 5 A terakhir

    def __init__(self, knowledge_base, nama_bot="BNSP-AIBot"):
        self.nama = nama_bot
        self.kb = knowledge_base
        self._corpus = [c['teks'] for c in knowledge_base]
        self._vect = TfidfVectorizer(ngram_range=(1, 2)).fit(self._corpus)
        self._tfidf_mat = self._vect.transform(self._corpus)
        self.chat_history = []  # memory percakapan
        self.llm_client = DummyLLMAPIClient(model_name=f"{nama_bot}-model")
        print(f"    [ChatbotRAG] {self.nama} SIAP dengan KB={len(self.kb)} dokumen.")

    def _retrieve(self, q, k=3):
        vec_q = self._vect.transform([q])
        sims = cos_sim(vec_q, self._tfidf_mat).flatten()
        top_idx = np.argsort(sims)[::-1][:k]
        return [
            {**self.kb[i], 'score': float(sims[i])} for i in top_idx if sims[i] > 0.03
        ]

    def _tambah_memory(self, role, content):
        self.chat_history.append({"role": role, "content": content})
        # Trim memory
        if len(self.chat_history) > self.MEMORY_MAX_LEN:
            self.chat_history = self.chat_history[-self.MEMORY_MAX_LEN:]

    def chat(self, pertanyaan_user, gunakan_rag=True, tampilkan_sumber=True):
        self._tambah_memory('user', pertanyaan_user)

        # Step 1: Retrieve
        retrieved = self._retrieve(pertanyaan_user) if gunakan_rag else []

        # Step 2: Build messages (termasuk chat history memory pendek)
        messages = [
            {"role": "system",
             "content": (f"Kamu adalah {self.nama}, asisten QA untuk pelatihan BNSP. "
                         "Jawab dengan Bahasa Indonesia yang sopan dan jelas.")}
        ]
        # Masukkan memory (hanya 5 pesan terakhir sebelum ini)
        for msg in self.chat_history[-6:-1]:
            messages.append(msg)
        # Masukkan RAG context jika ada
        if retrieved:
            ctx = "\n\n".join([f"[{r['sumber']}]: {r['teks'][:250]}" for r in retrieved])
            messages.append({
                "role": "system",
                "content": f"KONTEKS DARI KNOWLEDGE BASE:\n{ctx}\n"
                           f"Jawab berdasarkan konteks di atas jika relevan."
            })
        messages.append({"role": "user", "content": pertanyaan_user})

        # Step 3: Generate (pakai dummy client)
        response = self.llm_client.chat_completion(messages, temperature=0.3, max_tokens=400)
        jawaban = response['choices'][0]['message']['content']

        # Tambah sumber
        if tampilkan_sumber and retrieved:
            jawaban += "\n\n📚 Sumber Referensi:\n"
            for r in retrieved:
                jawaban += f"  • [{r['score']:.3f}] {r['sumber']}\n"

        self._tambah_memory('assistant', jawaban)
        return jawaban

# Demo jalankan chatbot
bot = SimpleChatbotRAG(knowledge_base_chunks, nama_bot="BNSP-AI-QA")

riwayat_chat_demo = [
    "Halo, perkenalkan dirimu dong!",
    "Apa saja bukti portofolio yang dibutuhkan untuk BNSP AI Engineer?",
    "Oke, lalu untuk proyek ML klasifikasi, bagaimana cara split dataset yang benar?",
    "Terima kasih informasinya!",
]
print("\n>>> 🗨️  Demo Percakapan Chatbot 4 Turn (dengan Memory + RAG):")
for i, q in enumerate(riwayat_chat_demo):
    print(f"\n  👤 User {i+1}: {q}")
    jawaban_bot = bot.chat(q, gunakan_rag=True)
    print(f"  🤖 {bot.nama}: {jawaban_bot[:220]}" + ("..." if len(jawaban_bot) > 220 else ""))

print(f"\n    📜 Memory chat sekarang: {len(bot.chat_history)} pesan tersimpan")

# ============================================================
# BAGIAN 10 - EVALUASI OUTPUT GENERATIVE AI & RESPONSIBLE LLM AI
# ============================================================
print("\n" + "=" * 80)
print("BAGIAN 10: EVALUASI OUTPUT GENERATIVE AI & RESPONSIBLE LLM AI")
print("=" * 80)

print("""
----------------------------------------------------------------------
10.1 EVALUASI OUTPUT LLM (LEBIH KOMPLEKS DARI ML TRADISIONAL!)
----------------------------------------------------------------------
Masalah: Output LLM = teks panjang terbuka, bukan label kelas → butuh
         metrik khusus dan seringkali MELIBATKAN MANUSIA (Human-in-the-Loop).

3 KATEGORI EVALUASI LLM:

A. EVALUASI OBYEKTIF / METRIK OTOMATIS (tidak butuh manusia):
   • BLEU-4 / ROUGE-L / METEOR      → Machine Translation, Summarization
   • BERTScore / BLEURT             → Similaritas semantik dengan embedding
   • LLM-as-Judge (pakai GPT-4o menilai output LLM lain) → trending!
   • Exact Match / F1 SQuAD         → QA (jawaban singkat)
   • Pass@k (code generation)       → Code LLM (HumanEval benchmark)

B. EVALUASI DIMENSI SUBYEKTIF (rating manusia 1-5 skala Likert):
   • ✅ FAKTUALITAS / HALLUCINATION RATE
   • ✅ RELEVANSI jawaban dengan pertanyaan
   • ✅ KOHERENSI & kelancaran bahasa
   • ✅ KEBERGUNAAN (Groundedness = didukung sumber?)
   • ✅ KREATIVITAS (untuk tugas creative)
   • ✅ KEAMANAN (tidak menghasilkan konten berbahaya)

C. EVALUASI RAG SPESIFIK (RAG TRIAD METRIC - Amazon AWS):
   • 🔍 CONTEXT RECALL    : Seberapa banyak info penting dari ground truth
                            TERCAKUP di context yang berhasil di-retrieve?
   • 🔍 CONTEXT PRECISION : Seberapa BERSIH context yang di-retrieve dari
                            hal yang TIDAK RELEVAN?
   • 🧠 ANSWER RELEVANCE  : Seberapa RELEVAN jawaban akhir dengan pertanyaan?
   • 🧠 ANSWER CORRECTNESS: Seberapa BENAR jawaban dibanding ground truth?
   • 🧠 GROUNDEDNESS      : Berapa banyak pernyataan di jawaban BISA DIBUKTIKAN
                            dengan sumber context di RAG?
""")

# --- Simulasi Data Evaluasi Dimensi ---
dimensi = [
    'Faktualitas\n(Anti-Hallucinasi)', 'Relevansi\nPertanyaan',
    'Koherensi\nBahasa', 'Kegunaan\n(Groundedness)', 'Keamanan\n(Safety)',
    'Kreativitas', 'Context Recall\n(RAG)', 'Context Precision\n(RAG)',
]
skor_model1 = [0.88, 0.92, 0.95, 0.82, 0.97, 0.68, 0.80, 0.75]  # LLM Akurat tapi kaku
skor_model2 = [0.76, 0.85, 0.90, 0.70, 0.88, 0.94, 0.68, 0.62]  # LLM Kreatif tapi sering ngawur

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(dimensi))
width = 0.35
bars1 = ax.bar(x - width/2, [s * 100 for s in skor_model1], width,
               label='LLM-A (Balita-Jujur / Akurat)', color='#4C78A8', edgecolor='white')
bars2 = ax.bar(x + width/2, [s * 100 for s in skor_model2], width,
               label='LLM-B (Kreatif-Halusinasi)', color='#E45756', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(dimensi, fontsize=10)
ax.set_ylabel('Skor Persentase (%)', fontsize=11)
ax.set_ylim(50, 105)
ax.set_title('EVALUASI 8 DIMENSI PERBANDINGAN 2 MODEL LLM\n'
             '(Simulasi Rating 1-5 × 20 responden manusia)',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(axis='y', alpha=0.3)
for bar_group in [bars1, bars2]:
    for bar in bar_group:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_llm_evaluation_8dimensions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Visualisasi evaluasi LLM disimpan: 07_llm_evaluation_8dimensions.png")

print("""
----------------------------------------------------------------------
10.2 RESPONSIBLE AI KHUSUS UNTUK GENERATIVE AI & LLM
----------------------------------------------------------------------
Masalah Generative AI yang TIDAK ditemukan (atau minim) di AI tradisional:
  ❌ HALLUCINATION : LLM "ngarang" fakta yang terdengar masuk akal tapi SALAH
  ❌ DEEPFAKE & HAK CIPTA : Gambar/audio/video sintetis meniru manusia
  ❌ BIASA YANG TERAMPLIFIKASI : Model memperkuat stereotipe gender/ras
  ❌ PRIVASI DATA TRAINING : Model "mengingat" data privat dari training
  ❌ KECURANGAN AKADEMIK & PEKERJAAN : Plagiat esai/kode oleh LLM
  ❌ JAHAT (JAILBREAK) : Prompt rekayasa agar model bypass safety guardrails

5 STRATEGI MITIGASI RESPONSIBLE LLM (WAJIB DIKETAHUI BNSP):

1. 🛡️ GUARDRAILS (Keamanan Input + Output)
   • Input filter  : Cek pertanyaan user → blok jika harmful/jailbreak
   • Output filter : Cek jawaban LLM → blok/tolak jika berbahaya, PII bocor
   • Tool: NVIDIA NeMo Guardrails, AWS Bedrock Guardrails, Llama Guard

2. 🧠 HALLUCINATION MITIGATION
   • Implementasikan RAG SELALU untuk domain spesifik (jangan andalkan training data)
   • Aktifkan CITATION / forced grounding dengan sumber
   • Gunakan LLM-as-Judge untuk mengecek self-consistency
   • Temperature rendah (0.0 - 0.3) untuk tugas faktual

3. ⚖️ FAIRNESS & BIAS EVALUATION
   • Test dengan dataset benchmark bias (BBQ, CrowS-Pairs, StereoSet)
   • Periksa output untuk gender, ras, agama, kelompok rentan
   • Prompt engineering: "Pastikan jawaban tidak mengandung stereotipe apapun."

4. 🔒 PRIVASI & DATA PROTECTION (PDP Law No 27/2022 Indonesia)
   • JANGAN kirim data rahasia perusahaan ke API LLM public tanpa NDA & DPA
   • Gunakan LOCAL LLM (Ollama + LLaMA 3 / Mistral) untuk data super sensitif
   • Differential Privacy saat training fine-tuning
   • PII redaction otomatis (hapus nomor KTP, email, HP sebelum masuk prompt)

5. 📜 TRANSPARANSI & AKUNTABILITAS
   • Tampilkan WATERMARKING untuk output AI (tanda bukti sintetis)
   • Simpan LOG audit lengkap (timestamp, user, prompt, output, model, cost)
   • Dokumentasikan batasan kemampuan model dengan jelas kepada pengguna
   • Adakan tim redaksi/manusia reviewer untuk output critical (medis, hukum)
""")

# --- Simulasi Hallucination Rate Pre vs Post RAG ---
print("\n>>> Simulasi Perbaikan Hallucination dengan RAG (dummy data evaluasi):")
simulasi_rag_eval = pd.DataFrame({
    'Skenario': ['LLM Langsung\n(No RAG)', 'LLM + RAG\n(Context 3 dok)',
                 'LLM + RAG + LLM Judge\n+ Citasi Wajib'],
    'Hallucination_Rate_Persen': [38.5, 11.2, 3.7],
    'Avg_Groundedness_Skor': [0.52, 0.84, 0.96],
    'Avg_Response_Time_S': [0.8, 1.9, 3.2],
})
print(simulasi_rag_eval.to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
bars1 = ax1.bar(simulasi_rag_eval['Skenario'],
                simulasi_rag_eval['Hallucination_Rate_Persen'],
                color=['#E45756', '#F58518', '#54A24B'], edgecolor='white', linewidth=2)
ax1.set_ylabel('Hallucination Rate (%)', fontsize=11)
ax1.set_title('PENGURANGAN HALLUCINATION BERKAT RAG', fontsize=12, fontweight='bold')
for bar, val in zip(bars1, simulasi_rag_eval['Hallucination_Rate_Persen']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val}%', ha='center', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

bars2 = ax2.bar(simulasi_rag_eval['Skenario'],
                simulasi_rag_eval['Avg_Groundedness_Skor'] * 100,
                color=['#E45756', '#F58518', '#54A24B'], edgecolor='white', linewidth=2)
ax2.set_ylabel('Groundedness Skor (%)', fontsize=11)
ax2.set_title('KENAIKAN KEBENARAN DIDUKUNG SUMBER', fontsize=12, fontweight='bold')
for bar, val in zip(bars2, simulasi_rag_eval['Avg_Groundedness_Skor'] * 100):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_rag_improves_hallucination_groundedness.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("✅ Visualisasi perbaikan hallucination disimpan: 08_rag_improves_hallucination_groundedness.png")

# ============================================================
# RINGKASAN FINAL MODUL 9
# ============================================================
print("\n" + "=" * 80)
print("📋 RINGKASAN AKHIR MODUL 9 - GENERATIVE AI & LLM")
print("=" * 80)
print("""
+===============================================================+
|  Poin Kunci yang Wajib Dikuasai untuk Uji Kompetensi BNSP     |
+===============================================================+
| 1. Perbedaan mendasar Traditional AI vs Generative AI         |
| 2. Arsitektur Transformer (Encoder/Decoder/MHSA/PosEnc)       |
| 3. Teknik Prompting: Zero/Few-Shot, CoT, ReAct, Structured   |
| 4. RAG Architecture end-to-end (Ingesti 4 langkah, Query 5)   |
| 5. Konsep Embedding + Vector DB (HNSW index, provider list)   |
| 6. Pattern panggilan LLM API (messages, temperature, dll)     |
| 7. Evaluasi LLM: RAG Triad, LLM-as-Judge, Hallucination Rate |
| 8. Responsible LLM: Guardrails, Privacy PDP 27/2022, Bias     |
+===============================================================+
""")
print(f"Total visualisasi di folder output_charts: "
      f"{len(os.listdir(OUTPUT_DIR))} file PNG")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  ✔ {f}")

# ============================================================
# BLOK LATIHAN PRAKTIK - PROYEK BERBASIS BELAJAR BNSP
# ============================================================
print("\n" + "=" * 80)
print("🏋️  LATIHAN PRAKTIK MANDIRI (MATERI PORTOFOLIO BNSP)")
print("=" * 80)

print("""
LATIHAN 1 (Pemahaman Prompt Engineering)
   Buat 3 variasi prompt untuk tugas yang sama:
   Tugas = "Ekstrak 5 keyword + ringkasan 3 kalimat dari berita berikut:
   [berita panjang tentang kecelakaan AI di industri manufaktur]"
   Buat versi: (a) Zero-shot, (b) Few-shot 2 contoh, (c) Chain-of-Thought.
   Bandingkan output ketiganya dalam tabel: akurasi, format, hallucination.
   Simpan dalam file: latihan_09_01_prompt_engineering.py

LATIHAN 2 (Structured Prompting)
   Buatlah sebuah prompt yang memaksa LLM mengeluarkan output HANYA
   format JSON dengan skema PYDANTIC berikut:
   class KuesionerKepuasan(BaseModel):
       nim: str
       nama: str
       skor_pemahaman: Literal[1,2,3,4,5]
       saran_perbaikan: list[str] (maks 3 saran)
       rekomendasi_lanjut: bool
   Simulasikan parsing JSON output dan hitung rata-rata skor untuk 5 responden dummy.

LATIHAN 3 (Full RAG Pipeline - Proyek Mini)
   Ambil 3 PDF modul pelatihan Anda (Modul 1, Modul 3, Modul 5), lalu:
   a. Load setiap halaman dengan library PyPDF2/pdfplumber
   b. Chunk menjadi 500 karakter dengan 10% overlap
   c. Vectorize dengan TF-IDF (atau SentenceTransformers jika terinstall)
   d. Bangun aplikasi CLI chatbot yang menerima pertanyaan, retrieve top-3
      dokumen, lalu print jawaban simulasi + nama sumber PDF + halaman.
   Ini adalah BUKTI PORTOFOLIO RAG untuk BNSP yang kuat!

LATIHAN 4 (Evaluasi & Responsible LLM)
   Buat dataset 20 pertanyaan + ground truth jawaban tentang materi
   Modul 1-9. Buat 2 skenario:
   Skenario A: Jawaban dihasilkan oleh LLM TANPA RAG (simulasi sendiri)
   Skenario B: Jawaban dihasilkan DENGAN RAG dari materi (pakai kode RAG di atas)
   Minta 2 teman Anda menilai dengan skala 1-5 untuk dimensi:
   (Faktualitas, Relevansi, Koherensi, Keberadaan Sitasi).
   Buat grouped bar chart perbandingan kedua skenario seperti Gambar 07 dan 08.

LATIHAN 5 (Mitigasi Bahaya LLM - Guardian Script)
   Buatlah fungsi Python:
       def llm_safety_guardrails(prompt: str) -> tuple[bool, str]:
   Yang mengembalikan:
   • (True, "Aman") jika prompt lolos
   • (False, "Alasan pemblokiran") jika mengandung:
     - Kata kunci jahat (jailbreak, lupakan instruksi, buat malware, dll)
     - Nomor KTP / NIK / KK (regex 16 digit angka)
     - Nomor rekening bank (10-16 digit)
   Uji fungsi dengan 10 contoh prompt: 5 aman, 5 berbahaya.
   Dokumentasikan confusion matrix hasil pengujian untuk bukti portofolio.
""")

print("\n" + "=" * 80)
print("✅ MODUL 9 GENERATIVE AI & LLM SELESAI!")
print("   Semua 10 sub-topik BNSP telah tercakup beserta 8 visualisasi chart.")
print("=" * 80)
