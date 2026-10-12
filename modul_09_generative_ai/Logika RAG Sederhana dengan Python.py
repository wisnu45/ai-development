import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- FASE 1: INGESTI (Memasukkan Dokumen Internal Perusahaan) ---
knowledge_base = [
    "Skema Sertifikasi BNSP AI Engineer memiliki 12 unit kompetensi utama.",
    "Data preparation memakan waktu 80% dari total pekerjaan AI Engineer.",
    "Untuk evaluasi data imbalanced, gunakan metrik F1-Score, jangan Akurasi.",
    "Model deployment di industri biasanya menggunakan FastAPI dan Docker."
]

# Ubah teks menjadi Vektor Matematis (Simulasi Embedding)
vectorizer = TfidfVectorizer()
vector_database = vectorizer.fit_transform(knowledge_base)

# --- FASE 2: RETRIEVAL & GENERATION (Saat User Bertanya) ---
pertanyaan_user = "Metrik apa yang bagus untuk data yang tidak seimbang?"

# 1. Ubah pertanyaan jadi vektor
vector_tanya = vectorizer.transform([pertanyaan_user])

# 2. Cari dokumen termirip (Retrieval / Cosine Similarity)
skor_mirip = cosine_similarity(vector_tanya, vector_database).flatten()
index_terbaik = np.argmax(skor_mirip)
dokumen_referensi = knowledge_base[index_terbaik]

print(">>> Proses RAG Berjalan <<<")
print(f"🔍 Pertanyaan User  : '{pertanyaan_user}'")
print(f"📚 Dokumen Ditemukan: '{dokumen_referensi}' (Skor: {skor_mirip[index_terbaik]:.2f})")

# 3. Augmentasi Prompt ke LLM
prompt_ke_llm = f"""
Sistem: Kamu adalah Asisten AI yang jujur. Jawab pertanyaan BERDASARKAN KONTEKS SAJA.
Jika jawaban tidak ada di konteks, bilang "Saya tidak tahu".

KONTEKS: {dokumen_referensi}
PERTANYAAN: {pertanyaan_user}
JAWABAN:
"""
# Di dunia nyata, 'prompt_ke_llm' ini akan kita lempar ke API OpenAI/Claude.
print(f"\n📨 Prompt yang dikirim ke LLM:\n{prompt_ke_llm}")