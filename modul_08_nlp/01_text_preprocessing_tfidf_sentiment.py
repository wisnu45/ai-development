print("="*60)
print("MODUL 8: NATURAL LANGUAGE PROCESSING (NLP)")
print("Bagian 1: Text Preprocessing, TF-IDF, Sentiment Analysis & Word Embedding")
print("="*60)

import os
import re
import time
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
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.decomposition import PCA
except ImportError as e:
    print("=" * 80)
    print("⚠️  MODUL 8 GAGAL BERJALAN - DEPENDENSI TIDAK DITEMUKAN")
    print("=" * 80)
    print(f"Penyebab: {e}")
    print("\nSolusi (CMD):")
    print("  cd /d C:\\ai-engineer")
    print("  venv_bnsp\\Scripts\\activate")
    print("  pip install -r requirements.txt")
    print("=" * 80)
    import sys
    sys.exit(1)

output_dir = os.path.join(os.path.dirname(__file__), "output_charts")
os.makedirs(output_dir, exist_ok=True)

sns.set_style("whitegrid")
np.random.seed(42)

print("""
┌─────────────────────────────────────────────────────────────────┐
│        NATURAL LANGUAGE PROCESSING (PEMROSESAN BAHASA ALAM)     │
├─────────────────────────────────────────────────────────────────┤
│  PIPELINE NLP:                                                   │
│                                                                 │
│  [1] TEXT ACQUISITION → Scraping, API, file (txt/csv/pdf/docx)  │
│                                                                 │
│  [2] TEXT PREPROCESSING ← TOPIK UTAMA (PENTING BNSP!)           │
│      ├─ a. Case Folding        → Lowercase / Uppercase semua    │
│      ├─ b. Cleaning            → Hapus HTML, URL, hashtag, @,   │
│      │                           angka, punctuation, emoji      │
│      ├─ c. Tokenization        → Pecah teks → token (kata)      │
│      ├─ d. Stopword Removal    → Buang kata tidak informatif    │
│      │   (contoh: yang, di, ke, dan, adalah, the, is, a, ...)   │
│      ├─ e. Stemming            → Potong kata ke kata dasar      │
│      │   (meminjamkan → pinjam; walking → walk)                 │
│      ├─ f. Lemmatization       → Kembalikan ke lemma (kamus)    │
│      │   (better → good; ran → run; was → be)                   │
│      ├─ g. Normalization       → Slang → baku, typo correction  │
│      └─ h. POS Tag / NER       → Label: kata benda/kerja/nama   │
│                                                                 │
│  [3] FEATURE EXTRACTION TEKS (Text → Vektor Angka):             │
│      ├─ a. Bag of Words (BoW) → Frekuensi kata (hitungan)      │
│      ├─ b. TF-IDF = Term Frequency x Inverse Doc Frequency     │
│      │   ⚡ PALING UMUM dipakai + SANGAT RELEVAN DI BNSP!       │
│      ├─ c. Word Embedding = d-dimensional vektor dense          │
│      │   (Word2Vec CBOW/Skipgram, GloVe, FastText)             │
│      ├─ d. Contextual Embedding = Transformer (BERT dkk)       │
│                                                                 │
│  [4] HIGH-LEVEL NLP TASK:                                        │
│      ├─ Text Classification: Sentiment, Topic, Spam, dll       │
│      ├─ Named Entity Recognition (NER): Orang/Lokasi/Org/Tgl   │
│      ├─ Machine Translation: ID ↔ EN, dll                      │
│      ├─ Text Summarization: Ekstraktif / Abstraktif            │
│      ├─ Question Answering, Chatbot, Speech2Text, dll.          │
└─────────────────────────────────────────────────────────────────┘
""")

print("\n>>> 1. LOAD DATASET SENTIMEN - REVIEW PRODUK E-COMMERCE INDONESIA (SINTETIS)")
print("-" * 60)

np.random.seed(42)
n = 600
positive_words = ["bagus", "sangat bagus", "cepat", "pengiriman cepat", "ori", "asli", "berkualitas",
                  "memuaskan", "ramah", "recommended", "worth it", "sesuai", "tepat waktu",
                  "barang sesuai foto", "seller ramah", "packing rapi", "mulus", "barang baru",
                  "suka", "puas", "terima kasih", "respon cepat"]
negative_words = ["rusak", "palsu", "kw", "kw super", "lambat", "pengiriman lambat",
                  "tidak sesuai", "penipuan", "scam", "barang cacat", "pecah", "lecet",
                  "seller tidak ramah", "respon lambat", "tidak recommended", "kecewa",
                  "tidak ori", "bedanya tipis", "barang lama", "salah kirim", "hilang"]
neutral_words = ["standar", "biasa saja", "ok", "lumayan", "tidak ada komentar", "cukup",
                 "sesuai deskripsi", "normal"]

def generate_review(sentiment, n_words_range=(10, 40)):
    if sentiment == "positive":
        words = np.random.choice(positive_words, size=np.random.randint(3, 8), replace=False).tolist()
    elif sentiment == "negative":
        words = np.random.choice(negative_words, size=np.random.randint(3, 8), replace=False).tolist()
    else:
        words = np.random.choice(neutral_words, size=np.random.randint(2, 5), replace=False).tolist()

    fillers = ["produknya", "pengirimannya", "pelayanannya", "barangnya", "packingnya",
               "overall", "jujur saja", "setelah dipakai", "kedepannya", "untuk harganya",
               "saya pikir", "ternyata", "sejauh ini", "cuma"]
    structures = [
        lambda: f"Overall {' '.join(words)}. {np.random.choice(fillers)} {np.random.choice(words if len(words) > 2 else neutral_words)} juga. Semoga awet!",
        lambda: f"{' '.join([w.title() for w in words[:2]])} banget! Sisanya {' dan '.join(words[2:]) if len(words) > 2 else np.random.choice(fillers)}. {'Baguslah untuk harga segini' if sentiment != 'negative' else 'Kecewa berat saya beli di sini.'}",
        lambda: f"Pesan {np.random.choice(['minggu lalu', 'kemarin', 'hari senin'])}, hari ini datang. Kondisi barang: {' '.join(words)}. {'Saya beli lagi nanti.' if sentiment == 'positive' else ('Next time lebih teliti.' if sentiment == 'negative' else 'Mungkin nanti coba lagi.')}",
        lambda: f"{'⭐' * (5 if sentiment == 'positive' else (3 if sentiment == 'neutral' else 2))} Review jujur: {' '.join(words)}. {np.random.choice(['Terima kasih','Saran: perbaiki packaging','Ya sudahlah'])} seller!",
        lambda: f"Untuk harganya {' '.join(words)}. Kualitas {' dan '.join(np.random.choice(words, max(2, len(words)//2), replace=False))}. Shipping {np.random.choice(['ok','cepet','lambat','normal'])}",
    ]
    text = np.random.choice(structures)()
    # tambah noise: beberapa huruf kapital acak, tanda baca berlebih
    if np.random.random() < 0.4:
        text += np.random.choice(["", "", "!!!", " !!", "....", " ???", "👍", "👎", "😊", "😡", "  "])
    return text

texts, labels = [], []
for i in range(n):
    sentiment = np.random.choice(["positive", "neutral", "negative"], p=[0.45, 0.20, 0.35])
    texts.append(generate_review(sentiment))
    labels.append(sentiment)

df = pd.DataFrame({"review_text": texts, "sentiment": labels})
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["sentiment_label"] = df["sentiment"].map(label_map)

print(f"Shape dataset review: {df.shape}")
print(f"Distribusi sentimen:")
for s, cnt in df["sentiment"].value_counts().items():
    print(f"  {s.upper():<10}: {cnt:>3} ({cnt/len(df)*100:.1f}%)")
print("\nContoh 3 review per kelas:")
for s in ["negative", "neutral", "positive"]:
    print(f"\n[{s.upper()}]")
    for t in df[df["sentiment"]==s]["review_text"].head(3).tolist():
        print(f"   • {t[:100]}{'...' if len(t)>100 else ''}")

print("\n>>> 2. TEXT PREPROCESSING PIPELINE - 6 LANGKAH BAHASA INDONESIA")
print("-" * 60)

STOPWORDS_ID = set("""
yang dan di ke dari pada untuk dengan ini itu adalah dalam juga tidak ada atau saya
akan oleh kita bisa lebih dia seperti sudah setelah akan mereka menjadi kamu
dapat tetapi karena hanya satu harus beberapa menjadi antara jika tanpa maka
pun sendiri apakah sangat sehingga namun ataupun maupun tak apa sih lagi
lah kok ya dong nah saja kan bang kak sis mas mba pak bu umi abi tau tanya
wkwkwk wkwk haha hehe hihi btw tpi tp ya udh udeh gak ga dah org sdh krn
yg dg dgn utk dlm pd org kalo klo kpd krn sbg tdk ato tpi tgl dll nih
nggak nggak enggak bikin buat bilang kasih lanjut nyata nyata
""".split())

def stemming_id_sederhana(kata):
    """Sederhana aturan imbuhan Bahasa Indonesia (tidak selengkapnya Nazief-Adriani)"""
    awalan = [("meng", ""), ("meny", "s"), ("men", ""), ("mem", ""), ("me", ""),
              ("peng", ""), ("peny", "s"), ("pen", ""), ("pem", ""), ("pe", ""),
              ("ber", ""), ("bel", ""), ("be", ""), ("per", ""), ("pel", ""),
              ("ter", ""), ("ke", ""), ("di", ""), ("se", "")]
    akhiran = [("lah", ""), ("kah", ""), ("tah", ""), ("pun", ""), ("nya", ""),
               ("ku", ""), ("mu", ""), ("kan", ""), ("an", ""), ("i", "")]

    kata = kata.lower().strip()
    # Awalan
    for pref, ganti in awalan:
        if kata.startswith(pref) and len(kata) > len(pref) + 2:
            kata = kata[len(pref):] + (ganti if ganti else "")
            break
    # Akhiran - 2x loop untuk akhiran ganda (contoh: di+...+kan+nya)
    for _ in range(2):
        for suff, ganti in akhiran:
            if kata.endswith(suff) and len(kata) > len(suff) + 2:
                kata = kata[:-len(suff)]
                break
    return kata if len(kata) >= 3 else kata

def preprocess_text(text):
    steps = {}
    original = str(text)
    # Step 1: Case folding (lowercase)
    t1 = original.lower()
    steps["1_lower"] = t1
    # Step 2: Cleaning - hapus URL, tag HTML, hashtag, @, tanda baca berlebih
    t2 = re.sub(r"https?://\S+", " ", t1)
    t2 = re.sub(r"<.*?>", " ", t2)
    t2 = re.sub(r"@\w+|#\w+", " ", t2)
    t2 = re.sub(r"[^\w\s]", " ", t2)
    t2 = re.sub(r"\d+", " ", t2)
    t2 = re.sub(r"\s+", " ", t2).strip()
    steps["2_clean"] = t2
    # Step 3: Tokenization
    tokens = t2.split()
    steps["3_tokenized"] = tokens
    # Step 4: Stopword removal (kata umum dibuang)
    t4 = [w for w in tokens if w not in STOPWORDS_ID]
    steps["4_stopword_removed"] = t4
    # Step 5: Stemming Bahasa Indonesia
    t5 = [stemming_id_sederhana(w) for w in t4]
    steps["5_stemmed"] = t5
    # Step 6: Join kembali ke string (untuk TF-IDF input)
    t6 = " ".join([w for w in t5 if len(w) >= 2])
    steps["6_final_text"] = t6
    return steps

# Test pada contoh 1 review
sample_text = df["review_text"].iloc[5]
pp = preprocess_text(sample_text)
print("CONTOH PREPROCESSING 1 REVIEW:")
print(f"  [ORIGINAL]      : {sample_text[:120]}...")
for step, val in pp.items():
    if isinstance(val, list):
        print(f"  [{step.upper():<20}]: {val}")
    else:
        print(f"  [{step.upper():<20}]: {val[:100]}{'...' if len(val)>100 else ''}")

# Preprocessing seluruh dataset
t0 = time.time()
df["clean_tokens"] = df["review_text"].apply(lambda t: preprocess_text(t)["5_stemmed"])
df["clean_text"] = df["review_text"].apply(lambda t: preprocess_text(t)["6_final_text"])
df["text_length_raw"] = df["review_text"].apply(lambda x: len(x.split()))
df["text_length_clean"] = df["clean_text"].apply(lambda x: len(x.split()))
print(f"\n✅ Full preprocessing selesai: {time.time()-t0:.2f}s")
print(f"   Rata-rata panjang: {df['text_length_raw'].mean():.1f} → {df['text_length_clean'].mean():.1f} kata (hemat {100-(df['text_length_clean'].mean()/df['text_length_raw'].mean()*100):.1f}%)")

print("\n>>> 3. FEATURE EXTRACTION - Bag of Words vs TF-IDF")
print("-" * 60)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

X_text = df["clean_text"].values
y_sent = df["sentiment_label"].values
class_names = ["negative", "neutral", "positive"]

# Bag of Words (CountVectorizer)
cv = CountVectorizer(max_features=1500, min_df=2, max_df=0.90, ngram_range=(1, 2))
X_bow = cv.fit_transform(X_text).toarray()
bow_features = cv.get_feature_names_out()

# TF-IDF Vectorizer (bobot kata berdasarkan pentingnya!)
tfidf = TfidfVectorizer(max_features=1500, min_df=2, max_df=0.90, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(X_text).toarray()
tfidf_features = tfidf.get_feature_names_out()

print(f"Bag of Words Shape   : {X_bow.shape} (vocab size={len(bow_features)})")
print(f"TF-IDF Shape         : {X_tfidf.shape} (vocab size={len(tfidf_features)})")

# Top 15 TF-IDF per kelas
print("\nTop 15 TF-IDF tertinggi per kelas (kata paling khas tiap sentimen):")
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
for i, (cls, cls_name) in enumerate(zip([0, 1, 2], class_names)):
    mean_tfidf = X_tfidf[y_sent == cls].mean(axis=0)
    top_idx = np.argsort(mean_tfidf)[::-1][:15]
    top_words = [tfidf_features[i] for i in top_idx]
    top_scores = mean_tfidf[top_idx]

    print(f"\n  [{cls_name.upper()}]")
    for w, sc in zip(top_words, top_scores):
        print(f"    • {w:<20} TF-IDF={sc:.4f}")

    colors = ["#dc2626", "#f59e0b", "#16a34a"][i]
    sns.barplot(x=top_scores, y=top_words, color=colors, ax=axes[i], hue=top_words, legend=False)
    axes[i].set_title(f"Top TF-IDF: {cls_name.upper()}", fontweight="bold", fontsize=11)
    axes[i].set_xlabel("Mean TF-IDF Score")

plt.suptitle("PERBANDINGAN 15 KATA TERPENTING (TF-IDF) SETIAP KELAS SENTIMEN", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{output_dir}\\01_tfidf_top_words_per_class.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Chart Top TF-IDF tersimpan")

print("\n>>> 4. KLASIFIKASI SENTIMEN - Bandingkan BoW vs TF-IDF + 3 Algoritma")
print("-" * 60)

X_train_bow, X_test_bow, y_tr, y_te = train_test_split(X_bow, y_sent, test_size=0.25, random_state=42, stratify=y_sent)
X_train_tf, X_test_tf, _, _ = train_test_split(X_tfidf, y_sent, test_size=0.25, random_state=42, stratify=y_sent)

models_to_try = {
    "Multinomial NB (Umum untuk NLP!)": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "Random Forest (150 trees)": RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1),
}

results_nlp = []
for fe_name, X_train_f, X_test_f in [("Bag of Words (Count)", X_train_bow, X_test_bow),
                                    ("TF-IDF Vectorizer", X_train_tf, X_test_tf)]:
    for mdl_name, mdl in models_to_try.items():
        t0 = time.time()
        mdl.fit(X_train_f, y_tr)
        t_tr = time.time() - t0
        ypr = mdl.predict(X_test_f)
        acc = accuracy_score(y_te, ypr)
        f1 = f1_score(y_te, ypr, average="weighted")
        results_nlp.append({"Feature": fe_name, "Model": mdl_name, "Train_Time_s": t_tr,
                            "Test_Acc": acc, "Test_F1_weighted": f1})
        print(f"  [{fe_name:<22}] {mdl_name:<30} | Acc={acc:.4f} | F1={f1:.4f} | {t_tr:.2f}s")

df_res_nlp = pd.DataFrame(results_nlp).sort_values("Test_F1_weighted", ascending=False)
print("\n🏆 Perbandingan SEMUA kombinasi Feature Extraction + Model (urutan by F1):")
print(df_res_nlp.round(4).to_string(index=False))

# Pilih kombinasi TERBAIK
best_row = df_res_nlp.iloc[0]
print(f"\n🏆 KOMBINASI TERBAIK: {best_row['Feature']} + {best_row['Model']}")
print(f"   Acc={best_row['Test_Acc']:.4f} | F1 Weighted={best_row['Test_F1_weighted']:.4f}")

# Train best combo untuk confusion matrix
fe_best = best_row["Feature"]
mdl_best_name = best_row["Model"]
if fe_best.startswith("Bag"):
    Xtr, Xte = X_train_bow, X_test_bow
else:
    Xtr, Xte = X_train_tf, X_test_tf

from sklearn.base import clone
best_full_model = clone([m for n, m in models_to_try.items() if n == mdl_best_name][0])
best_full_model.fit(Xtr, y_tr)
y_pred_best = best_full_model.predict(Xte)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
cm = confusion_matrix(y_te, y_pred_best)
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=axes[0],
            xticklabels=class_names, yticklabels=class_names, cbar=False, linewidths=0.5,
            annot_kws={"fontsize": 13, "fontweight": "bold"})
axes[0].set_title(f"Confusion Matrix Sentimen\n{fe_best.split(' (')[0]} + {mdl_best_name.split(' (')[0]}\nF1={best_row['Test_F1_weighted']:.4f}",
                  fontweight="bold")
axes[0].set_xlabel("Predicted Sentiment"); axes[0].set_ylabel("True Sentiment")

pivot_fe = df_res_nlp.pivot_table(index="Model", columns="Feature", values="Test_F1_weighted")
sns.heatmap(pivot_fe, annot=True, fmt=".4f", cmap="viridis", ax=axes[1], linewidths=0.5,
            annot_kws={"fontsize": 11, "fontweight": "bold"})
axes[1].set_title("Heatmap: F1 Weighted (Feature Extraction vs Model)", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{output_dir}\\02_sentiment_classification_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart Perbandingan Sentimen tersimpan")

print(f"\nClassification Report (Kombinasi Terbaik):\n")
print(classification_report(y_te, y_pred_best, target_names=class_names, digits=4))

print("\n>>> 5. WORD EMBEDDING SEDERHANA + Cosine Similarity (Kemiripan Kata)")
print("-" * 60)

print("""
Word Embedding = Setiap kata → Vektor DENSE dimensi D (umum 50/100/200/300)
    Contoh: "bagus" → [0.12, -0.35, 0.87, 0.02, ... ] (300 angka)
    Kata dengan MAKNA SERUPA → VEKTOR DEKAT (cosine similarity tinggi)

    Contoh Hubungan Semantik (analogi):
        raja - pria + wanita = ratu   (man → woman + queen = king)
        indonesia - jakarta + london = inggris

    Jenis populer:
    • Word2Vec (Google 2013): CBOW (prediksi kata dari konteks)
                              Skip-gram (prediksi konteks dari kata)
    • GloVe (Stanford 2014)  : Global + Local co-occurrence matrix
    • FastText (Facebook): Subword info, BISA OOV + bahasa morfologi!
    • BERT/GPT dkk : Contextual embedding (kata beda → beda konteks!)
""")

# Train Word2Vec SEDERHANA dari dataset review menggunakan Gensim jika tersedia
try:
    from gensim.models import Word2Vec
    GENSIM_OK = True
    print("✅ Gensim TERSEDIA! Melatih Word2Vec dari review Indonesia...")
    sentences = df["clean_tokens"].tolist() + [["<pad>"]*5]
    w2v = Word2Vec(sentences=sentences, vector_size=50, window=5, min_count=2,
                   workers=4, sg=1, epochs=25, seed=42)
    vocab_len = len(w2v.wv)
    print(f"   Word2Vec selesai. Vocab size={vocab_len}, vector dim=50")

    # Test kemiripan kata
    test_words = ["bagus", "cepat", "rusak", "seller", "barang", "ori"]
    print(f"\nTop 5 Kata TERDEKAT (Cosine Similarity tertinggi):")
    for w in test_words:
        if w in w2v.wv:
            similar = w2v.wv.most_similar(w, topn=5)
            print(f"  • '{w:<12}' → ", end="")
            print(", ".join([f"{s[0]}({s[1]:.3f})" for s in similar]))

    # Visualisasi 2D Word Embedding dengan PCA
    from sklearn.decomposition import PCA
    n_plot = 100
    all_words = list(w2v.wv.key_to_index.keys())[:n_plot]
    vectors = np.array([w2v.wv[w] for w in all_words])
    pca = PCA(n_components=2, random_state=42)
    vec_2d = pca.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(14, 10))
    colors_w = []
    for w in all_words:
        is_pos = w in [stemming_id_sederhana(x) for x in positive_words]
        is_neg = w in [stemming_id_sederhana(x) for x in negative_words]
        colors_w.append("#16a34a" if is_pos else ("#dc2626" if is_neg else "#6b7280"))
    ax.scatter(vec_2d[:, 0], vec_2d[:, 1], c=colors_w, s=70, alpha=0.85, edgecolors="white")
    for i, w in enumerate(all_words):
        if np.random.random() < 0.55 or len(w) >= 5:
            ax.annotate(w, (vec_2d[i, 0], vec_2d[i, 1]), fontsize=8, alpha=0.9)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker="o", color="w", label="Positif (hijau)", markerfacecolor="#16a34a", markersize=11),
                       Line2D([0], [0], marker="o", color="w", label="Negatif (merah)", markerfacecolor="#dc2626", markersize=11),
                       Line2D([0], [0], marker="o", color="w", label="Netral (abu)", markerfacecolor="#6b7280", markersize=11)]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.set_title(f"Word2Vec Visualisasi (PCA 2D) {n_plot} Kata dari Review Sentimen\nHijau=Kata Positif, Merah=Kata Negatif", fontweight="bold")
    ax.set_xlabel(f"PCA 1 (var={pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PCA 2 (var={pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}\\03_word2vec_pca_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Chart Word2Vec Visualization tersimpan")
except ImportError:
    GENSIM_OK = False
    print("⚠️  Gensim TIDAK terinstall. Install untuk Word2Vec: pip install gensim")
    print("   Membuat dummy embedding sederhana sebagai pengganti visual...")
    words_dummy = ["bagus", "cepat", "ori", "puas", "ramah", "rusak", "palsu", "kecewa", "lambat", "salah kirim",
                   "standar", "ok", "lumayan", "normal", "biasa", "seller", "barang", "harga", "kirim", "packing"]
    vec_dummy = np.random.randn(len(words_dummy), 50)
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(vec_dummy[:, :10], cmap="coolwarm", center=0, yticklabels=words_dummy,
                annot=False, cbar=True, ax=ax)
    ax.set_title("DUMMY Word Embedding Matrix (20 Kata x 10 Dimensi Pertama)\n(Install gensim untuk Word2Vec asli)", fontweight="bold")
    ax.set_xlabel("Dimensi Embedding (d=10 dari 50)")
    ax.set_ylabel("Kata / Token")
    plt.tight_layout()
    plt.savefig(f"{output_dir}\\03_dummy_embedding_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Chart Dummy Embedding tersimpan")

print("\n>>> 6. CONTOH NER & INTRO TRANSFORMER ARCHITECTURE")
print("-" * 60)

print("""
CONTOH NER (Named Entity Recognition) sederhana pada kalimat:
    > "PT Sejahtera Indonesia (persero) baru saja menandatangani kontrak
       senilai Rp 50 Miliar dengan Bapak Bambang Santoso di Jakarta
       pada tanggal 10 November 2026."

    ENTITAS yang dideteksi oleh model NER:
    ┌────────────────────────────────┬───────────────────────┐
    │ FRASA                          │ LABEL NER             │
    ├────────────────────────────────┼───────────────────────┤
    │ PT Sejahtera Indonesia (persero)│ ORGANIZATION (ORG)   │
    │ Rp 50 Miliar                   │ MONEY / MONETARY      │
    │ Bapak Bambang Santoso          │ PERSON (PER)          │
    │ Jakarta                        │ LOCATION (LOC)        │
    │ 10 November 2026               │ DATE / TIME           │
    └────────────────────────────────┴───────────────────────┘

ARSITEKTUR TRANSFORMER (Attention Is All You Need, 2017) → Cikal bakal LLM!
┌───────────────────────────────────────────────────────────────────────┐
│ Input: "Saya suka bakso enak sekali" → Token Embedding + Positional     │
│ Encoding                                                               │
│                                                                         │
│ ┌─── ENCODER BLOCK (diulang N = 6/12/24 kali) ──────────────────┐    │
│ │  ┌─────────────────────────────┐   ┌─────────────────────┐   │    │
│ │  │ MULTI-HEAD SELF-ATTENTION   │→ │ FEED FORWARD NETWORK │   │    │
│ │  │ Q * K^T /√dk → Softmax * V  │   │ (2 Dense + GELU)    │   │    │
│ │  │ 8 / 16 attention heads      │   │ Residual + LayerNorm│   │    │
│ │  │ + Residual + LayerNorm      │   │                     │   │    │
│ │  └─────────────────────────────┘   └─────────────────────┘   │    │
│ └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│ Output = Hidden states tiap token (dipakai untuk NER/classification)  │
│          ATAU <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token untuk classification (BERT)                │
│          ATAU diumpan ke Decoder (Machine Translation / GPT style)    │
└───────────────────────────────────────────────────────────────────────┘
    ⚡ ATTENTION MECHANISME = "Fokus ke token MANA yang PENTING"
       Self-Attention: Q=K=V = semua token di sequence (internal)
       Cross-Attention: Q dari decoder, K/V dari encoder (GPT baru!?)

    VARIAN TRANSFORMER POPULER:
    • BERT      = Encoder-only → Pahami KONTEKS (QA, NER, Classification)
    • GPT series= Decoder-only → GENERASI teks (next token prediction)
    • T5/BART   = Encoder+Decoder → Translation, Summarization
    • Vision Transformer (ViT) = Untuk gambar! (patch = token!)
""")

print("""
┌─────────────────────────────────────────────────────────────────┐
│  RANGKUMAN NLP MODUL 8 BAGIAN 1 UNTUK BNSP:                      │
├─────────────────────────────────────────────────────────────────┤
│  Preprocessing 6 langkah: Lower → Clean → Token → Stopword →   │
│                 Stemming → Join Lagi (Bahasa Indonesia!)        │
│  Feature Extraction: BoW (Count) vs TF-IDF (UMUMNYA LEBIH BAIK) │
│  Sentiment Classifier: Multinomial NB + TF-IDF = GOLD STANDARD! │
│  Word Embedding: Word2Vec/GloVe/FastText (dense, makna = dekat)│
│  Advanced: NER, POS Tag, + TRANSFORMER (BERT/GPT sebagai base LLM)│
└─────────────────────────────────────────────────────────────────┘
""")

print("""\n>>> LATIHAN:
1. Tambahkan 2 langkah preprocessing: Lemmatization menggunakan pkg Sastrawi
   (Stemmer ID yang lebih lengkap), bandingkan hasil stemming sederhana
   kita vs Sastrawi!
2. Tambahkan parameter 'ngram_range=(1,3)' ke TF-IDF, apakah akurasi naik?
3. Tambahkan feature 'panjang_review' + 'jumlah_emoji' ke pipeline
   (gunakan ColumnTransformer + FeatureUnion), bandingkan akurasi!
4. Jika terinstall transformers: Coba pipeline('sentiment-analysis') dari
   HuggingFace IndoBERT untuk prediksi sentimen, bandingkan dengan model kita!
5. Buat Word Cloud per kelas sentimen (library wordcloud) — visualisasi
   frekuensi kata!
6. (Lanjut) Buat aplikasi SEDERHANA prediksi sentimen dengan Flask/FastAPI
   + menerima input teks user! → latihan Modul 10 Deployment nanti
""")

print("\n✓ Bagian 1 Modul 8 Selesai: NLP Fundamental + TF-IDF + Sentimen + Embedding")
