import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# (Asumsikan kita punya DataFrame 'df' dengan kolom 'review_text' dan 'sentiment_label')
# Misal df sudah ada dari proses scraping atau dataset sintetis
# 0=Negatif, 1=Netral, 2=Positif

# 1. Terapkan Preprocessing ke seluruh data
print("Sedang melakukan preprocessing data...")
df['clean_text'] = df['review_text'].apply(preprocess_text)

X_text = df['clean_text'].values
y_label = df['sentiment_label'].values

# 2. Split Data (Train/Test) SEBELUM vektorisasi untuk mencegah data leakage
X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    X_text, y_label, test_size=0.25, random_state=42, stratify=y_label
)

# 3. Inisialisasi TF-IDF Vectorizer
# max_features=1000: Hanya ambil 1000 kata terpenting
# ngram_range=(1, 2): Ambil kata tunggal (unigram) dan pasangan dua kata (bigram)
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))

# 4. Fit pada data Train, lalu Transform data Train dan Test
print("Melakukan vektorisasi TF-IDF...")
X_train_tfidf = tfidf.fit_transform(X_train_txt) # Pelajari kosakata dari train
X_test_tfidf = tfidf.transform(X_test_txt)       # Ubah test pakai kosakata train

print(f"Shape Matriks TF-IDF Train: {X_train_tfidf.shape}")
# Contoh 10 kata fitur yang didapat
print(f"Contoh Fitur Kata: {tfidf.get_feature_names_out()[:10]}")

# 5. Training Model Klasifikasi (Multinomial Naive Bayes - Populer untuk Teks)
print("\nMelatih model Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# 6. Evaluasi Model
y_pred = nb_model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Analisis Sentimen: {acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negatif", "Netral", "Positif"]))

# 7. Coba Prediksi Kalimat Baru
kalimat_baru = ["Barangnya jelek banget, nyesel beli di sini!", "Lumayan lah sesuai harga"]
kalimat_bersih = [preprocess_text(k) for k in kalimat_baru]
vektor_baru = tfidf.transform(kalimat_bersih)
prediksi = nb_model.predict(vektor_baru)
map_label = {0: "Negatif", 1: "Netral", 2: "Positif"}
print(f"\nPrediksi Kalimat Baru:")
for k, p in zip(kalimat_baru, prediksi):
    print(f"  '{k}' -> Sentimen: {map_label[p]}")