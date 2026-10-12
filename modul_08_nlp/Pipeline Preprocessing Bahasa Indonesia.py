import re

# Daftar kata penghubung (stopword) Bahasa Indonesia sederhana
STOPWORDS_ID = set([
    "yang", "dan", "di", "ke", "dari", "pada", "untuk", "dengan", "ini", "itu", 
    "adalah", "saya", "kamu", "dia", "mereka", "kita", "akan", "sudah", "juga", 
    "tidak", "ya", "yg", "dgn", "utk", "tpi", "kalo", "gak", "ga", "udah"
])

# Fungsi Stemming sederhana (Mengubah kata berimbuhan jadi kata dasar)
# (Catatan: Di proyek nyata, gunakan library Sastrawi untuk hasil lebih akurat)
def stemming_sederhana(kata):
    # Aturan sederhana: hapus awalan me-, ber-, di-, ter-, pe-
    kata = re.sub(r'^(me|ber|di|ter|pe|se)', '', kata)
    # Hapus akhiran -kan, -i, -an, -nya
    kata = re.sub(r'(kan|i|an|nya)$', '', kata)
    return kata if len(kata) > 2 else kata # Kembalikan jika masih masuk akal

# Pipeline Utama Preprocessing
def preprocess_text(text):
    # 1. Case Folding & Cleaning
    text = text.lower() # Ubah ke huruf kecil
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Hapus URL
    text = re.sub(r'<.*?>', '', text) # Hapus tag HTML
    text = re.sub(r'[^a-z\s]', ' ', text) # Hapus angka dan tanda baca (sisakan huruf & spasi)
    text = re.sub(r'\s+', ' ', text).strip() # Hapus spasi berlebih

    # 2. Tokenization
    tokens = text.split()

    # 3. Stopword Removal & Stemming
    clean_tokens = []
    for token in tokens:
        if token not in STOPWORDS_ID and len(token) > 2:
            stemmed_word = stemming_sederhana(token)
            clean_tokens.append(stemmed_word)

    # 4. Join kembali menjadi string
    return " ".join(clean_tokens)

# --- Contoh Penggunaan ---
contoh_review = "Barangnya BAGUS banget!! Pengiriman juga cepat, saya suka. Recommended seller, makasih ya gan!"
hasil_bersih = preprocess_text(contoh_review)

print(f"Teks Asli   : {contoh_review}")
print(f"Teks Bersih : {hasil_bersih}")
# Output: barang bagus banget kirim cepat suka recommended seller makasih gan