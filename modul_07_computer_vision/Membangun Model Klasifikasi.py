import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# (Asumsikan fungsi ekstraksi fitur: extract_features(img) sudah dibuat)
# Fungsi ini menggabungkan: Piksel yang diratakan + LBP + Histogram Warna

print("Sedang mengekstrak fitur dan memperbanyak data (Augmentasi)...")
X_data = []
y_data = []

# Kita perbanyak 4 gambar asli menjadi ratusan gambar dengan augmentasi
# (Kode augmentasi disederhanakan untuk ilustrasi)
for shape, label in classes.items():
    orig = Image.open(f"{sample_dir}/sample_{shape}.png")
    for i in range(50): # Buat 50 variasi per bentuk
        # Lakukan rotasi acak, dll (simulasi fungsi augment_image)
        aug_img = orig.rotate(np.random.randint(-30, 30)).resize((64, 64))
        
        # EKSTRAKSI FITUR MANUAL DI SINI!
        features = extract_features(aug_img) # Mengembalikan array 1D fitur
        
        X_data.append(features)
        y_data.append(label)

X = np.array(X_data)
y = np.array(y_data)
print(f"Data siap: {X.shape[0]} sampel, {X.shape[1]} fitur per gambar.")

# Split Data (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# --- Training Model ML Klasik ---
print("Melatih Random Forest...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# --- Evaluasi ---
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n>>> Hasil Klasifikasi (Metode Fitur Manual) <<<")
print(f"Akurasi Test Set: {acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=classes.keys()))

# (Analisis: Untuk bentuk geometris sederhana, metode klasik ini seringkali sudah sangat akurat dan cepat!)