from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load Dataset Gambar (MNIST) ---
digits = load_digits()
X_images = digits.images # Data gambar asli (8x8 piksel)
y_labels = digits.target # Label digit (0-9)

print(f"Dataset Gambar MNIST: {len(X_images)} sampel, ukuran {X_images.shape[1:]}")

# Visualisasi satu contoh gambar
plt.imshow(X_images[0], cmap='gray')
plt.title(f"Contoh Gambar Digit: {y_labels[0]}")
plt.axis('off')
plt.show()

# --- 2. Preprocessing Gambar untuk MLP ---
# MLP tidak bisa menerima input 2D (8x8). Kita harus meratakannya (Flatten) jadi 1D (64 fitur).
X_flat = X_images.reshape(len(X_images), -1)

# PENTING: Normalisasi data gambar!
# Nilai piksel biasanya 0-255 (atau 0-16 di dataset ini).
# Kita ubah jadi range 0-1 agar training ANN stabil.
X_norm = X_flat / 16.0

# Split Data
X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
    X_norm, y_labels, test_size=0.25, random_state=42, stratify=y_labels
)

# --- 3. Training MLP untuk "Computer Vision" Sederhana ---
# Kita anggap setiap piksel adalah satu fitur independen.
print("\nMelatih MLP untuk mengenali digit...")
mlp_vision = MLPClassifier(
    hidden_layer_sizes=(256, 128), # Arsitektur yang cukup besar
    activation="relu",
    solver="adam",
    max_iter=200,
    random_state=42
)
mlp_vision.fit(X_train_img, y_train_img)

# --- 4. Evaluasi ---
y_pred_img = mlp_vision.predict(X_test_img)
acc_vision = accuracy_score(y_test_img, y_pred_img)
print(f"Akurasi Test Set: {acc_vision:.2%}")

# Visualisasi Kesalahan (Di mana model bingung?)
cm = confusion_matrix(y_test_img, y_pred_img)
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix (Digit Mana yang Sering Tertukar?)")
plt.xlabel("Prediksi Model")
plt.ylabel("Label Asli")
plt.colorbar()
plt.show()
# (Analisis: Lihat angka di luar diagonal utama. Misal, model sering salah memprediksi 8 sebagai 1, atau 9 sebagai 7).