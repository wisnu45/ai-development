print("="*60)
print("MODUL 6: DEEP LEARNING")
print("Bagian 2: Convolutional Neural Network (CNN) + Computer Vision Fundamentals")
print("="*60)

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = "c:\\ai-engineer\\modul_06_deep_learning\\output_charts"
os.makedirs(output_dir, exist_ok=True)

sns.set_style("whitegrid")
np.random.seed(42)

print("""
┌─────────────────────────────────────────────────────────────────┐
│    COMPUTER VISION & Convolutional Neural Network (CNN)         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  KOMPONEN DASAR COMPUTER VISION:                                │
│  1. Image Acquisition  → Ambil gambar (camera / file)          │
│  2. Image Preprocessing→ Resize, Normalize, Denoise, Augment    │
│  3. Feature Extraction → Otomatis dengan CNN (tanpa manual!)    │
│  4. Model Inference    → Classification / Detection / Segm     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  LAYER KHUSUS DI CNN (PENTING BUAT BNSP!):               │  │
│  │                                                          │  │
│  │  1. CONVOLUTIONAL LAYER (Conv2D)                         │  │
│  │     • Fungsi: Mengekstrak FITUR LOKAL dari gambar        │  │
│  │       (garis, tepi, sudut, texture, bentuk, objek)       │  │
│  │     • Kernel / Filter = Matriks kecil (3x3, 5x5, 7x7)    │  │
│  │       yang "digeser" ke seluruh gambar (sliding window)  │  │
│  │     • Parameter Penting:                                 │  │
│       - filters=32  → Jumlah kernel = jumlah peta fitur     │  │
│       - kernel_size=(3,3) → Ukuran kernel                  │  │
│       - strides=(1,1)  → Langkah geser kernel               │  │
│       - padding='same' → Tambah nol pinggir (output ukuran │  │
│         SAMA dengan input)                                  │  │
│       - padding='valid' → TIDAK padding (output KECIL)     │  │
│  │                                                          │  │
│  │  2. POOLING LAYER (MaxPool2D / AvgPool2D)               │  │
│  │     • Fungsi: DOWNSAMPLING (perkecil spatial size)     │  │
│  │       → Kurangi komputasi, buat fitur robust translasi │  │
│  │     • Max Pool = Ambil nilai MAX per jendela (PALING    │  │
│  │       UMUM, ambil fitur PALING MENONJOL)                 │  │
│  │     • Avg Pool = Ambil nilai rata-rata                  │  │
│  │     • Biasanya pool_size=(2,2) → Ukuran berkurang 1/2!  │  │
│  │                                                          │  │
│  │  3. FULLY CONNECTED LAYER (Dense Layer)                 │  │
│  │     • Fungsi: Klasifikasi AKHIR berdasarkan fitur      │  │
│  │       yang diekstrak CNN                                │  │
│  │     • Sebelum masuk Flatten() → Ubah 2D → 1D vektor     │  │
│  │                                                          │  │
│  │  4. DROPOUT LAYER                                       │  │
│  │     • Fungsi: REGULARIZATION, prevent overfitting       │  │
│  │       → Secara acak "matikan" sebagian neuron saat      │  │
│         training (biasanya rate=0.2 ~ 0.5)                  │  │
│  │                                                          │  │
│  │  5. BATCH NORMALIZATION LAYER                           │  │
│  │     • Stabilkan distribusi input ke layer berikutnya     │  │
│  │       → Training LEBIH CEPAT & stabil, mengurangi        │  │
│         kebutuhan regularization yang berat                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ARSITEKTUR CNN KLASIK (untuk referensi portofolio!):          │
│  1. LeNet-5      (1998) → LeCun → (Digits) 2 Conv + 2 Pool    │
│  2. AlexNet      (2012) → ImageNet Winner! 5 Conv + 3 FC       │
│  3. VGG16/VGG19  (2014) → SANGAR: 3x3 conv bertumpuk 16/19 ly │
│  4. GoogLeNet/Inception (2014) → Inception module, hemat param │
│  5. ResNet (2015)  → SKIP CONNECTION (residual), 152 layer!    │
│     → SOLUSI VANISHING GRADIENT di Deep Network                │
│  6. EfficientNet (2019) → Compound Scaling (depth+width+res)   │
│  7. MobileNet (2017) → Depthwise Separable Conv, HP & EDGE!    │
│                                                                 │
│  TRANSFER LEARNING = (PENTING BUAT DATA KECIL!)                 │
│  = Pakai PRE-TRAINED MODEL (VGG, ResNet, EfficientNet yang     │
│    SUDAH DILATIH di jutaan gambar ImageNet) → Ganti HANYA      │
│    layer classification TERAKHIR (FINE-TUNING!)                 │
│  Manfaat: BUTUH DATA SEDIKIT, AKURASI TINGGI, TRAINING CEPAT!   │
└─────────────────────────────────────────────────────────────────┘
""")

input("TEKAN ENTER UNTUK MULAI PRAKTIK COMPUTER VISION (MNIST digits)...")

print("\n>>> 1. LOAD DATASET - MNIST Handwritten Digits (via Scikit-learn)")
print("-" * 60)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

digits = load_digits()
X = digits.images
y = digits.target

print(f"Shape X gambar mentah: {X.shape} → {len(X)} gambar ukuran {X.shape[1]}x{X.shape[2]} piksel, 8-bit grayscale")
print(f"Shape y label        : {y.shape} → {len(np.unique(y))} kelas (digit 0-9)")

fig, axes = plt.subplots(2, 10, figsize=(16, 4))
for img_idx in range(10):
    for row in range(2):
        sample = np.random.choice(np.where(y == img_idx)[0])
        axes[row][img_idx].imshow(X[sample], cmap="gray")
        axes[row][img_idx].set_title(f"Label: {y[sample]}", fontsize=9, fontweight="bold")
        axes[row][img_idx].axis("off")
plt.suptitle("Contoh Dataset MNIST: Digit Tulisan Tangan 0-9", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{output_dir}\\04_mnist_dataset_samples.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart sample MNIST tersimpan")

print("\n>>> 2. PREPROCESSING IMAGE - Normalisasi + Train/Test Split")
print("-" * 60)

X_flat = X.reshape(len(X), -1)
print(f"Setelah FLATTEN: {X_flat.shape} (8x8 → 64 piksel linear)")
print(f"Rentang nilai piksel ASLI: min={X_flat.min()}, max={X_flat.max()}")

X_norm = X_flat / 16.0
print(f"Rentang nilai SETELAH NORMALISASI (dibagi 16): min={X_norm.min():.1f}, max={X_norm.max():.1f}")

X_img_train, X_img_test, y_img_train, y_img_test = train_test_split(
    X_norm, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Train: {len(X_img_train)}, Test: {len(X_img_test)}")
print(f"Distribusi kelas di test set (stratified):")
digit_count = np.bincount(y_img_test)
for i in range(10):
    print(f"  Digit {i}: {digit_count[i]:>3} sampel ({digit_count[i]/len(y_img_test)*100:.1f}%)")

print("\n>>> 3. TRAINING: MLP (Fully Connected) vs Klasifikasi MNIST")
print("-" * 60)

mlp_vision = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=64,
    learning_rate_init=0.001,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
    verbose=0,
)
t0 = time.time()
mlp_vision.fit(X_img_train, y_img_train)
t = time.time() - t0

y_pred_img = mlp_vision.predict(X_img_test)
acc_mlp = accuracy_score(y_img_test, y_pred_img)
print(f"✅ MLP (256,128,64) ReLU → selesai {t:.1f}d, {mlp_vision.n_iter_} epochs")
print(f"   Test Accuracy: {acc_mlp:.4f} ({acc_mlp*100:.2f}%)")
print(f"\nClassification Report MLP (MNIST):")
print(classification_report(y_img_test, y_pred_img, digits=4))

print("\n>>> 4. VISUALISASI CONFUSION MATRIX MNIST 10x10")
print("-" * 60)

cm_mnist = confusion_matrix(y_img_test, y_pred_img)
plt.figure(figsize=(11, 9))
sns.heatmap(cm_mnist, annot=True, fmt=",d", cmap="Blues", cbar=True,
            xticklabels=[str(i) for i in range(10)],
            yticklabels=[str(i) for i in range(10)], linewidths=0.3,
            annot_kws={"fontsize": 10})
plt.xlabel("Predicted Digit", fontsize=12)
plt.ylabel("True Digit", fontsize=12)
plt.title(f"Confusion Matrix MNIST - MLP (Acc={acc_mlp:.4f})\nKesalahan umum: digit mirip (3 vs 5, 8 vs 1, 9 vs 7)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{output_dir}\\05_cnn_confusion_matrix_mnist.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart Confusion Matrix MNIST tersimpan")

print("\n>>> 5. VISUALISASI MISCLASSIFICATION (KESALAHAN MODEL)")
print("-" * 60)

errors_mask = y_img_test != y_pred_img
X_errors = X_img_test[errors_mask]
y_true_err = y_img_test[errors_mask]
y_pred_err = y_pred_img[errors_mask]
print(f"Total kesalahan klasifikasi: {errors_mask.sum()} / {len(y_img_test)} gambar")

max_show = min(20, len(X_errors))
cols = 5
rows = int(np.ceil(max_show / cols))
fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
axes = axes.flatten() if rows > 1 else [axes]
for i in range(max_show):
    axes[i].imshow(X_errors[i].reshape(8, 8), cmap="gray")
    axes[i].set_title(f"TRUE: {y_true_err[i]} → PRED: {y_pred_err[i]}",
                      fontsize=9, fontweight="bold",
                      color="green" if False else "red")
    axes[i].axis("off")
for j in range(max_show, len(axes)):
    axes[j].axis("off")
plt.suptitle(f"Visualisasi {max_show} KESALAHAN KLASIFIKASI MNIST", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{output_dir}\\06_misclassification_samples.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart misclassification tersimpan")

print("\n>>> 6. VISUALISASI KONVOLUSI DAN POOLING - Langkah demi langkah")
print("-" * 60)

sample_img = X[np.random.choice(np.where(y == 3)[0])]
print(f"Contoh gambar: digit '3', shape={sample_img.shape}")

kernel_vertical = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
kernel_horizontal = np.array([[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]])
kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
kernel_blur = np.ones((3, 3)) / 9

def conv2d(img, kernel, padding=0):
    kh, kw = kernel.shape
    if padding > 0:
        img = np.pad(img, padding, mode="constant")
    ih, iw = img.shape
    oh, ow = ih - kh + 1, iw - kw + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(img[i:i+kh, j:j+kw] * kernel)
    return out

def max_pool2d(img, pool_size=(2, 2)):
    ph, pw = pool_size
    oh = img.shape[0] // ph
    ow = img.shape[1] // pw
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.max(img[i*ph:(i+1)*ph, j*pw:(j+1)*pw])
    return out

fig, axes = plt.subplots(2, 5, figsize=(17, 7))
axes[0][0].imshow(sample_img, cmap="gray")
axes[0][0].set_title(f"Input Asli\nShape: {sample_img.shape}", fontweight="bold", fontsize=9)

conv_v = conv2d(sample_img, kernel_vertical, padding=1)
axes[0][1].imshow(conv_v, cmap="gray")
axes[0][1].set_title(f"Conv Sobel Vertikal (Tepi V)\n{conv_v.shape}", fontsize=8, fontweight="bold")

conv_h = conv2d(sample_img, kernel_horizontal, padding=1)
axes[0][2].imshow(conv_h, cmap="gray")
axes[0][2].set_title(f"Conv Sobel Horizontal (Tepi H)\n{conv_h.shape}", fontsize=8, fontweight="bold")

conv_s = conv2d(sample_img, kernel_sharpen, padding=1)
axes[0][3].imshow(conv_s, cmap="gray")
axes[0][3].set_title(f"Conv Sharpen (Lebih Tajam)\n{conv_s.shape}", fontsize=8, fontweight="bold")

conv_b = conv2d(sample_img, kernel_blur, padding=1)
axes[0][4].imshow(conv_b, cmap="gray")
axes[0][4].set_title(f"Conv Blur (Smoothing)\n{conv_b.shape}", fontsize=8, fontweight="bold")

axes[1][0].axis("off")
axes[1][0].text(0.5, 0.5, "↓\nMax Pooling 2x2\n↓", ha="center", va="center", fontsize=12, fontweight="bold", color="#dc2626")

pool_v = max_pool2d(conv_v, (2, 2))
axes[1][1].imshow(pool_v, cmap="gray")
axes[1][1].set_title(f"MaxPool Tepi V\n{pool_v.shape}", fontsize=8, fontweight="bold")

pool_h = max_pool2d(conv_h, (2, 2))
axes[1][2].imshow(pool_h, cmap="gray")
axes[1][2].set_title(f"MaxPool Tepi H\n{pool_h.shape}", fontsize=8, fontweight="bold")

pool_s = max_pool2d(conv_s, (2, 2))
axes[1][3].imshow(pool_s, cmap="gray")
axes[1][3].set_title(f"MaxPool Sharpen\n{pool_s.shape}", fontsize=8, fontweight="bold")

pool_b = max_pool2d(conv_b, (2, 2))
axes[1][4].imshow(pool_b, cmap="gray")
axes[1][4].set_title(f"MaxPool Blur\n{pool_b.shape}", fontsize=8, fontweight="bold")

plt.suptitle("VISUALISASI LANGKAH CNN: CONV (4 Kernel berbeda) → MAX POOLING 2x2", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{output_dir}\\07_conv_pool_visualized.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart visualisasi Conv + Pooling tersimpan")

print("\n>>> 7. PERBANDINGAN MLP vs ARSITEKTUR CNN SEJATI (jika TF terinstall)")
print("-" * 60)

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, utils
    TF_AVAILABLE = True
    print(f"✅ TensorFlow/Keras TERSEDIA! Versi: {tf.__version__}")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow BELUM terinstall. Install dengan:")
    print("    pip install tensorflow==2.15.0 pillow")
    print("    (Gunakan file requirements.txt, baris TF di-comment)")

if TF_AVAILABLE:
    print("\n>>> MEMBANGUN MODEL CNN SEJATI dengan TensorFlow Keras")
    print("-" * 60)

    X_cnn_train = X_img_train.reshape(-1, 8, 8, 1)
    X_cnn_test = X_img_test.reshape(-1, 8, 8, 1)
    y_cnn_train = utils.to_categorical(y_img_train, 10)
    y_cnn_test = utils.to_categorical(y_img_test, 10)

    cnn_model = models.Sequential([
        layers.Input(shape=(8, 8, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])

    cnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    cnn_model.summary(print_fn=lambda x: print(f"   {x}"))

    early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

    t0_cnn = time.time()
    history_cnn = cnn_model.fit(
        X_cnn_train, y_cnn_train,
        batch_size=32, epochs=50,
        validation_split=0.15,
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
    )
    t_cnn = time.time() - t0_cnn

    test_loss, test_acc_cnn = cnn_model.evaluate(X_cnn_test, y_cnn_test, verbose=0)
    print(f"\n✅ CNN Keras selesai! {t_cnn:.1f}s, {len(history_cnn.history['loss'])} epochs")
    print(f"   CNN Test Accuracy: {test_acc_cnn:.4f} ({test_acc_cnn*100:.2f}%)")
    print(f"   MLP Test Accuracy: {acc_mlp:.4f} ({acc_mlp*100:.2f}%)")
    print(f"   CNN menang sebesar: {(test_acc_cnn - acc_mlp)*100:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    h = history_cnn.history
    epochs_arr = np.arange(1, len(h["loss"]) + 1)
    axes[0].plot(epochs_arr, h["loss"], "o-", label="Train Loss", color="#3b82f6")
    axes[0].plot(epochs_arr, h["val_loss"], "s-", label="Val Loss", color="#dc2626")
    axes[0].set_title(f"CNN Loss Curve (Early Stop di epoch {len(h['loss'])})", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_arr, h["accuracy"], "o-", label="Train Acc", color="#16a34a")
    axes[1].plot(epochs_arr, h["val_accuracy"], "s-", label="Val Acc", color="#7c3aed")
    axes[1].set_title(f"CNN Accuracy Curve → Final Test Acc: {test_acc_cnn*100:.2f}%", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}\\08_tf_keras_cnn_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Chart training curve CNN TF/Keras tersimpan")

print("""
┌─────────────────────────────────────────────────────────────────┐
│  RANGKUMAN COMPUTER VISION & CNN untuk BNSP:                    │
├─────────────────────────────────────────────────────────────────┤
│  📸 Tahapan Preprocessing Gambar:                               │
│  1. Resize ke input size model (misal 224x224 untuk ImageNet)   │
│  2. Normalisasi piksel: dibagi 255 → [0,1] ATAU                │
│     ImageNet mean/std: (mean=[0.485,0.456,0.406],              │
│                           std=[0.229,0.224,0.225])              │
│  3. Data Augmentation (jika data sedikit):                      │
│     Random Rotasi, Flip, Zoom, Brightness, Shear, Shift         │
│  4. One-Hot encoding label (Categorical Cross-Entropy)          │
│                                                                 │
│  ⚡ TRANSFER LEARNING WORKFLOW (Data Kecil tapi akurat tinggi): │
│  1. Ambil base_model=ResNet50/EfficientNet/VGG(weights=ImageNet)│
│  2. base_model.trainable = False (FREEZE SEMUA layer CNN)       │
│  3. Tambah head: Flatten() + Dense(256,relu) + Dropout(0.5)    │
│     + Dense(n_classes, softmax)                                 │
│  4. TRAIN HEAD SAJA (epoch sedikit, lr biasa)                  │
│  5. UNFREEZE sebagian atas base_model (3-5 block terakhir)     │
│  6. FINE-TUNE dengan LR SANGAT KECIL (1e-5 s/d 1e-6)          │
│     → model menyesuaikan dengan domain gambar KITA!             │
└─────────────────────────────────────────────────────────────────┘
""")

print("""\n>>> LATIHAN:
1. Coba MLP dengan hidden layer KECIL: (32,) vs BESAR (512,256)
   Manakah akurasi terbaik di MNIST? Apakah yang besar selalu bagus?
2. Buat AUGMENTASI data MNIST manual: rotate gambar 15 derajat,
   shift 1 piksel kanan/kiri. Apakah akurasi meningkat?
3. Jika TF terinstall, coba tambah BatchNormalization SEBELUM
   setiap Conv block! Bandingkan kecepatan training dan akurasi.
4. TRANSFER LEARNING: Jika TF terinstall dan ada gambar sendiri,
   coba ImageDataGenerator + ResNet50 pre-trained untuk klasifikasi
   2-3 kelas data sendiri (kucing vs anjing, dll.)
5. Visualisasikan FEATURE MAP (output Conv layer) untuk 1 gambar MNIST
   → 32 channel filter pertama, apa yang dipelajari masing-masing filter?
""")

print("\n✓ Bagian 2 Modul 6 Selesai: CNN & Computer Vision")
print("\n" + "="*60)
print("MODUL 6 SELESAI - Deep Learning (ANN/MLP + CNN)")
print("="*60)
