print("="*60)
print("MODUL 7: COMPUTER VISION")
print("Bagian 1: Image Processing Fundamental & Image Classification Pipeline")
print("="*60)

import os
import io
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
    from PIL import Image, ImageOps, ImageFilter, ImageEnhance
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
except ImportError as e:
    print("=" * 80)
    print("⚠️  MODUL 7 GAGAL BERJALAN - DEPENDENSI TIDAK DITEMUKAN")
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
sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)

sns.set_style("whitegrid")
np.random.seed(42)

print("""
┌─────────────────────────────────────────────────────────────────┐
│     COMPUTER VISION FUNDAMENTAL (UNTUK PORTFOLIO BNSP!)         │
├─────────────────────────────────────────────────────────────────┤
│  PIPELINE COMPUTER VISION:                                       │
│                                                                 │
│  [1] IMAGE ACQUISITION → Ambil gambar: kamera, file (jpg/png),  │
│                          web scraping, dataset public           │
│                                                                 │
│  [2] IMAGE PREPROCESSING ← TOPIK UTAMA BAGIAN INI!              │
│      ├─ a. Resize & Rescale → Standarisasi ukuran              │
│      ├─ b. Normalization   → Rentang piksel: [0,1] / Z-score    │
│      ├─ c. Color Space Conv→ RGB↔Grayscale, HSV, YCbCr        │
│      ├─ d. Noise Reduction → Blur (Gaussian, Median)           │
│      ├─ e. Contrast Enhanc → Histogram Equalization / CLAHE     │
│      ├─ f. Morphology Ops  → Erosion, Dilation (binary)         │
│      └─ g. Data Augmentation→ Rotate, Flip, Zoom, Shear dll    │
│                                                                 │
│  [3] FEATURE EXTRACTION:                                        │
│      • Manual: HOG, SIFT, SURF, ORB, LBP, Haar Cascade (JADUL) │
│      • Otomatis: CNN (feature maps langsung belajar) - MODERN  │
│                                                                 │
│  [4] HIGH-LEVEL CV TASK:                                        │
│      ├─ Image Classification = "gambar ini APA?" (1 label/gbr) │
│      ├─ Object Detection    = "APA objek DIMANA?" (bbox)       │
│      ├─ Semantic Segmentation = tiap PIKSEL diberi label kelas │
│      ├─ Instance Segmentation = objek sama dipisah instance     │
│      ├─ Face Detection/Recognition, OCR (baca tulisan), dll    │
└─────────────────────────────────────────────────────────────────┘
""")

print("\n>>> 1. MEMBUAT 4 SAMPLE GAMBAR SINTETIS (untuk latihan tanpa data luar)")
print("-" * 60)

def generate_sample_image(shape, size=(256, 256)):
    img = np.zeros((*size, 3), dtype=np.uint8)
    cx, cy = size[0]//2, size[1]//2
    Y, X = np.ogrid[:size[0], :size[1]]
    if shape == "circle":
        dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
        img[dist <= 80] = [220, 38, 38]
        img[dist <= 40] = [254, 215, 170]
    elif shape == "square":
        img[cy-70:cy+70, cx-70:cx+70] = [37, 99, 235]
        img[cy-40:cy+40, cx-40:cx+40] = [253, 224, 71]
    elif shape == "triangle":
        for y in range(size[0]):
            for x in range(size[1]):
                if abs(x-cx) <= (cy - y) * 0.9 and cy >= y >= cy-140:
                    img[y, x] = [34, 197, 94]
    elif shape == "cross":
        img[cy-80:cy+80, cx-25:cx+25] = [168, 85, 247]
        img[cy-25:cy+25, cx-80:cx+80] = [168, 85, 247]
    # add soft gradient background
    bg = np.stack([np.linspace(220, 250, size[0])]*size[1], axis=1)
    bg = np.stack([bg]*3, axis=-1).astype(np.uint8)
    # composite: if img not black, use img, else bg*noise
    mask = (img.sum(axis=-1) > 0)
    result = img.copy()
    noise = np.random.randint(0, 30, (*size, 3), dtype=np.uint8)
    bg_noisy = np.clip(bg.astype(int) + noise - 15, 0, 255).astype(np.uint8)
    result[~mask] = bg_noisy[~mask]
    return result

classes = {"circle": 0, "square": 1, "triangle": 2, "cross": 3}
samples = []
for shape, label in classes.items():
    arr = generate_sample_image(shape)
    pil_img = Image.fromarray(arr)
    path = f"{sample_dir}\\sample_{shape}.png"
    pil_img.save(path)
    samples.append({"path": path, "label": label, "label_name": shape})
    print(f"  ✅ Dibuat: {path} ({arr.shape})")

fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for i, s in enumerate(samples):
    axes[i].imshow(np.array(Image.open(s["path"])))
    axes[i].set_title(f"Class: {s['label_name']}\nLabel: {s['label']}", fontweight="bold")
    axes[i].axis("off")
plt.suptitle("4 Kelas Dataset Sintetis: Lingkaran / Persegi / Segitiga / Salib", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{output_dir}\\01_4_classes_dataset.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart 4 kelas dataset tersimpan")

print("\n>>> 2. PREPROCESSING IMAGE - 9 Teknik Utama (Step by Step)")
print("-" * 60)

pil_original = Image.open(samples[0]["path"])
arr_orig = np.array(pil_original)
steps = [("Original", arr_orig)]

pil_resize = pil_original.resize((128, 128), Image.Resampling.LANCZOS)
steps.append(("1. Resize 256→128px", np.array(pil_resize)))

pil_gray = ImageOps.grayscale(pil_original)
steps.append(("2. Convert Grayscale", np.array(pil_gray), True))

pil_blur = pil_original.filter(ImageFilter.GaussianBlur(radius=3))
steps.append(("3. Gaussian Blur (denoise)", np.array(pil_blur)))

enh = ImageEnhance.Contrast(pil_original)
pil_contrast = enh.enhance(1.8)
steps.append(("4. Contrast Enhanced (x1.8)", np.array(pil_contrast)))

enh = ImageEnhance.Brightness(pil_original)
pil_bright = enh.enhance(0.7)
steps.append(("5. Brightness Reduced (0.7)", np.array(pil_bright)))

pil_rot = pil_original.rotate(30, expand=True, fillcolor=(240, 240, 240))
steps.append(("6. Rotation 30°", np.array(pil_rot)))

pil_flip = ImageOps.mirror(pil_original)
steps.append(("7. Horizontal Flip", np.array(pil_flip)))

pil_crop = pil_original.crop((60, 60, 200, 200)).resize((256, 256), Image.Resampling.LANCZOS)
steps.append(("8. Random Crop + Resize", np.array(pil_crop)))

arr_gray = np.array(pil_gray)
eq = np.stack([np.array(ImageOps.equalize(Image.fromarray(arr_gray)))]*3, axis=-1)
steps.append(("9. Histogram Equalization (Contrast)", eq))

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()
for i, s in enumerate(steps):
    title = s[0]
    arr = s[1]
    cmap = "gray" if (len(s) > 2 and s[2]) else None
    axes[i].imshow(arr, cmap=cmap)
    axes[i].set_title(title, fontsize=9, fontweight="bold")
    axes[i].axis("off")
    shape_info = f"{arr.shape[0]}x{arr.shape[1]}"
    axes[i].text(0.5, -0.08, shape_info, ha="center", va="top", transform=axes[i].transAxes, fontsize=7, color="gray")

for j in range(len(steps), len(axes)):
    axes[j].axis("off")

plt.suptitle("9 TEKNIK PREPROCESSING & AUGMENTASI GAMBAR (PENTING BNSP!)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{output_dir}\\02_preprocessing_pipeline_9_teknik.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart 9 teknik preprocessing tersimpan")

print("\n>>> 3. FEATURE EXTRACTION MANUAL + KLASIFIKASI (ML Tradisional)")
print("-" * 60)

print("""
FEATURE TRADISIONAL (sebelum era CNN):
┌───────────────────────────────────────────────────────────┐
│ HOG = Histogram of Oriented Gradients                     │
│       → Histogram arah gradien (tekstur/bentuk)           │
│       → Umum untuk deteksi manusia / vehicle              │
│ LBP = Local Binary Pattern                                │
│       → Texture descriptor (bandingkan pixel tetangga)   │
│ Color Histogram → Distribusi warna R/G/B                  │
│ Pixel Flatten → Tiap piksel jadi fitur langsung! (sederhana)│
└───────────────────────────────────────────────────────────┘
""")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

n_per_class = 120
np.random.seed(42)

def augment_image(pil_img, idx):
    """19 augmentasi berbeda per gambar asli untuk jadi 20 gambar total"""
    if idx == 0:
        return pil_img
    op = idx % 8
    rot = np.random.uniform(-25, 25)
    result = pil_img.rotate(rot, expand=False, fillcolor=(240, 240, 240), resample=Image.Resampling.BILINEAR)
    if op == 0:
        result = ImageOps.mirror(result)
    elif op == 1:
        enh = ImageEnhance.Contrast(result)
        result = enh.enhance(np.random.uniform(0.6, 1.8))
    elif op == 2:
        enh = ImageEnhance.Brightness(result)
        result = enh.enhance(np.random.uniform(0.6, 1.5))
    elif op == 3:
        enh = ImageEnhance.Color(result)
        result = enh.enhance(np.random.uniform(0.5, 1.5))
    elif op == 4:
        r = np.random.choice([1, 2])
        result = result.filter(ImageFilter.GaussianBlur(radius=r))
    elif op == 5:
        crop = max(20, int(256*np.random.uniform(0.08, 0.18)))
        result = result.crop((crop, crop, 256-crop, 256-crop)).resize((256, 256), Image.Resampling.LANCZOS)
    return result.resize((64, 64), Image.Resampling.LANCZOS)

def extract_lbp(arr_gray):
    """Sederhana implementasi LBP 3x3 neighborhood"""
    h, w = arr_gray.shape
    padded = np.pad(arr_gray, 1, mode="edge")
    lbp = np.zeros_like(arr_gray, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            center = padded[i+1, j+1]
            bits = (padded[i:i+3, j:j+3].flatten() >= center).astype(int)
            bits = np.delete(bits, 4)
            val = 0
            for k, b in enumerate(bits):
                val += b * (1 << k)
            lbp[i, j] = val
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    return hist / (hist.sum() + 1e-9)

def extract_color_hist(arr_rgb):
    hists = []
    for c in range(3):
        h, _ = np.histogram(arr_rgb[:, :, c].ravel(), bins=16, range=(0, 256))
        hists.append(h / (h.sum() + 1e-9))
    return np.concatenate(hists)

def extract_features(img_small):
    arr = np.array(img_small)
    g = np.array(ImageOps.grayscale(img_small))
    f_pix = g.flatten() / 255.0
    f_lbp = extract_lbp(g)
    f_col = extract_color_hist(arr)
    return np.concatenate([f_pix, f_lbp, f_col])

X_data = []
y_data = []
meta = []
for shape, label in classes.items():
    orig = Image.open(f"{sample_dir}\\sample_{shape}.png")
    for i in range(n_per_class):
        aug = augment_image(orig, i)
        feat = extract_features(aug)
        X_data.append(feat)
        y_data.append(label)
        meta.append(f"{shape}_{i}")

X_data = np.array(X_data)
y_data = np.array(y_data)
print(f"Dataset built: {X_data.shape} samples x {X_data.shape[1]} features (Pixel Flat + LBP + Color Hist)")
print(f"Class balance: {np.bincount(y_data)}")

X_tr, X_te, y_tr, y_te, meta_tr, meta_te = train_test_split(
    X_data, y_data, meta, test_size=0.25, random_state=42, stratify=y_data
)
scaler_fe = StandardScaler()
X_tr_s = scaler_fe.fit_transform(X_tr)
X_te_s = scaler_fe.transform(X_te)

clf_cv = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
t0 = time.time()
clf_cv.fit(X_tr_s, y_tr)
t = time.time() - t0
yp = clf_cv.predict(X_te_s)

class_names = list(classes.keys())
acc_cv = accuracy_score(y_te, yp)
print(f"\nRandom Forest (Manual HOG-like + LBP + Color Hist):")
print(f"  Training time: {t:.1f}s")
print(f"  Test Accuracy: {acc_cv:.4f} ({acc_cv*100:.1f}%)")
print(classification_report(y_te, yp, target_names=class_names, digits=3))

cm = confusion_matrix(y_te, yp)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=axes[0],
            xticklabels=class_names, yticklabels=class_names, cbar=False, linewidths=0.5)
axes[0].set_title(f"Confusion Matrix (Acc={acc_cv:.3f})", fontweight="bold")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

importances = clf_cv.feature_importances_
split_feat = {"Pixel Flat": importances[:64*64].sum(),
              "LBP Hist": importances[64*64:64*64+256].sum(),
              "Color Hist (RGB x 16 bin)": importances[64*64+256:].sum()}
bars = axes[1].bar(list(split_feat.keys()), list(split_feat.values()),
                   color=["#3b82f6", "#16a34a", "#f59e0b"])
axes[1].bar_label(bars, fmt="%.4f", fontsize=11, fontweight="bold")
axes[1].set_title("Feature Importance per Grup Feature", fontweight="bold")
axes[1].set_ylabel("Sum of Importance Scores")
plt.xticks(rotation=15, ha="right")

plt.tight_layout()
plt.savefig(f"{output_dir}\\03_manual_feature_classification.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart hasil klasifikasi manual feature tersimpan")

print("\n>>> 4. TEKNIK DETEKSI TEPI & EDGE (Sobel, Prewitt, Laplacian, Canny-like)")
print("-" * 60)

arr_gray = np.array(ImageOps.grayscale(Image.open(samples[2]["path"]))).astype(float)

def conv2d_simple(img, kernel):
    kh, kw = kernel.shape
    h, w = img.shape
    padded = np.pad(img, (kh//2, kw//2), mode="edge")
    out = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            out[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return out

sobel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_h = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

gx = conv2d_simple(arr_gray, sobel_v)
gy = conv2d_simple(arr_gray, sobel_h)
sobel_edge = np.sqrt(gx**2 + gy**2)
lapl_edge = np.abs(conv2d_simple(arr_gray, laplacian))
threshold = sobel_edge.mean() + 1.2 * sobel_edge.std()
canny_like = (sobel_edge > threshold).astype(np.uint8) * 255

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes[0, 0].imshow(arr_gray, cmap="gray"); axes[0, 0].set_title("Input Grayscale", fontweight="bold")
axes[0, 1].imshow(gx, cmap="gray"); axes[0, 1].set_title("Sobel Vertical Edge (Gx)", fontweight="bold")
axes[0, 2].imshow(gy, cmap="gray"); axes[0, 2].set_title("Sobel Horizontal Edge (Gy)", fontweight="bold")
axes[1, 0].imshow(sobel_edge, cmap="gray"); axes[1, 0].set_title("Sobel Magnitude (√Gx²+Gy²)", fontweight="bold")
axes[1, 1].imshow(lapl_edge, cmap="gray"); axes[1, 1].set_title("Laplacian (2nd Derivative Edge)", fontweight="bold")
axes[1, 2].imshow(canny_like, cmap="gray"); axes[1, 2].set_title(f"Canny-Like (Threshold)", fontweight="bold")
for ax in axes.flat:
    ax.axis("off")
plt.suptitle("OPERASI KONVOLUSI UNTUK DETEKSI TEPI (EDGE DETECTION)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{output_dir}\\04_edge_detection_kernels.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart edge detection tersimpan")

print("\n>>> 5. Histogram of Grayscale & RGB (Analisis Distribusi Intensitas)")
print("-" * 60)

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for j, s in enumerate(samples):
    arr = np.array(Image.open(s["path"]))
    axes[0, j].imshow(arr)
    axes[0, j].set_title(f"{s['label_name'].title()}", fontweight="bold")
    axes[0, j].axis("off")
    g = np.array(ImageOps.grayscale(Image.fromarray(arr)))
    axes[1, j].hist(g.ravel(), bins=32, range=(0, 256), alpha=0.8, color="gray", label="Grayscale", edgecolor="white")
    colors = ["#dc2626", "#16a34a", "#2563eb"]
    for c, col in enumerate(colors):
        axes[1, j].hist(arr[:, :, c].ravel(), bins=32, range=(0, 256), alpha=0.5, color=col,
                       label=f"{'R/G/B'[c]} Channel")
    axes[1, j].set_title(f"Histogram {s['label_name']}", fontweight="bold", fontsize=9)
    axes[1, j].set_xlabel("Pixel Intensity 0-255")
    axes[1, j].set_ylabel("Count")
    axes[1, j].legend(fontsize=6)

plt.suptitle("HISTOGRAM ANALISIS INTENSITAS PIKSEL: GRAYSCALE + R/G/B CHANNEL", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{output_dir}\\05_histogram_analysis_rgb.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart histogram tersimpan")

print("\n>>> 6. Membuat & Mengevaluasi METRIC EVALUASI DETEKSI OBJEK (Simulasi)")
print("-" * 60)

print("""
EVALUASI OBJECT DETECTION (PENTING DI BNSP jika ditanyakan!):
┌───────────────────────────────────────────────────────────────┐
│ IoU (Intersection over Union) = Area Irisan / Area Gabungan   │
│   IoU ≥ 0.5 → prediksi dianggap BENAR (TP)                    │
│                                                               │
│ TP = IoU ≥ threshold & label benar                            │
│ FP = IoU ≥ threshold & label salah ATAU duplikat box          │
│ FN = Ground Truth yang TIDAK punya pasangan prediksi          │
│                                                               │
│ mAP (mean Average Precision) → Metric UTAMA deteksi objek     │
│   = rata-rata Precision pada tiap Recall (PR curve area)      │
│   dihitung PER KELAS → dirata-ratakan (m = mean)              │
│   mAP@0.5 = IoU threshold 0.5, mAP@[0.5:0.95] = 10 thresh   │
└───────────────────────────────────────────────────────────────┘
""")

def iou(boxA, boxB):
    xa = max(boxA[0], boxB[0]); ya = max(boxA[1], boxB[1])
    xb = min(boxA[2], boxB[2]); yb = min(boxA[3], boxB[3])
    iw, ih = max(0, xb - xa), max(0, yb - ya)
    inter = iw * ih
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-9)

gt_boxes = [
    {"class": 0, "box": [50, 50, 150, 150]},
    {"class": 1, "box": [160, 40, 230, 120]},
    {"class": 0, "box": [10, 180, 90, 250]},
]
pred_boxes = [
    {"class": 0, "box": [55, 48, 152, 155], "conf": 0.95},
    {"class": 1, "box": [165, 45, 235, 125], "conf": 0.80},
    {"class": 0, "box": [20, 190, 95, 255], "conf": 0.40},
    {"class": 2, "box": [100, 20, 130, 60], "conf": 0.60},
]

ious_all = []
for p in pred_boxes:
    ious = [iou(p["box"], g["box"]) for g in gt_boxes]
    best_gt = np.argmax(ious)
    ious_all.append((p["conf"], max(ious), gt_boxes[best_gt]["class"] == p["class"]))
    print(f"  Pred Class={p['class']}, Conf={p['conf']:.2f} → Best IoU={max(ious):.3f} dengan GT Class={gt_boxes[best_gt]['class']} (Match {'OK ✅' if (gt_boxes[best_gt]['class'] == p['class'] and max(ious) >= 0.5) else '❌'})")

print(f"\nRata-rata IoU simulasi: {np.mean([i for _, i, _ in ious_all]):.3f}")

fig, ax = plt.subplots(figsize=(8, 8))
demo_img = np.ones((280, 280, 3), dtype=np.uint8) * 248
for g in gt_boxes:
    x1, y1, x2, y2 = g["box"]
    demo_img[y1:y1+3, x1:x2] = [34, 197, 94]
    demo_img[y2-3:y2, x1:x2] = [34, 197, 94]
    demo_img[y1:y2, x1:x1+3] = [34, 197, 94]
    demo_img[y1:y2, x2-3:x2] = [34, 197, 94]
    ax.text(x1, y1-6, f"GT Class {g['class']}", color="#16a34a", fontweight="bold", fontsize=9)
for p in pred_boxes:
    x1, y1, x2, y2 = p["box"]
    demo_img[y1:y1+2, x1:x2] = [220, 38, 38]
    demo_img[y2-2:y2, x1:x2] = [220, 38, 38]
    demo_img[y1:y2, x1:x1+2] = [220, 38, 38]
    demo_img[y1:y2, x2-2:x2] = [220, 38, 38]
    ax.text(x1, y2+18, f"Pred {p['class']} ({p['conf']:.2f})", color="#dc2626",
            fontweight="bold", fontsize=8)
ax.imshow(demo_img)
ax.set_title("Simulasi Ground Truth (Hijau) vs Prediction (Merah) → Hitung IoU/mAP", fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.savefig(f"{output_dir}\\06_object_detection_iou.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Chart simulasi Object Detection + IoU tersimpan")

print("""
┌─────────────────────────────────────────────────────────────────┐
│  RANGKUMAN MODUL 7 BAGIAN 1: COMPUTER VISION FUNDAMENTAL       │
├─────────────────────────────────────────────────────────────────┤
│  9 Preprocessing: Resize, Gray, Blur, Contrast, Brightness,    │
│                  Rotation, Flip, Crop, Histogram Equalization   │
│  Feature Engineering Manual: Pixel Flat + LBP + Color Hist      │
│                  → Random Forest: ~99% akurat di dataset ini    │
│  Edge Detection: Sobel, Laplacian, Canny-like thresholding      │
│  Eval Object Detection: IoU, TP/FP/FN, mAP@0.5                  │
└─────────────────────────────────────────────────────────────────┘
""")

print("""\n>>> LATIHAN:
1. Tambahkan 2 augmentasi baru: Random Shear & Color Jitter
2. Ekstrak fitur HOG asli dari skimage.feature.hog()
3. Buat 500 gambar total, lalu bandingkan:
   a. Akurasi Random Forest vs SVM vs MLP di feature manual
   b. (Jika TF terinstall) Akurasi CNN kecil vs feature manual
4. Tambahkan 2 kelas baru (contoh: Diamond, Star) di dataset sintetis,
   lalu lihat apakah akurasi tetap tinggi?
5. Buat gambar simulasi Intersection over Union (IoU) berulang:
   iou=0.1, iou=0.5, iou=0.75, iou=0.95 dalam 1 figure!
""")

print("\n✓ Bagian 1 Modul 7 Selesai: Computer Vision Fundamental & Image Classification")
