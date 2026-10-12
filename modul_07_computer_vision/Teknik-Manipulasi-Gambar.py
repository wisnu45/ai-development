from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt

# Buka gambar sampel (misal: lingkaran)
pil_original = Image.open(samples[0]["path"])

# --- KOLEKSI TEKNIK MANIPULASI ---
transformations = [
    ("Asli", pil_original),
    # 1. Resize: Standarisasi ukuran adalah wajib di CV
    ("1. Resize (128x128)", pil_original.resize((128, 128))),
    # 2. Grayscale: Mengurangi kompleksitas warna (3 channel jadi 1)
    ("2. Grayscale", ImageOps.grayscale(pil_original)),
    # 3. Blur (Gaussian): Mengurangi noise / detail yang tidak perlu
    ("3. Gaussian Blur", pil_original.filter(ImageFilter.GaussianBlur(radius=3))),
    # 4. Enhance Contrast: Mempertegas perbedaan objek dan background
    ("4. Contrast Up (x1.8)", ImageEnhance.Contrast(pil_original).enhance(1.8)),
    # 5. Brightness: Mensimulasikan kondisi cahaya berbeda
    ("5. Brightness Down (x0.7)", ImageEnhance.Brightness(pil_original).enhance(0.7)),
    # 6. Rotation (Augmentation): Objek bisa miring di dunia nyata
    ("6. Rotate 30°", pil_original.rotate(30, expand=True)),
    # 7. Flip (Augmentation): Objek bisa terbalik kiri-kanan
    ("7. Horizontal Flip", ImageOps.mirror(pil_original)),
    # 8. Random Crop (Augmentation): Simulasi objek hanya terlihat sebagian
    # (Disederhanakan: crop tengah lalu resize balik)
    ("8. Center Crop+Resize", pil_original.crop((60, 60, 200, 200)).resize((256, 256))),
    # 9. Histogram Equalization: Meratakan distribusi pencahayaan
    ("9. Hist. Equalization", ImageOps.equalize(ImageOps.grayscale(pil_original)))
]

# Visualisasi Hasil
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()
for i, (title, img) in enumerate(transformations):
    # Jika gambar grayscale, gunakan colormap 'gray'
    cmap = 'gray' if img.mode == 'L' else None
    axes[i].imshow(img, cmap=cmap)
    axes[i].set_title(title, fontsize=9)
    axes[i].axis("off")

plt.tight_layout()
plt.show()