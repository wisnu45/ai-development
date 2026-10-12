import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# (Asumsikan fungsi generate_sample_image() sudah didefinisikan seperti di file .py)
# Fungsi ini membuat array NumPy 3D yang merepresentasikan gambar RGB

# Direktori untuk menyimpan gambar
sample_dir = "output_images/sample"
os.makedirs(sample_dir, exist_ok=True)

classes = {"circle": 0, "square": 1, "triangle": 2, "cross": 3}
samples = []

print("Membuat dataset gambar sintetis...")
fig, axes = plt.subplots(1, 4, figsize=(12, 3))

for i, (shape, label) in enumerate(classes.items()):
    # 1. Generate gambar dalam bentuk array NumPy
    arr = generate_sample_image(shape)
    
    # 2. Ubah jadi objek Gambar PIL agar mudah disimpan/dimanipulasi
    pil_img = Image.fromarray(arr)
    
    # 3. Simpan ke file
    path = f"{sample_dir}/sample_{shape}.png"
    pil_img.save(path)
    samples.append({"path": path, "label": label, "shape": shape})
    print(f"  ✅ Berhasil membuat: {shape}")
    
    # Visualisasi
    axes[i].imshow(arr)
    axes[i].set_title(f"{shape.title()} (Label: {label})")
    axes[i].axis("off")

plt.tight_layout()
plt.show()
print("\nDataset siap digunakan!")