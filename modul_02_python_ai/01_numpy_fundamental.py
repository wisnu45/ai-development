import numpy as np

# --- Perbedaan List vs Array ---
# List Python bersifat dinamis tapi lambat untuk matematika
list_python = [1, 2, 3, 4, 5]
# NumPy Array bersifat statis (tipe data seragam) tapi sangat cepat
array_np = np.array([1, 2, 3, 4, 5])

print(f"NumPy Array: {array_np} (type: {type(array_np)})")

# --- KONSEP INTI: Vektorisasi ---
# Bayangkan kita punya data fitur 'a' dan bobot 'b'.
# Kita bisa mengoperasikannya langsung tanpa loop 'for'.
a = np.array([10, 20, 30, 40])
b = np.array([1, 2, 3, 4])

print(f"Perkalian Element-wise (a * b) : {a * b}")
print(f"Kuadrat (a ** 2)               : {a ** 2}")
print(f"Fungsi Matematika (sin(a))    : {np.sin(a)}")

# --- Multidimensional Array (Matriks/Tensor) ---
# Data di AI jarang sekali 1 dimensi. Gambar adalah 3D (tinggi, lebar, warna).
matriks = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(f"\nMatriks 3x3:\n{matriks}")
print(f"Shape (Bentuk dimensi): {matriks.shape}")

# --- Filtering dengan Boolean Masking ---
# Teknik ini sangat sering dipakai untuk memilih data berdasarkan kondisi
data = np.array([100, 550, 300, 750, 500])
print(f"\nData > 500 saja: {data[data > 500]}")