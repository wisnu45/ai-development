import pandas as pd 
# 1. Buat DataFrame sederhana dari Dictionary 
data = { 
    'Barang': ['Buku', 'Pensil', 'Buku', 'Pulpen'], 
    'Harga': [5000, 2000, 5500, 3000], 
    'Terjual': [10, 50, 5, 20] 
} 
df = pd.DataFrame(data) 
print("Data Awal:\n", df) 
# 2. Tambah kolom 'Pendapatan' (Harga x Terjual) 
df['Pendapatan'] = df['Harga'] * df['Terjual'] 
print("\nSetelah tambah kolom Pendapatan:\n", df) 
# 3. Filter: Tampilkan hanya barang 'Buku' 
print("\nHanya Data 'Buku':\n", df[df['Barang'] == 'Buku']) 
# 4. Agregasi: Total Pendapatan semua barang 
total_uang = df['Pendapatan'].sum() 
print(f"\nTotal Pendapatan Keseluruhan: Rp {total_uang}") 