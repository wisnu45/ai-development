import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 
# 1. Buat data dummy sederhana 
df_nilai = pd.DataFrame({ 
'Jam_Belajar': [2, 3, 5, 7, 8, 1, 4, 6], 
'Nilai_Ujian': [60, 65, 75, 85, 90, 55, 70, 80] 
}) 
# 2. Buat Scatter Plot (Melihat hubungan Jam Belajar vs Nilai) 
plt.figure(figsize=(4, 3)) # Ukuran gambar kecil 
sns.scatterplot(data=df_nilai, x='Jam_Belajar', y='Nilai_Ujian') 
plt.title('Hubungan: Jam Belajar vs Nilai') 
plt.show() # Tampilkan plot pertama 
# 3. Buat Histogram (Melihat sebaran Nilai Ujian) 
plt.figure(figsize=(4, 3)) 
sns.histplot(data=df_nilai, x='Nilai_Ujian', bins=4) # Bins = jumlah batang 
plt.title('Distribusi Nilai Ujian') 
plt.show() # Tampilkan plot kedua 