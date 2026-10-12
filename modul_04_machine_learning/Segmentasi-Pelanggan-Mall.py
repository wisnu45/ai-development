# --- Setup Awal ---
# (Asumsikan data mall sudah dimuat di df_mall)
cols_cluster = ['Usia', 'Pendapatan_Thn_Juta', 'Skor_Belanja_1_100']
X_mall = df_mall[cols_cluster]

# SANGAT PENTING di Clustering: Scaling data!
# Karena K-Means berbasis jarak, jika satu fitur skalanya ribuan (Pendapatan)
# dan fitur lain puluhan (Usia), fitur Pendapatan akan mendominasi total.
scaler_cluster = StandardScaler()
X_scaled_mall = scaler_cluster.fit_transform(X_mall)

# --- 1. Mencari K Optimal (Elbow Method & Silhouette) ---
print("Mencari jumlah cluster (K) terbaik...")
inertias = []
silhouette_scores = []
K_range = range(2, 11) # Coba dari 2 sampai 10 cluster

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    km.fit(X_scaled_mall)
    inertias.append(km.inertia_) # Inertia = Jarak total titik ke pusat clusternya
    score = silhouette_score(X_scaled_mall, km.labels_)
    silhouette_scores.append(score)
    print(f"  K={k} | Silhouette Score: {score:.4f}")

# (Di sini biasanya kita plot grafik Elbow dan Silhouette.
#  Misal, dari plot terlihat K=5 adalah yang paling optimal.)
k_best = 5
print(f"\nMemutuskan menggunakan K={k_best} berdasarkan evaluasi.")

# --- 2. Final Clustering & Profiling ---
km_final = KMeans(n_clusters=k_best, random_state=42, n_init='auto')
df_mall['Cluster_ID'] = km_final.fit_predict(X_scaled_mall)

# Profiling: Melihat karakteristik rata-rata setiap cluster
print(f"\nProfil Rata-rata per Cluster (K={k_best}):")
profil = df_mall.groupby('Cluster_ID')[cols_cluster].mean().round(1)
profil['Jumlah_Orang'] = df_mall['Cluster_ID'].value_counts()
print(profil)

# --- 3. Interpretasi Bisnis (Contoh) ---
print("\nInterpretasi Bisnis (Contoh dari Profil di atas):")
# Misal Cluster 0 punya Pendapatan Tinggi tapi Skor Belanja Rendah
print("👉 Cluster 0 ('Si Hemat Kaya'): Pendapatan tinggi, tapi jarang belanja. Target: Promo eksklusif barang mewah.")
# Misal Cluster 2 punya Usia Muda, Pendapatan Rendah, Skor Belanja Tinggi
print("👉 Cluster 2 ('Si Muda Boros'): Pendapatan pas-pasan, tapi hobi belanja. Target: Diskon/Cashback kecil tapi sering.")