# --- Setup Awal ---
# (Asumsikan data Churn sudah siap sebagai X_pp dan y)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint

# --- 1. Demonstrasi Cross-Validation ---
# Mari kita lihat bagaimana skor bisa berbeda di tiap lipatan (fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base_rf = RandomForestClassifier(n_estimators=100, random_state=42)

scores = cross_val_score(base_rf, X_pp, y, cv=skf, scoring='f1') # Gunakan F1 karena data churn mungkin imbalanced
print(f">>> Hasil Stratified K-Fold CV (5 folds) - F1 Score:")
print(f"Skor tiap fold: {scores.round(4)}")
print(f"Rata-rata Skor : {scores.mean():.4f} ± {scores.std():.4f} (Ini estimasi performa yang lebih stabil)")


# --- 2. Hyperparameter Tuning: Grid Search vs Random Search ---
# Bagi data dulu menjadi Train dan Test untuk evaluasi final nanti
X_train, X_test, y_train, y_test = train_test_split(X_pp, y, test_size=0.2, random_state=42, stratify=y)

# A. Define Parameter Space (Ruang Pencarian)
# Untuk Grid Search (Daftar nilai spesifik)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
# Total kombinasi = 2 * 3 * 2 = 12 kombinasi.
# Total fitting = 12 komb * 5 fold CV = 60 kali training!

# B. Eksekusi Grid Search
print("\n[Proses] Memulai GridSearchCV (Mungkin agak lama)...")
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True), # Gunakan CV di dalam search
    scoring='f1', # Optimalkan berdasarkan F1-Score
    n_jobs=-1, # Gunakan semua core CPU
    verbose=1
)
grid_search.fit(X_train, y_train) # Fit pada data train saja!

print(f"\n🏆 Hasil Terbaik Grid Search:")
print(f"Best Params: {grid_search.best_params_}")
print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

# C. Evaluasi Final Model Terbaik di Test Set
best_model_grid = grid_search.best_estimator_
test_f1 = f1_score(y_test, best_model_grid.predict(X_test))
print(f"Final Test F1 Score: {test_f1:.4f}")

# --- (Opsional: Random Search) ---
# Jika ruang parameter sangat besar, gunakan RandomizedSearchCV.
# Ia menggunakan distribusi (seperti randint) bukan daftar nilai tetap.
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': randint(2, 20)
}
# random_search = RandomizedSearchCV(..., param_distributions=param_dist, n_iter=50, ...)
# random_search.fit(X_train, y_train)
# Ini akan mencoba 50 kombinasi acak, jauh lebih cepat dari Grid Search penuh.