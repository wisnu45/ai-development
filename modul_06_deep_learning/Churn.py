# --- Setup Awal ---
# (Asumsikan data Churn sudah di-preprocess menjadi X_train, X_test, y_train, y_test)
# Pastikan data numerik sudah di-SCALE (StandardScaler)! ANN sangat sensitif terhadap skala.

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

# --- 1. Mendefinisikan Arsitektur Jaringan ---
# Kita coba membuat jaringan dengan 3 Hidden Layer (64 neuron, 32 neuron, 16 neuron)
# Ini adalah arsitektur "Deep" yang sederhana.
mlp_churn = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16), # Arsitektur "corong"
    activation="relu",     # Fungsi aktivasi standar modern
    solver="adam",         # Optimizer yang paling umum dan efisien
    alpha=0.0001,          # Regularisasi L2 untuk mencegah overfitting
    batch_size=64,         # Belajar dari 64 data sekaligus
    learning_rate_init=0.001, # Kecepatan belajar awal
    max_iter=500,          # Maksimal epoch (putaran belajar)
    early_stopping=True,   # Stop otomatis jika tidak ada kemajuan (hemat waktu!)
    random_state=42,
    verbose=True # Tampilkan proses training
)

# --- 2. Proses Training (Backpropagation berjalan di sini!) ---
print("Memulai Training MLP...")
mlp_churn.fit(X_train, y_train)
print(f"Training selesai dalam {mlp_churn.n_iter_} epoch.")

# --- 3. Evaluasi Model ---
y_pred_test = mlp_churn.predict(X_test)
train_acc = accuracy_score(y_train, mlp_churn.predict(X_train))
test_acc = accuracy_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)

print(f"\nHasil Evaluasi MLP Churn:")
print(f"  Train Accuracy: {train_acc:.2%}")
print(f"  Test Accuracy : {test_acc:.2%} (Cek gap dengan train untuk overfitting)")
print(f"  Test F1-Score : {test_f1:.4f} (Penting untuk data churn!)")

# Visualisasi Loss Curve (Opsional tapi sangat disarankan)
import matplotlib.pyplot as plt
plt.plot(mlp_churn.loss_curve_)
plt.title("Training Loss Curve (Semakin turun semakin baik)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()