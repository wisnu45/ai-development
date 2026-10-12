# --- Setup Awal ---
# (Asumsikan data fraud imbalanced sudah dimuat sebagai X_train, X_test, y_train, y_test)
# Kita latih model Random Forest sebagai contoh
rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1] # Probabilitas kelas positif (Fraud)

# --- 1. Jebakan Akurasi vs Metrik Lain ---
print(">>> PERBANDINGAN METRIK (Data Imbalanced 1% Fraud) <<<")
print(f"Akurasi : {accuracy_score(y_test, y_pred):.4f} (Terlihat tinggi?)")
print("-" * 30)
# Jika kita cuma nebak "Semua Aman", akurasinya adalah proporsi kelas mayoritas
baseline_acc = (y_test == 0).mean()
print(f"Baseline (Tebak 'Aman' Semua): {baseline_acc:.4f} (Ternyata sama!)")
print("\nMetrik yang LEBIH JUJUR:")
print(f"Precision (Ketepatan Prediksi Fraud): {precision_score(y_test, y_pred):.4f}")
print(f"Recall (Daya Tangkap Fraud Asli)    : {recall_score(y_test, y_pred):.4f} <-- PENTING!")
print(f"F1-Score (Keseimbangan P & R)       : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score                       : {roc_auc_score(y_test, y_proba):.4f}")

# --- 2. Visualisasi Confusion Matrix ---
# Mari lihat di mana letak kesalahannya
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Aman", "Fraud"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(cmap="Blues", ax=ax, values_format="d")
plt.title("Confusion Matrix - Random Forest")
plt.grid(False) # Matikan grid agar bersih
plt.show()
# Analisis: Fokus pada kotak kanan bawah (True Positive/Fraud tertangkap)
# dan kiri bawah (False Negative/Fraud lolos).

# --- 3. Visualisasi ROC Curve ---
# Melihat performa model di berbagai titik potong (threshold) probabilitas
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Garis tebakan acak
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Alarm Palsu)')
plt.ylabel('True Positive Rate (Recall/Daya Tangkap)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
# Semakin kurva melengkung ke pojok kiri atas, semakin baik modelnya.