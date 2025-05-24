# evaluate_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

# Veriyi oku
df = pd.read_csv("data/processed_data.csv")
df = df.replace([float('inf'), float('-inf')], pd.NA)
df = df.dropna()

# Özellikler ve hedef
features = ['Avg_players', 'Gain', 'Peak_Players', 'Gain_Ratio', 'Gain_Direction', 'Volatility']
X = df[features]
y = df['churn']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Modeli yeniden eğitmeden yüklemek istiyorsan:
# model = load_model("model.h5")

# Veya son eğittiğin modeli yeniden eğit burada:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.005), metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Tahmin
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("📊 Confusion Matrix:\n", cm)

# Sınıflandırma Raporu
print("\n🧾 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Tahminleri CSV olarak kaydet
results_df = pd.DataFrame(X_test, columns=features)
results_df['Actual'] = y_test.values
results_df['Predicted'] = y_pred
results_df.to_csv("results/predictions.csv", index=False)
print("\n✅ Tahmin sonuçları results/predictions.csv dosyasına kaydedildi.")
