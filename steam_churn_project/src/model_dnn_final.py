# model_dnn_final.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Veriyi yükle
df = pd.read_csv("data/processed_data.csv")

# 🚨 Eksik ve sonsuz verileri temizle
df = df.replace([float('inf'), float('-inf')], pd.NA)
df = df.dropna()

# Özellikleri ve hedefi ayır
features = ['Avg_players', 'Gain', 'Peak_Players', 'Gain_Ratio', 'Gain_Direction', 'Volatility']
X = df[features]
y = df['churn']

# Normalize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test seti
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Model mimarisi
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Derleme
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.005), metrics=['accuracy'])

# Eğitim
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Sonuç
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n📊 Final Model Test Doğruluğu: {accuracy:.4f}")

# Eğitim grafiği
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Final Model Eğitim Grafiği')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("results/dnn_final_loss_plot.png")
plt.show()
