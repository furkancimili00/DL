# model_dnn_strong.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Veriyi oku ve hazırla
df = pd.read_csv("Valve_Player_Data.csv")
df['Percent_Gain'] = df['Percent_Gain'].str.replace('%', '').astype(float)
df['churn'] = df['Percent_Gain'].apply(lambda x: 1 if x < 0 else 0)

X = df[['Avg_players', 'Gain', 'Peak_Players']]
y = df['churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Yeni güçlü DNN modeli
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.005), metrics=['accuracy'])

# Eğitim
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Değerlendirme
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nYeni Model Test Doğruluğu: {accuracy:.4f}")

# Eğitim grafiği
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp (Loss)')
plt.title('Yeni DNN Eğitim Grafiği')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("results/dnn_strong_loss_plot.png")
plt.show()
