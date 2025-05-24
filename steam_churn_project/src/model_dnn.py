# model_dnn.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Veriyi yükle ve hazırla
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

# DNN modeli oluştur
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification için sigmoid

# Modeli derle
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Modeli eğit
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Sonuçları yazdır
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Doğruluğu: {accuracy:.4f}")
