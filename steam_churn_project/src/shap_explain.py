# shap_explain.py

import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# SHAP için uyarıyı kapat
shap.initjs()

# Veriyi oku
df = pd.read_csv("data/processed_data.csv")
df = df.replace([float('inf'), float('-inf')], pd.NA)
df = df.dropna()

features = ['Avg_players', 'Gain', 'Peak_Players', 'Gain_Ratio', 'Gain_Direction', 'Volatility']
X = df[features]
y = df['churn']

# Normalize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim böl
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Modeli yeniden eğit (yüklemiyoruz çünkü SHAP için erişmek gerekiyor)
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.005), metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# SHAP DeepExplainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test[:100])  # İlk 100 örnek

# Özet grafiği
shap.summary_plot(shap_values, X_test[:100], feature_names=features, show=False)
plt.tight_layout()
plt.savefig("results/shap_summary.png")
print("\n✅ SHAP summary plot kaydedildi: results/shap_summary.png")
