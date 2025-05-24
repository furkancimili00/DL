# prepare_data.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Veriyi yükle
df = pd.read_csv("Valve_Player_Data.csv")
df['Percent_Gain'] = df['Percent_Gain'].str.replace('%', '').astype(float)
df['churn'] = df['Percent_Gain'].apply(lambda x: 1 if x < 0 else 0)

# Özellikleri ve hedefi ayır
X = df[['Avg_players', 'Gain', 'Peak_Players']]
y = df['churn']

# Özellikleri normalize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test seti
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Kontrol
print("Eğitim örnek sayısı:", len(X_train))
print("Test örnek sayısı:", len(X_test))
