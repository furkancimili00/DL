# feature_engineering.py

import pandas as pd

# Veriyi oku
df = pd.read_csv("Valve_Player_Data.csv")

# Temel dönüşümler
df['Percent_Gain'] = df['Percent_Gain'].str.replace('%', '').astype(float)
df['churn'] = df['Percent_Gain'].apply(lambda x: 1 if x < 0 else 0)

# 📌 Yeni Özellikler

# 1. Gain_Ratio: Göreli değişim
df['Gain_Ratio'] = df['Gain'] / df['Avg_players']

# 2. Gain_Direction: Artış mı, azalış mı? (binary)
df['Gain_Direction'] = df['Gain'].apply(lambda x: 1 if x > 0 else 0)

# 3. Volatility: Değişim ne kadar büyük?
df['Volatility'] = abs(df['Gain_Ratio'])

# Yeni sütunları ve ilk satırları göster
print(df[['Avg_players', 'Gain', 'Peak_Players', 'Gain_Ratio', 'Gain_Direction', 'Volatility', 'churn']].head())

# Yeni CSV dosyasına kaydet (model için kullanılacak)
df.to_csv("data/processed_data.csv", index=False)
print("\n✅ Yeni özellikler başarıyla eklendi ve processed_data.csv dosyası oluşturuldu.")
