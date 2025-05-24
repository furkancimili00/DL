# load_data.py

import pandas as pd

# CSV dosyasını oku (dosya adını kendi dosyana göre yazdın)
df = pd.read_csv("Valve_Player_Data.csv")

# Sütun isimlerini ve ilk 5 satırı yazdır
print("Sütunlar:")
print(df.columns)
print("\nİlk 5 satır:")
print(df.head())

# Percent_Gain sütunundaki '%' işaretini silip float'a çevir
df['Percent_Gain'] = df['Percent_Gain'].str.replace('%', '').astype(float)

# churn sütununu oluştur: eğer oyuncu yüzdesi düşmüşse (yani negatifse) churn = 1
df['churn'] = df['Percent_Gain'].apply(lambda x: 1 if x < 0 else 0)

# Kontrol için ilk 10 satırı göster
print("\nchurn sütunu eklenmiş hali (ilk 10 satır):")
print(df[['Month_Year', 'Avg_players', 'Percent_Gain', 'churn']].head(10))
