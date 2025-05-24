# eda_check.py

import pandas as pd

df = pd.read_csv("Valve_Player_Data.csv")
df['Percent_Gain'] = df['Percent_Gain'].str.replace('%', '').astype(float)
df['churn'] = df['Percent_Gain'].apply(lambda x: 1 if x < 0 else 0)

print("Churn dağılımı:")
print(df['churn'].value_counts())
print("\nOranlar (%):")
print(df['churn'].value_counts(normalize=True) * 100)
