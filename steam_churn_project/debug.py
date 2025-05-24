import pandas as pd

df = pd.read_csv("data/processed_data.csv")
print("NaN var mı:\n", df.isnull().sum())
print("\nSonsuz değer var mı:\n", df.replace([float('inf'), float('-inf')], pd.NA).isnull().sum())
