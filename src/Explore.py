import pandas as pd

# Load dataset
df = pd.read_csv("../data/poverty_data.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nDATASET INFORMATION:")
print(df.info())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDATASET SHAPE:")
print(df.shape)