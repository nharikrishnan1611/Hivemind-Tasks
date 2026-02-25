# =========================
# PART 1: Load Dataset & Detect Column Types
# =========================

import pandas as pd

# 1. Load the dataset
df = pd.read_csv("data/ecommerce.csv")

# Show first 5 rows
print("First 5 rows of dataset:")
print(df.head())

print("\n-------------------------\n")

# 2. Detect data types
print("Data types of each column:")
print(df.dtypes)

print("\n-------------------------\n")

# 3. Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

print("Numerical Columns:")
print(numerical_cols)

print("\nCategorical Columns:")
print(categorical_cols)
