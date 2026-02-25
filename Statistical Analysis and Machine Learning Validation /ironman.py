# ===============================
# STATISTICAL ANALYSIS + ML VALIDATION
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("your_dataset.csv")

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

print("\nDataset Preview:\n")
print(df.head())

# -------------------------------
# 2. STATISTICAL ANALYSIS
# -------------------------------
print("\nStatistical Summary:\n")
print(df.describe())

print("\nChecking Missing Values:\n")
print(df.isnull().sum())

# -------------------------------
# 3. STATIONARITY TEST (ADF Test)
# -------------------------------
result = adfuller(df.iloc[:,0])

print("\nADF Test Results:")
print(f"ADF Statistic: {result[0]}")
print(f"P-Value: {result[1]}")

if result[1] < 0.05:
    print("Data is Stationary")
else:
    print("Data is NOT Stationary")

# -------------------------------
# 4. FEATURE ENGINEERING
# -------------------------------
df['Lag_1'] = df.iloc[:,0].shift(1)
df.dropna(inplace=True)

X = df[['Lag_1']]
y = df.iloc[:,0]

# -------------------------------
# 5. TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

# -------------------------------
# 6. MACHINE LEARNING MODEL
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# -------------------------------
# 7. VALIDATION METRICS
# -------------------------------
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nModel Validation Results:")
print(f"MAE  : {mae}")
print(f"MSE  : {mse}")
print(f"RMSE : {rmse}")
print(f"R2   : {r2}")

# -------------------------------
# 8. VISUALIZATION
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, predictions, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted")
plt.show()
