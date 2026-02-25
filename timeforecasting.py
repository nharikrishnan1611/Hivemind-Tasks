import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
import yfinance as yf

df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
df = df[['Close']]
df = df.sort_index()

print("Dataset Loaded Successfully")
print(df.head())

print("\nMissing Values Before Handling:")
print(df.isnull().sum())

df.fillna(method='ffill', inplace=True)

print("\nMissing Values After Handling:")
print(df.isnull().sum())

plt.figure()
plt.plot(df['Close'])
plt.title("Stock Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

df['7_day_avg'] = df['Close'].rolling(window=7).mean()
df['30_day_avg'] = df['Close'].rolling(window=30).mean()

plt.figure()
plt.plot(df['Close'])
plt.plot(df['7_day_avg'])
plt.plot(df['30_day_avg'])
plt.title("Rolling Averages")
plt.show()

decomposition = seasonal_decompose(df['Close'], model='additive', period=30)
decomposition.plot()
plt.show()

df['lag_1'] = df['Close'].shift(1)
df['lag_7'] = df['Close'].shift(7)

df.dropna(inplace=True)

train_size = int(len(df) * 0.8)

train = df[:train_size]
test = df[train_size:]

X_train = train[['lag_1', 'lag_7']]
y_train = train['Close']

X_test = test[['lag_1', 'lag_7']]
y_test = test['Close']
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print("\nModel Evaluation Results:")
print("RMSE:", rmse)
print("MAE:", mae)

plt.figure()
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, predictions, label="Predicted")
plt.title("Actual vs Predicted")
plt.legend()
plt.show()