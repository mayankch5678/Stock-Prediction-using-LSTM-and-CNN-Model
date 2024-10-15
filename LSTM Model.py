import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

data = pd.read_csv("C:/Users/alexb/OneDrive/Desktop/Dataset Starbucks.csv")
data = data[['Date', 'Close', 'Open']]

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data[['Close', 'Open']])
sequence_length = 60
X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 0])  

X, y = np.array(X), np.array(y)

train_size = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 2)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=70, batch_size=128)

predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(np.hstack((predicted_prices, X_test[:, -1, 1].reshape(-1, 1))))[:, 0]

actual_prices = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), X_test[:, -1, 1].reshape(-1, 1))))[:, 0]

rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
print(f"Root Mean Squared Error: {rmse}")

mape = mean_absolute_percentage_error(actual_prices, predicted_prices)
print(f"Mean Absolute Percentage Error: {mape:.2f}%")

plt.figure(figsize=(16, 6))
plt.plot(actual_prices, label='Actual Prices', color='blue')
plt.plot(predicted_prices, label='Predicted Prices', color='red', linestyle='dashed')
plt.title('Actual vs. Predicted Stock Prices')
plt.xlabel('Time (in days)')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
