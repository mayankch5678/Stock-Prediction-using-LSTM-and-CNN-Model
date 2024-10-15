import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D

data = pd.read_csv("C:/Users/alexb/OneDrive/Desktop/Dataset Apple.csv")
data = data[['Date', 'Close', 'Open']]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close', 'Open']])

sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)


train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 2)),
    MaxPooling1D(pool_size=2),
    LSTM(units=50, return_sequences=True),
    LSTM(units=50),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

predicted_stock_price = model.predict(X_test)
predicted_stock_price_transformed = scaler.inverse_transform(np.column_stack((predicted_stock_price, np.zeros_like(predicted_stock_price))))[:, 0]

real_stock_price = data['Close'].iloc[train_size+sequence_length:].values

# Calculate RMSE and MAPE
rmse = np.sqrt(mean_squared_error(real_stock_price, predicted_stock_price_transformed))
print(f"Root Mean Squared Error (RMSE): {rmse}")

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(real_stock_price, predicted_stock_price_transformed)
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

plt.figure(figsize=(16, 6))
plt.plot(real_stock_price, label='Actual Prices', color='blue')
plt.plot(predicted_stock_price_transformed, label='Predicted Prices', color='red', linestyle='dashed')
plt.title('Actual vs. Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
