import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

data = pd.read_csv("C:/Users/alexb/OneDrive/Desktop/Dataset Starbucks.csv")
data = data[["Date", "Close", "Open"]]
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date")

scaler_close = MinMaxScaler()
scaler_open = MinMaxScaler()
data["Close"] = scaler_close.fit_transform(data[["Close"]])
data["Open"] = scaler_open.fit_transform(data[["Open"]])

# Define train_size to split the dataset
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 30
X_train, y_train = create_dataset(train[["Close", "Open"]], train["Close"], TIME_STEPS)
X_test, y_test = create_dataset(test[["Close", "Open"]], test["Close"], TIME_STEPS)

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
history = model.fit(X_train, y_train, epochs=120, batch_size=64, validation_split=0.1, shuffle=False)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(scaler_close.inverse_transform(y_test.reshape(-1, 1)), scaler_close.inverse_transform(y_pred))

print("Root Mean Squared Error:", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)

plt.figure(figsize=(16, 6))
real_stock_price = scaler_close.inverse_transform(y_test.reshape(-1, 1))
predicted_stock_price_transformed = scaler_close.inverse_transform(y_pred)

plt.plot(real_stock_price, label='Actual Prices', color='blue')
plt.plot(predicted_stock_price_transformed, label='Predicted Prices', color='red', linestyle='dashed')
plt.title('Actual vs. Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
