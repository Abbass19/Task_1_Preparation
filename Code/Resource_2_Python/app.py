import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam


def preprocess_data(data):
    close_prices = data['Close'].values

    min_price = np.min(close_prices)
    max_price = np.max(close_prices)
    normalized_prices = (close_prices - min_price) / (max_price - min_price)

    return normalized_prices


def build_model(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(output_shape, activation='linear'))
    model.compile(optimizer=Adam(), loss='mse')
    return model


def train_model(model, X_train, y_train, epochs=10):
    model.fit(X_train, y_train, epochs=epochs, verbose=1)


def predict_action(model, state):
    action = model.predict(state)
    return action


symbol = 'AAPL'
start_date = '2022-01-01'
end_date = '2023-07-01'

stock_data = yf.download(symbol, start=start_date, end=end_date)

preprocessed_data = preprocess_data(stock_data)

train_data = preprocessed_data[:int(0.8 * len(preprocessed_data))]
test_data = preprocessed_data[int(0.8 * len(preprocessed_data)):]

input_shape = (None, 1)
output_shape = 1

model = build_model(input_shape, output_shape)

X_train = np.expand_dims(train_data[:-1], axis=1)
y_train = np.expand_dims(train_data[1:], axis=1)

train_model(model, X_train, y_train, epochs=10)

X_test = np.expand_dims(test_data[:-1], axis=1)
y_test = np.expand_dims(test_data[1:], axis=1)

loss = model.evaluate(X_test, y_test)
print(f'Testing loss: {loss}')

actions = predict_action(model, X_test)

plt.figure(figsize=(12, 6))
plt.plot(test_data[1:], label='True Price', color='blue')
plt.plot(actions, label='Predicted Price', color='red')
plt.title(f'{symbol} Stock Price Prediction with Deep RL')
plt.xlabel('Time')
plt.ylabel('Normalized Price')
plt.legend()
plt.show()