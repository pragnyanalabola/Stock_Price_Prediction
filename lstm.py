import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime

# Function to fetch stock data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

# Function to preprocess data
def preprocess_data(data, time_steps=60):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

# Function to create LSTM model
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Streamlit UI
st.title('Stock Price Prediction using LSTM')

ticker = st.text_input('Enter Stock Ticker:', 'AAPL')
start_date = st.date_input('Start Date:', datetime.date(2015, 1, 1))
end_date = st.date_input('End Date:', datetime.date.today())

if st.button('Load Data'):
    data = load_data(ticker, start_date, end_date)
    st.subheader('Stock Price Data')
    st.write(data.tail())
    
    # Plot historical data
    st.subheader('Stock Closing Price History')
    fig, ax = plt.subplots()
    ax.plot(data['Close'], label='Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)
    
    # Preprocessing
    X, y, scaler = preprocess_data(data)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Train LSTM Model
    model = create_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=5, batch_size=32)
    
    # Prepare test data
    test_data = data['Close'][-60:].values.reshape(-1,1)
    test_data = scaler.transform(test_data)
    X_test = [test_data]
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Predict next day price
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    st.subheader('Predicted Stock Price for Next Day')
    st.write(predicted_price[0][0])
