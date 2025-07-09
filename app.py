import streamlit as st
import numpy as np
import pandas as pd
from binance_data import fetch_data
from model_utils import create_dataset, build_model
from sklearn.preprocessing import MinMaxScaler
import datetime

st.set_page_config(page_title="Crypto Forecast", layout="centered")
st.title("📈 Real-Time Crypto Price Forecast")
symbol = st.selectbox("Select a Crypto Pair", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])

df = fetch_data(symbol=symbol, interval="1h", lookback=1000)
data = df[['Close']].values

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

X, y = create_dataset(scaled_data)
X = X.reshape(X.shape[0], X.shape[1], 1)

model = build_model((X.shape[1], 1))
model.fit(X, y, epochs=5, batch_size=32, verbose=0)

last_60 = scaled_data[-60:].reshape(1, 60, 1)
prediction = model.predict(last_60)
predicted_price = scaler.inverse_transform(prediction)[0][0]

st.subheader("Last 1000 Price Points")
st.line_chart(df['Close'])

st.subheader("🔮 Forecast")
st.metric(label=f"Next Hour Forecast for {symbol}", value=f"${predicted_price:.2f}")
