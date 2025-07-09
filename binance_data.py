from binance.client import Client
import pandas as pd

client = Client()

def fetch_data(symbol="BTCUSDT", interval="1h", lookback=1000):
    frame = pd.DataFrame(client.get_klines(symbol=symbol, interval=interval, limit=int(lookback)))
    frame = frame.iloc[:, :6]
    frame.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    frame['Time'] = pd.to_datetime(frame['Time'], unit='ms')
    frame.set_index('Time', inplace=True)
    frame = frame.astype(float)
    return frame
