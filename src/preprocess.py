"""Simple preprocessing: resample daily, fill missing, return series of 'Close'"""
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path('data/processed')

def load_series(ticker):
    p = PROCESSED_DIR / f"{ticker}_processed.csv"
    df = pd.read_csv(p, parse_dates=['Date'], index_col='Date')
    df = df.asfreq('B')  # business day frequency
    df['Close'] = df['Close'].fillna(method='ffill')
    return df['Close']

if __name__ == '__main__':
    s = load_series('AAPL')
    print(s.tail())