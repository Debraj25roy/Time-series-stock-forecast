"""Download stock data using yfinance and save raw + processed CSVs."""
import yfinance as yf
import argparse
from pathlib import Path
import pandas as pd

RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def download_ticker(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker} in range {start} to {end}")
    # save raw
    df.to_csv(RAW_DIR / f"{ticker}.csv")
    # keep relevant columns and save processed
    df = df[['Open','High','Low','Close','Adj Close','Volume']].copy()
    df.index.name = 'Date'
    df.to_csv(PROCESSED_DIR / f"{ticker}_processed.csv")
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='+', required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    args = parser.parse_args()
    for t in args.tickers:
        print(f"Downloading {t}...")
        download_ticker(t, args.start, args.end)
        print(f"Saved {t}")