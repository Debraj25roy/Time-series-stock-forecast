"""Command-line training script that supports arima, prophet, lstm."""
import argparse
from pathlib import Path
from src.preprocess import load_series
from src.models import train_arima, train_prophet, train_lstm
import joblib
import os

MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True, parents=True)

def main(ticker, models):
    s = load_series(ticker)
    results = {}
    if 'arima' in models:
        print('Training ARIMA...')
        ar = train_arima(s)
        joblib.dump(ar, MODELS_DIR / f"{ticker}_arima.pkl")
        results['arima'] = str(MODELS_DIR / f"{ticker}_arima.pkl")
    if 'prophet' in models:
        print('Training Prophet...')
        pr = train_prophet(s.rename('Close'))
        joblib.dump(pr, MODELS_DIR / f"{ticker}_prophet.pkl")
        results['prophet'] = str(MODELS_DIR / f"{ticker}_prophet.pkl")
    if 'lstm' in models:
        print('Training LSTM (this may take a while)...')
        lst = train_lstm(s)
        lst.save(MODELS_DIR / f"{ticker}_lstm.h5")
        results['lstm'] = str(MODELS_DIR / f"{ticker}_lstm.h5")
    print('Saved models to', MODELS_DIR)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--models', nargs='+', default=['arima','prophet','lstm'])
    parser.add_argument('--save_dir', default='models/')
    args = parser.parse_args()
    main(args.ticker, args.models)