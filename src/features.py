"""Feature engineering: lags and rolling stats"""
import pandas as pd

def create_lag_features(series, lags=[1,2,3,5,10]):
    df = pd.DataFrame({'y': series})
    for l in lags:
        df[f'lag_{l}'] = df['y'].shift(l)
    df['rolling_mean_5'] = df['y'].shift(1).rolling(window=5).mean()
    df = df.dropna()
    return df