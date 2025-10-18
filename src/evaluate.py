"""Evaluate saved models and produce simple RMSE/MAE metrics + plots"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from src.preprocess import load_series

def rmse(a,b): return np.sqrt(((a-b)**2).mean())
def mae(a,b): return np.abs(a-b).mean()

def evaluate_arima(path, series, out_dir):
    res = joblib.load(path)
    pred = res.predict(start=series.index[0], end=series.index[-1])
    return pred

def evaluate_prophet(path, series, out_dir):
    m = joblib.load(path)
    future = pd.DataFrame({'ds': series.index})
    fcst = m.predict(future)
    return pd.Series(fcst['yhat'].values, index=series.index)

def evaluate_lstm(path, series, out_dir):
    from tensorflow.keras.models import load_model
    model = load_model(path)
    # naive full-window forecast for demo: use last n_lags to predict forward one-step repeatedly
    n_lags = 10
    arr = series.values.astype('float32')
    preds = []
    window = arr[-n_lags:].tolist()
    for _ in range(len(arr)-n_lags):
        import numpy as np
        x = np.array(window[-n_lags:]).reshape(1,n_lags,1)
        p = model.predict(x, verbose=0)[0,0]
        preds.append(p)
        window.append(p)
    idx = series.index[n_lags:]
    s = pd.Series([None]*n_lags + list(preds), index=series.index)
    return s

def main(ticker, models_dir='models', out_dir='reports/figures'):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    series = load_series(ticker)
    results = {}
    # search for files
    import glob
    for p in glob.glob(f"{models_dir}/{ticker}_*.pkl") + glob.glob(f"{models_dir}/{ticker}_*.h5"):
        p = Path(p)
        name = p.stem
        if name.endswith('arima'):
            pred = evaluate_arima(p, series, out)
        elif name.endswith('prophet'):
            pred = evaluate_prophet(p, series, out)
        elif name.endswith('lstm'):
            pred = evaluate_lstm(p, series, out)
        else:
            continue
        # align
        pred = pred.reindex(series.index)
        # compute metrics on overlapping window
        mask = ~pred.isna()
        results[name] = {'rmse': rmse(series[mask], pred[mask]), 'mae': mae(series[mask], pred[mask])}
        # simple plot
        plt.figure(figsize=(8,4))
        plt.plot(series.index, series.values, label='actual')
        plt.plot(pred.index, pred.values, label='pred')
        plt.legend()
        plt.title(name)
        plt.savefig(out / f"{name}.png")
        plt.close()
    print('Results:', results)
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--models_dir', default='models/')
    parser.add_argument('--out_dir', default='reports/figures/')
    args = parser.parse_args()
    main(args.ticker, args.models_dir, args.out_dir)