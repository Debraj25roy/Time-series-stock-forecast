"""Train ARIMA/SARIMA, Prophet, and a simple LSTM. Models are saved/loaded externally."""
import joblib
import numpy as np
import pandas as pd

# ARIMA / SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Prophet
from prophet import Prophet

# LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def train_arima(series, order=(5,1,0)):
    model = ARIMA(series, order=order)
    res = model.fit()
    return res

def train_sarima(series, order=(1,1,1), seasonal_order=(0,0,0,0)):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    res = model.fit(disp=False)
    return res

def train_prophet(series):
    df = series.reset_index().rename(columns={'Date':'ds','Close':'y'}) if 'Date' in series.index.names or isinstance(series.index, pd.DatetimeIndex) else series
    if isinstance(series, pd.Series):
        df = series.reset_index()
        df.columns = ['ds','y']
    m = Prophet()
    m.fit(df)
    return m

def train_lstm(series, n_lags=10, epochs=30, batch_size=16):
    # create supervised dataset
    arr = series.values.astype('float32')
    X, y = [], []
    for i in range(n_lags, len(arr)):
        X.append(arr[i-n_lags:i])
        y.append(arr[i])
    X = np.array(X)[:, :, np.newaxis]
    y = np.array(y)
    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
    return model