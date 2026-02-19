import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# -------------------------------
# Base model abstraction
# -------------------------------

class BaseTSModel:
    name = "BaseModel"

    def fit(self, y_train: pd.Series):
        raise NotImplementedError

    def forecast(self, steps: int):
        raise NotImplementedError


# -------------------------------
# SARIMAX
# -------------------------------

class SARIMAXModel(BaseTSModel):
    def __init__(self, order, seasonal_order):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None
        self.name = f"SARIMAX{order}{seasonal_order}"

    def fit(self, y_train: pd.Series):
        model = SARIMAX(
            y_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_fit = model.fit(disp=False)

    def forecast(self, steps: int):
        return self.model_fit.forecast(steps=steps).values


# -------------------------------
# ETS
# -------------------------------

class ETSWrapper(BaseTSModel):
    def __init__(self, trend, seasonal, seasonal_periods):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model_fit = None
        self.name = "ETS"

    def fit(self, y_train: pd.Series):
        model = ETSModel(
            y_train,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        )
        self.model_fit = model.fit()

    def forecast(self, steps: int):
        return self.model_fit.forecast(steps=steps).values


# -------------------------------
# XGBoost
# -------------------------------

def create_windowed(series, window: int):
    X, y = [], []
    arr = np.asarray(series)
    for i in range(len(arr) - window):
        X.append(arr[i : i + window])
        y.append(arr[i + window])
    return np.array(X), np.array(y)


class XGBWrapper(BaseTSModel):
    def __init__(self, window: int, n_estimators: int, max_depth: int):
        self.window = window
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        self.train_series = None
        self.name = f"XGBoost(window={window})"

    def fit(self, y_train: pd.Series):
        self.train_series = y_train.values
        X, y = create_windowed(self.train_series, self.window)
        self.model.fit(X, y)

    def forecast(self, steps: int):
        history = list(self.train_series)
        preds = []
        for _ in range(steps):
            seq = np.array(history[-self.window :]).reshape(1, -1)
            yhat = self.model.predict(seq)[0]
            preds.append(yhat)
            history.append(yhat)
        return np.array(preds)


# -------------------------------
# LSTM
# -------------------------------

class LSTMWrapper(BaseTSModel):
    def __init__(self, window: int, units: int, epochs: int, batch_size: int):
        self.window = window
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_series = None
        self.model = None
        self.name = f"LSTM(window={window})"

    def fit(self, y_train: pd.Series):
        self.train_series = y_train.values
        X, y = create_windowed(self.train_series, self.window)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(self.units, input_shape=(self.window, 1)))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.model = model

    def forecast(self, steps: int):
        history = list(self.train_series)
        preds = []
        for _ in range(steps):
            seq = np.array(history[-self.window :]).reshape(1, self.window, 1)
            yhat = self.model.predict(seq, verbose=0)[0, 0]
            preds.append(yhat)
            history.append(yhat)
        return np.array(preds)
