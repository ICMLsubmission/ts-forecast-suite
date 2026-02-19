
"""
Streamlit Time Series Forecasting Suite

Models:
- Statistical: SARIMAX, ETS
- ML: XGBoost
- DL: LSTM

Features:
- Drag-and-drop CSV/Excel upload
- Auto detection of datetime and target column (with sensible defaults)
- Train / Validation / Test split (time-based)
- Model training on train, ranking on validation
- Best model + weighted ensemble (weights ∝ 1 / RMSE on validation)
- Evaluation on test set
- Metrics: RMSE, MAPE
- Plots: Actual vs Forecast with Prediction Intervals

This is a POC-style app: readable, modular, easy for the team to extend.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def train_val_test_split_series(
    y: pd.Series,
    train_size: float = 0.6,
    val_size: float = 0.2
):
    """
    Time-based split of a univariate series into train/val/test.
    """
    n = len(y)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)

    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:val_end]
    y_test = y.iloc[val_end:]

    return y_train, y_val, y_test


def create_windowed_dataset(series, window_size: int, horizon: int = 1):
    """
    Create supervised data (X, y) from a univariate time series by windowing.

    X[t] = [y[t-window_size], ..., y[t-1]]
    y[t] = y[t + horizon - 1] (for horizon=1)
    """
    series = np.asarray(series)
    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size + horizon - 1])
    return np.array(X), np.array(y)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred, eps: float = 1e-8):
    """
    Mean Absolute Percentage Error with epsilon for stability.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.clip(np.abs(y_true), eps, None)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0


def gaussian_prediction_interval(y_pred, residuals, alpha: float = 0.05):
    """
    Simple Gaussian prediction interval from residuals:
    y_pred ± z * sigma.
    """
    y_pred = np.asarray(y_pred)
    residuals = np.asarray(residuals)

    sigma = np.std(residuals)
    # 95% PI ≈ 1.96
    z = 1.96 if abs(alpha - 0.05) < 1e-6 else 1.96
    lower = y_pred - z * sigma
    upper = y_pred + z * sigma
    return lower, upper


# -------------------------------------------------------------------
# Base model interface
# -------------------------------------------------------------------

class BaseTSModel:
    """
    Simple base interface to standardize fit() and forecast().
    """

    name = "BaseModel"

    def fit(self, y_train: pd.Series):
        raise NotImplementedError

    def forecast(self, steps: int) -> np.ndarray:
        raise NotImplementedError


# -------------------------------------------------------------------
# Statistical models: SARIMAX and ETS
# -------------------------------------------------------------------

class SARIMAXModel(BaseTSModel):
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
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

    def forecast(self, steps: int) -> np.ndarray:
        return self.model_fit.forecast(steps=steps).values


class ETSModelWrapper(BaseTSModel):
    def __init__(self, error="add", trend="add", seasonal=None, seasonal_periods=None):
        self.error = error
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model_fit = None
        self.name = "ETS"

    def fit(self, y_train: pd.Series):
        model = ETSModel(
            y_train,
            error=self.error,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        )
        self.model_fit = model.fit()

    def forecast(self, steps: int) -> np.ndarray:
        return self.model_fit.forecast(steps=steps).values


# -------------------------------------------------------------------
# ML model: XGBoost (with lag features)
# -------------------------------------------------------------------

class XGBoostTSModel(BaseTSModel):
    def __init__(self, window_size: int = 12):
        self.window_size = window_size
        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        self.train_series = None
        self.name = f"XGBoost(window={window_size})"

    def fit(self, y_train: pd.Series):
        self.train_series = np.asarray(y_train)
        X, y = create_windowed_dataset(self.train_series, self.window_size, horizon=1)
        self.model.fit(X, y)

    def forecast(self, steps: int) -> np.ndarray:
        """
        Recursive multi-step forecasting.
        """
        history = list(self.train_series)
        preds = []
        for _ in range(steps):
            if len(history) < self.window_size:
                window = [history[0]] * (self.window_size - len(history)) + history
            else:
                window = history[-self.window_size:]
            X_input = np.array(window).reshape(1, -1)
            yhat = self.model.predict(X_input)[0]
            preds.append(yhat)
            history.append(yhat)
        return np.array(preds)


# -------------------------------------------------------------------
# DL model: LSTM (with lag features)
# -------------------------------------------------------------------

class LSTMTSModel(BaseTSModel):
    def __init__(self, window_size: int = 12, n_units: int = 64, epochs: int = 50, batch_size: int = 32):
        self.window_size = window_size
        self.n_units = n_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.train_series = None
        self.name = f"LSTM(window={window_size})"

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(self.n_units, input_shape=(self.window_size, 1)))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
        return model

    def fit(self, y_train: pd.Series):
        self.train_series = np.asarray(y_train)
        X, y = create_windowed_dataset(self.train_series, self.window_size, horizon=1)
        # reshape for LSTM: (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        self.model = self._build_model()
        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        # simple holdout inside train for DL stability
        split = int(len(X) * 0.8) if len(X) > 10 else len(X)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        if len(X_val) == 0:
            val_data = None
        else:
            val_data = (X_val, y_val)

        self.model.fit(
            X_tr,
            y_tr,
            validation_data=val_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=[es],
        )

    def forecast(self, steps: int) -> np.ndarray:
        """
        Recursive multi-step forecast.
        """
        history = list(self.train_series)
        preds = []
        for _ in range(steps):
            if len(history) < self.window_size:
                window = [history[0]] * (self.window_size - len(history)) + history
            else:
                window = history[-self.window_size:]
            X_input = np.array(window).reshape(1, self.window_size, 1)
            yhat = self.model.predict(X_input, verbose=0)[0, 0]
            preds.append(yhat)
            history.append(yhat)
        return np.array(preds)


# -------------------------------------------------------------------
# Forecasting Suite
# -------------------------------------------------------------------

class ForecastingSuite:
    def __init__(
        self,
        models,
        train_size: float = 0.6,
        val_size: float = 0.2,
        metric: str = "rmse",
        logger=print,  # logger function (for Streamlit, pass st.write)
    ):
        self.models = models
        self.train_size = train_size
        self.val_size = val_size
        self.metric = metric
        self.logger = logger

        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.model_results_val = {}
        self.best_model = None
        self.ensemble_weights = None

    def fit_and_select(self, y: pd.Series):
        """
        1. Split y into train/val/test
        2. Fit each model on train and evaluate on val
        3. Choose best model and compute ensemble weights
        """
        self.y_train, self.y_val, self.y_test = train_val_test_split_series(
            y, train_size=self.train_size, val_size=self.val_size
        )

        self.logger(f"Train size: {len(self.y_train)}, Val size: {len(self.y_val)}, Test size: {len(self.y_test)}")

        if len(self.y_train) < 30:
            self.logger("⚠️ Very small training set. Results might be unstable.")

        metrics = {}

        for model in self.models:
            self.logger(f"\nFitting {model.name} on train, evaluating on validation...")
            try:
                model.fit(self.y_train)

                y_val_pred = model.forecast(steps=len(self.y_val))
                y_val_true = self.y_val.values

                model_rmse = rmse(y_val_true, y_val_pred)
                model_mape = mape(y_val_true, y_val_pred)

                self.model_results_val[model.name] = {
                    "rmse": model_rmse,
                    "mape": model_mape,
                    "y_val_pred": y_val_pred,
                }

                self.logger(f"  RMSE (val): {model_rmse:.4f}, MAPE (val): {model_mape:.2f}%")

                metrics[model.name] = model_rmse if self.metric == "rmse" else model_mape

            except Exception as e:
                self.logger(f"  ERROR fitting {model.name}: {e}")

        if not metrics:
            raise RuntimeError("No model successfully fitted. Check configurations and data.")

        best_name = min(metrics, key=metrics.get)
        self.best_model = next(m for m in self.models if m.name == best_name)
        self.logger(f"\nBest model based on validation {self.metric.upper()}: {self.best_model.name}")

        # Ensemble weights based on inverse RMSE
        rmse_values = np.array([
            self.model_results_val[m.name]["rmse"]
            for m in self.models
            if m.name in self.model_results_val
        ])
        inv_rmse = 1 / (rmse_values + 1e-8)
        weights = inv_rmse / inv_rmse.sum()
        self.ensemble_weights = {m.name: w for m, w in zip(self.models, weights)}

        self.logger("\nEnsemble weights (∝ 1 / RMSE on validation):")
        for name, w in self.ensemble_weights.items():
            if name in self.model_results_val:
                self.logger(f"  {name}: {w:.3f}")

    def _retrain_on_train_val(self, model: BaseTSModel):
        """
        Retrain the model on train + validation before final test evaluation.
        """
        y_train_val = pd.concat([self.y_train, self.y_val])
        model.fit(y_train_val)

    def evaluate_on_test(self):
        """
        Retrain models on train+val and evaluate on test.
        """
        if self.best_model is None:
            raise RuntimeError("You must run fit_and_select() first.")

        y_test_true = self.y_test.values

        # ---- Best model ----
        self.logger(f"\nRetraining best model ({self.best_model.name}) on train+val and evaluating on test...")
        self._retrain_on_train_val(self.best_model)
        y_test_pred_best = self.best_model.forecast(steps=len(self.y_test))

        val_residuals_best = self.y_val.values - self.model_results_val[self.best_model.name]["y_val_pred"]
        lower_best, upper_best = gaussian_prediction_interval(y_test_pred_best, val_residuals_best)

        best_rmse_test = rmse(y_test_true, y_test_pred_best)
        best_mape_test = mape(y_test_true, y_test_pred_best)
        self.logger(f"Best model test RMSE: {best_rmse_test:.4f}, MAPE: {best_mape_test:.2f}%")

        # ---- Ensemble ----
        self.logger("\nRetraining all models for ensemble on train+val and evaluating on test...")
        ensemble_preds = []
        valid_models_for_ensemble = []
        for model in self.models:
            if model.name not in self.ensemble_weights:
                continue
            try:
                self._retrain_on_train_val(model)
                yhat = model.forecast(steps=len(self.y_test))
                ensemble_preds.append(yhat)
                valid_models_for_ensemble.append(model)
            except Exception as e:
                self.logger(f"  Skipping {model.name} in ensemble due to error: {e}")

        if ensemble_preds:
            ensemble_preds = np.array(ensemble_preds)  # (n_models, n_steps)
            weights = np.array([self.ensemble_weights[m.name] for m in valid_models_for_ensemble]).reshape(-1, 1)
            y_test_pred_ens = (ensemble_preds * weights).sum(axis=0)

            # ensemble residuals on validation
            val_preds = np.zeros_like(self.y_val.values, dtype=float)
            for m in valid_models_for_ensemble:
                w = self.ensemble_weights[m.name]
                val_preds += w * self.model_results_val[m.name]["y_val_pred"]
            val_residuals_ens = self.y_val.values - val_preds
            lower_ens, upper_ens = gaussian_prediction_interval(y_test_pred_ens, val_residuals_ens)

            ens_rmse_test = rmse(y_test_true, y_test_pred_ens)
            ens_mape_test = mape(y_test_true, y_test_pred_ens)
            self.logger(f"Ensemble test RMSE: {ens_rmse_test:.4f}, MAPE: {ens_mape_test:.2f}%")
        else:
            y_test_pred_ens = None
            lower_ens, upper_ens = None, None
            ens_rmse_test = None
            ens_mape_test = None
            self.logger("No valid models for ensemble.")

        results = {
            "best_model_name": self.best_model.name,
            "best": {
                "y_test_true": y_test_true,
                "y_test_pred": y_test_pred_best,
                "lower": lower_best,
                "upper": upper_best,
                "rmse": best_rmse_test,
                "mape": best_mape_test,
            },
            "ensemble": {
                "y_test_true": y_test_true,
                "y_test_pred": y_test_pred_ens,
                "lower": lower_ens,
                "upper": upper_ens,
                "rmse": ens_rmse_test,
                "mape": ens_mape_test,
                "weights": self.ensemble_weights,
            },
        }
        return results

    @staticmethod
    def plot_forecasts(
        index,
        y_true,
        y_pred,
        lower=None,
        upper=None,
        title="Actual vs Forecast"
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(index, y_true, label="Actual", color="black")
        ax.plot(index, y_pred, label="Forecast", color="blue")
        if lower is not None and upper is not None:
            ax.fill_between(index, lower, upper, color="blue", alpha=0.2, label="Prediction Interval")
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        return fig


# -------------------------------------------------------------------
# Streamlit App
# -------------------------------------------------------------------

def auto_detect_datetime_column(df: pd.DataFrame):
    """
    Try to automatically detect a datetime column.
    """
    # if already datetime dtypes
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    if datetime_cols:
        return datetime_cols[0]

    # heuristic on column names
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "datetime", "ds", "timestamp"])]
    for c in candidates:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            continue
    return None


def auto_detect_target_column(df: pd.DataFrame):
    """
    Try to automatically select a numeric target column.
    Fallback to last column.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        return num_cols[-1]
    return df.columns[-1]


def main():
    st.set_page_config(page_title="Time Series Forecasting Suite", layout="wide")
    st.title("🧠 Time Series Forecasting Suite (SARIMAX / ETS / XGBoost / LSTM)")
    st.markdown(
        """
        Drag & drop your time series data and let the suite:
        - Split into Train / Validation / Test (time-based)
        - Train SARIMAX, ETS, XGBoost, LSTM
        - Pick the best model + build a weighted ensemble
        - Show MAPE, RMSE, and Actual vs Forecast plots with prediction intervals

        **Assumptions** (POC-style):
        - Univariate target (one main column to forecast)
        - Optional datetime column for proper ordering
        """
    )

    st.sidebar.header("1. Upload Data")
    file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

    if file is None:
        st.info("⬆️ Upload a CSV/Excel file to get started.")
        st.stop()

    # Read data
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Auto-detect datetime and target columns
    dt_col_guess = auto_detect_datetime_column(df)
    target_col_guess = auto_detect_target_column(df)

    st.sidebar.header("2. Basic Settings (auto-filled)")
    datetime_col = st.sidebar.selectbox(
        "Datetime column (optional but recommended)",
        options=["<None>"] + list(df.columns),
        index=(["<None>"] + list(df.columns)).index(dt_col_guess) if dt_col_guess in df.columns else 0,
    )

    target_col = st.sidebar.selectbox(
        "Target column to forecast",
        options=list(df.columns),
        index=list(df.columns).index(target_col_guess) if target_col_guess in df.columns else 0,
    )

    st.sidebar.header("3. Split Settings")
    train_size = st.sidebar.slider("Train fraction", 0.4, 0.8, 0.6, 0.05)
    val_size = st.sidebar.slider("Validation fraction", 0.1, 0.4, 0.2, 0.05)
    metric = st.sidebar.selectbox("Model ranking metric", ["rmse", "mape"], index=0)

    if train_size + val_size >= 0.95:
        st.sidebar.warning("Train + Validation fraction should leave some data for Test set.")

    # Prepare time series
    data = df.copy()

    if datetime_col != "<None>":
        # Parse datetime and sort
        data[datetime_col] = pd.to_datetime(data[datetime_col], errors="coerce")
        data = data.dropna(subset=[datetime_col])
        data = data.sort_values(datetime_col)
        data = data.set_index(datetime_col)
        index = data.index
    else:
        # Use existing index as time order
        index = data.index

    y = data[target_col].astype(float)
    st.markdown(f"**Using target column:** `{target_col}`")
    if datetime_col != "<None>":
        st.markdown(f"**Using datetime column:** `{datetime_col}` (sorted as index)")
    else:
        st.markdown("⚠️ **No datetime column selected.** Using row order as time.")

    if len(y) < 60:
        st.warning("Dataset is quite small (< 60 points). Deep learning model may be unstable, but we'll still run it.")

    st.markdown("---")
    run_button = st.button("🚀 Run Forecasting Suite")

    if not run_button:
        st.stop()

    # Run the forecasting suite
    with st.spinner("Running SARIMAX / ETS / XGBoost / LSTM... this might take a bit for bigger data."):
        log_container = st.empty()

        def logger(msg):
            log_container.write(str(msg)))

        # Define models (POC hyperparameters)
        models = [
            SARIMAXModel(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)),
            ETSModelWrapper(error="add", trend="add", seasonal=None, seasonal_periods=None),
            XGBoostTSModel(window_size=12),
            LSTMTSModel(window_size=12, n_units=32, epochs=30, batch_size=32),
        ]

        suite = ForecastingSuite(
            models=models,
            train_size=train_size,
            val_size=val_size,
            metric=metric,
            logger=logger,
        )

        try:
            suite.fit_and_select(y)
            results = suite.evaluate_on_test()
        except Exception as e:
            st.error(f"Error while running forecasting suite: {e}")
            st.stop()

    # ----------------- Display metrics -----------------

    st.markdown("## 📊 Model Validation Metrics (on Validation Set)")
    val_rows = []
    for name, res in suite.model_results_val.items():
        val_rows.append(
            {
                "Model": name,
                "RMSE (val)": round(res["rmse"], 4),
                "MAPE (val) %": round(res["mape"], 2),
            }
        )
    val_df = pd.DataFrame(val_rows).sort_values("RMSE (val)")
    st.dataframe(val_df.reset_index(drop=True))

    st.markdown("## 🧪 Test Metrics")

    best = results["best"]
    ens = results["ensemble"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### Best Model: `{results['best_model_name']}`")
        st.metric("RMSE (test)", f"{best['rmse']:.4f}")
        st.metric("MAPE (test) %", f"{best['mape']:.2f}")

    with col2:
        st.markdown("### Ensemble (Weighted)")
        if ens["rmse"] is not None:
            st.metric("RMSE (test)", f"{ens['rmse']:.4f}")
            st.metric("MAPE (test) %", f"{ens['mape']:.2f}")
        else:
            st.write("Ensemble not available.")

    # ----------------- Plots -----------------

    st.markdown("## 📈 Actual vs Forecast (Test Set)")

    test_index = suite.y_test.index

    # Best model plot
    fig_best = suite.plot_forecasts(
        test_index,
        best["y_test_true"],
        best["y_test_pred"],
        best["lower"],
        best["upper"],
        title=f"Best Model: {results['best_model_name']}",
    )
    st.pyplot(fig_best)

    # Ensemble plot
    if ens["y_test_pred"] is not None:
        fig_ens = suite.plot_forecasts(
            test_index,
            ens["y_test_true"],
            ens["y_test_pred"],
            ens["lower"],
            ens["upper"],
            title="Ensemble Forecast",
        )
        st.pyplot(fig_ens)

    # Show ensemble weights
    st.markdown("## ⚖️ Ensemble Weights (Validation-based)")
    weights_rows = []
    for name, w in ens["weights"].items():
        if name in suite.model_results_val:
            weights_rows.append({"Model": name, "Weight": round(float(w), 3)})
    if weights_rows:
        st.dataframe(pd.DataFrame(weights_rows))


if __name__ == "__main__":
    main()
