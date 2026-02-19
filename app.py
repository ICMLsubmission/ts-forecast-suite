# ===========================
# app.py — CHUNK 1/5
# ===========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.metrics import mean_squared_error

from eda_utils import (
    plot_time_series,
    plot_histogram,
    plot_acf_pacf,
    stationarity_tests,
    ets_decomposition,
    plot_fft,
    detect_seasonality,
)

from model_utils import (
    SARIMAXModel,
    ETSWrapper,
    XGBWrapper,
    LSTMWrapper,
)

# --------- basic utilities ----------

def train_val_test_split_series(y: pd.Series, train_frac: float, val_frac: float):
    """Time-based split: returns y_train, y_val, y_test."""
    n = len(y)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    if val_end >= n:
        val_end = n - 1
    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:val_end]
    y_test = y.iloc[val_end:]
    return y_train, y_val, y_test


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred, eps: float = 1e-8):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def gaussian_prediction_interval(y_pred, residuals, alpha: float = 0.05):
    """
    Simple Gaussian PI: y_pred ± z * sigma, using residual std from validation.
    """
    y_pred = np.asarray(y_pred)
    residuals = np.asarray(residuals)
    sigma = np.std(residuals)
    z = 1.96 if abs(alpha - 0.05) < 1e-6 else 1.96
    lower = y_pred - z * sigma
    upper = y_pred + z * sigma
    return lower, upper

# ===========================
# app.py — CHUNK 2/5
# ===========================

st.set_page_config(
    page_title="Time Series Forecasting Suite",
    layout="wide"
)

st.title("📈 Time Series Forecasting Suite (EDA + SARIMAX + ETS + XGBoost + LSTM)")
st.markdown(
    """
    This app lets you:
    - Explore your time series (EDA tab)
    - Configure basic hyperparameters
    - Train SARIMAX, ETS, XGBoost, LSTM
    - Compare models on validation
    - Evaluate best model on test with prediction intervals
    """
)

# ------------- Sidebar: data upload -------------

st.sidebar.header("1️⃣ Upload Time Series Data")
file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if file is None:
    st.info("⬅️ Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(file)

# ------------- Sidebar: basic column selection -------------

st.sidebar.header("2️⃣ Basic Settings")
datetime_col = st.sidebar.selectbox("Datetime column", df.columns)
target_col = st.sidebar.selectbox("Target column to forecast", df.columns)

# Convert and sort
df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
df = df.dropna(subset=[datetime_col])
df = df.sort_values(datetime_col)
df = df.set_index(datetime_col)

y = df[target_col].astype(float)

st.write("#### Data Preview")
st.dataframe(df[[target_col]].head())

st.markdown(f"**Using datetime column:** `{datetime_col}`")
st.markdown(f"**Using target column:** `{target_col}`")

# ------------- Sidebar: split + metric -------------

st.sidebar.header("3️⃣ Split & Ranking")
train_frac = st.sidebar.slider("Train fraction", 0.4, 0.8, 0.6, 0.05)
val_frac = st.sidebar.slider("Validation fraction", 0.1, 0.4, 0.2, 0.05)
ranking_metric = st.sidebar.selectbox("Model ranking metric", ["rmse", "mape"], index=0)

y_train, y_val, y_test = train_val_test_split_series(y, train_frac, val_frac)

st.write(
    f"**Split sizes** → Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}"
)
if len(y_test) < 5:
    st.warning("⚠️ Very small test set. Metrics may be unstable.")

# ------------- Sidebar: hyperparameters -------------

st.sidebar.header("4️⃣ Model Hyperparameters")

# SARIMAX
st.sidebar.subheader("SARIMAX")
p = st.sidebar.number_input("p", min_value=0, max_value=5, value=1, step=1)
d = st.sidebar.number_input("d", min_value=0, max_value=2, value=1, step=1)
q = st.sidebar.number_input("q", min_value=0, max_value=5, value=1, step=1)
P = st.sidebar.number_input("P (seasonal)", min_value=0, max_value=5, value=0, step=1)
D = st.sidebar.number_input("D (seasonal)", min_value=0, max_value=2, value=0, step=1)
Q = st.sidebar.number_input("Q (seasonal)", min_value=0, max_value=5, value=0, step=1)
S = st.sidebar.number_input("Seasonal period S", min_value=0, max_value=60, value=0, step=1)

# ETS
st.sidebar.subheader("ETS")
ets_trend = st.sidebar.selectbox("Trend", [None, "add", "mul"], index=1)
ets_seasonal = st.sidebar.selectbox("Seasonal", [None, "add", "mul"], index=0)
ets_periods = st.sidebar.number_input("Seasonal periods", min_value=1, max_value=60, value=12, step=1)

# XGBoost
st.sidebar.subheader("XGBoost")
xgb_window = st.sidebar.number_input("Lag window (XGB)", min_value=3, max_value=60, value=12, step=1)
xgb_depth = st.sidebar.number_input("Max depth", min_value=1, max_value=10, value=3, step=1)
xgb_estimators = st.sidebar.number_input("n_estimators", min_value=50, max_value=1000, value=300, step=50)

# LSTM
st.sidebar.subheader("LSTM")
lstm_window = st.sidebar.number_input("Lag window (LSTM)", min_value=3, max_value=60, value=12, step=1)
lstm_units = st.sidebar.number_input("Units", min_value=10, max_value=256, value=64, step=8)
lstm_epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=100, value=30, step=1)
lstm_batch = st.sidebar.number_input("Batch size", min_value=8, max_value=256, value=32, step=8)


# ===========================
# app.py — CHUNK 3/5 (EDA tab)
# ===========================

tab_eda, tab_models = st.tabs(["🔍 EDA", "🤖 Modeling & Forecasting"])

with tab_eda:
    st.header("🔍 Exploratory Data Analysis (on Train Set)")

    st.markdown("#### 1️⃣ Train Series Line Plot")
    st.pyplot(plot_time_series(y_train))

    st.markdown("#### 2️⃣ Distribution (Histogram)")
    st.pyplot(plot_histogram(y_train))

    st.markdown("#### 3️⃣ ACF & PACF")
    st.pyplot(plot_acf_pacf(y_train))

    st.markdown("#### 4️⃣ ETS Decomposition")
    # crude period guess – if you know domain frequency, you can override
    period_guess = max(2, min(len(y_train) // 5, 60))
    try:
        decomp = ets_decomposition(y_train, period_guess)
        fig, axes = plt.subplots(4, 1, figsize=(10, 8))
        decomp.observed.plot(ax=axes[0]); axes[0].set_title("Observed")
        decomp.trend.plot(ax=axes[1]); axes[1].set_title("Trend")
        decomp.seasonal.plot(ax=axes[2]); axes[2].set_title("Seasonal")
        decomp.resid.plot(ax=axes[3]); axes[3].set_title("Residual")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"ETS decomposition failed: {e}")

    st.markdown("#### 5️⃣ Fourier Transform Spectrum")
    st.pyplot(plot_fft(y_train))

    st.markdown("#### 6️⃣ Stationarity Tests (ADF / KPSS)")
    stat = stationarity_tests(y_train)
    st.write(stat)

    st.markdown("#### 7️⃣ Seasonality Detection")
    seas = detect_seasonality(y_train)
    st.write(seas)

    # Summary insights
    st.markdown("### 🧠 Summary Insights")

    adf_comment = "likely **stationary**" if stat["ADF_p"] < 0.05 else "likely **non‑stationary**"
    kpss_comment = (
        "KPSS suggests **stationary**" if stat["KPSS_p"] > 0.05 else "KPSS suggests **non‑stationary**"
    ) if not np.isnan(stat["KPSS_p"]) else "KPSS not available."

    st.write(f"- ADF p‑value = **{stat['ADF_p']:.4f}** → {adf_comment}")
    if not np.isnan(stat["KPSS_p"]):
        st.write(f"- KPSS p‑value = **{stat['KPSS_p']:.4f}** → {kpss_comment}")
    st.write(
        f"- Seasonality detected: **{seas['seasonal']}**, "
        f"period guess ≈ **{seas['period_guess']}** (in time steps)"
    )


# ===========================
# app.py — CHUNK 4/5 (Modeling setup & validation)
# ===========================

with tab_models:
    st.header("🤖 Modeling & Forecasting")

    st.write(
        f"Train: **{len(y_train)}** points | "
        f"Validation: **{len(y_val)}** | "
        f"Test: **{len(y_test)}**"
    )

    run_button = st.button("🚀 Run Forecasting Suite")
    if not run_button:
        st.info("Configure hyperparameters in the sidebar, then click **Run Forecasting Suite**.")
        st.stop()

    progress = st.progress(0)
    log_area = st.empty()
    logs = []

    def log(msg):
        logs.append(str(msg))
        log_area.write("\n".join(logs))

    progress.progress(5)
    log("Starting pipeline...")

    # ----------------- Build models based on sidebar params -----------------

    # SARIMAX seasonal order handling
    if S is not None and S > 1 and (P + D + Q) > 0:
        seasonal_order = (P, D, Q, S)
    else:
        seasonal_order = (0, 0, 0, 0)

    models = []

    models.append(
        SARIMAXModel(
            order=(p, d, q),
            seasonal_order=seasonal_order
        )
    )

    # ETS config
    ets_trend_param = ets_trend if ets_trend is not None else None
    ets_seasonal_param = ets_seasonal if ets_seasonal is not None else None
    ets_period_param = ets_periods if ets_seasonal_param is not None else None

    models.append(
        ETSWrapper(
            trend=ets_trend_param,
            seasonal=ets_seasonal_param,
            seasonal_periods=ets_period_param
        )
    )

    # XGBoost
    models.append(
        XGBWrapper(
            window=xgb_window,
            n_estimators=int(xgb_estimators),
            max_depth=int(xgb_depth),
        )
    )

    # LSTM
    models.append(
        LSTMWrapper(
            window=int(lstm_window),
            units=int(lstm_units),
            epochs=int(lstm_epochs),
            batch_size=int(lstm_batch),
        )
    )

    log("Models initialized: " + ", ".join(m.name for m in models))
    progress.progress(10)

    # ----------------- Train & validate each model -----------------

    results_val = {}
    rows = []

    for idx, model in enumerate(models, start=1):
        log(f"Fitting {model.name} on train...")
        try:
            model.fit(y_train)
            log(f"Forecasting on validation with {model.name}...")
            y_val_pred = model.forecast(len(y_val))

            val_rmse = rmse(y_val.values, y_val_pred)
            val_mape = mape(y_val.values, y_val_pred)

            results_val[model.name] = {
                "rmse": val_rmse,
                "mape": val_mape,
                "y_val_pred": y_val_pred,
            }

            rows.append(
                {
                    "Model": model.name,
                    "RMSE (val)": round(val_rmse, 4),
                    "MAPE (val) %": round(val_mape, 2),
                }
            )

            log(f"{model.name} → RMSE={val_rmse:.4f}, MAPE={val_mape:.2f}%")

        except Exception as e:
            log(f"❌ {model.name} failed: {e}")

        progress.progress(10 + int(idx * (60 / len(models))))

    if not results_val:
        st.error("All models failed on validation. Check data / hyperparameters.")
        st.stop()

    # ----------------- Show validation metrics & pick best model -----------------

    st.subheader("📊 Validation Metrics")
    val_df = pd.DataFrame(rows).sort_values("RMSE (val)")
    st.dataframe(val_df.reset_index(drop=True))

    if ranking_metric == "rmse":
        key_metric = "rmse"
    else:
        key_metric = "mape"

    best_name = min(results_val.keys(), key=lambda k: results_val[k][key_metric])
    best_model = next(m for m in models if m.name == best_name)

    st.success(f"🏆 Best model on validation ({ranking_metric.upper()}): **{best_name}**")

    progress.progress(75)
    log(f"Best model selected: {best_name}. Retraining on train+val...")


# ===========================
# app.py — CHUNK 5/5 (Test evaluation & plots)
# ===========================

    # Retrain best model on combined train+val
    y_train_val = pd.concat([y_train, y_val])
    best_model.fit(y_train_val)

    log("Forecasting on test with best model...")
    y_test_pred = best_model.forecast(len(y_test))

    # Use validation residuals for PI
    val_residuals = y_val.values - results_val[best_name]["y_val_pred"]
    lower_pi, upper_pi = gaussian_prediction_interval(y_test_pred, val_residuals)

    test_rmse = rmse(y_test.values, y_test_pred)
    test_mape = mape(y_test.values, y_test_pred)

    progress.progress(90)
    log(f"Test RMSE: {test_rmse:.4f}, MAPE: {test_mape:.2f}%")

    st.subheader("🧪 Test Set Performance (Best Model)")
    c1, c2 = st.columns(2)
    c1.metric("RMSE (test)", f"{test_rmse:.4f}")
    c2.metric("MAPE (test) %", f"{test_mape:.2f}")

    st.subheader("📈 Actual vs Forecast (Test Set) with Prediction Interval")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_test.index, y_test.values, label="Actual", color="black")
    ax.plot(y_test.index, y_test_pred, label="Forecast", color="blue")
    ax.fill_between(
        y_test.index,
        lower_pi,
        upper_pi,
        color="blue",
        alpha=0.2,
        label="Prediction Interval (approx.)",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)

    progress.progress(100)
    log("✅ Pipeline completed.")
