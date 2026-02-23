import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf as sm_plot_acf, plot_pacf as sm_plot_pacf
from scipy.fft import fft

# -------------------------------
# Time Series EDA Utilities
# -------------------------------

def plot_time_series(y: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y.index, y.values, color="blue")
    ax.set_title("Time Series")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_histogram(y: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(y.values, bins=30, color="gray", edgecolor="black")
    ax.set_title("Distribution")
    fig.tight_layout()
    return fig


def plot_acf_pacf(y: pd.Series, lags: int = 40):
    """
    Robust ACF & PACF using statsmodels' plot_acf/plot_pacf.
    No manual matplotlib.stem calls to avoid TypeError issues.
    """
    y = pd.Series(y).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if len(y) < 15:
        # Not enough data – show message instead of crashing
        for ax in axes:
            ax.text(
                0.5, 0.5,
                "Not enough data for ACF/PACF\n(need ≥ 15 points)",
                ha="center", va="center"
            )
            ax.set_axis_off()
        fig.tight_layout()
        return fig

    max_lags = min(lags, len(y) // 2)

    try:
        sm_plot_acf(y, lags=max_lags, ax=axes[0])
        axes[0].set_title("ACF")
        axes[0].grid(True)
    except Exception as e:
        axes[0].text(0.5, 0.5, f"ACF error: {e}", ha="center", va="center")
        axes[0].set_axis_off()

    try:
        sm_plot_pacf(y, lags=max_lags, ax=axes[1], method="ywm")
        axes[1].set_title("PACF")
        axes[1].grid(True)
    except Exception as e:
        axes[1].text(0.5, 0.5, f"PACF error: {e}", ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()
    return fig


def ets_decomposition(y: pd.Series, period: int = None):
    """
    Seasonal decomposition using additive model.
    """
    y = pd.Series(y).dropna()
    if len(y) < 10:
        raise ValueError("Not enough data for decomposition (need ≥ 10 points).")

    if period is None or period < 2:
        period = max(2, min(len(y) // 5, 60))

    result = seasonal_decompose(y, model="additive", period=period)
    return result


def plot_fft(y: pd.Series):
    """
    Simple Fourier spectrum.
    """
    y = pd.Series(y).dropna()
    n = len(y)
    fig, ax = plt.subplots(figsize=(10, 4))

    if n < 4:
        ax.text(0.5, 0.5, "Not enough data for FFT (need ≥ 4 points).",
                ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    yf = fft(y.values)
    xf = np.fft.fftfreq(n)

    ax.plot(xf[: n // 2], np.abs(yf[: n // 2]))
    ax.set_title("Fourier Transform Spectrum")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    fig.tight_layout()
    return fig


def stationarity_tests(y: pd.Series):
    """
    Run ADF and KPSS tests and return p-values.
    """
    y = pd.Series(y).dropna()

    try:
        _, adf_p, *_ = adfuller(y)
    except Exception:
        adf_p = np.nan

    try:
        _, kpss_p, *_ = kpss(y, nlags="auto")
    except Exception:
        kpss_p = np.nan

    return {"ADF_p": adf_p, "KPSS_p": kpss_p}


def detect_seasonality(y: pd.Series):
    """
    Simple seasonality detection using autocorrelation.
    """
    from statsmodels.tsa.stattools import acf

    y = pd.Series(y).dropna()
    if len(y) < 20:
        return {"seasonal": False, "period_guess": 0}

    ac_values = acf(y, nlags=min(60, len(y) // 2))
    lags = np.arange(len(ac_values))

    # Ignore lag 0
    lags = lags[1:]
    ac_values = ac_values[1:]

    best_lag = int(lags[np.argmax(ac_values)])
    is_seasonal = ac_values.max() > 0.3

    return {"seasonal": bool(is_seasonal), "period_guess": best_lag}
