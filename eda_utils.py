import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
import streamlit as st


# -------------------------------
# Basic plots
# -------------------------------

def plot_time_series(y: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y.index, y.values, color="blue")
    ax.set_title("Time Series (Train)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_histogram(y: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(y.values, bins=30, color="gray", edgecolor="black")
    ax.set_title("Distribution (Train)")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def plot_acf_pacf(y: pd.Series, lags: int = 40):
    ac_vals = acf(y, nlags=min(lags, len(y) - 1))
    pac_vals = pacf(y, nlags=min(lags, len(y) - 1), method="yw")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].stem(range(len(ac_vals)), ac_vals, use_line_collection=True)
    axes[0].set_title("Autocorrelation (ACF)")
    axes[0].set_xlabel("Lag")

    axes[1].stem(range(len(pac_vals)), pac_vals, use_line_collection=True)
    axes[1].set_title("Partial Autocorrelation (PACF)")
    axes[1].set_xlabel("Lag")

    fig.tight_layout()
    return fig


# -------------------------------
# ETS decomposition
# -------------------------------

def ets_decomposition(y: pd.Series, period: int = None):
    if period is None:
        # fallback: 10% of length, at least 2
        period = max(2, int(len(y) * 0.1))

    result = seasonal_decompose(y, model="additive", period=period)
    return result


def plot_ets_decomposition(result):
    fig = result.plot()
    fig.set_size_inches(10, 8)
    fig.tight_layout()
    return fig


# -------------------------------
# Fourier / frequency analysis
# -------------------------------

def plot_fft(y: pd.Series):
    n = len(y)
    y_centered = y - y.mean()
    fft_vals = np.fft.fft(y_centered)
    freqs = np.fft.fftfreq(n)

    # Only positive frequencies
    mask = freqs > 0
    freqs = freqs[mask]
    power = np.abs(fft_vals[mask])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, power)
    ax.set_title("Fourier Spectrum (Train)")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


# -------------------------------
# Stationarity & seasonality
# -------------------------------

def stationarity_tests(y: pd.Series):
    """
    Returns p-values for ADF and KPSS.
    ADF: H0 = non-stationary (unit root present)
    KPSS: H0 = stationary
    """
    adf_stat, adf_p, *_ = adfuller(y)
    try:
        kpss_stat, kpss_p, *_ = kpss(y, nlags="auto")
    except Exception:
        kpss_p = np.nan

    return {"ADF_p": adf_p, "KPSS_p": kpss_p}


def interpret_stationarity(adf_p: float, kpss_p: float, alpha: float = 0.05):
    """
    Simple text interpretation combining ADF and KPSS.
    """
    adf_stationary = adf_p < alpha
    kpss_stationary = (np.isnan(kpss_p) or kpss_p > alpha)

    if adf_stationary and kpss_stationary:
        msg = "Series is **likely stationary** (ADF rejects unit root, KPSS does not reject stationarity)."
    elif (not adf_stationary) and (not kpss_stationary):
        msg = "Both ADF and KPSS suggest **non-stationarity**. Differencing is likely needed."
    elif not adf_stationary and kpss_stationary:
        msg = "Mixed evidence, but leaning **non-stationary** (ADF does not reject unit root)."
    else:
        msg = "Mixed evidence, but leaning **stationary** (ADF rejects unit root)."

    return msg


def detect_seasonality(y: pd.Series, max_lag: int = 60, threshold: float = 0.3):
    """
    Very simple seasonality detection via ACF peaks.
    Returns whether seasonal and a guessed period.
    """
    nlags = min(max_lag, len(y) // 2)
    ac_vals = acf(y, nlags=nlags)
    # ignore lag 0
    ac_vals[0] = 0.0

    peak_lag = int(np.argmax(ac_vals))
    peak_val = ac_vals[peak_lag]

    is_seasonal = peak_val > threshold and peak_lag > 1

    return {
        "seasonal": bool(is_seasonal),
        "period_guess": int(peak_lag),
        "peak_autocorr": float(peak_val),
    }
