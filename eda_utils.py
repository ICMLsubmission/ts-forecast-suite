import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from scipy.fft import fft
import streamlit as st

# -------------------------------
# Time Series EDA Utilities
# -------------------------------

def plot_time_series(y):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y.index, y.values, color="blue")
    ax.set_title("Time Series")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True)
    return fig


def plot_histogram(y):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(y.values, bins=30, color="gray")
    ax.set_title("Distribution")
    return fig


def plot_acf_pacf(y, lags=40):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].stem(acf(y, nlags=lags), use_line_collection=True)
    axes[0].set_title("ACF")

    axes[1].stem(pacf(y, nlags=lags), use_line_collection=True)
    axes[1].set_title("PACF")

    return fig


def ets_decomposition(y, period=None):
    if period is None:
        period = max(2, int(len(y) * 0.1))  # fallback
    result = seasonal_decompose(y, model="additive", period=period)
    return result


def plot_fft(y):
    n = len(y)
    freq = np.fft.fftfreq(n)
    magnitude = np.abs(fft(y))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freq[:n//2], magnitude[:n//2])
    ax.set_title("Fourier Transform Spectrum")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Amplitude")
    return fig


def stationarity_tests(y):
    adf_stat, adf_p, *_ = adfuller(y)
    try:
        kpss_stat, kpss_p, *_ = kpss(y, nlags="auto")
    except:
        kpss_stat, kpss_p = np.nan, np.nan

    return {
        "ADF_p": adf_p,
        "KPSS_p": kpss_p
    }


def detect_seasonality(y):
    ac_values = acf(y, nlags=min(60, len(y)//2))
    peaks = np.argsort(ac_values)[-3:]  # extract top peaks
    period_guess = peaks[-1]

    is_seasonal = ac_values[period_guess] > 0.3

    return {
        "seasonal": bool(is_seasonal),
        "period_guess": int(period_guess)
    }
