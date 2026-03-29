"""技术指标计算工具"""
from __future__ import annotations

import numpy as np
import pandas as pd


def moving_average(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均线 (SMA)"""
    return series.rolling(window=window, min_periods=1).mean()


def exponential_moving_average(series: pd.Series, span: int) -> pd.Series:
    """指数移动平均线 (EMA)"""
    return series.ewm(span=span, adjust=False).mean()


def macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD 指标

    Returns:
        (dif, dea, macd_hist)
    """
    ema_fast = exponential_moving_average(series, fast_period)
    ema_slow = exponential_moving_average(series, slow_period)
    dif = ema_fast - ema_slow
    dea = exponential_moving_average(dif, signal_period)
    macd_hist = 2 * (dif - dea)
    return dif, dea, macd_hist


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """相对强弱指标 (RSI)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    return 100 - (100 / (1 + rs))


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """布林带

    Returns:
        (upper_band, middle_band, lower_band)
    """
    middle = moving_average(series, window)
    std = series.rolling(window=window, min_periods=1).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """平均真实波幅 (ATR)"""
    prev_close = close.shift(1)
    true_range = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=1).mean()
