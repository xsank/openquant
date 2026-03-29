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
    # When avg_loss is 0 (pure uptrend), RSI should be 100
    # When avg_gain is 0 (pure downtrend), RSI should be 0
    rsi_values = pd.Series(np.where(
        avg_loss == 0,
        np.where(avg_gain == 0, 50.0, 100.0),
        100 - (100 / (1 + avg_gain / avg_loss)),
    ), index=series.index)
    return rsi_values


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


def kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    fastk_period: int = 9,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """KDJ 随机指标

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        fastk_period: RSV 周期
        slowk_period: K 值平滑周期
        slowd_period: D 值平滑周期

    Returns:
        (K, D, J)
    """
    lowest_low = low.rolling(window=fastk_period, min_periods=1).min()
    highest_high = high.rolling(window=fastk_period, min_periods=1).max()
    denominator = highest_high - lowest_low
    rsv = pd.Series(
        np.where(denominator == 0, 50.0, (close - lowest_low) / denominator * 100),
        index=close.index,
    )
    k_value = rsv.ewm(alpha=1.0 / slowk_period, adjust=False).mean()
    d_value = k_value.ewm(alpha=1.0 / slowd_period, adjust=False).mean()
    j_value = 3 * k_value - 2 * d_value
    return k_value, d_value, j_value


def donchian_channel(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """唐奇安通道 (Donchian Channel)

    Args:
        high: 最高价序列
        low: 最低价序列
        period: 回看周期

    Returns:
        (upper, middle, lower)
    """
    upper = high.rolling(window=period, min_periods=1).max()
    lower = low.rolling(window=period, min_periods=1).min()
    middle = (upper + lower) / 2
    return upper, middle, lower


def rate_of_change(series: pd.Series, period: int = 12) -> pd.Series:
    """变动率指标 (ROC)

    衡量价格在 period 周期内的变化百分比。
    """
    shifted = series.shift(period)
    return pd.Series(
        np.where(shifted == 0, 0.0, (series - shifted) / shifted * 100),
        index=series.index,
    )
