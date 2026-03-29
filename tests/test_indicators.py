"""技术指标单元测试"""
import pytest
import numpy as np
import pandas as pd

from openquant.utils.indicators import (
    moving_average,
    exponential_moving_average,
    macd,
    rsi,
    bollinger_bands,
    atr,
    kdj,
    donchian_channel,
    rate_of_change,
)

def _make_price_series(prices: list[float]) -> pd.Series:
    return pd.Series(prices, dtype=float)

class TestMovingAverage:
    def test_basic(self):
        series = _make_price_series([1, 2, 3, 4, 5])
        result = moving_average(series, 3)
        assert abs(result.iloc[-1] - 4.0) < 1e-9
        assert abs(result.iloc[-2] - 3.0) < 1e-9

    def test_window_equals_length(self):
        series = _make_price_series([10, 20, 30])
        result = moving_average(series, 3)
        assert abs(result.iloc[-1] - 20.0) < 1e-9

    def test_single_element(self):
        series = _make_price_series([42.0])
        result = moving_average(series, 5)
        assert abs(result.iloc[0] - 42.0) < 1e-9

class TestEMA:
    def test_basic(self):
        series = _make_price_series([1, 2, 3, 4, 5])
        result = exponential_moving_average(series, 3)
        assert len(result) == 5
        # EMA should be closer to recent values
        assert result.iloc[-1] > result.iloc[0]

    def test_constant_series(self):
        series = _make_price_series([10, 10, 10, 10])
        result = exponential_moving_average(series, 3)
        for val in result:
            assert abs(val - 10.0) < 1e-9

class TestMACD:
    def test_output_shape(self):
        series = _make_price_series(list(range(1, 31)))
        dif, dea, hist = macd(series)
        assert len(dif) == 30
        assert len(dea) == 30
        assert len(hist) == 30

    def test_hist_is_2x_diff(self):
        series = _make_price_series(list(range(1, 31)))
        dif, dea, hist = macd(series)
        for i in range(len(hist)):
            assert abs(hist.iloc[i] - 2 * (dif.iloc[i] - dea.iloc[i])) < 1e-9

class TestRSI:
    def test_range(self):
        series = _make_price_series([10 + i * 0.5 for i in range(30)])
        result = rsi(series, 14)
        for val in result.dropna():
            assert 0 <= val <= 100

    def test_uptrend_high_rsi(self):
        # Use a longer series to ensure RSI stabilizes
        series = _make_price_series(list(range(1, 51)))
        result = rsi(series, 14)
        # In a steady uptrend, RSI should be high (above 50)
        assert result.iloc[-1] > 50

class TestBollingerBands:
    def test_output_shape(self):
        series = _make_price_series(list(range(1, 25)))
        upper, middle, lower = bollinger_bands(series, 20)
        assert len(upper) == 24
        assert len(middle) == 24
        assert len(lower) == 24

    def test_band_ordering(self):
        series = _make_price_series([10 + np.sin(i) for i in range(30)])
        upper, middle, lower = bollinger_bands(series, 10)
        # Skip first element where std is NaN (only 1 data point)
        for i in range(1, len(upper)):
            if not (np.isnan(upper.iloc[i]) or np.isnan(lower.iloc[i])):
                assert upper.iloc[i] >= middle.iloc[i] >= lower.iloc[i]

class TestATR:
    def test_basic(self):
        high = _make_price_series([12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        low = _make_price_series([8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        close = _make_price_series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        result = atr(high, low, close, period=5)
        assert len(result) == 10
        for val in result:
            assert val > 0

class TestKDJ:
    def test_output_shape(self):
        high = _make_price_series([10 + i for i in range(20)])
        low = _make_price_series([8 + i for i in range(20)])
        close = _make_price_series([9 + i for i in range(20)])
        k, d, j = kdj(high, low, close)
        assert len(k) == 20
        assert len(d) == 20
        assert len(j) == 20

    def test_j_formula(self):
        high = _make_price_series([10 + i for i in range(20)])
        low = _make_price_series([8 + i for i in range(20)])
        close = _make_price_series([9 + i for i in range(20)])
        k, d, j = kdj(high, low, close)
        for i in range(len(j)):
            assert abs(j.iloc[i] - (3 * k.iloc[i] - 2 * d.iloc[i])) < 1e-9

class TestDonchianChannel:
    def test_basic(self):
        high = _make_price_series([10, 12, 14, 13, 15, 11, 16, 18, 17, 20])
        low = _make_price_series([8, 9, 10, 11, 12, 9, 13, 14, 15, 16])
        upper, middle, lower = donchian_channel(high, low, period=5)
        assert len(upper) == 10
        # Last 5 highs: 15, 11, 16, 18, 17, 20 -> max = 20
        assert upper.iloc[-1] == 20.0

class TestROC:
    def test_basic(self):
        series = _make_price_series([100, 110, 120, 130, 140])
        result = rate_of_change(series, period=2)
        # ROC at index 2: (120 - 100) / 100 * 100 = 20
        assert abs(result.iloc[2] - 20.0) < 1e-9

    def test_zero_shifted(self):
        series = _make_price_series([0, 10, 20])
        result = rate_of_change(series, period=1)
        # At index 1: shifted = 0, should return 0 (division by zero protection)
        assert result.iloc[1] == 0.0
