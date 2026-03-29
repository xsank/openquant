"""绩效指标单元测试"""
import pytest
import numpy as np
import pandas as pd

from openquant.utils.metrics import calculate_metrics, _calculate_max_drawdown

class TestCalculateMetrics:
    def test_empty_series(self):
        result = calculate_metrics(pd.Series(dtype=float))
        assert result == {}

    def test_single_value(self):
        result = calculate_metrics(pd.Series([100.0]))
        assert result == {}

    def test_constant_equity(self):
        equity = pd.Series(
            [100.0] * 10,
            index=pd.date_range("2025-01-01", periods=10),
        )
        result = calculate_metrics(equity)
        assert result["total_return"] == 0.0
        assert result["annual_return"] == 0.0
        assert result["max_drawdown"] == 0.0

    def test_positive_return(self):
        equity = pd.Series(
            [100.0, 102.0, 105.0, 108.0, 110.0],
            index=pd.date_range("2025-01-01", periods=5),
        )
        result = calculate_metrics(equity)
        assert result["total_return"] == 10.0
        assert result["total_trading_days"] == 5

    def test_negative_return(self):
        equity = pd.Series(
            [100.0, 98.0, 95.0, 90.0, 85.0],
            index=pd.date_range("2025-01-01", periods=5),
        )
        result = calculate_metrics(equity)
        assert result["total_return"] == -15.0

    def test_sharpe_ratio_positive(self):
        # Steadily increasing equity should have positive Sharpe
        equity = pd.Series(
            [100 + i * 0.5 for i in range(252)],
            index=pd.date_range("2025-01-01", periods=252),
        )
        result = calculate_metrics(equity)
        assert result["sharpe_ratio"] > 0

    def test_win_rate(self):
        # 3 up days, 1 down day
        equity = pd.Series(
            [100.0, 101.0, 102.0, 103.0, 101.0],
            index=pd.date_range("2025-01-01", periods=5),
        )
        result = calculate_metrics(equity)
        assert result["win_rate"] == 75.0

class TestMaxDrawdown:
    def test_no_drawdown(self):
        equity = pd.Series([100, 101, 102, 103, 104])
        dd, duration = _calculate_max_drawdown(equity)
        assert dd == 0.0
        assert duration == 0

    def test_simple_drawdown(self):
        equity = pd.Series([100, 110, 105, 108, 115])
        dd, duration = _calculate_max_drawdown(equity)
        # Max drawdown: (105 - 110) / 110 = -0.04545...
        assert abs(dd - (-5 / 110)) < 1e-6

    def test_drawdown_duration(self):
        equity = pd.Series([100, 110, 105, 100, 95, 110, 120])
        dd, duration = _calculate_max_drawdown(equity)
        # Drawdown from 110 to 95: 3 bars in drawdown (105, 100, 95)
        assert duration == 3
