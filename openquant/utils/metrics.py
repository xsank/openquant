"""回测绩效指标计算"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.03) -> dict:
    """根据权益曲线计算回测绩效指标

    Args:
        equity_curve: 每日权益值序列（index 为日期）
        risk_free_rate: 年化无风险利率

    Returns:
        包含各项绩效指标的字典
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return {}

    daily_returns = equity_curve.pct_change().dropna()
    total_days = len(equity_curve)
    annual_factor = 252

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    annual_return = (1 + total_return) ** (annual_factor / total_days) - 1
    annual_volatility = daily_returns.std() * np.sqrt(annual_factor)

    sharpe_ratio = 0.0
    if annual_volatility > 0:
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

    max_drawdown, max_drawdown_duration = _calculate_max_drawdown(equity_curve)

    win_days = (daily_returns > 0).sum()
    total_trading_days = len(daily_returns)
    win_rate = win_days / total_trading_days if total_trading_days > 0 else 0

    avg_win = daily_returns[daily_returns > 0].mean() if win_days > 0 else 0
    loss_days = (daily_returns < 0).sum()
    avg_loss = abs(daily_returns[daily_returns < 0].mean()) if loss_days > 0 else 0
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(annual_factor) if len(downside_returns) > 0 else 0
    sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0

    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        "total_return": round(total_return * 100, 2),
        "annual_return": round(annual_return * 100, 2),
        "annual_volatility": round(annual_volatility * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "sortino_ratio": round(sortino_ratio, 4),
        "calmar_ratio": round(calmar_ratio, 4),
        "max_drawdown": round(max_drawdown * 100, 2),
        "max_drawdown_duration": max_drawdown_duration,
        "win_rate": round(win_rate * 100, 2),
        "profit_loss_ratio": round(profit_loss_ratio, 4),
        "total_trading_days": total_days,
        "total_trades": total_trading_days,
    }


def _calculate_max_drawdown(equity_curve: pd.Series) -> tuple[float, int]:
    """计算最大回撤和最大回撤持续天数"""
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()

    in_drawdown = drawdown < 0
    if not in_drawdown.any():
        return 0.0, 0

    duration = 0
    max_duration = 0
    for is_dd in in_drawdown:
        if is_dd:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0

    return max_drawdown, max_duration
