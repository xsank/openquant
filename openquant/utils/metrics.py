"""回测绩效指标计算

包含基础绩效指标、基准对比指标（Alpha/Beta/信息比率/跟踪误差）等。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


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
        "total_trades": 0,
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


def calculate_benchmark_metrics(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    risk_free_rate: float = 0.03,
) -> dict:
    """计算策略相对于基准的对比指标

    Args:
        strategy_equity: 策略每日权益值序列（index 为日期）
        benchmark_equity: 基准每日权益值序列（index 为日期，需与策略对齐）
        risk_free_rate: 年化无风险利率

    Returns:
        包含 Alpha、Beta、信息比率、跟踪误差等指标的字典
    """
    if strategy_equity.empty or benchmark_equity.empty or len(strategy_equity) < 2:
        return {}

    # 对齐日期索引
    common_index = strategy_equity.index.intersection(benchmark_equity.index)
    if len(common_index) < 2:
        return {}

    strategy_aligned = strategy_equity.loc[common_index]
    benchmark_aligned = benchmark_equity.loc[common_index]

    strategy_returns = strategy_aligned.pct_change().dropna()
    benchmark_returns = benchmark_aligned.pct_change().dropna()

    # 再次对齐（pct_change 后可能索引不同）
    common_returns_index = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_returns = strategy_returns.loc[common_returns_index]
    benchmark_returns = benchmark_returns.loc[common_returns_index]

    if len(strategy_returns) < 2:
        return {}

    annual_factor = 252

    # Beta: Cov(Rs, Rb) / Var(Rb)
    covariance = np.cov(strategy_returns.values, benchmark_returns.values)
    beta = covariance[0, 1] / covariance[1, 1] if covariance[1, 1] != 0 else 0.0

    # Alpha (Jensen's Alpha): 年化策略收益 - [Rf + Beta * (年化基准收益 - Rf)]
    total_days = len(strategy_aligned)
    strategy_total_return = (strategy_aligned.iloc[-1] / strategy_aligned.iloc[0]) - 1
    benchmark_total_return = (benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0]) - 1
    strategy_annual_return = (1 + strategy_total_return) ** (annual_factor / total_days) - 1
    benchmark_annual_return = (1 + benchmark_total_return) ** (annual_factor / total_days) - 1
    alpha = strategy_annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))

    # 跟踪误差 (Tracking Error)
    excess_returns = strategy_returns - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(annual_factor)

    # 信息比率 (Information Ratio)
    information_ratio = (
        (strategy_annual_return - benchmark_annual_return) / tracking_error
        if tracking_error > 0
        else 0.0
    )

    # 相关系数
    correlation = strategy_returns.corr(benchmark_returns)

    # R² (决定系数)
    r_squared = correlation ** 2 if not np.isnan(correlation) else 0.0

    # Treynor 比率
    treynor_ratio = (
        (strategy_annual_return - risk_free_rate) / beta
        if beta != 0
        else 0.0
    )

    # 超额收益
    excess_return = strategy_total_return - benchmark_total_return
    excess_annual_return = strategy_annual_return - benchmark_annual_return

    # 基准自身指标
    benchmark_volatility = benchmark_returns.std() * np.sqrt(annual_factor)
    benchmark_max_drawdown, _ = _calculate_max_drawdown(benchmark_aligned)

    return {
        "alpha": round(alpha * 100, 4),
        "beta": round(beta, 4),
        "tracking_error": round(tracking_error * 100, 4),
        "information_ratio": round(information_ratio, 4),
        "correlation": round(correlation, 4),
        "r_squared": round(r_squared, 4),
        "treynor_ratio": round(treynor_ratio, 4),
        "excess_return": round(excess_return * 100, 2),
        "excess_annual_return": round(excess_annual_return * 100, 2),
        "benchmark_total_return": round(benchmark_total_return * 100, 2),
        "benchmark_annual_return": round(benchmark_annual_return * 100, 2),
        "benchmark_volatility": round(benchmark_volatility * 100, 2),
        "benchmark_max_drawdown": round(benchmark_max_drawdown * 100, 2),
    }


def calculate_rolling_alpha_beta(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    window: int = 60,
    risk_free_rate: float = 0.03,
) -> pd.DataFrame:
    """计算滚动 Alpha 和 Beta

    Args:
        strategy_equity: 策略权益曲线
        benchmark_equity: 基准权益曲线
        window: 滚动窗口大小（交易日）
        risk_free_rate: 年化无风险利率

    Returns:
        DataFrame with columns: alpha, beta（index 为日期）
    """
    common_index = strategy_equity.index.intersection(benchmark_equity.index)
    if len(common_index) < window + 1:
        return pd.DataFrame(columns=["alpha", "beta"])

    strategy_returns = strategy_equity.loc[common_index].pct_change().dropna()
    benchmark_returns = benchmark_equity.loc[common_index].pct_change().dropna()

    common_returns_index = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_returns = strategy_returns.loc[common_returns_index]
    benchmark_returns = benchmark_returns.loc[common_returns_index]

    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    alphas = []
    betas = []
    dates = []

    for i in range(window, len(strategy_returns)):
        window_strategy = strategy_returns.iloc[i - window:i]
        window_benchmark = benchmark_returns.iloc[i - window:i]

        slope, intercept, _, _, _ = stats.linregress(
            window_benchmark.values, window_strategy.values,
        )
        betas.append(slope)
        alphas.append((intercept - daily_rf * (1 - slope)) * 252)
        dates.append(strategy_returns.index[i])

    return pd.DataFrame({"alpha": alphas, "beta": betas}, index=dates)
