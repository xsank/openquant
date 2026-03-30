"""通用股票分析脚本

自动完成：全策略回测 → 最优策略筛选 → 参数优化 → 技术分析 → 交易建议

用法:
    python scripts/analyze_stock.py --symbol 09988 --market hk_stock
    python scripts/analyze_stock.py --symbol 300750 --market a_share
    python scripts/analyze_stock.py --symbol 105.GOOG --market us_stock
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta

from openquant.core.models import MarketType
from openquant.datasource.factory import DataSourceFactory
from openquant.engine.backtest_engine import BacktestEngine
from openquant.optimizer.param_optimizer import ParameterOptimizer
from openquant.storage.sqlite_storage import SqliteStorage
from openquant.strategy.bollinger_strategy import BollingerBandStrategy
from openquant.strategy.dual_momentum_strategy import DualMomentumStrategy
from openquant.strategy.kdj_strategy import KDJStrategy
from openquant.strategy.ma_cross import MACrossStrategy
from openquant.strategy.macd_strategy import MACDStrategy
from openquant.strategy.rsi_strategy import RSIReversalStrategy
from openquant.strategy.turtle_strategy import TurtleStrategy
from openquant.strategy.volume_breakout_strategy import VolumeBreakoutStrategy
from openquant.utils.indicators import bollinger_bands
from openquant.utils.md_to_pdf import convert_md_to_pdf

STRATEGY_REGISTRY = {
    "ma_cross": MACrossStrategy,
    "macd": MACDStrategy,
    "rsi_reversal": RSIReversalStrategy,
    "bollinger_band": BollingerBandStrategy,
    "turtle": TurtleStrategy,
    "kdj": KDJStrategy,
    "dual_momentum": DualMomentumStrategy,
    "volume_breakout": VolumeBreakoutStrategy,
}

MARKET_MAP = {
    "a_share": MarketType.A_SHARE,
    "hk_stock": MarketType.HK_STOCK,
    "us_stock": MarketType.US_STOCK,
}

CURRENCY_MAP = {
    "a_share": "¥",
    "hk_stock": "HK$",
    "us_stock": "$",
}

# 各策略的参数搜索空间
PARAM_SEARCH_SPACE = {
    "ma_cross": [
        ("short_window", "int", 3, 20, 2),
        ("long_window", "int", 10, 60, 5),
    ],
    "macd": [
        ("fast_period", "int", 8, 16, 2),
        ("slow_period", "int", 20, 32, 3),
        ("signal_period", "int", 5, 12, 1),
    ],
    "rsi_reversal": [
        ("rsi_period", "int", 6, 20, 2),
        ("oversold", "float", 20, 35, 5),
        ("overbought", "float", 65, 80, 5),
    ],
    "bollinger_band": [
        ("window", "int", 10, 30, 2),
        ("num_std", "float", 1.5, 3.0, 0.25),
    ],
    "turtle": [
        ("entry_period", "int", 10, 30, 5),
        ("exit_period", "int", 5, 20, 5),
    ],
    "kdj": [
        ("fastk_period", "int", 5, 15, 2),
        ("slowk_period", "int", 2, 5, 1),
        ("slowd_period", "int", 2, 5, 1),
    ],
    "dual_momentum": [
        ("roc_period", "int", 10, 30, 5),
        ("short_ma_period", "int", 5, 15, 5),
        ("long_ma_period", "int", 20, 40, 5),
    ],
    "volume_breakout": [
        ("ma_period", "int", 10, 30, 5),
        ("volume_multiplier", "float", 1.0, 2.5, 0.25),
    ],
}


def fetch_data(symbol: str, market: MarketType, days: int = 400) -> pd.DataFrame:
    """获取历史数据"""
    DataSourceFactory.register_defaults()
    datasource_name = "akshare" if market != MarketType.A_SHARE else "baostock"
    # 非A股强制使用 akshare
    if market != MarketType.A_SHARE:
        datasource_name = "akshare"
    data_source = DataSourceFactory.get(datasource_name)

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    return data_source.fetch_daily_bars(symbol, start_date, end_date, market)


def optimize_all_strategies(
    symbol: str,
    df: pd.DataFrame,
    market: MarketType,
    capital: float = 100000,
) -> dict[str, dict]:
    """对所有策略并行进行参数优化，返回每个策略的最优参数和指标。

    每个策略都在其完整参数空间中搜索最优参数，避免因默认参数不佳
    而导致潜在优秀策略被过早淘汰。
    """
    results = {}
    data_feeds = [(symbol, df, market)]

    for strategy_name, strategy_class in STRATEGY_REGISTRY.items():
        param_space = PARAM_SEARCH_SPACE.get(strategy_name, [])

        if not param_space:
            # 无参数空间，使用默认参数回测一次
            storage = SqliteStorage("data/analyze_temp.db")
            storage.initialize()
            storage.save_bars(df, symbol, market)
            strategy = strategy_class()
            engine = BacktestEngine(
                initial_capital=capital,
                commission_rate=0.0003,
                slippage_rate=0.001,
                storage=storage,
            )
            engine.set_strategy(strategy)
            engine.add_data(symbol, df, market)
            engine.run()
            metrics = engine.get_results()
            storage.close()
            if os.path.exists("data/analyze_temp.db"):
                os.remove("data/analyze_temp.db")
            results[strategy_name] = {
                "name": strategy.get_name(),
                "best_params": {},
                "metrics": metrics,
            }
            continue

        print(f"  ⏳ 优化 {strategy_name} ({len(param_space)} 个参数)...")
        optimizer = ParameterOptimizer(
            strategy_class=strategy_class,
            data_feeds=data_feeds,
            initial_capital=capital,
            commission_rate=0.0003,
            slippage_rate=0.001,
        )

        for param_name, param_type, low, high, step in param_space:
            optimizer.add_parameter(param_name, param_type, low=low, high=high, step=step)

        optimizer.grid_search("sharpe_ratio", maximize=True)

        best_params = optimizer.get_best_params()
        opt_results = optimizer.get_results()
        best_metrics = opt_results[0].metrics if opt_results else {}
        total_combos = len(opt_results)

        # 用最优参数构造策略名称
        optimized_strategy = strategy_class(**best_params)
        optimized_name = optimized_strategy.get_name()

        results[strategy_name] = {
            "name": optimized_name,
            "best_params": best_params,
            "metrics": best_metrics,
            "total_combos": total_combos,
        }
        sharpe = best_metrics.get("sharpe_ratio", 0)
        total_return = best_metrics.get("total_return", 0)
        print(f"    ✅ {optimized_name}: 收益 {total_return:+.2f}%, 夏普 {sharpe:.4f} (搜索 {total_combos} 组)")

    return results


def find_best_strategy(results: dict[str, dict]) -> tuple[str, dict]:
    """根据夏普比率找出最优策略"""
    best_name = None
    best_sharpe = float("-inf")
    for strategy_name, result in results.items():
        sharpe = result["metrics"].get("sharpe_ratio", float("-inf"))
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_name = strategy_name
    return best_name, results[best_name]


def calculate_percentile_rank(df: pd.DataFrame) -> dict[str, float]:
    """计算当前价格在历史中的百分位"""
    current_price = df["close"].iloc[-1]
    periods = {
        "1年百分位": min(250, len(df)),
        "6个月百分位": min(125, len(df)),
        "3个月百分位": min(63, len(df)),
        "1个月百分位": min(22, len(df)),
    }
    result = {}
    for label, window in periods.items():
        data = df["close"].tail(window)
        result[label] = (data < current_price).sum() / len(data) * 100
    return result


def calculate_bollinger_signals(
    df: pd.DataFrame, window: int = 20, num_std: float = 2.0
) -> dict:
    """计算布林带信号"""
    close_series = df["close"]
    upper, middle, lower = bollinger_bands(close_series, window, num_std)

    current_close = close_series.iloc[-1]
    prev_close = close_series.iloc[-2]
    current_upper = upper.iloc[-1]
    current_lower = lower.iloc[-1]
    current_middle = middle.iloc[-1]
    prev_upper = upper.iloc[-2]
    prev_lower = lower.iloc[-2]

    band_width = current_upper - current_lower
    band_position = (
        (current_close - current_lower) / band_width if band_width > 0 else 0.5
    )

    if prev_close <= prev_lower and current_close > current_lower:
        signal = "🟢 买入信号（价格从下轨回升）"
    elif prev_close >= prev_upper and current_close < current_upper:
        signal = "🔴 卖出信号（价格从上轨回落）"
    elif current_close < current_lower:
        signal = "⚠️ 价格在下轨下方，等待回升确认后买入"
    elif current_close > current_upper:
        signal = "⚠️ 价格在上轨上方，等待回落确认后卖出"
    elif current_close < current_middle:
        signal = "🟡 价格在中轨下方，偏空观望"
    else:
        signal = "🟡 价格在中轨上方，偏多观望"

    return {
        "上轨": round(current_upper, 2),
        "中轨": round(current_middle, 2),
        "下轨": round(current_lower, 2),
        "带宽": round(band_width, 2),
        "带内位置": round(band_position, 4),
        "信号": signal,
    }


def calculate_rsi(close_series: pd.Series, period: int = 14) -> pd.Series:
    """计算RSI"""
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(
    close_series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """计算MACD"""
    ema_fast = close_series.ewm(span=fast, adjust=False).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_support_resistance(
    df: pd.DataFrame, lookback: int = 60
) -> tuple[list, list]:
    """计算支撑位和阻力位"""
    recent = df.tail(lookback)
    current_price = df["close"].iloc[-1]

    period_high = recent["high"].max()
    period_low = recent["low"].min()
    q25 = np.percentile(recent["close"], 25)
    q50 = np.percentile(recent["close"], 50)
    q75 = np.percentile(recent["close"], 75)

    ma5 = df["close"].rolling(5).mean().iloc[-1]
    ma10 = df["close"].rolling(10).mean().iloc[-1]
    ma20 = df["close"].rolling(20).mean().iloc[-1]
    ma60 = df["close"].rolling(60).mean().iloc[-1] if len(df) >= 60 else None

    all_levels = {
        "MA5": ma5,
        "MA10": ma10,
        "MA20": ma20,
        "25%分位": q25,
        "50%分位(中位数)": q50,
        "75%分位": q75,
        "近期最高": period_high,
        "近期最低": period_low,
    }
    if ma60 is not None:
        all_levels["MA60"] = ma60

    support_levels = []
    resistance_levels = []
    for name, level in sorted(all_levels.items(), key=lambda x: x[1]):
        if level < current_price:
            support_levels.append((name, round(level, 2)))
        else:
            resistance_levels.append((name, round(level, 2)))

    return support_levels, resistance_levels


def calculate_volatility(df: pd.DataFrame, window: int = 20) -> dict:
    """计算波动率"""
    returns = df["close"].pct_change().dropna()
    daily_vol = returns.tail(window).std()
    annual_vol = daily_vol * np.sqrt(252)

    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift(1))
    low_close = abs(df["low"] - df["close"].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_value = true_range.rolling(window=14).mean().iloc[-1]

    return {
        "日波动率": round(daily_vol * 100, 2),
        "年化波动率": round(annual_vol * 100, 2),
        "ATR(14)": round(atr_value, 2),
    }


def estimate_success_probability(
    df: pd.DataFrame,
    bollinger_info: dict,
    rsi_value: float,
    macd_hist: float,
) -> int:
    """基于多指标综合估算买入成功概率"""
    score = 50

    band_pos = bollinger_info["带内位置"]
    if band_pos < 0:
        score += 15
    elif band_pos < 0.3:
        score += 10
    elif band_pos < 0.5:
        score += 5
    elif band_pos > 0.8:
        score -= 10
    elif band_pos > 1.0:
        score -= 15

    if rsi_value < 30:
        score += 15
    elif rsi_value < 40:
        score += 8
    elif rsi_value < 50:
        score += 3
    elif rsi_value > 70:
        score -= 15
    elif rsi_value > 60:
        score -= 5

    if macd_hist > 0:
        score += 5
    else:
        score -= 5

    if len(df) > 6:
        recent_return_5d = (df["close"].iloc[-1] / df["close"].iloc[-6] - 1) * 100
        if recent_return_5d < -5:
            score += 8
        elif recent_return_5d > 5:
            score -= 5

    return max(10, min(90, score))


def print_header(symbol: str, market_key: str):
    """打印报告头"""
    currency = CURRENCY_MAP.get(market_key, "")
    print("=" * 70)
    print(f"  {symbol} 综合量化分析报告")
    print(f"  市场: {market_key}  |  货币: {currency}")
    print(f"  分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


def print_optimization_ranking(results: dict[str, dict]):
    """打印所有策略参数优化后的排名对比"""
    print(f"\n{'='*70}")
    print("  📊 全策略参数优化结果排名 (过去1年)")
    print(f"{'='*70}")

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["metrics"].get("sharpe_ratio", float("-inf")),
        reverse=True,
    )

    print(
        f"  {'排名':>4s}  {'策略(最优参数)':<28s}  {'收益率':>8s}  {'夏普比率':>8s}  {'最大回撤':>8s}  {'交易次数':>6s}"
    )
    print("  " + "-" * 70)

    for rank, (strategy_name, result) in enumerate(sorted_results, 1):
        metrics = result["metrics"]
        name = result["name"]
        total_return = metrics.get("total_return", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd = metrics.get("max_drawdown", 0)
        trades = metrics.get("total_trades", 0)
        medal = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(
            f"  {medal}{rank:>2d}  {name:<28s}  {total_return:>+7.2f}%  {sharpe:>8.4f}  {max_dd:>+7.2f}%  {trades:>6d}"
        )

    # 打印冠军策略详情
    best_name, best_result = sorted_results[0]
    best_params = best_result.get("best_params", {})
    best_metrics = best_result["metrics"]
    params_str = ", ".join(f"{k}={v}" for k, v in best_params.items()) if best_params else "默认参数"

    print(f"\n  🏆 冠军策略: {best_result['name']}")
    print(f"  最优参数: {params_str}")
    print(
        f"  回测表现: 收益 {best_metrics.get('total_return', 0):+.2f}%, "
        f"夏普 {best_metrics.get('sharpe_ratio', 0):.4f}, "
        f"回撤 {best_metrics.get('max_drawdown', 0):.2f}%, "
        f"卡玛 {best_metrics.get('calmar_ratio', 0):.4f}"
    )
    if best_result.get("total_combos"):
        print(f"  参数搜索空间: {best_result['total_combos']} 组")


def print_technical_analysis(
    df: pd.DataFrame,
    symbol: str,
    currency: str,
    bollinger_window: int,
    bollinger_std: float,
):
    """打印技术分析"""
    current_price = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]
    price_change = current_price - prev_close
    price_change_pct = (price_change / prev_close) * 100

    print(f"\n{'='*70}")
    print(f"  📈 当前行情")
    print(f"{'='*70}")
    print(f"  最新收盘价: {currency}{current_price:.2f}")
    print(f"  涨跌: {price_change:+.2f} ({price_change_pct:+.2f}%)")
    print(
        f"  开盘: {df['open'].iloc[-1]:.2f}  最高: {df['high'].iloc[-1]:.2f}  最低: {df['low'].iloc[-1]:.2f}"
    )
    print(f"  成交量: {df['volume'].iloc[-1]:,.0f}")

    # 历史百分位
    print(f"\n{'='*70}")
    print("  📊 历史价格位点分析")
    print(f"{'='*70}")
    percentiles = calculate_percentile_rank(df)
    for period, pct in percentiles.items():
        bar_len = int(pct / 2)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"  {period:>10s}: {pct:5.1f}%  |{bar}|")

    # 布林带分析
    print(f"\n{'='*70}")
    print(
        f"  📉 布林带分析 (最优参数: window={bollinger_window}, num_std={bollinger_std})"
    )
    print(f"{'='*70}")
    bollinger_info = calculate_bollinger_signals(df, bollinger_window, bollinger_std)
    print(f"  上轨: {bollinger_info['上轨']}")
    print(f"  中轨: {bollinger_info['中轨']}")
    print(f"  下轨: {bollinger_info['下轨']}")
    print(f"  带宽: {bollinger_info['带宽']}")
    print(f"  带内位置: {bollinger_info['带内位置']:.2%}")
    print(f"  信号: {bollinger_info['信号']}")

    # RSI
    print(f"\n{'='*70}")
    print("  📊 RSI 分析")
    print(f"{'='*70}")
    rsi_series = calculate_rsi(df["close"], 14)
    rsi_value = rsi_series.iloc[-1]
    rsi_status = "超卖区" if rsi_value < 30 else "超买区" if rsi_value > 70 else "中性区"
    print(f"  RSI(14): {rsi_value:.2f} ({rsi_status})")
    print(f"  RSI(6):  {calculate_rsi(df['close'], 6).iloc[-1]:.2f}")

    # MACD
    print(f"\n{'='*70}")
    print("  📊 MACD 分析")
    print(f"{'='*70}")
    macd_line, signal_line, histogram = calculate_macd(df["close"])
    hist_val = histogram.iloc[-1]
    prev_hist = histogram.iloc[-2]
    if hist_val > 0 and hist_val > prev_hist:
        macd_trend = "多头增强"
    elif hist_val > 0:
        macd_trend = "多头减弱"
    elif hist_val < 0 and hist_val > prev_hist:
        macd_trend = "空头减弱"
    else:
        macd_trend = "空头增强"
    print(f"  MACD线:  {macd_line.iloc[-1]:.4f}")
    print(f"  信号线:  {signal_line.iloc[-1]:.4f}")
    print(f"  柱状图:  {hist_val:.4f} ({macd_trend})")

    # 均线系统
    print(f"\n{'='*70}")
    print("  📊 均线系统")
    print(f"{'='*70}")
    for period in [5, 10, 20, 60]:
        if len(df) >= period:
            ma = df["close"].rolling(period).mean().iloc[-1]
            diff_pct = (current_price / ma - 1) * 100
            status = "↑ 多头" if current_price > ma else "↓ 空头"
            print(f"  MA{period:>2d}: {ma:>8.2f}  偏离: {diff_pct:>+6.2f}%  {status}")

    # 支撑阻力位
    print(f"\n{'='*70}")
    print("  📊 支撑位与阻力位")
    print(f"{'='*70}")
    support_levels, resistance_levels = calculate_support_resistance(df)

    print("  🔴 阻力位 (由近到远):")
    for name, level in resistance_levels:
        diff = (level / current_price - 1) * 100
        print(f"    {name:>16s}: {level:>8.2f}  (距当前 {diff:>+6.2f}%)")

    print(f"\n  当前价格: {currency}{current_price:.2f}")
    print()

    print("  🟢 支撑位 (由近到远):")
    for name, level in reversed(support_levels):
        diff = (level / current_price - 1) * 100
        print(f"    {name:>16s}: {level:>8.2f}  (距当前 {diff:>+6.2f}%)")

    # 波动率
    print(f"\n{'='*70}")
    print("  📊 波动率分析")
    print(f"{'='*70}")
    vol_info = calculate_volatility(df)
    for key, val in vol_info.items():
        print(f"  {key}: {val}")

    return bollinger_info, rsi_value, hist_val, vol_info


def print_trading_advice(
    df: pd.DataFrame,
    symbol: str,
    currency: str,
    best_strategy_name: str,
    best_params: dict,
    best_metrics: dict,
    bollinger_info: dict,
    rsi_value: float,
    macd_hist: float,
    vol_info: dict,
):
    """打印交易建议"""
    current_price = df["close"].iloc[-1]
    atr_value = vol_info["ATR(14)"]

    # 成功概率
    print(f"\n{'='*70}")
    print("  🎯 买入成功概率估算")
    print(f"{'='*70}")
    success_prob = estimate_success_probability(df, bollinger_info, rsi_value, macd_hist)

    if success_prob >= 70:
        prob_label = "✅ 高概率买入机会"
    elif success_prob >= 55:
        prob_label = "🟡 中等概率，可小仓位试探"
    elif success_prob >= 40:
        prob_label = "⚠️ 概率偏低，建议观望"
    else:
        prob_label = "🔴 不建议买入"

    print(f"  综合评分: {success_prob}/100")
    print(f"  判断: {prob_label}")

    # 交易建议
    print(f"\n{'='*70}")
    print("  💡 今日交易建议")
    print(f"{'='*70}")

    # 近期走势
    return_periods = [(5, "5日"), (20, "20日"), (60, "60日")]
    print(f"\n  近期走势:")
    for days, label in return_periods:
        if len(df) > days + 1:
            ret = (current_price / df["close"].iloc[-(days + 1)] - 1) * 100
            print(f"    {label}涨跌: {ret:>+6.2f}%")

    # 基于布林带的建议
    ideal_buy = bollinger_info["下轨"]
    aggressive_buy = bollinger_info["中轨"]
    stop_loss = round(ideal_buy - 1.5 * atr_value, 2)
    take_profit = bollinger_info["上轨"]

    params_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
    print(f"\n  最优策略: {best_strategy_name} ({params_str})")
    print(
        f"  回测表现: 收益 {best_metrics.get('total_return', 0):+.2f}%, "
        f"夏普 {best_metrics.get('sharpe_ratio', 0):.4f}, "
        f"回撤 {best_metrics.get('max_drawdown', 0):.2f}%"
    )

    print(f"\n  基于布林带的价格参考:")
    print(f"    📌 当前价格:     {currency}{current_price:.2f}")
    print(f"    🟢 理想买入价:   {currency}{ideal_buy:.2f} (布林带下轨)")
    print(f"    🟡 激进买入价:   {currency}{aggressive_buy:.2f} (布林带中轨)")
    print(f"    🔴 止损价:       {currency}{stop_loss:.2f} (下轨 - 1.5×ATR)")
    print(f"    🎯 目标止盈价:   {currency}{take_profit:.2f} (布林带上轨)")

    if take_profit > ideal_buy and ideal_buy > stop_loss:
        risk_reward = (take_profit - ideal_buy) / (ideal_buy - stop_loss)
        print(f"    📊 盈亏比:       {risk_reward:.2f}:1")

    print(f"\n  操作建议:")
    band_pos = bollinger_info["带内位置"]
    if band_pos < 0.1:
        print("    ★ 价格接近/低于布林带下轨，是理想买入区域")
        print(f"    ★ 建议在 {currency}{ideal_buy:.2f} 附近分批建仓")
        print(f"    ★ 严格设置止损在 {currency}{stop_loss:.2f}")
    elif band_pos < 0.4:
        print("    ★ 价格在布林带中下区域，可考虑轻仓试探")
        print(f"    ★ 建议在 {currency}{ideal_buy:.2f} 挂限价单等待")
        print(f"    ★ 或在当前价位小仓位(30%)介入，跌到下轨再加仓")
    elif band_pos < 0.6:
        print("    ★ 价格在布林带中轨附近，信号不明确")
        print("    ★ 建议观望，等待价格回调到下轨区域再考虑买入")
    elif band_pos < 0.9:
        print("    ★ 价格偏向布林带上轨，不建议追高")
        print("    ★ 如持有仓位，可考虑部分止盈")
    else:
        print("    ★ 价格接近/超过布林带上轨，短期有回调风险")
        print("    ★ 不建议买入，持仓者建议止盈")

    print(f"\n  ⚠️ 风险提示:")
    print(f"    - 以上分析基于历史数据回测，不构成投资建议")
    print(f"    - 过去表现不代表未来收益，市场存在不确定性")
    print(f"    - 建议控制单只股票仓位不超过总资产的 20%")
    print(f"    - 严格执行止损纪律，单笔亏损不超过总资产的 2%")
    print("=" * 70)


def generate_charts(
    df: pd.DataFrame,
    symbol: str,
    charts_dir: str,
    all_results: dict,
    bollinger_window: int = 20,
    bollinger_std: float = 2.0,
):
    """生成所有技术分析图表并保存到指定目录"""
    Path(charts_dir).mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(df["datetime"])
    close = df["close"].values

    # 1. 价格走势 + 布林带图
    upper, middle, lower = bollinger_bands(df["close"], bollinger_window, bollinger_std)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, close, linewidth=1.2, color="#1976D2", label="Close")
    ax.plot(dates, upper, linewidth=0.8, color="#E53935", linestyle="--", label="Upper Band")
    ax.plot(dates, middle, linewidth=0.8, color="#FF9800", linestyle="-.", label="Middle Band")
    ax.plot(dates, lower, linewidth=0.8, color="#43A047", linestyle="--", label="Lower Band")
    ax.fill_between(dates, upper, lower, alpha=0.08, color="#90CAF9")
    ax.set_title(f"{symbol} Price & Bollinger Bands", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(f"{charts_dir}/price_bollinger.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. RSI 指标图
    rsi14 = calculate_rsi(df["close"], 14)
    rsi6 = calculate_rsi(df["close"], 6)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates, rsi14, linewidth=1.2, color="#1976D2", label="RSI(14)")
    ax.plot(dates, rsi6, linewidth=0.8, color="#FF9800", alpha=0.7, label="RSI(6)")
    ax.axhline(y=70, color="#E53935", linestyle="--", linewidth=0.8, label="Overbought (70)")
    ax.axhline(y=30, color="#43A047", linestyle="--", linewidth=0.8, label="Oversold (30)")
    ax.axhline(y=50, color="gray", linestyle=":", linewidth=0.5)
    ax.fill_between(dates, rsi14, 70, where=(rsi14 >= 70), alpha=0.15, color="#E53935")
    ax.fill_between(dates, rsi14, 30, where=(rsi14 <= 30), alpha=0.15, color="#43A047")
    ax.set_title(f"{symbol} RSI Analysis", fontsize=14, fontweight="bold")
    ax.set_ylabel("RSI", fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(f"{charts_dir}/rsi_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. MACD 图
    macd_line, signal_line, histogram = calculate_macd(df["close"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(dates, close, linewidth=1.2, color="#1976D2", label="Close")
    ax1.set_title(f"{symbol} MACD Analysis", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Price", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    colors_hist = ["#43A047" if v >= 0 else "#E53935" for v in histogram]
    ax2.bar(dates, histogram, color=colors_hist, alpha=0.7, width=1.5)
    ax2.plot(dates, macd_line, linewidth=1.0, color="#1976D2", label="MACD")
    ax2.plot(dates, signal_line, linewidth=1.0, color="#FF9800", label="Signal")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_ylabel("MACD", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(f"{charts_dir}/macd_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. 均线系统图
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, close, linewidth=1.2, color="#1976D2", label="Close")
    ma_colors = {"MA5": "#FF9800", "MA10": "#9C27B0", "MA20": "#E53935", "MA60": "#43A047"}
    for period, color in zip([5, 10, 20, 60], ma_colors.values()):
        if len(df) >= period:
            ma = df["close"].rolling(period).mean()
            ax.plot(dates, ma, linewidth=0.8, color=color, alpha=0.8, label=f"MA{period}")
    ax.set_title(f"{symbol} Moving Average System", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(f"{charts_dir}/ma_system.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. 支撑阻力位图
    support_levels, resistance_levels = calculate_support_resistance(df)
    current_price = df["close"].iloc[-1]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, close, linewidth=1.2, color="#1976D2", label="Close")
    for name, level in resistance_levels:
        ax.axhline(y=level, color="#E53935", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.text(dates.iloc[-1], level, f" {name}: {level}", fontsize=8, color="#E53935", va="bottom")
    for name, level in support_levels:
        ax.axhline(y=level, color="#43A047", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.text(dates.iloc[-1], level, f" {name}: {level}", fontsize=8, color="#43A047", va="top")
    ax.axhline(y=current_price, color="#FF9800", linewidth=1.0, label=f"Current: {current_price:.2f}")
    ax.set_title(f"{symbol} Support & Resistance", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(f"{charts_dir}/support_resistance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 6. 策略排名柱状图
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["metrics"].get("sharpe_ratio", float("-inf")),
        reverse=True,
    )
    names = [r[1]["name"] for r in sorted_results]
    returns = [r[1]["metrics"].get("total_return", 0) for r in sorted_results]
    sharpes = [r[1]["metrics"].get("sharpe_ratio", 0) for r in sorted_results]
    x = np.arange(len(names))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(14, 6))
    colors_bar = ["#43A047" if r >= 0 else "#E53935" for r in returns]
    bars = ax1.bar(x - width / 2, returns, width, color=colors_bar, alpha=0.85, label="Return (%)")
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, sharpes, width, color="#1976D2", alpha=0.7, label="Sharpe Ratio")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Return (%)", fontsize=11)
    ax2.set_ylabel("Sharpe Ratio", fontsize=11)
    ax1.set_title(f"{symbol} Strategy Ranking", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{charts_dir}/strategy_ranking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n📊 图表已保存到: {charts_dir}/")


def generate_markdown_report(
    symbol: str,
    market_key: str,
    timestamp: str,
    md_dir: str,
    charts_dir: str,
    df: pd.DataFrame,
    all_results: dict,
    best_strategy_name: str,
    best_params: dict,
    best_metrics: dict,
    bollinger_info: dict,
    rsi_value: float,
    macd_hist: float,
    vol_info: dict,
    bollinger_window: int,
    bollinger_std: float,
):
    """生成 Markdown 格式的综合分析报告"""
    Path(md_dir).mkdir(parents=True, exist_ok=True)
    currency = CURRENCY_MAP.get(market_key, "")
    current_price = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]
    price_change = current_price - prev_close
    price_change_pct = (price_change / prev_close) * 100
    md_path = f"{md_dir}/analyze_{symbol}.md"

    # 策略排名表格
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["metrics"].get("sharpe_ratio", float("-inf")),
        reverse=True,
    )
    ranking_rows = ""
    for rank, (strategy_name, result) in enumerate(sorted_results, 1):
        metrics = result["metrics"]
        medal = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else str(rank)
        ranking_rows += (
            f"| {medal} | {result['name']} | {metrics.get('total_return', 0):+.2f}% "
            f"| {metrics.get('sharpe_ratio', 0):.4f} | {metrics.get('max_drawdown', 0):+.2f}% "
            f"| {metrics.get('total_trades', 0)} |\n"
        )

    # 百分位
    percentiles = calculate_percentile_rank(df)
    percentile_rows = ""
    for period, pct in percentiles.items():
        percentile_rows += f"| {period} | {pct:.1f}% |\n"

    # 均线
    ma_rows = ""
    for period in [5, 10, 20, 60]:
        if len(df) >= period:
            ma = df["close"].rolling(period).mean().iloc[-1]
            diff_pct = (current_price / ma - 1) * 100
            status = "多头" if current_price > ma else "空头"
            ma_rows += f"| MA{period} | {ma:.2f} | {diff_pct:+.2f}% | {status} |\n"

    # 支撑阻力位
    support_levels, resistance_levels = calculate_support_resistance(df)
    resistance_rows = ""
    for name, level in resistance_levels:
        diff = (level / current_price - 1) * 100
        resistance_rows += f"| {name} | {level:.2f} | {diff:+.2f}% |\n"
    support_rows = ""
    for name, level in reversed(support_levels):
        diff = (level / current_price - 1) * 100
        support_rows += f"| {name} | {level:.2f} | {diff:+.2f}% |\n"

    # MACD 趋势
    macd_line_val, signal_line_val, histogram = calculate_macd(df["close"])
    hist_val = histogram.iloc[-1]
    prev_hist = histogram.iloc[-2]
    if hist_val > 0 and hist_val > prev_hist:
        macd_trend = "多头增强"
    elif hist_val > 0:
        macd_trend = "多头减弱"
    elif hist_val < 0 and hist_val > prev_hist:
        macd_trend = "空头减弱"
    else:
        macd_trend = "空头增强"

    # RSI 状态
    rsi_status = "超卖区" if rsi_value < 30 else "超买区" if rsi_value > 70 else "中性区"
    rsi6_value = calculate_rsi(df["close"], 6).iloc[-1]

    # 成功概率
    success_prob = estimate_success_probability(df, bollinger_info, rsi_value, macd_hist)
    if success_prob >= 70:
        prob_label = "高概率买入机会"
    elif success_prob >= 55:
        prob_label = "中等概率，可小仓位试探"
    elif success_prob >= 40:
        prob_label = "概率偏低，建议观望"
    else:
        prob_label = "不建议买入"

    # 交易建议
    atr_value = vol_info["ATR(14)"]
    ideal_buy = bollinger_info["下轨"]
    stop_loss = round(ideal_buy - 1.5 * atr_value, 2)
    take_profit = bollinger_info["上轨"]
    params_str = ", ".join(f"{k}={v}" for k, v in best_params.items()) if best_params else "默认参数"

    # 近期走势
    recent_returns = ""
    for days, label in [(5, "5日"), (20, "20日"), (60, "60日")]:
        if len(df) > days + 1:
            ret = (current_price / df["close"].iloc[-(days + 1)] - 1) * 100
            recent_returns += f"| {label} | {ret:+.2f}% |\n"

    md_content = f"""# {symbol} 综合量化分析报告

**市场**: {market_key} | **货币**: {currency} | **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 全策略参数优化结果排名 (过去1年)

| 排名 | 策略(最优参数) | 收益率 | 夏普比率 | 最大回撤 | 交易次数 |
|------|---------------|--------|----------|----------|----------|
{ranking_rows}

### 冠军策略详情

| 项目 | 值 |
|------|-----|
| 策略名称 | {sorted_results[0][1]['name']} |
| 最优参数 | {', '.join(f'{k}={v}' for k, v in sorted_results[0][1].get('best_params', {}).items()) or '默认参数'} |
| 收益率 | {sorted_results[0][1]['metrics'].get('total_return', 0):+.2f}% |
| 夏普比率 | {sorted_results[0][1]['metrics'].get('sharpe_ratio', 0):.4f} |
| 最大回撤 | {sorted_results[0][1]['metrics'].get('max_drawdown', 0):.2f}% |

![策略排名](../../charts/{timestamp}/strategy_ranking.png)

---

## 当前行情

| 指标 | 值 |
|------|-----|
| 最新收盘价 | {currency}{current_price:.2f} |
| 涨跌 | {price_change:+.2f} ({price_change_pct:+.2f}%) |
| 开盘 | {df['open'].iloc[-1]:.2f} |
| 最高 | {df['high'].iloc[-1]:.2f} |
| 最低 | {df['low'].iloc[-1]:.2f} |
| 成交量 | {df['volume'].iloc[-1]:,.0f} |

---

## 历史价格百分位分析

| 周期 | 百分位 |
|------|--------|
{percentile_rows}

---

## 布林带分析 (window={bollinger_window}, num_std={bollinger_std})

| 指标 | 值 |
|------|-----|
| 上轨 | {bollinger_info['上轨']} |
| 中轨 | {bollinger_info['中轨']} |
| 下轨 | {bollinger_info['下轨']} |
| 带宽 | {bollinger_info['带宽']} |
| 带内位置 | {bollinger_info['带内位置']:.2%} |
| 信号 | {bollinger_info['信号']} |

![布林带分析](../../charts/{timestamp}/price_bollinger.png)

---

## RSI 分析

| 指标 | 值 |
|------|-----|
| RSI(14) | {rsi_value:.2f} ({rsi_status}) |
| RSI(6) | {rsi6_value:.2f} |

![RSI分析](../../charts/{timestamp}/rsi_analysis.png)

---

## MACD 分析

| 指标 | 值 |
|------|-----|
| MACD线 | {macd_line_val.iloc[-1]:.4f} |
| 信号线 | {signal_line_val.iloc[-1]:.4f} |
| 柱状图 | {hist_val:.4f} ({macd_trend}) |

![MACD分析](../../charts/{timestamp}/macd_analysis.png)

---

## 均线系统

| 均线 | 价格 | 偏离 | 方向 |
|------|------|------|------|
{ma_rows}

![均线系统](../../charts/{timestamp}/ma_system.png)

---

## 支撑位与阻力位

### 阻力位 (由近到远)

| 名称 | 价格 | 距当前 |
|------|------|--------|
{resistance_rows}

**当前价格**: {currency}{current_price:.2f}

### 支撑位 (由近到远)

| 名称 | 价格 | 距当前 |
|------|------|--------|
{support_rows}

![支撑阻力位](../../charts/{timestamp}/support_resistance.png)

---

## 波动率分析

| 指标 | 值 |
|------|-----|
| 日波动率 | {vol_info['日波动率']}% |
| 年化波动率 | {vol_info['年化波动率']}% |
| ATR(14) | {vol_info['ATR(14)']} |

---

## 买入成功概率估算

| 项目 | 值 |
|------|-----|
| 综合评分 | {success_prob}/100 |
| 判断 | {prob_label} |

---

## 今日交易建议

### 近期走势

| 周期 | 涨跌 |
|------|------|
{recent_returns}

### 最优策略

**{best_strategy_name}** ({params_str})

回测表现: 收益 {best_metrics.get('total_return', 0):+.2f}%, 夏普 {best_metrics.get('sharpe_ratio', 0):.4f}, 回撤 {best_metrics.get('max_drawdown', 0):.2f}%

### 基于布林带的价格参考

| 项目 | 价格 |
|------|------|
| 当前价格 | {currency}{current_price:.2f} |
| 理想买入价 (下轨) | {currency}{ideal_buy:.2f} |
| 激进买入价 (中轨) | {currency}{bollinger_info['中轨']:.2f} |
| 止损价 (下轨-1.5×ATR) | {currency}{stop_loss:.2f} |
| 目标止盈价 (上轨) | {currency}{take_profit:.2f} |

---

## ⚠️ 风险提示

- 以上分析基于历史数据回测，不构成投资建议
- 过去表现不代表未来收益，市场存在不确定性
- 建议控制单只股票仓位不超过总资产的 20%
- 严格执行止损纪律，单笔亏损不超过总资产的 2%

---

*报告由 OpenQuant 通用股票分析工具自动生成*
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"📄 Markdown 报告已保存到: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="OpenQuant 通用股票分析工具 - 全策略回测 + 参数优化 + 交易建议"
    )
    parser.add_argument("--symbol", required=True, help="股票代码 (如 09988, 300750, 105.GOOG)")
    parser.add_argument(
        "--market",
        required=True,
        choices=list(MARKET_MAP.keys()),
        help="市场类型: a_share, hk_stock, us_stock",
    )
    parser.add_argument("--capital", type=float, default=100000, help="回测初始资金 (默认 100000)")
    args = parser.parse_args()

    symbol = args.symbol
    market_key = args.market
    market = MARKET_MAP[market_key]
    currency = CURRENCY_MAP.get(market_key, "")
    capital = args.capital

    print_header(symbol, market_key)

    # ========== 第一步: 获取数据 ==========
    print("\n📊 正在获取历史数据...")
    df = fetch_data(symbol, market, days=400)
    if df.empty:
        print("❌ 未获取到数据，请检查股票代码和市场类型")
        return
    print(f"  获取到 {len(df)} 条K线数据")

    # 回测使用最近1年的数据
    backtest_days = min(250, len(df))
    backtest_df = df.tail(backtest_days).copy()
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    print(f"  回测区间: 最近 {backtest_days} 个交易日 (~{start_date} ~ {end_date})")

    # ========== 第二步: 全策略并行参数优化 ==========
    print("\n⏳ 正在对所有策略进行参数优化（避免默认参数淘汰潜力策略）...")
    all_results = optimize_all_strategies(symbol, backtest_df, market, capital)
    print_optimization_ranking(all_results)

    # ========== 第三步: 确定最优策略和参数 ==========
    best_strategy_name, best_result = find_best_strategy(all_results)
    best_params = best_result.get("best_params", {})
    optimized_metrics = best_result["metrics"]

    # 确定布林带参数（如果最优策略是布林带则用优化后的参数，否则用默认值）
    if best_strategy_name == "bollinger_band" and best_params:
        bollinger_window = best_params.get("window", 20)
        bollinger_std = best_params.get("num_std", 2.0)
    else:
        bollinger_window = 20
        bollinger_std = 2.0

    # ========== 第五步: 技术分析 ==========
    bollinger_info, rsi_value, macd_hist, vol_info = print_technical_analysis(
        df, symbol, currency, bollinger_window, bollinger_std
    )

    # ========== 第六步: 交易建议 ==========
    final_metrics = optimized_metrics if optimized_metrics else best_result["metrics"]
    print_trading_advice(
        df,
        symbol,
        currency,
        best_strategy_name,
        best_params if best_params else {},
        final_metrics,
        bollinger_info,
        rsi_value,
        macd_hist,
        vol_info,
    )

    # ========== 第七步: 生成图表和 Markdown 报告 ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    charts_dir = f"output/charts/{timestamp}"
    md_dir = f"output/md/{timestamp}"

    print("\n🎨 正在生成图表...")
    generate_charts(
        df, symbol, charts_dir, all_results,
        bollinger_window, bollinger_std,
    )

    print("📝 正在生成 Markdown 报告...")
    generate_markdown_report(
        symbol=symbol,
        market_key=market_key,
        timestamp=timestamp,
        md_dir=md_dir,
        charts_dir=charts_dir,
        df=df,
        all_results=all_results,
        best_strategy_name=best_strategy_name,
        best_params=best_params if best_params else {},
        best_metrics=final_metrics,
        bollinger_info=bollinger_info,
        rsi_value=rsi_value,
        macd_hist=macd_hist,
        vol_info=vol_info,
        bollinger_window=bollinger_window,
        bollinger_std=bollinger_std,
    )

    # ========== 第八步: 导出 PDF ==========
    pdf_dir = f"output/pdf/{timestamp}"
    md_file_path = f"{md_dir}/analyze_{symbol}.md"
    pdf_file_path = f"{pdf_dir}/analyze_{symbol}.pdf"
    print("📑 正在导出 PDF...")
    try:
        convert_md_to_pdf(md_file_path, pdf_file_path)
        print(f"📑 PDF 已保存到: {pdf_file_path}")
    except Exception as exc:
        print(f"⚠️ PDF 导出失败: {exc}")

    print(f"\n{'='*70}")
    print(f"  ✅ 分析完成!")
    print(f"  📁 图表目录: {charts_dir}/")
    print(f"  📁 报告目录: {md_dir}/")
    print(f"  📁 PDF 目录:  {pdf_dir}/")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()