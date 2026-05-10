"""PDF 报告生成模块

为 recommend 命令生成包含以下内容的 PDF 报告：
1. 综合分析汇总表
2. 各股票最优策略的历史股价时序图 + B/S 操作标记 + 策略辅助线
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

from openquant.utils.indicators import (
    bollinger_bands,
    donchian_channel,
    exponential_moving_average,
    kdj,
    macd,
    moving_average,
    rate_of_change,
    rsi,
)

logger = logging.getLogger(__name__)

CJK_FONT_PATH = "/usr/share/fonts/google-droid/DroidSansFallback.ttf"

# 策略是否需要子图
_STRATEGIES_WITH_SUB_PLOT = {"macd", "rsi_reversal", "kdj", "volume_breakout"}


def _get_font() -> FontProperties:
    if os.path.exists(CJK_FONT_PATH):
        return FontProperties(fname=CJK_FONT_PATH)
    return FontProperties()


def generate_recommend_pdf(
    recommendations: list,
    output_dir: str = "output/recommend",
) -> str | None:
    """生成推荐报告 PDF，返回文件路径"""
    if not recommendations:
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    pdf_path = os.path.join(output_dir, f"recommend_{timestamp}.pdf")
    font_prop = _get_font()

    try:
        with PdfPages(pdf_path) as pdf:
            _draw_summary_page(pdf, recommendations, font_prop)
            for rec in recommendations:
                if rec.price_data is not None and not rec.price_data.empty:
                    _draw_stock_chart(pdf, rec, font_prop)
        logger.info("PDF 报告已保存: %s", pdf_path)
        print(f"\n  📄 PDF 报告已保存: {pdf_path}")
        return pdf_path
    except Exception as exc:
        logger.error("生成 PDF 报告失败: %s", exc)
        return None


def _draw_summary_page(pdf: PdfPages, recommendations: list, font_prop: FontProperties) -> None:
    """绘制综合分析汇总表页"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("off")

    title = "股票多策略综合分析报告"
    subtitle = f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}    验证方式: {recommendations[0].rolling_rounds}轮滚动窗口验证"
    ax.text(0.5, 0.97, title, transform=ax.transAxes, fontsize=18, fontproperties=font_prop,
            ha="center", va="top", fontweight="bold")
    ax.text(0.5, 0.93, subtitle, transform=ax.transAxes, fontsize=10, fontproperties=font_prop,
            ha="center", va="top", color="gray")

    columns = ["排名", "股票", "最优策略", "交易胜率", "盈亏比", "期望收益", "总收益", "持仓天", "最新价", "建议"]
    rows = []
    for rank, rec in enumerate(recommendations, 1):
        best_result = next((r for r in rec.strategy_results if r.strategy_name == rec.best_strategy_name), None)
        if best_result:
            if best_result.latest_buy_signal and not best_result.latest_sell_signal:
                action = "买入"
            elif best_result.latest_sell_signal and not best_result.latest_buy_signal:
                action = "卖出"
            else:
                action = "持有"
        else:
            action = "持有"

        pf_str = f"{rec.profit_factor:.2f}" if rec.profit_factor > 0 else "N/A"
        rows.append([
            str(rank), rec.display_name, rec.best_strategy_name,
            f"{rec.trade_win_rate:.1f}%", pf_str,
            f"{rec.expected_value:+.3f}%", f"{rec.backtest_return:+.2f}%",
            f"{rec.avg_holding_days:.1f}", f"{rec.latest_close:.2f}", action,
        ])

    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center",
                     bbox=[0.02, 0.05, 0.96, 0.83])
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for key, cell in table.get_celld().items():
        row_idx, col_idx = key
        if row_idx == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontproperties=font_prop, fontweight="bold")
        else:
            cell.set_text_props(fontproperties=font_prop)
            if col_idx == 9:
                text = cell.get_text().get_text()
                if text == "买入":
                    cell.set_facecolor("#E2EFDA")
                elif text == "卖出":
                    cell.set_facecolor("#FCE4D6")
            if row_idx % 2 == 0:
                if cell.get_facecolor()[:3] == (1.0, 1.0, 1.0):
                    cell.set_facecolor("#F2F2F2")

    table.scale(1, 1.6)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _draw_stock_chart(pdf: PdfPages, rec, font_prop: FontProperties) -> None:
    """绘制单只股票的股价时序图 + B/S 操作标记 + 策略辅助线"""
    df = rec.price_data
    trades = rec.best_strategy_trades
    strategy = rec.best_strategy_name

    dates = pd.to_datetime(df["datetime"])
    close_series = pd.Series(df["close"].values, index=range(len(df)))
    closes = df["close"].values

    needs_sub = strategy in _STRATEGIES_WITH_SUB_PLOT
    if needs_sub:
        fig, (ax_price, ax_sub) = plt.subplots(
            2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True,
        )
    else:
        fig, ax_price = plt.subplots(figsize=(14, 6))
        ax_sub = None

    # --- 主图：价格 ---
    ax_price.plot(dates, closes, color="#4472C4", linewidth=1.2, label="收盘价", zorder=2)
    ax_price.fill_between(dates, closes, alpha=0.06, color="#4472C4")

    # --- 策略辅助线（主图叠加） ---
    _draw_strategy_overlays(ax_price, ax_sub, df, dates, close_series, strategy, font_prop)

    # --- B/S 标记 ---
    _draw_trade_markers(ax_price, trades)

    # --- 标题和指标信息 ---
    title_text = f"{rec.display_name} ({rec.symbol}) - 最优策略: {rec.best_strategy_name}"
    ax_price.set_title(title_text, fontproperties=font_prop, fontsize=14, fontweight="bold", pad=12)

    info_text = (
        f"EV={rec.expected_value:+.3f}%  "
        f"胜率={rec.trade_win_rate:.1f}%  "
        f"盈亏比={rec.profit_factor:.2f}  "
        f"总收益={rec.backtest_return:+.2f}%  "
        f"持仓={rec.avg_holding_days:.1f}天"
    )
    ax_price.text(0.5, 1.02, info_text, transform=ax_price.transAxes, fontsize=9,
                  fontproperties=font_prop, ha="center", va="bottom", color="gray")

    ax_price.set_ylabel("价格", fontproperties=font_prop, fontsize=10)
    ax_price.legend(loc="upper left", prop=font_prop, fontsize=8, ncol=2)
    ax_price.grid(True, alpha=0.3)

    bottom_ax = ax_sub if ax_sub is not None else ax_price
    bottom_ax.set_xlabel("日期", fontproperties=font_prop, fontsize=10)

    fig.autofmt_xdate()
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _draw_trade_markers(ax, trades: list[dict]) -> None:
    """在主图上绘制 B/S 标记"""
    buy_dates, buy_prices, sell_dates, sell_prices = [], [], [], []
    for trade in trades:
        trade_dt = pd.to_datetime(trade["datetime"])
        if trade["side"] == "BUY":
            buy_dates.append(trade_dt)
            buy_prices.append(trade["price"])
        elif trade["side"] == "SELL":
            sell_dates.append(trade_dt)
            sell_prices.append(trade["price"])

    if buy_dates:
        ax.scatter(buy_dates, buy_prices, marker="^", color="#2E7D32", s=120, zorder=5, label="Buy")
        for dt, price in zip(buy_dates, buy_prices):
            ax.annotate("B", (dt, price), textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=10, fontweight="bold", color="#2E7D32")
    if sell_dates:
        ax.scatter(sell_dates, sell_prices, marker="v", color="#C62828", s=120, zorder=5, label="Sell")
        for dt, price in zip(sell_dates, sell_prices):
            ax.annotate("S", (dt, price), textcoords="offset points", xytext=(0, -16),
                        ha="center", fontsize=10, fontweight="bold", color="#C62828")


def _draw_strategy_overlays(ax_price, ax_sub, df, dates, close_series, strategy, font_prop) -> None:
    """根据策略类型绘制辅助线"""
    high_series = pd.Series(df["high"].values, index=range(len(df)))
    low_series = pd.Series(df["low"].values, index=range(len(df)))
    volume_series = pd.Series(df["volume"].values, index=range(len(df)))

    if strategy == "ma_cross":
        ma_short = moving_average(close_series, 5)
        ma_long = moving_average(close_series, 20)
        ax_price.plot(dates, ma_short.values, color="#FF8C00", linewidth=1.0, alpha=0.8, label="MA(5)")
        ax_price.plot(dates, ma_long.values, color="#DC143C", linewidth=1.0, alpha=0.8, label="MA(20)")

    elif strategy == "macd":
        dif, dea, macd_hist = macd(close_series, 12, 26, 9)
        if ax_sub is not None:
            ax_sub.plot(dates, dif.values, color="#FF8C00", linewidth=1.0, label="DIF")
            ax_sub.plot(dates, dea.values, color="#4472C4", linewidth=1.0, label="DEA")
            colors = ["#C62828" if v < 0 else "#2E7D32" for v in macd_hist.values]
            ax_sub.bar(dates, macd_hist.values, color=colors, alpha=0.6, width=1.0)
            ax_sub.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
            ax_sub.set_ylabel("MACD", fontproperties=font_prop, fontsize=9)
            ax_sub.legend(loc="upper left", prop=font_prop, fontsize=7)
            ax_sub.grid(True, alpha=0.2)

    elif strategy == "rsi_reversal":
        rsi_values = rsi(close_series, 14)
        if ax_sub is not None:
            ax_sub.plot(dates, rsi_values.values, color="#8B008B", linewidth=1.0, label="RSI(14)")
            ax_sub.axhline(y=70, color="#C62828", linewidth=0.8, linestyle="--", alpha=0.7, label="超买(70)")
            ax_sub.axhline(y=30, color="#2E7D32", linewidth=0.8, linestyle="--", alpha=0.7, label="超卖(30)")
            ax_sub.fill_between(dates, 30, 70, alpha=0.05, color="gray")
            ax_sub.set_ylabel("RSI", fontproperties=font_prop, fontsize=9)
            ax_sub.set_ylim(0, 100)
            ax_sub.legend(loc="upper left", prop=font_prop, fontsize=7)
            ax_sub.grid(True, alpha=0.2)

    elif strategy == "bollinger_band":
        upper, middle, lower = bollinger_bands(close_series, 20, 2.0)
        ax_price.plot(dates, upper.values, color="#C62828", linewidth=0.8, alpha=0.7, linestyle="--", label="上轨")
        ax_price.plot(dates, middle.values, color="#FF8C00", linewidth=0.8, alpha=0.7, label="中轨(MA20)")
        ax_price.plot(dates, lower.values, color="#2E7D32", linewidth=0.8, alpha=0.7, linestyle="--", label="下轨")
        ax_price.fill_between(dates, upper.values, lower.values, alpha=0.06, color="#FF8C00")

    elif strategy == "volume_breakout":
        price_ma = moving_average(close_series, 20)
        ax_price.plot(dates, price_ma.values, color="#FF8C00", linewidth=1.0, alpha=0.8, label="MA(20)")
        if ax_sub is not None:
            vol_ma = moving_average(volume_series, 20)
            ax_sub.bar(dates, volume_series.values, color="#4472C4", alpha=0.4, width=1.0, label="成交量")
            ax_sub.plot(dates, vol_ma.values, color="#FF8C00", linewidth=1.0, label="成交量MA(20)")
            threshold = vol_ma.values * 1.5
            ax_sub.plot(dates, threshold, color="#C62828", linewidth=0.8, linestyle="--", alpha=0.6, label="1.5×均量")
            ax_sub.set_ylabel("成交量", fontproperties=font_prop, fontsize=9)
            ax_sub.legend(loc="upper left", prop=font_prop, fontsize=7)
            ax_sub.grid(True, alpha=0.2)

    elif strategy == "kdj":
        k_val, d_val, j_val = kdj(high_series, low_series, close_series, 9, 3, 3)
        if ax_sub is not None:
            ax_sub.plot(dates, k_val.values, color="#FF8C00", linewidth=1.0, label="K")
            ax_sub.plot(dates, d_val.values, color="#4472C4", linewidth=1.0, label="D")
            ax_sub.plot(dates, j_val.values, color="#8B008B", linewidth=0.8, alpha=0.6, label="J")
            ax_sub.axhline(y=80, color="#C62828", linewidth=0.8, linestyle="--", alpha=0.5, label="超买(80)")
            ax_sub.axhline(y=20, color="#2E7D32", linewidth=0.8, linestyle="--", alpha=0.5, label="超卖(20)")
            ax_sub.set_ylabel("KDJ", fontproperties=font_prop, fontsize=9)
            ax_sub.legend(loc="upper left", prop=font_prop, fontsize=7, ncol=3)
            ax_sub.grid(True, alpha=0.2)

    elif strategy == "dual_momentum":
        ma_short = moving_average(close_series, 10)
        ma_long = moving_average(close_series, 30)
        ax_price.plot(dates, ma_short.values, color="#FF8C00", linewidth=1.0, alpha=0.8, label="MA(10)")
        ax_price.plot(dates, ma_long.values, color="#DC143C", linewidth=1.0, alpha=0.8, label="MA(30)")

    elif strategy == "trend_follow":
        ema_short = exponential_moving_average(close_series, 10)
        ema_long = exponential_moving_average(close_series, 50)
        ax_price.plot(dates, ema_short.values, color="#FF8C00", linewidth=1.0, alpha=0.8, label="EMA(10)")
        ax_price.plot(dates, ema_long.values, color="#DC143C", linewidth=1.0, alpha=0.8, label="EMA(50)")

    elif strategy == "turtle":
        upper_ch, _, _ = donchian_channel(high_series, low_series, 20)
        _, _, lower_ch = donchian_channel(high_series, low_series, 10)
        ax_price.plot(dates, upper_ch.values, color="#C62828", linewidth=0.8, alpha=0.7, linestyle="--", label="入场通道(20)")
        ax_price.plot(dates, lower_ch.values, color="#2E7D32", linewidth=0.8, alpha=0.7, linestyle="--", label="离场通道(10)")
        ax_price.fill_between(dates, upper_ch.values, lower_ch.values, alpha=0.05, color="#4472C4")
