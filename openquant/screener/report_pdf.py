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


def _generate_signal_reason(rec, df: pd.DataFrame) -> tuple[str, str]:
    """根据最优策略和最新数据生成当前建议及原因说明

    建议仅以最优策略信号为准（与回测一致）：
    - 最优策略有买入信号 → 买入
    - 最优策略有卖出信号 → 卖出
    - 最优策略无信号 → 观望
    其他策略不一致的信号仅作为辅助参考在文字中说明。

    Returns:
        (action, reason): 建议动作和原因文字
    """
    strategy = rec.best_strategy_name
    best_result = next((r for r in rec.strategy_results if r.strategy_name == strategy), None)
    if best_result is None:
        return "观望", ""

    has_buy = best_result.latest_buy_signal
    has_sell = best_result.latest_sell_signal

    # 建议仅以最优策略为准
    if has_buy and not has_sell:
        action = "买入"
    elif has_sell and not has_buy:
        action = "卖出"
    elif has_buy and has_sell:
        action = "冲突"
    else:
        action = "观望"

    # 收集其他策略的辅助信号
    buy_strategies = [r.strategy_name for r in rec.strategy_results
                      if r.latest_buy_signal and r.strategy_name != strategy]
    sell_strategies = [r.strategy_name for r in rec.strategy_results
                       if r.latest_sell_signal and r.strategy_name != strategy]

    close_series = pd.Series(df["close"].values, index=range(len(df)))
    high_series = pd.Series(df["high"].values, index=range(len(df)))
    low_series = pd.Series(df["low"].values, index=range(len(df)))
    volume_series = pd.Series(df["volume"].values, index=range(len(df)))
    current_close = close_series.iloc[-1]
    prev_close = close_series.iloc[-2] if len(close_series) > 1 else current_close

    reason = ""

    if strategy == "volume_breakout":
        price_ma = moving_average(close_series, 20)
        vol_ma = moving_average(volume_series, 20)
        cur_vol = volume_series.iloc[-1]
        cur_vol_ma = vol_ma.iloc[-1] if not np.isnan(vol_ma.iloc[-1]) else 1
        vol_ratio = cur_vol / cur_vol_ma if cur_vol_ma > 0 else 0
        cur_price_ma = price_ma.iloc[-1]
        if has_buy:
            reason = (f"价格{current_close:.2f}突破MA(20)={cur_price_ma:.2f}，"
                      f"成交量为均量的{vol_ratio:.1f}倍(阈值1.5倍)")
        elif has_sell:
            reason = f"价格{current_close:.2f}跌破MA(20)={cur_price_ma:.2f}，趋势转弱"
        else:
            reason = f"价格{current_close:.2f} vs MA(20)={cur_price_ma:.2f}，成交量{vol_ratio:.1f}倍均量，等待突破确认"

    elif strategy == "bollinger_band":
        upper, middle, lower = bollinger_bands(close_series, 20, 2.0)
        cur_upper = upper.iloc[-1]
        cur_lower = lower.iloc[-1]
        cur_middle = middle.iloc[-1]
        prev_lower_val = lower.iloc[-2] if len(lower) > 1 else cur_lower
        prev_upper_val = upper.iloc[-2] if len(upper) > 1 else cur_upper
        if has_buy:
            reason = (f"价格从下轨({prev_lower_val:.2f})下方回到带内，"
                      f"当前{current_close:.2f}，BOLL({cur_lower:.2f}/{cur_middle:.2f}/{cur_upper:.2f})，超跌反弹信号")
        elif has_sell:
            reason = (f"价格从上轨({prev_upper_val:.2f})上方回到带内，"
                      f"当前{current_close:.2f}，BOLL({cur_lower:.2f}/{cur_middle:.2f}/{cur_upper:.2f})，超涨回落信号")
        else:
            reason = f"当前{current_close:.2f}，BOLL({cur_lower:.2f}/{cur_middle:.2f}/{cur_upper:.2f})，处于带内等待信号"

    elif strategy == "ma_cross":
        ma_short = moving_average(close_series, 5)
        ma_long = moving_average(close_series, 20)
        cur_ma5 = ma_short.iloc[-1]
        cur_ma20 = ma_long.iloc[-1]
        if has_buy:
            reason = f"MA(5)={cur_ma5:.2f}上穿MA(20)={cur_ma20:.2f}，金叉买入信号"
        elif has_sell:
            reason = f"MA(5)={cur_ma5:.2f}下穿MA(20)={cur_ma20:.2f}，死叉卖出信号"
        else:
            reason = f"MA(5)={cur_ma5:.2f} vs MA(20)={cur_ma20:.2f}，等待交叉信号"

    elif strategy == "macd":
        dif, dea, _ = macd(close_series, 12, 26, 9)
        cur_dif = dif.iloc[-1]
        cur_dea = dea.iloc[-1]
        if has_buy:
            reason = f"DIF={cur_dif:.3f}上穿DEA={cur_dea:.3f}，MACD金叉买入"
        elif has_sell:
            reason = f"DIF={cur_dif:.3f}下穿DEA={cur_dea:.3f}，MACD死叉卖出"
        else:
            reason = f"DIF={cur_dif:.3f} vs DEA={cur_dea:.3f}，等待交叉"

    elif strategy == "rsi_reversal":
        rsi_values = rsi(close_series, 14)
        cur_rsi = rsi_values.iloc[-1]
        if has_buy:
            reason = f"RSI(14)={cur_rsi:.1f}，从超卖区(<30)回升，反弹买入"
        elif has_sell:
            reason = f"RSI(14)={cur_rsi:.1f}，从超买区(>70)回落，获利卖出"
        else:
            reason = f"RSI(14)={cur_rsi:.1f}，处于中性区间，等待信号"

    elif strategy == "kdj":
        k_val, d_val, j_val = kdj(high_series, low_series, close_series, 9, 3, 3)
        cur_k = k_val.iloc[-1]
        cur_d = d_val.iloc[-1]
        if has_buy:
            reason = f"K={cur_k:.1f}上穿D={cur_d:.1f}(超卖区)，KDJ金叉买入"
        elif has_sell:
            reason = f"K={cur_k:.1f}下穿D={cur_d:.1f}(超买区)，KDJ死叉卖出"
        else:
            reason = f"K={cur_k:.1f}, D={cur_d:.1f}，等待交叉信号"

    elif strategy == "dual_momentum":
        ma_short = moving_average(close_series, 10)
        ma_long = moving_average(close_series, 30)
        cur_ma10 = ma_short.iloc[-1]
        cur_ma30 = ma_long.iloc[-1]
        if has_buy:
            reason = f"MA(10)={cur_ma10:.2f}上穿MA(30)={cur_ma30:.2f}+动量确认，双动量买入"
        elif has_sell:
            reason = f"MA(10)={cur_ma10:.2f}下穿MA(30)={cur_ma30:.2f}，动量衰减卖出"
        else:
            reason = f"MA(10)={cur_ma10:.2f} vs MA(30)={cur_ma30:.2f}，等待动量确认"

    elif strategy == "trend_follow":
        ema_short = exponential_moving_average(close_series, 10)
        ema_long = exponential_moving_average(close_series, 50)
        cur_ema10 = ema_short.iloc[-1]
        cur_ema50 = ema_long.iloc[-1]
        if has_buy:
            reason = f"EMA(10)={cur_ema10:.2f}上穿EMA(50)={cur_ema50:.2f}，趋势确认买入"
        elif has_sell:
            reason = f"EMA(10)={cur_ema10:.2f}下穿EMA(50)={cur_ema50:.2f}，趋势反转卖出"
        else:
            reason = f"EMA(10)={cur_ema10:.2f} vs EMA(50)={cur_ema50:.2f}，等待趋势确认"

    elif strategy == "turtle":
        upper_ch, _, _ = donchian_channel(high_series, low_series, 20)
        _, _, lower_ch = donchian_channel(high_series, low_series, 10)
        cur_upper_ch = upper_ch.iloc[-1]
        cur_lower_ch = lower_ch.iloc[-1]
        if has_buy:
            reason = f"价格{current_close:.2f}突破20日高点{cur_upper_ch:.2f}，海龟入场"
        elif has_sell:
            reason = f"价格{current_close:.2f}跌破10日低点{cur_lower_ch:.2f}，海龟离场"
        else:
            reason = f"价格{current_close:.2f}，20日高点{cur_upper_ch:.2f}/10日低点{cur_lower_ch:.2f}，等待突破"

    elif strategy == "event_enhanced_ma":
        ma_short = moving_average(close_series, 5)
        ma_long = moving_average(close_series, 20)
        cur_ma5 = ma_short.iloc[-1]
        cur_ma20 = ma_long.iloc[-1]
        if has_buy:
            reason = f"MA(5)={cur_ma5:.2f}上穿MA(20)={cur_ma20:.2f}+事件增强，买入"
        elif has_sell:
            reason = f"MA(5)={cur_ma5:.2f}下穿MA(20)={cur_ma20:.2f}，卖出"
        else:
            reason = f"MA(5)={cur_ma5:.2f} vs MA(20)={cur_ma20:.2f}，等待信号"

    else:
        if has_buy:
            reason = f"策略{strategy}发出买入信号，价格{current_close:.2f}"
        elif has_sell:
            reason = f"策略{strategy}发出卖出信号，价格{current_close:.2f}"
        else:
            reason = f"策略{strategy}当前无明确信号，价格{current_close:.2f}"

    # 附加仓位信息
    pos_pct = int(rec.position_ratio_used * 100)
    if action in ("买入", "卖出"):
        reason += f"  |  建议仓位: {pos_pct}%"

    # 其他策略的辅助参考信息（与最优策略信号不一致时补充说明）
    aux_parts = []
    if buy_strategies:
        aux_parts.append(f"另有{len(buy_strategies)}个策略建议买入({', '.join(buy_strategies[:3])})")
    if sell_strategies:
        aux_parts.append(f"另有{len(sell_strategies)}个策略建议卖出({', '.join(sell_strategies[:3])})")
    if aux_parts:
        reason += "  [辅助参考: " + "; ".join(aux_parts) + "]"

    return action, reason


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

        # 建议仅以最优策略信号为准（与回测一致）
        if best_result:
            if best_result.latest_buy_signal and not best_result.latest_sell_signal:
                action = "买入"
            elif best_result.latest_sell_signal and not best_result.latest_buy_signal:
                action = "卖出"
            elif best_result.latest_buy_signal and best_result.latest_sell_signal:
                action = "冲突"
            else:
                action = "观望"
        else:
            action = "观望"

        # 其他策略辅助参考（不改变主建议，仅标注）
        other_buy_count = sum(1 for r in rec.strategy_results
                             if r.latest_buy_signal and r.strategy_name != rec.best_strategy_name)
        other_sell_count = sum(1 for r in rec.strategy_results
                              if r.latest_sell_signal and r.strategy_name != rec.best_strategy_name)

        if other_buy_count > 0:
            aux_tag = f"+{other_buy_count}买"
        elif other_sell_count > 0:
            aux_tag = f"+{other_sell_count}卖"
        else:
            aux_tag = ""

        # 附加仓位比例
        pos_pct = int(rec.position_ratio_used * 100)
        action_display = f"{action} {pos_pct}%仓"
        if aux_tag:
            action_display += f" {aux_tag}"

        pf_str = f"{rec.profit_factor:.2f}" if rec.profit_factor > 0 else "N/A"
        rows.append([
            str(rank), rec.display_name, rec.best_strategy_name,
            f"{rec.trade_win_rate:.1f}%", pf_str,
            f"{rec.expected_value:+.3f}%", f"{rec.backtest_return:+.2f}%",
            f"{rec.avg_holding_days:.1f}", f"{rec.latest_close:.2f}", action_display,
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
                if text.startswith("买入"):
                    cell.set_facecolor("#E2EFDA")
                elif text.startswith("卖出"):
                    cell.set_facecolor("#FCE4D6")
                elif text.startswith("观望"):
                    cell.set_facecolor("#FFF2CC")
            if row_idx % 2 == 0:
                if cell.get_facecolor()[:3] == (1.0, 1.0, 1.0):
                    cell.set_facecolor("#F2F2F2")

    table.scale(1, 1.6)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _draw_stock_chart(pdf: PdfPages, rec, font_prop: FontProperties) -> None:
    """绘制单只股票的股价时序图 + B/S 操作标记 + 策略辅助线 + 当前建议标注"""
    df = rec.price_data
    trades = rec.best_strategy_trades
    strategy = rec.best_strategy_name

    dates = pd.to_datetime(df["datetime"])
    close_series = pd.Series(df["close"].values, index=range(len(df)))
    closes = df["close"].values

    # 生成当前建议和原因
    signal_action, signal_reason = _generate_signal_reason(rec, df)

    needs_sub = strategy in _STRATEGIES_WITH_SUB_PLOT
    if needs_sub:
        fig, (ax_price, ax_sub) = plt.subplots(
            2, 1, figsize=(14, 9), height_ratios=[3, 1], sharex=True,
        )
    else:
        fig, ax_price = plt.subplots(figsize=(14, 7))
        ax_sub = None

    # --- 主图：价格 ---
    ax_price.plot(dates, closes, color="#4472C4", linewidth=1.2, label="收盘价", zorder=2)
    ax_price.fill_between(dates, closes, alpha=0.06, color="#4472C4")

    # --- 策略辅助线（主图叠加） ---
    _draw_strategy_overlays(ax_price, ax_sub, df, dates, close_series, strategy, font_prop)

    # --- B/S 标记 ---
    _draw_trade_markers(ax_price, trades)

    # --- 在最新价格处标注当前建议 ---
    latest_date = dates.iloc[-1]
    latest_close = closes[-1]
    if signal_action == "买入":
        marker_color = "#2E7D32"
        marker_symbol = "★买入"
    elif signal_action == "卖出":
        marker_color = "#C62828"
        marker_symbol = "★卖出"
    else:
        marker_color = "#FF8C00"
        marker_symbol = f"★{signal_action}"

    ax_price.annotate(
        marker_symbol,
        xy=(latest_date, latest_close),
        xytext=(15, 20),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
        color=marker_color,
        fontproperties=font_prop,
        arrowprops=dict(arrowstyle="->", color=marker_color, lw=1.5),
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=marker_color, alpha=0.9),
    )

    # --- 标题和指标信息（使用 fig.suptitle + fig.text 避免重叠） ---
    title_text = f"{rec.display_name} ({rec.symbol}) - 最优策略: {rec.best_strategy_name}"
    fig.suptitle(title_text, fontproperties=font_prop, fontsize=14, fontweight="bold", y=0.98)

    info_text = (
        f"EV={rec.expected_value:+.3f}%  "
        f"胜率={rec.trade_win_rate:.1f}%  "
        f"盈亏比={rec.profit_factor:.2f}  "
        f"总收益={rec.backtest_return:+.2f}%  "
        f"持仓={rec.avg_holding_days:.1f}天"
    )
    fig.text(0.5, 0.94, info_text, ha="center", va="top", fontsize=9,
             fontproperties=font_prop, color="gray")

    ax_price.set_ylabel("价格", fontproperties=font_prop, fontsize=10)
    ax_price.legend(loc="upper left", prop=font_prop, fontsize=8, ncol=2)
    ax_price.grid(True, alpha=0.3)

    bottom_ax = ax_sub if ax_sub is not None else ax_price
    bottom_ax.set_xlabel("日期", fontproperties=font_prop, fontsize=10)

    # --- 底部建议原因说明文字 ---
    if signal_reason:
        reason_label = f"[当前建议: {signal_action}]  原因: {signal_reason}"
        fig.text(
            0.5, 0.01, reason_label,
            ha="center", va="bottom", fontsize=9,
            fontproperties=font_prop, color="#333333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", edgecolor="#CCCCCC", alpha=0.95),
        )

    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
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
