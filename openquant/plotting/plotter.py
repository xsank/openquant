"""回测结果绘图工具

提供多种图表绘制功能，用于可视化回测结果分析。
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 配置中文字体支持
import matplotlib.font_manager as fm

_CHINESE_FONT_PATH = "/usr/share/fonts/google-droid/DroidSansFallback.ttf"
_CHINESE_FONT_PROP = fm.FontProperties(fname=_CHINESE_FONT_PATH)
fm.fontManager.addfont(_CHINESE_FONT_PATH)
_FONT_NAME = _CHINESE_FONT_PROP.get_name()

plt.rcParams["font.sans-serif"] = [_FONT_NAME, "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "sans-serif"


def plot_equity_curve(
    equity_df: pd.DataFrame,
    title: str = "权益曲线",
    save_path: str | None = None,
) -> None:
    """绘制单只标的的权益曲线

    Args:
        equity_df: 包含 datetime 和 equity 列的 DataFrame
        title: 图表标题
        save_path: 保存路径，为 None 则不保存
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(equity_df["datetime"], equity_df["equity"], linewidth=1.5, color="#2196F3")
    ax.fill_between(
        equity_df["datetime"],
        equity_df["equity"],
        equity_df["equity"].iloc[0],
        alpha=0.15,
        color="#2196F3",
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("权益曲线已保存: %s", save_path)
    plt.close(fig)


def plot_multi_equity_curves(
    equity_data: dict[str, pd.DataFrame],
    title: str = "Multiple Equity Curves",
    save_path: str | None = None,
) -> None:
    """绘制多只标的的权益曲线（归一化对比）

    Args:
        equity_data: {标的名称: equity_df} 字典
        title: 图表标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(equity_data)))

    for (name, df), color in zip(equity_data.items(), colors):
        if df.empty:
            continue
        normalized = df["equity"] / df["equity"].iloc[0] * 100
        ax.plot(df["datetime"], normalized, linewidth=1.5, label=name, color=color)

    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Equity (Base=100)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("多标的权益曲线已保存: %s", save_path)
    plt.close(fig)


def plot_returns_comparison(
    results: dict[str, dict],
    title: str = "Returns Comparison",
    save_path: str | None = None,
) -> None:
    """绘制各标的收益率对比柱状图

    Args:
        results: {标的名称: metrics_dict} 字典
        title: 图表标题
        save_path: 保存路径
    """
    names = list(results.keys())
    total_returns = [results[name].get("total_return", 0) for name in names]
    annual_returns = [results[name].get("annual_return", 0) for name in names]

    x_positions = np.arange(len(names))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(16, 8))
    bars_total = ax.bar(
        x_positions - bar_width / 2, total_returns, bar_width,
        label="Total Return (%)", color="#4CAF50", alpha=0.85,
    )
    bars_annual = ax.bar(
        x_positions + bar_width / 2, annual_returns, bar_width,
        label="Annual Return (%)", color="#FF9800", alpha=0.85,
    )

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar_group in [bars_total, bars_annual]:
        for bar in bar_group:
            height = bar.get_height()
            vertical_offset = 0.5 if height >= 0 else -1.5
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, vertical_offset),
                textcoords="offset points",
                ha="center", va="bottom" if height >= 0 else "top",
                fontsize=7,
            )

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("收益对比图已保存: %s", save_path)
    plt.close(fig)


def plot_risk_metrics(
    results: dict[str, dict],
    title: str = "Risk Metrics Comparison",
    save_path: str | None = None,
) -> None:
    """绘制风险指标对比图（夏普比率、最大回撤、波动率）

    Args:
        results: {标的名称: metrics_dict} 字典
        title: 图表标题
        save_path: 保存路径
    """
    names = list(results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 夏普比率
    sharpe_values = [results[name].get("sharpe_ratio", 0) for name in names]
    colors_sharpe = ["#4CAF50" if v > 0 else "#F44336" for v in sharpe_values]
    axes[0].barh(names, sharpe_values, color=colors_sharpe, alpha=0.85)
    axes[0].set_title("Sharpe Ratio", fontsize=12, fontweight="bold")
    axes[0].axvline(x=0, color="black", linewidth=0.8)
    axes[0].grid(True, axis="x", alpha=0.3)

    # 最大回撤
    drawdown_values = [results[name].get("max_drawdown", 0) for name in names]
    axes[1].barh(names, drawdown_values, color="#F44336", alpha=0.85)
    axes[1].set_title("Max Drawdown (%)", fontsize=12, fontweight="bold")
    axes[1].grid(True, axis="x", alpha=0.3)

    # 年化波动率
    volatility_values = [results[name].get("annual_volatility", 0) for name in names]
    axes[2].barh(names, volatility_values, color="#FF9800", alpha=0.85)
    axes[2].set_title("Annual Volatility (%)", fontsize=12, fontweight="bold")
    axes[2].grid(True, axis="x", alpha=0.3)

    for ax in axes:
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("风险指标对比图已保存: %s", save_path)
    plt.close(fig)


def plot_drawdown(
    equity_df: pd.DataFrame,
    title: str = "Drawdown",
    save_path: str | None = None,
) -> None:
    """绘制回撤曲线

    Args:
        equity_df: 包含 datetime 和 equity 列的 DataFrame
        title: 图表标题
        save_path: 保存路径
    """
    peak = equity_df["equity"].expanding().max()
    drawdown = (equity_df["equity"] - peak) / peak * 100

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(equity_df["datetime"], drawdown, 0, alpha=0.4, color="#F44336")
    ax.plot(equity_df["datetime"], drawdown, linewidth=1, color="#D32F2F")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("回撤曲线已保存: %s", save_path)
    plt.close(fig)


def plot_summary_table(
    results: dict[str, dict],
    title: str = "Backtest Summary",
    save_path: str | None = None,
) -> None:
    """绘制回测结果汇总表格图

    Args:
        results: {标的名称: metrics_dict} 字典
        title: 图表标题
        save_path: 保存路径
    """
    columns = [
        "Total Return(%)", "Annual Return(%)", "Volatility(%)",
        "Sharpe", "Sortino", "Max DD(%)", "Win Rate(%)", "Trades",
    ]
    table_data = []
    for name, metrics in results.items():
        row = [
            f"{metrics.get('total_return', 0):.2f}",
            f"{metrics.get('annual_return', 0):.2f}",
            f"{metrics.get('annual_volatility', 0):.2f}",
            f"{metrics.get('sharpe_ratio', 0):.4f}",
            f"{metrics.get('sortino_ratio', 0):.4f}",
            f"{metrics.get('max_drawdown', 0):.2f}",
            f"{metrics.get('win_rate', 0):.2f}",
            f"{metrics.get('total_trades', 0)}",
        ]
        table_data.append(row)

    row_labels = list(results.keys())
    row_count = len(row_labels)
    fig_height = max(4, 1.5 + row_count * 0.45)
    fig, ax = plt.subplots(figsize=(18, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == -1:
            cell.set_facecolor("#ECEFF1")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#FAFAFA" if row % 2 == 0 else "white")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("汇总表格已保存: %s", save_path)
    plt.close(fig)


def generate_full_report(
    all_results: dict[str, dict],
    all_equity_curves: dict[str, pd.DataFrame],
    output_dir: str = "output/charts",
) -> list[str]:
    """生成完整的回测报告图表集

    Args:
        all_results: {标的名称: metrics_dict} 字典
        all_equity_curves: {标的名称: equity_df} 字典
        output_dir: 输出目录

    Returns:
        生成的图表文件路径列表
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated_files = []

    # 1. 多标的权益曲线对比
    filepath = str(output_path / "equity_curves_comparison.png")
    plot_multi_equity_curves(
        all_equity_curves,
        title="Equity Curves Comparison (Normalized)",
        save_path=filepath,
    )
    generated_files.append(filepath)

    # 2. 收益率对比柱状图
    filepath = str(output_path / "returns_comparison.png")
    plot_returns_comparison(
        all_results,
        title="Returns Comparison",
        save_path=filepath,
    )
    generated_files.append(filepath)

    # 3. 风险指标对比
    filepath = str(output_path / "risk_metrics.png")
    plot_risk_metrics(
        all_results,
        title="Risk Metrics Comparison",
        save_path=filepath,
    )
    generated_files.append(filepath)

    # 4. 汇总表格
    filepath = str(output_path / "summary_table.png")
    plot_summary_table(
        all_results,
        title="Backtest Summary",
        save_path=filepath,
    )
    generated_files.append(filepath)

    # 5. 每只标的的单独权益曲线和回撤图
    for name, equity_df in all_equity_curves.items():
        if equity_df.empty:
            continue
        safe_name = name.replace("/", "_").replace(".", "_").replace(" ", "_")

        filepath = str(output_path / f"equity_{safe_name}.png")
        plot_equity_curve(equity_df, title=f"Equity Curve - {name}", save_path=filepath)
        generated_files.append(filepath)

        filepath = str(output_path / f"drawdown_{safe_name}.png")
        plot_drawdown(equity_df, title=f"Drawdown - {name}", save_path=filepath)
        generated_files.append(filepath)

    logger.info("报告生成完成，共 %d 张图表，输出目录: %s", len(generated_files), output_dir)
    return generated_files


def plot_strategy_heatmap(
    all_results: dict[str, dict[str, dict]],
    metric_key: str = "total_return",
    title: str = "Strategy × Stock Heatmap",
    save_path: str | None = None,
) -> None:
    """绘制策略×标的的热力图

    Args:
        all_results: {strategy_name: {stock_name: metrics_dict}}
        metric_key: 要展示的指标键名
        title: 图表标题
        save_path: 保存路径
    """
    strategy_names = list(all_results.keys())
    stock_names = []
    for strategy_result in all_results.values():
        for stock_name in strategy_result:
            if stock_name not in stock_names:
                stock_names.append(stock_name)

    data_matrix = np.zeros((len(strategy_names), len(stock_names)))
    for row_idx, strategy in enumerate(strategy_names):
        for col_idx, stock in enumerate(stock_names):
            metrics = all_results.get(strategy, {}).get(stock, {})
            data_matrix[row_idx, col_idx] = metrics.get(metric_key, 0)

    fig, ax = plt.subplots(figsize=(max(14, len(stock_names) * 1.5), max(6, len(strategy_names) * 0.8)))
    im = ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(np.arange(len(stock_names)))
    ax.set_yticks(np.arange(len(strategy_names)))
    ax.set_xticklabels(stock_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(strategy_names, fontsize=9)

    for row_idx in range(len(strategy_names)):
        for col_idx in range(len(stock_names)):
            value = data_matrix[row_idx, col_idx]
            text_color = "white" if abs(value) > np.max(np.abs(data_matrix)) * 0.6 else "black"
            ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center",
                    color=text_color, fontsize=8, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8, label=metric_key.replace("_", " ").title())
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("热力图已保存: %s", save_path)
    plt.close(fig)


def plot_best_strategy_per_stock(
    all_results: dict[str, dict[str, dict]],
    title: str = "Best Strategy per Stock (by Total Return)",
    save_path: str | None = None,
) -> None:
    """绘制每只标的的最佳策略对比图

    Args:
        all_results: {strategy_name: {stock_name: metrics_dict}}
        title: 图表标题
        save_path: 保存路径
    """
    stock_names = []
    for strategy_result in all_results.values():
        for stock_name in strategy_result:
            if stock_name not in stock_names:
                stock_names.append(stock_name)

    best_strategies = []
    best_returns = []
    worst_returns = []

    for stock in stock_names:
        best_strategy = None
        best_return = -float("inf")
        worst_return = float("inf")
        for strategy_name, strategy_result in all_results.items():
            total_return = strategy_result.get(stock, {}).get("total_return", 0)
            if total_return > best_return:
                best_return = total_return
                best_strategy = strategy_name
            if total_return < worst_return:
                worst_return = total_return
        best_strategies.append(best_strategy or "N/A")
        best_returns.append(best_return)
        worst_returns.append(worst_return)

    fig, ax = plt.subplots(figsize=(max(14, len(stock_names) * 1.2), 8))
    x_positions = np.arange(len(stock_names))
    bar_width = 0.35

    colors_best = ["#4CAF50" if r >= 0 else "#F44336" for r in best_returns]
    colors_worst = ["#81C784" if r >= 0 else "#E57373" for r in worst_returns]

    ax.bar(x_positions - bar_width / 2, best_returns, bar_width,
           label="Best Strategy", color=colors_best, alpha=0.9, edgecolor="white")
    ax.bar(x_positions + bar_width / 2, worst_returns, bar_width,
           label="Worst Strategy", color=colors_worst, alpha=0.6, edgecolor="white")

    for idx, (strategy, ret) in enumerate(zip(best_strategies, best_returns)):
        vertical_offset = 1 if ret >= 0 else -3
        ax.annotate(f"{strategy}\n{ret:.1f}%",
                    xy=(x_positions[idx] - bar_width / 2, ret),
                    xytext=(0, vertical_offset), textcoords="offset points",
                    ha="center", va="bottom" if ret >= 0 else "top", fontsize=7)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Total Return (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stock_names, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("最佳策略对比图已保存: %s", save_path)
    plt.close(fig)


def plot_strategy_radar(
    all_results: dict[str, dict[str, dict]],
    stock_name: str,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """绘制单只标的在各策略下的雷达图

    Args:
        all_results: {strategy_name: {stock_name: metrics_dict}}
        stock_name: 标的名称
        title: 图表标题
        save_path: 保存路径
    """
    metrics_keys = ["total_return", "sharpe_ratio", "win_rate", "profit_loss_ratio"]
    metrics_labels = ["Total Return", "Sharpe Ratio", "Win Rate", "P/L Ratio"]

    strategies_with_data = []
    for strategy_name, strategy_result in all_results.items():
        if stock_name in strategy_result:
            strategies_with_data.append(strategy_name)

    if not strategies_with_data:
        return

    raw_values = {}
    for strategy in strategies_with_data:
        metrics = all_results[strategy][stock_name]
        raw_values[strategy] = [metrics.get(key, 0) for key in metrics_keys]

    # 归一化到 0-1
    all_vals = np.array(list(raw_values.values()))
    min_vals = all_vals.min(axis=0)
    max_vals = all_vals.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1

    num_vars = len(metrics_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies_with_data)))

    for (strategy, values), color in zip(raw_values.items(), colors):
        normalized = (np.array(values) - min_vals) / ranges
        normalized = normalized.tolist()
        normalized += normalized[:1]
        ax.plot(angles, normalized, linewidth=1.5, label=strategy, color=color)
        ax.fill(angles, normalized, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels, fontsize=10)
    ax.set_ylim(0, 1.1)

    chart_title = title or f"Strategy Comparison - {stock_name}"
    ax.set_title(chart_title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("雷达图已保存: %s", save_path)
    plt.close(fig)


def plot_multi_strategy_summary_table(
    all_results: dict[str, dict[str, dict]],
    title: str = "Multi-Strategy Backtest Summary",
    save_path: str | None = None,
) -> None:
    """绘制多策略×多标的汇总表格

    Args:
        all_results: {strategy_name: {stock_name: metrics_dict}}
        title: 图表标题
        save_path: 保存路径
    """
    stock_names = []
    for strategy_result in all_results.values():
        for stock_name in strategy_result:
            if stock_name not in stock_names:
                stock_names.append(stock_name)

    columns = ["Strategy"] + stock_names
    table_data = []

    for strategy_name, strategy_result in all_results.items():
        row = [strategy_name]
        for stock in stock_names:
            metrics = strategy_result.get(stock, {})
            total_return = metrics.get("total_return", 0)
            sharpe = metrics.get("sharpe_ratio", 0)
            row.append(f"{total_return:+.1f}% | SR:{sharpe:.2f}")
        table_data.append(row)

    row_count = len(table_data)
    col_count = len(columns)
    fig_width = max(18, col_count * 2.2)
    fig_height = max(4, 1.5 + row_count * 0.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 0:
            cell.set_facecolor("#ECEFF1")
            cell.set_text_props(fontweight="bold")
        else:
            cell_text = cell.get_text().get_text()
            if "+" in cell_text:
                cell.set_facecolor("#E8F5E9")
            elif "-" in cell_text:
                cell.set_facecolor("#FFEBEE")
            else:
                cell.set_facecolor("#FAFAFA" if row % 2 == 0 else "white")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("多策略汇总表格已保存: %s", save_path)
    plt.close(fig)


def generate_multi_strategy_report(
    all_results: dict[str, dict[str, dict]],
    all_equity_curves: dict[str, dict[str, pd.DataFrame]],
    output_dir: str = "output/charts",
) -> list[str]:
    """生成多策略×多标的的完整回测报告

    Args:
        all_results: {strategy_name: {stock_name: metrics_dict}}
        all_equity_curves: {strategy_name: {stock_name: equity_df}}
        output_dir: 输出目录

    Returns:
        生成的图表文件路径列表
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated_files = []

    # 1. 收益率热力图
    filepath = str(output_path / "return_heatmap.png")
    plot_strategy_heatmap(
        all_results, metric_key="total_return",
        title="Total Return Heatmap (Strategy × Stock)",
        save_path=filepath,
    )
    generated_files.append(filepath)

    # 2. 夏普比率热力图
    filepath = str(output_path / "sharpe_heatmap.png")
    plot_strategy_heatmap(
        all_results, metric_key="sharpe_ratio",
        title="Sharpe Ratio Heatmap (Strategy × Stock)",
        save_path=filepath,
    )
    generated_files.append(filepath)

    # 3. 最大回撤热力图
    filepath = str(output_path / "drawdown_heatmap.png")
    plot_strategy_heatmap(
        all_results, metric_key="max_drawdown",
        title="Max Drawdown Heatmap (Strategy × Stock)",
        save_path=filepath,
    )
    generated_files.append(filepath)

    # 4. 每只标的的最佳/最差策略对比
    filepath = str(output_path / "best_strategy_per_stock.png")
    plot_best_strategy_per_stock(all_results, save_path=filepath)
    generated_files.append(filepath)

    # 5. 多策略汇总表格
    filepath = str(output_path / "multi_strategy_summary.png")
    plot_multi_strategy_summary_table(all_results, save_path=filepath)
    generated_files.append(filepath)

    # 6. 按策略分组：每个策略的多标的权益曲线对比
    for strategy_name, equity_data in all_equity_curves.items():
        if not equity_data:
            continue
        safe_strategy = strategy_name.replace("/", "_").replace(" ", "_")

        filepath = str(output_path / f"equity_{safe_strategy}.png")
        plot_multi_equity_curves(
            equity_data,
            title=f"Equity Curves - {strategy_name}",
            save_path=filepath,
        )
        generated_files.append(filepath)

        # 该策略的收益率对比
        strategy_results = all_results.get(strategy_name, {})
        if strategy_results:
            filepath = str(output_path / f"returns_{safe_strategy}.png")
            plot_returns_comparison(
                strategy_results,
                title=f"Returns Comparison - {strategy_name}",
                save_path=filepath,
            )
            generated_files.append(filepath)

            filepath = str(output_path / f"risk_{safe_strategy}.png")
            plot_risk_metrics(
                strategy_results,
                title=f"Risk Metrics - {strategy_name}",
                save_path=filepath,
            )
            generated_files.append(filepath)

    # 7. 按标的分组：每只标的在各策略下的雷达图
    stock_names = []
    for strategy_result in all_results.values():
        for stock_name in strategy_result:
            if stock_name not in stock_names:
                stock_names.append(stock_name)

    for stock_name in stock_names:
        safe_stock = stock_name.replace("/", "_").replace(".", "_").replace(" ", "_")
        filepath = str(output_path / f"radar_{safe_stock}.png")
        plot_strategy_radar(all_results, stock_name, save_path=filepath)
        generated_files.append(filepath)

    logger.info(
        "多策略报告生成完成，共 %d 张图表，输出目录: %s",
        len(generated_files), output_dir,
    )
    return generated_files
