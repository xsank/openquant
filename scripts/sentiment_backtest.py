"""情绪分析对比回测脚本

对同一标的、同一策略，分别运行"原始策略"和"情绪增强策略"，
对比两者的回测指标，验证情绪分析模块是否带来正向提升。

重要说明（港股/新闻数据限制）：
    akshare 的新闻接口（stock_news_em）只能获取最近约 10 条新闻，
    无法按历史日期范围查询。因此对于历史回测：
    - 新闻数据可能全部在回测区间之外（最新日期）
    - 这些新闻仍会被加载，并以时间衰减权重参与情绪计算
    - --show-events 会扩展扫描范围至所有事件覆盖的日期，并标注 * 提示

使用示例：
    # 基础用法（使用 MA Cross 策略对比）
    python scripts/sentiment_backtest.py --symbol 600519 --start-date 2022-01-01 --end-date 2024-12-31

    # 港股 + 新闻情绪 + 显示事件明细
    python scripts/sentiment_backtest.py \\
        --symbol 09988 \\
        --start-date 2025-03-01 --end-date 2026-03-01 \\
        --strategy macd --market hk_stock \\
        --with-news --show-events

    # 多标的对比
    python scripts/sentiment_backtest.py \\
        --symbols 600519 000858 601318 \\
        --start-date 2022-01-01 --end-date 2024-12-31 \\
        --strategy ma_cross
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from openquant.core.models import EventFactor, EventSentiment, MarketType
from openquant.datasource.factory import DataSourceFactory
from openquant.datasource.sentiment_event_source import SentimentEventSource
from openquant.engine.backtest_engine import BacktestEngine
from openquant.sentiment.config import SentimentConfig
from openquant.sentiment.sentiment_analyzer import SentimentAnalyzer
from openquant.sentiment.strategy_wrapper import SentimentStrategyWrapper
from openquant.storage.sqlite_storage import SqliteStorage
from openquant.strategy.bollinger_strategy import BollingerBandStrategy
from openquant.strategy.dual_momentum_strategy import DualMomentumStrategy
from openquant.strategy.kdj_strategy import KDJStrategy
from openquant.strategy.ma_cross import MACrossStrategy
from openquant.strategy.macd_strategy import MACDStrategy
from openquant.strategy.rsi_strategy import RSIReversalStrategy
from openquant.strategy.turtle_strategy import TurtleStrategy
from openquant.strategy.volume_breakout_strategy import VolumeBreakoutStrategy
from openquant.utils.logger import setup_logger

logger = setup_logger()

_STRATEGY_REGISTRY = {
    "ma_cross": MACrossStrategy,
    "macd": MACDStrategy,
    "rsi_reversal": RSIReversalStrategy,
    "bollinger_band": BollingerBandStrategy,
    "turtle": TurtleStrategy,
    "kdj": KDJStrategy,
    "dual_momentum": DualMomentumStrategy,
    "volume_breakout": VolumeBreakoutStrategy,
}

_SENTIMENT_CONFIG_MAP = {
    "default": SentimentConfig(),
    "conservative": SentimentConfig.conservative(),
    "aggressive": SentimentConfig.aggressive(),
}

_MARKET_MAP = {
    "a_share": MarketType.A_SHARE,
    "hk_stock": MarketType.HK_STOCK,
    "us_stock": MarketType.US_STOCK,
}

# 对比时关注的核心指标
_KEY_METRICS = [
    ("total_return", "总收益率(%)", True),
    ("annual_return", "年化收益率(%)", True),
    ("sharpe_ratio", "夏普比率", True),
    ("max_drawdown", "最大回撤(%)", False),
    ("win_rate", "胜率(%)", True),
    ("profit_loss_ratio", "盈亏比", True),
    ("calmar_ratio", "卡玛比率", True),
    ("total_trades", "交易次数", None),
]


def run_single_backtest(
    symbol: str,
    df: pd.DataFrame,
    market: MarketType,
    strategy_class,
    sentiment_config: SentimentConfig | None,
    events: list[EventFactor],
    initial_capital: float,
    commission_rate: float,
    slippage_rate: float,
    storage: SqliteStorage,
) -> dict:
    """运行单次回测，返回结果指标字典"""
    strategy = strategy_class()

    if sentiment_config is not None:
        strategy = SentimentStrategyWrapper.wrap(strategy, sentiment_config)

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        storage=storage,
    )
    engine.set_strategy(strategy)
    engine.add_data(symbol, df, market)

    if events:
        engine.add_events(symbol, events)

    engine.run()
    results = engine.get_results()
    results["strategy_display_name"] = strategy.get_name()
    return results


def compare_sentiment_impact(
    symbol: str,
    df: pd.DataFrame,
    market: MarketType,
    strategy_class,
    sentiment_modes: list[str],
    events: list[EventFactor],
    initial_capital: float,
    commission_rate: float,
    slippage_rate: float,
    storage: SqliteStorage,
) -> dict[str, dict]:
    """对同一标的运行原始策略 + 多种情绪模式，返回对比结果"""
    all_results: dict[str, dict] = {}

    logger.info("运行原始策略: %s × %s", strategy_class.__name__, symbol)
    baseline_result = run_single_backtest(
        symbol, df, market, strategy_class,
        sentiment_config=None,
        events=events,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        storage=storage,
    )
    all_results["baseline"] = baseline_result

    for mode in sentiment_modes:
        sentiment_config = _SENTIMENT_CONFIG_MAP.get(mode)
        if sentiment_config is None:
            logger.warning("未知情绪模式: %s，跳过", mode)
            continue

        logger.info("运行情绪增强策略: %s × %s (模式=%s)", strategy_class.__name__, symbol, mode)
        result = run_single_backtest(
            symbol, df, market, strategy_class,
            sentiment_config=sentiment_config,
            events=events,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            storage=storage,
        )
        all_results[f"sentiment_{mode}"] = result

    return all_results


def print_sentiment_events(
    symbol: str,
    events: list[EventFactor],
    start_date: str,
    end_date: str,
    sentiment_config: SentimentConfig | None = None,
) -> None:
    """打印情绪事件/新闻明细及逐日情绪分析结果

    输出三部分：
    1. 原始事件/新闻列表（时间点、类型、情绪方向、强度、内容摘要）
    2. 按周聚合的情绪走势（正向/负向得分，扩展至所有事件覆盖日期）
    3. 关键情绪拐点（得分绝对值最大的 Top 10 日期）
    """
    if not events:
        print(f"\n  ⚠️  {symbol} 未加载到任何情绪事件/新闻数据")
        print(f"  可能原因：")
        print(f"    1. akshare 新闻接口只返回最近约 10 条，历史区间内可能无数据")
        print(f"    2. 港股/美股事件数据源覆盖有限，可尝试 --with-events 加载历史事件")
        return

    config = sentiment_config or SentimentConfig()
    analyzer = SentimentAnalyzer(config)
    analyzer.load_events(symbol, events)

    sentiment_icons = {
        EventSentiment.BULLISH: "🟢",
        EventSentiment.BEARISH: "🔴",
        EventSentiment.NEUTRAL: "⚪",
    }

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # 计算实际扫描范围（扩展到所有事件覆盖的日期）
    all_event_dates = [pd.Timestamp(e.event_date) for e in events]
    scan_end_ts = max(end_ts, max(all_event_dates)) if all_event_dates else end_ts
    out_of_range_events = [e for e in events if pd.Timestamp(e.event_date) > end_ts]

    # ── 1. 原始事件明细 ──────────────────────────────────────────────
    print(f"\n{'='*92}")
    print(f"  📰 {symbol} 情绪事件/新闻明细  ({len(events)} 条，回测区间: {start_date} ~ {end_date})")
    print(f"{'='*92}")

    if out_of_range_events:
        print(f"  ℹ️  注意：{len(out_of_range_events)} 条新闻超出回测区间（标注 *），"
              f"因接口限制只能获取最近新闻，无法查询历史。")
        print(f"     这些新闻以最大权重（days_ago=0）参与回测末期情绪计算，"
              f"扫描范围已扩展至 {scan_end_ts.strftime('%Y-%m-%d')}。")

    print(f"\n  {'日期':<14} {'情绪':<8} {'强度':>5}  {'类型':<18} {'来源':<16} 内容摘要")
    print(f"  {'-'*90}")

    for event in sorted(events, key=lambda e: e.event_date):
        date_str = event.event_date.strftime("%Y-%m-%d")
        is_out_of_range = pd.Timestamp(event.event_date) > end_ts
        range_marker = " *" if is_out_of_range else "  "
        icon = sentiment_icons.get(event.sentiment, "⚪")
        event_type = event.event_type.value[:16]
        source = event.source[:14]
        description = event.description[:42] if event.description else "(无描述)"
        print(
            f"  {date_str}{range_marker:<2} {icon}{event.sentiment.value:<6} {event.strength:>4.2f}  "
            f"{event_type:<18} {source:<16} {description}"
        )

    # ── 2. 按周聚合情绪走势（扩展范围）────────────────────────────────
    print(f"\n{'='*92}")
    print(f"  📈 {symbol} 情绪得分走势（按周聚合，扫描至 {scan_end_ts.strftime('%Y-%m-%d')}）")
    print(f"{'='*92}")
    print(f"  {'周起始日':<16} {'综合得分':>10} {'正向得分':>10} {'负向得分':>10}  情绪强度可视化")
    print(f"  {'-'*72}")

    weekly_scores: list[tuple[str, float, float, float]] = []
    current = start_ts
    while current <= scan_end_ts:
        score = analyzer.compute_sentiment(symbol, current.to_pydatetime())
        if score.event_count > 0 or score.news_count > 0:
            is_future = current > end_ts
            date_label = current.strftime("%Y-%m-%d") + (" *" if is_future else "  ")
            weekly_scores.append((date_label, score.composite_score, score.bullish_score, score.bearish_score))
        current += pd.Timedelta(weeks=1)

    if not weekly_scores:
        print("  （无有效情绪数据）")
    else:
        for date_label, composite, bullish, bearish in weekly_scores:
            bar_len = min(int(abs(composite) * 4), 20)
            bar = ("🟢" if composite > 0 else "🔴") * bar_len if bar_len > 0 else "⚪"
            print(f"  {date_label:<16} {composite:>+10.3f} {bullish:>10.3f} {bearish:>10.3f}  {bar}")

    if out_of_range_events:
        print("  (* 超出回测区间，仅供参考)")

    # ── 3. 关键情绪拐点 Top 10 ───────────────────────────────────────
    print(f"\n{'='*92}")
    print(f"  🎯 {symbol} 关键情绪拐点（情绪得分绝对值 Top 10）")
    print(f"{'='*92}")
    print(f"  {'日期':<16} {'综合得分':>10} {'正向':>8} {'负向':>8} {'事件数':>6}  判断")
    print(f"  {'-'*74}")

    daily_scores: list[tuple[str, float, float, float, int]] = []
    current = start_ts
    while current <= scan_end_ts:
        score = analyzer.compute_sentiment(symbol, current.to_pydatetime())
        if score.event_count > 0 or score.news_count > 0:
            is_future = current > end_ts
            date_label = current.strftime("%Y-%m-%d") + (" *" if is_future else "  ")
            daily_scores.append((
                date_label,
                score.composite_score,
                score.bullish_score,
                score.bearish_score,
                score.event_count + score.news_count,
            ))
        current += pd.Timedelta(days=1)

    top_scores = sorted(daily_scores, key=lambda x: abs(x[1]), reverse=True)[:10]
    for date_label, composite, bullish, bearish, count in sorted(top_scores, key=lambda x: x[0]):
        if composite > config.bullish_block_buy_threshold:
            judgment = "✅ 利多（可加仓）"
        elif composite < config.bearish_force_sell_threshold:
            judgment = "🚨 极端利空（强制卖出）"
        elif composite < config.bearish_block_buy_threshold:
            judgment = "⛔ 利空（阻止买入）"
        else:
            judgment = "➖ 中性（未达阈值）"
        print(
            f"  {date_label:<16} {composite:>+10.3f} {bullish:>8.3f} {bearish:>8.3f} {count:>6}  {judgment}"
        )

    print(f"\n  阈值参考: 利多加仓 > {config.bullish_block_buy_threshold} | "
          f"阻止买入 < {config.bearish_block_buy_threshold} | "
          f"强制卖出 < {config.bearish_force_sell_threshold}")
    print(f"{'='*92}\n")


def print_comparison_table(symbol: str, all_results: dict[str, dict]) -> None:
    """打印策略对比表格"""
    print(f"\n{'='*80}")
    print(f"  情绪分析对比回测结果 — {symbol}")
    print(f"{'='*80}")

    run_keys = list(all_results.keys())
    col_width = 18
    header = f"{'指标':<20}" + "".join(
        f"{all_results[key].get('strategy_display_name', key)[:col_width]:<{col_width}}"
        for key in run_keys
    )
    print(header)
    print("-" * (20 + col_width * len(run_keys)))

    baseline = all_results.get("baseline", {})

    for metric_key, metric_label, higher_is_better in _KEY_METRICS:
        row = f"{metric_label:<20}"
        baseline_val = baseline.get(metric_key, 0)

        for run_key in run_keys:
            result = all_results[run_key]
            val = result.get(metric_key, 0)

            if isinstance(val, float):
                val_str = f"{val:.4f}" if abs(val) < 100 else f"{val:.2f}"
            else:
                val_str = str(val)

            if run_key != "baseline" and higher_is_better is not None and isinstance(val, (int, float)):
                delta = val - baseline_val
                if higher_is_better:
                    marker = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
                else:
                    marker = "↑" if delta < 0 else ("↓" if delta > 0 else "=")
                val_str = f"{val_str}{marker}"

            row += f"{val_str:<{col_width}}"

        print(row)

    print("=" * (20 + col_width * len(run_keys)))


def print_sentiment_summary(all_results: dict[str, dict]) -> None:
    """打印情绪增强效果总结"""
    baseline = all_results.get("baseline", {})
    if not baseline:
        return

    print("\n📊 情绪增强效果总结:")
    print("-" * 60)

    improvements: list[tuple[str, str, float]] = []

    for run_key, result in all_results.items():
        if run_key == "baseline":
            continue

        mode = run_key.replace("sentiment_", "")
        sharpe_delta = result.get("sharpe_ratio", 0) - baseline.get("sharpe_ratio", 0)
        return_delta = result.get("total_return", 0) - baseline.get("total_return", 0)
        drawdown_delta = result.get("max_drawdown", 0) - baseline.get("max_drawdown", 0)

        improvements.append((mode, result.get("strategy_display_name", run_key), sharpe_delta))

        print(f"  [{mode}] {result.get('strategy_display_name', run_key)}")
        print(f"    夏普比率: {baseline.get('sharpe_ratio', 0):.4f} → {result.get('sharpe_ratio', 0):.4f} "
              f"({'↑' if sharpe_delta > 0 else '↓'}{abs(sharpe_delta):.4f})")
        print(f"    总收益率: {baseline.get('total_return', 0):.2f}% → {result.get('total_return', 0):.2f}% "
              f"({'↑' if return_delta > 0 else '↓'}{abs(return_delta):.2f}%)")
        print(f"    最大回撤: {baseline.get('max_drawdown', 0):.2f}% → {result.get('max_drawdown', 0):.2f}% "
              f"({'↓' if drawdown_delta < 0 else '↑'}{abs(drawdown_delta):.2f}%)")
        print()

    if improvements:
        best_mode, best_name, best_delta = max(improvements, key=lambda x: x[2])
        verdict = "✅ 正向提升" if best_delta > 0 else "❌ 未见提升"
        print(f"  最佳情绪模式: [{best_mode}] {verdict} (夏普比率 Δ={best_delta:+.4f})")

    print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="情绪分析对比回测：验证情绪模块对策略的正向提升效果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--symbol", help="单个标的代码（与 --symbols 二选一）")
    parser.add_argument("--symbols", nargs="+", help="多个标的代码")
    parser.add_argument("--start-date", required=True, help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="结束日期 (YYYY-MM-DD)")
    parser.add_argument(
        "--strategy", default="ma_cross",
        choices=list(_STRATEGY_REGISTRY.keys()),
        help="策略名称（默认 ma_cross）",
    )
    parser.add_argument(
        "--sentiment-modes", nargs="+",
        default=["default", "conservative", "aggressive"],
        choices=list(_SENTIMENT_CONFIG_MAP.keys()),
        help="要对比的情绪模式列表（默认全部）",
    )
    parser.add_argument(
        "--with-events", action="store_true",
        help="是否加载历史事件因子（财报、分红、大宗交易等）",
    )
    parser.add_argument(
        "--with-news", action="store_true",
        help="是否加载新闻情绪数据（注意：接口只返回最近约10条，无法查询历史）",
    )
    parser.add_argument(
        "--show-events", action="store_true",
        help="打印情绪事件/新闻明细及逐日情绪分析结果（辅助分析用）",
    )
    parser.add_argument("--datasource", default="baostock", help="行情数据源（默认 baostock）")
    parser.add_argument("--market", default="a_share", choices=list(_MARKET_MAP.keys()))
    parser.add_argument("--capital", type=float, default=100000, help="初始资金")
    parser.add_argument("--commission", type=float, default=0.0003, help="佣金费率")
    parser.add_argument("--slippage", type=float, default=0.001, help="滑点费率")
    parser.add_argument("--db-path", default="data/sentiment_backtest.db", help="数据库路径")

    args = parser.parse_args()

    symbols: list[str] = []
    if args.symbol:
        symbols.append(args.symbol)
    if args.symbols:
        symbols.extend(args.symbols)
    if not symbols:
        parser.error("请通过 --symbol 或 --symbols 指定至少一个标的")

    market = _MARKET_MAP.get(args.market, MarketType.A_SHARE)
    strategy_class = _STRATEGY_REGISTRY[args.strategy]

    DataSourceFactory.register_defaults()
    datasource_name = args.datasource
    if market != MarketType.A_SHARE and datasource_name == "baostock":
        datasource_name = "akshare"
    data_source = DataSourceFactory.get(datasource_name)

    Path(args.db_path).parent.mkdir(parents=True, exist_ok=True)
    storage = SqliteStorage(args.db_path)
    storage.initialize()

    print(f"\n{'='*80}")
    print(f"  情绪分析对比回测")
    print(f"  策略: {strategy_class.__name__}  |  标的: {symbols}")
    print(f"  时间: {args.start_date} ~ {args.end_date}")
    print(f"  情绪模式: {args.sentiment_modes}")
    print(f"{'='*80}\n")

    all_symbol_results: dict[str, dict[str, dict]] = {}

    for symbol in symbols:
        logger.info("正在获取 %s 的行情数据...", symbol)
        df = data_source.fetch_daily_bars(symbol, args.start_date, args.end_date, market)

        if df.empty:
            logger.error("未获取到 %s 的数据，跳过", symbol)
            continue

        logger.info("获取到 %s 的 %d 条K线数据", symbol, len(df))
        storage.save_bars(df, symbol, market)

        # 加载情绪事件/新闻数据
        events: list[EventFactor] = []
        if args.with_events or args.with_news:
            logger.info("正在获取 %s 的情绪事件数据...", symbol)
            try:
                sentiment_source = SentimentEventSource(
                    enable_news=args.with_news,
                    enable_events=args.with_events,
                )
                events = sentiment_source.fetch_all(
                    symbol, args.start_date, args.end_date, market,
                )
                logger.info("加载了 %d 个情绪事件因子", len(events))
            except Exception as exc:
                logger.warning("获取情绪事件数据失败: %s", exc)

        # 打印情绪事件/新闻明细（辅助分析）
        if args.show_events:
            print_sentiment_events(
                symbol=symbol,
                events=events,
                start_date=args.start_date,
                end_date=args.end_date,
                sentiment_config=_SENTIMENT_CONFIG_MAP.get(args.sentiment_modes[0]),
            )

        # 运行对比回测
        symbol_results = compare_sentiment_impact(
            symbol=symbol,
            df=df,
            market=market,
            strategy_class=strategy_class,
            sentiment_modes=args.sentiment_modes,
            events=events,
            initial_capital=args.capital,
            commission_rate=args.commission,
            slippage_rate=args.slippage,
            storage=storage,
        )

        all_symbol_results[symbol] = symbol_results

        print_comparison_table(symbol, symbol_results)
        print_sentiment_summary(symbol_results)

    storage.close()

    # 多标的汇总
    if len(symbols) > 1:
        print(f"\n{'='*80}")
        print("  多标的情绪增强效果汇总")
        print(f"{'='*80}")
        print(f"  {'标的':<12} {'情绪模式':<16} {'夏普比率变化':>14} {'总收益率变化':>14} {'最大回撤变化':>14}")
        print("-" * 80)

        for symbol, symbol_results in all_symbol_results.items():
            baseline = symbol_results.get("baseline", {})
            for run_key, result in symbol_results.items():
                if run_key == "baseline":
                    continue
                mode = run_key.replace("sentiment_", "")
                sharpe_delta = result.get("sharpe_ratio", 0) - baseline.get("sharpe_ratio", 0)
                return_delta = result.get("total_return", 0) - baseline.get("total_return", 0)
                drawdown_delta = result.get("max_drawdown", 0) - baseline.get("max_drawdown", 0)
                print(
                    f"  {symbol:<12} {mode:<16} "
                    f"{sharpe_delta:>+13.4f}  {return_delta:>+13.2f}%  {drawdown_delta:>+13.2f}%"
                )

        print("=" * 80)


if __name__ == "__main__":
    main()
