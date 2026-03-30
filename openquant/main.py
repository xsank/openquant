"""OpenQuant 入口文件

提供命令行接口，支持回测和模拟交易。
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from openquant.core.models import MarketType
from openquant.datasource.factory import DataSourceFactory
from openquant.engine.backtest_engine import BacktestEngine
from openquant.optimizer.param_optimizer import ParameterOptimizer
from openquant.plotting.plotter import (
    generate_full_report,
    plot_benchmark_comparison,
    plot_rolling_alpha_beta,
    plot_benchmark_summary_table,
)
from openquant.storage.sqlite_storage import SqliteStorage
from openquant.strategy.bollinger_strategy import BollingerBandStrategy
from openquant.strategy.dual_momentum_strategy import DualMomentumStrategy
from openquant.strategy.kdj_strategy import KDJStrategy
from openquant.strategy.ma_cross import MACrossStrategy
from openquant.strategy.macd_strategy import MACDStrategy
from openquant.strategy.rsi_strategy import RSIReversalStrategy
from openquant.strategy.turtle_strategy import TurtleStrategy
from openquant.strategy.volume_breakout_strategy import VolumeBreakoutStrategy
from openquant.strategy.event_enhanced_ma_cross import EventEnhancedMACrossStrategy
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
    "event_ma_cross": EventEnhancedMACrossStrategy,
}

_MARKET_MAP = {
    "a_share": MarketType.A_SHARE,
    "hk_stock": MarketType.HK_STOCK,
    "us_stock": MarketType.US_STOCK,
    "fund": MarketType.FUND,
}


def run_backtest(args: argparse.Namespace) -> None:
    """执行历史回测"""
    DataSourceFactory.register_defaults()
    market = _MARKET_MAP.get(args.market, MarketType.A_SHARE)

    datasource_name = args.datasource
    if market != MarketType.A_SHARE and datasource_name == "baostock":
        logger.info("市场类型为 %s，baostock 仅支持 A 股，自动切换到 akshare 数据源", market.value)
        datasource_name = "akshare"

    data_source = DataSourceFactory.get(datasource_name)

    logger.info("正在获取 %s 的历史数据 (%s ~ %s)...", args.symbol, args.start_date, args.end_date)
    df = data_source.fetch_daily_bars(args.symbol, args.start_date, args.end_date, market)

    if df.empty:
        logger.error("未获取到数据，请检查标的代码和日期范围")
        return

    logger.info("获取到 %d 条K线数据", len(df))

    # 初始化存储
    storage = SqliteStorage(args.db_path)
    storage.initialize()
    storage.save_bars(df, args.symbol, market)

    # 创建策略
    strategy_class = _STRATEGY_REGISTRY.get(args.strategy)
    if strategy_class is None:
        logger.error("未知策略: %s，可用策略: %s", args.strategy, list(_STRATEGY_REGISTRY.keys()))
        return
    strategy = strategy_class()

    # 创建回测引擎
    engine = BacktestEngine(
        initial_capital=args.capital,
        commission_rate=args.commission,
        slippage_rate=args.slippage,
        storage=storage,
    )
    engine.set_strategy(strategy)
    engine.add_data(args.symbol, df, market)

    # 加载事件因子数据
    if getattr(args, "with_events", False):
        try:
            from openquant.datasource.composite_event_source import CompositeEventSource
            event_source = CompositeEventSource()
            logger.info(
                "正在获取 %s 的事件因子数据 (市场: %s, 数据源路由: %s)...",
                args.symbol, market.value,
                event_source.list_market_sources().get(market.value, []),
            )
            events = event_source.fetch_events(args.symbol, args.start_date, args.end_date, market=market)
            if events:
                engine.add_events(args.symbol, events)
                logger.info("加载了 %d 个事件因子", len(events))
            else:
                logger.info("未获取到事件因子数据")
        except Exception as exc:
            logger.warning("获取事件因子数据失败: %s", exc)

    # 加载基准数据
    benchmark_df = None
    if hasattr(args, "benchmark") and args.benchmark:
        logger.info("正在获取基准 %s 的历史数据...", args.benchmark)
        try:
            benchmark_df = data_source.fetch_daily_bars(args.benchmark, args.start_date, args.end_date, market)
            if not benchmark_df.empty:
                engine.set_benchmark(args.benchmark, benchmark_df)
                logger.info("基准数据加载成功: %d 条", len(benchmark_df))
            else:
                logger.warning("未获取到基准数据: %s", args.benchmark)
        except Exception as exc:
            logger.warning("获取基准数据失败: %s", exc)

    # 运行回测
    engine.run()

    # 输出结果
    results = engine.get_results()
    _print_results(results)

    # 生成基准对比图表
    if benchmark_df is not None and not benchmark_df.empty:
        output_dir = getattr(args, "output_dir", "output/charts")
        equity_df = engine.get_equity_curve()
        benchmark_equity_df = engine.get_benchmark_equity_curve()

        if not equity_df.empty and not benchmark_equity_df.empty:
            from pathlib import Path
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            plot_benchmark_comparison(
                equity_df, benchmark_equity_df,
                strategy_name=strategy.get_name(),
                benchmark_name=args.benchmark,
                save_path=f"{output_dir}/benchmark_comparison.png",
            )
            plot_rolling_alpha_beta(
                equity_df, benchmark_equity_df,
                window=60,
                save_path=f"{output_dir}/rolling_alpha_beta.png",
            )
            logger.info("基准对比图表已生成到 %s", output_dir)

    storage.close()


def run_simulation(args: argparse.Namespace) -> None:
    """执行模拟交易"""
    from openquant.engine.simulation_engine import SimulationEngine

    DataSourceFactory.register_defaults()
    data_source = DataSourceFactory.get(args.datasource)

    market = _MARKET_MAP.get(args.market, MarketType.A_SHARE)

    storage = SqliteStorage(args.db_path)
    storage.initialize()

    strategy_class = _STRATEGY_REGISTRY.get(args.strategy)
    if strategy_class is None:
        logger.error("未知策略: %s", args.strategy)
        return
    strategy = strategy_class()

    engine = SimulationEngine(
        data_source=data_source,
        initial_capital=args.capital,
        commission_rate=args.commission,
        slippage_rate=args.slippage,
        poll_interval=args.interval,
        max_rounds=args.max_rounds,
        storage=storage,
    )
    engine.set_strategy(strategy)

    for symbol in args.symbols:
        engine.add_symbol(symbol, market)

    logger.info("模拟交易启动，按 Ctrl+C 停止...")
    engine.run()

    results = engine.get_results()
    _print_results(results)

    storage.close()


def _print_results(results: dict) -> None:
    """格式化输出回测/模拟结果"""
    if not results:
        logger.warning("无结果数据")
        return

    print("\n" + "=" * 60)
    print(f"  策略: {results.get('strategy_name', 'N/A')}")
    print("=" * 60)
    print(f"  初始资金:       {results.get('initial_capital', 0):>14,.2f}")
    print(f"  最终权益:       {results.get('final_equity', 0):>14,.2f}")
    print(f"  总收益率:       {results.get('total_return', 0):>13.2f}%")
    print(f"  年化收益率:     {results.get('annual_return', 0):>13.2f}%")
    print(f"  年化波动率:     {results.get('annual_volatility', 0):>13.2f}%")
    print(f"  夏普比率:       {results.get('sharpe_ratio', 0):>14.4f}")
    print(f"  索提诺比率:     {results.get('sortino_ratio', 0):>14.4f}")
    print(f"  卡玛比率:       {results.get('calmar_ratio', 0):>14.4f}")
    print(f"  最大回撤:       {results.get('max_drawdown', 0):>13.2f}%")
    print(f"  最大回撤天数:   {results.get('max_drawdown_duration', 0):>14d}")
    print(f"  胜率:           {results.get('win_rate', 0):>13.2f}%")
    print(f"  盈亏比:         {results.get('profit_loss_ratio', 0):>14.4f}")
    print(f"  交易次数:       {results.get('total_trades', 0):>14d}")
    print(f"  总佣金:         {results.get('total_commission', 0):>14,.2f}")

    # 基准对比指标
    if "alpha" in results:
        print("-" * 60)
        print(f"  基准:           {results.get('benchmark_symbol', 'N/A'):>14s}")
        print(f"  Alpha:          {results.get('alpha', 0):>13.4f}%")
        print(f"  Beta:           {results.get('beta', 0):>14.4f}")
        print(f"  信息比率:       {results.get('information_ratio', 0):>14.4f}")
        print(f"  跟踪误差:       {results.get('tracking_error', 0):>13.4f}%")
        print(f"  Treynor比率:    {results.get('treynor_ratio', 0):>14.4f}")
        print(f"  超额收益:       {results.get('excess_return', 0):>13.2f}%")
        print(f"  超额年化收益:   {results.get('excess_annual_return', 0):>13.2f}%")
        print(f"  相关系数:       {results.get('correlation', 0):>14.4f}")
        print(f"  R²:             {results.get('r_squared', 0):>14.4f}")
        print(f"  基准总收益:     {results.get('benchmark_total_return', 0):>13.2f}%")
        print(f"  基准年化收益:   {results.get('benchmark_annual_return', 0):>13.2f}%")
        print(f"  基准波动率:     {results.get('benchmark_volatility', 0):>13.2f}%")
        print(f"  基准最大回撤:   {results.get('benchmark_max_drawdown', 0):>13.2f}%")

    print("=" * 60 + "\n")


def run_batch_backtest(args: argparse.Namespace) -> None:
    """执行批量回测：对多只标的使用多种策略分别回测并生成对比图表"""
    import json

    from openquant.plotting.plotter import generate_multi_strategy_report

    DataSourceFactory.register_defaults()

    # 解析标的配置：格式为 "market:symbol:name"
    stock_configs = []
    for item in args.stocks:
        parts = item.split(":")
        if len(parts) != 3:
            logger.error("标的格式错误: %s，应为 market:symbol:name", item)
            continue
        market_key, symbol, display_name = parts
        market = _MARKET_MAP.get(market_key)
        if market is None:
            logger.error("未知市场类型: %s", market_key)
            continue
        stock_configs.append((market, symbol, display_name))

    if not stock_configs:
        logger.error("无有效标的配置")
        return

    # 解析策略列表
    if args.strategies:
        strategy_names = args.strategies
    else:
        strategy_names = list(_STRATEGY_REGISTRY.keys())

    invalid_strategies = [s for s in strategy_names if s not in _STRATEGY_REGISTRY]
    if invalid_strategies:
        logger.error("未知策略: %s，可用策略: %s", invalid_strategies, list(_STRATEGY_REGISTRY.keys()))
        return

    # 预先获取所有标的数据（避免重复拉取）
    stock_data_cache: dict[str, pd.DataFrame] = {}
    storage = SqliteStorage(args.db_path)
    storage.initialize()

    for market, symbol, display_name in stock_configs:
        datasource_name = args.datasource
        if market != MarketType.A_SHARE and datasource_name == "baostock":
            datasource_name = "akshare"

        data_source = DataSourceFactory.get(datasource_name)

        try:
            logger.info("正在获取 %s (%s) 的数据...", display_name, symbol)
            df = data_source.fetch_daily_bars(symbol, args.start_date, args.end_date, market)
        except Exception as exc:
            logger.error("获取 %s 数据失败: %s", display_name, exc)
            continue

        if df.empty:
            logger.warning("未获取到 %s 的数据，跳过", display_name)
            continue

        logger.info("获取到 %s 的 %d 条K线数据", display_name, len(df))
        storage.save_bars(df, symbol, market)
        stock_data_cache[display_name] = (market, symbol, df)

    if not stock_data_cache:
        logger.error("所有标的数据获取失败")
        storage.close()
        return

    # 多策略 × 多标的 回测
    # all_results 结构: {strategy_name: {stock_name: metrics_dict}}
    all_results: dict[str, dict[str, dict]] = {}
    # all_equity_curves 结构: {strategy_name: {stock_name: equity_df}}
    all_equity_curves: dict[str, dict[str, pd.DataFrame]] = {}

    for strategy_name in strategy_names:
        strategy_class = _STRATEGY_REGISTRY[strategy_name]
        all_results[strategy_name] = {}
        all_equity_curves[strategy_name] = {}

        print(f"\n{'='*60}")
        print(f"  策略: {strategy_name}")
        print(f"{'='*60}")

        for display_name, (market, symbol, df) in stock_data_cache.items():
            logger.info("回测: %s × %s", strategy_name, display_name)

            strategy = strategy_class()
            engine = BacktestEngine(
                initial_capital=args.capital,
                commission_rate=args.commission,
                slippage_rate=args.slippage,
                storage=storage,
            )
            engine.set_strategy(strategy)
            engine.add_data(symbol, df, market)
            engine.run()

            results = engine.get_results()
            all_results[strategy_name][display_name] = results
            all_equity_curves[strategy_name][display_name] = engine.get_equity_curve()

            _print_results(results)

    storage.close()

    # 生成图表报告
    output_dir = args.output_dir
    logger.info("正在生成图表报告到 %s ...", output_dir)
    generated_files = generate_multi_strategy_report(
        all_results, all_equity_curves, output_dir
    )

    print(f"\n{'='*60}")
    print("  批量回测完成")
    print(f"{'='*60}")
    print(f"  标的数量:       {len(stock_data_cache)}")
    print(f"  策略数量:       {len(strategy_names)}")
    print(f"  总回测组合:     {len(stock_data_cache) * len(strategy_names)}")
    print(f"  生成图表数量:   {len(generated_files)}")
    print(f"  图表输出目录:   {output_dir}")
    print(f"{'='*60}")

    for filepath in generated_files:
        print(f"  📊 {filepath}")
    print()

    # 保存结果到 JSON
    results_json_path = f"{output_dir}/backtest_results.json"
    with open(results_json_path, "w", encoding="utf-8") as json_file:
        json.dump(all_results, json_file, ensure_ascii=False, indent=2)
    print(f"  📄 结果数据: {results_json_path}\n")

def run_optimize(args: argparse.Namespace) -> None:
    """执行策略参数优化"""
    import json

    DataSourceFactory.register_defaults()
    market = _MARKET_MAP.get(args.market, MarketType.A_SHARE)

    datasource_name = args.datasource
    if market != MarketType.A_SHARE and datasource_name == "baostock":
        logger.info("市场类型为 %s，自动切换到 akshare 数据源", market.value)
        datasource_name = "akshare"

    data_source = DataSourceFactory.get(datasource_name)

    logger.info("正在获取 %s 的历史数据 (%s ~ %s)...", args.symbol, args.start_date, args.end_date)
    df = data_source.fetch_daily_bars(args.symbol, args.start_date, args.end_date, market)

    if df.empty:
        logger.error("未获取到数据，请检查标的代码和日期范围")
        return

    logger.info("获取到 %d 条K线数据", len(df))

    strategy_class = _STRATEGY_REGISTRY.get(args.strategy)
    if strategy_class is None:
        logger.error("未知策略: %s，可用策略: %s", args.strategy, list(_STRATEGY_REGISTRY.keys()))
        return

    # 准备数据源
    data_feeds = [(args.symbol, df, market)]

    # 加载基准数据
    benchmark_data = None
    benchmark_symbol = None
    if args.benchmark:
        logger.info("正在获取基准 %s 的历史数据...", args.benchmark)
        try:
            benchmark_data = data_source.fetch_daily_bars(args.benchmark, args.start_date, args.end_date, market)
            benchmark_symbol = args.benchmark
            if benchmark_data.empty:
                logger.warning("未获取到基准数据: %s", args.benchmark)
                benchmark_data = None
            else:
                logger.info("基准数据加载成功: %d 条", len(benchmark_data))
        except Exception as exc:
            logger.warning("获取基准数据失败: %s", exc)

    # 创建优化器
    optimizer = ParameterOptimizer(
        strategy_class=strategy_class,
        data_feeds=data_feeds,
        initial_capital=args.capital,
        commission_rate=args.commission,
        slippage_rate=args.slippage,
        benchmark_data=benchmark_data,
        benchmark_symbol=benchmark_symbol,
    )

    # 解析参数空间：格式为 "name:type:low:high:step" 或 "name:choice:val1,val2,val3"
    for param_str in args.params:
        parts = param_str.split(":")
        if len(parts) < 3:
            logger.error("参数格式错误: %s，应为 name:type:low:high[:step] 或 name:choice:val1,val2,...", param_str)
            continue

        param_name = parts[0]
        param_type = parts[1]

        if param_type == "choice":
            choices_str = parts[2]
            choices = []
            for val in choices_str.split(","):
                val = val.strip()
                try:
                    choices.append(int(val))
                except ValueError:
                    try:
                        choices.append(float(val))
                    except ValueError:
                        choices.append(val)
            optimizer.add_parameter(param_name, "choice", choices=choices)
        elif param_type in ("int", "float"):
            if len(parts) < 4:
                logger.error("数值参数需要 low 和 high: %s", param_str)
                continue
            low = float(parts[2])
            high = float(parts[3])
            step = float(parts[4]) if len(parts) > 4 else None
            optimizer.add_parameter(param_name, param_type, low=low, high=high, step=step)
        else:
            logger.error("未知参数类型: %s，支持 int/float/choice", param_type)
            continue

    # 执行优化
    target_metric = args.target_metric
    maximize = not args.minimize

    print(f"\n{'='*60}")
    print(f"  参数优化: {args.strategy}")
    print(f"  目标指标: {target_metric} ({'最大化' if maximize else '最小化'})")
    print(f"  搜索方式: {args.search_method}")
    print(f"{'='*60}\n")

    if args.search_method == "grid":
        optimizer.grid_search(target_metric, maximize=maximize)
    else:
        optimizer.random_search(target_metric, num_trials=args.num_trials, maximize=maximize)

    # 输出结果
    results = optimizer.get_results()
    if not results:
        logger.error("优化未产生任何结果")
        return

    best_params = optimizer.get_best_params()
    best_result = results[0]

    print(f"\n{'='*60}")
    print("  优化结果")
    print(f"{'='*60}")
    print(f"  总组合数:       {len(results):>14d}")
    print(f"  最优 {target_metric}: {best_result.target_value:>14.4f}")
    print(f"  最优参数:")
    for param_name, param_value in best_params.items():
        print(f"    {param_name}: {param_value}")
    print(f"{'='*60}")

    # 输出 Top N 结果
    top_n = min(args.top_n, len(results))
    print(f"\n  Top {top_n} 参数组合:")
    print("-" * 60)
    for rank, result in enumerate(results[:top_n], 1):
        params_str = ", ".join(f"{k}={v}" for k, v in result.params.items())
        print(f"  #{rank:2d}  {target_metric}={result.target_value:>10.4f}  |  {params_str}")
    print("-" * 60)

    # 保存结果
    output_dir = args.output_dir
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results_df = optimizer.get_results_dataframe()
    csv_path = f"{output_dir}/optimization_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  📄 详细结果: {csv_path}")

    # 用最优参数运行一次完整回测并输出
    print(f"\n{'='*60}")
    print("  最优参数回测结果")
    print(f"{'='*60}")
    _print_results(best_result.metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenQuant 量化交易系统")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 回测命令
    backtest_parser = subparsers.add_parser("backtest", help="历史回测")
    backtest_parser.add_argument("--symbol", required=True, help="标的代码 (如 600519)")
    backtest_parser.add_argument("--start-date", required=True, help="开始日期 (YYYY-MM-DD)")
    backtest_parser.add_argument("--end-date", required=True, help="结束日期 (YYYY-MM-DD)")
    backtest_parser.add_argument("--strategy", default="ma_cross", choices=list(_STRATEGY_REGISTRY.keys()), help="策略名称")
    backtest_parser.add_argument("--datasource", default="baostock", help="数据源名称")
    backtest_parser.add_argument("--market", default="a_share", choices=list(_MARKET_MAP.keys()), help="市场类型")
    backtest_parser.add_argument("--capital", type=float, default=100000, help="初始资金")
    backtest_parser.add_argument("--commission", type=float, default=0.0003, help="佣金费率")
    backtest_parser.add_argument("--slippage", type=float, default=0.001, help="滑点费率")
    backtest_parser.add_argument("--db-path", default="data/openquant.db", help="数据库路径")
    backtest_parser.add_argument("--benchmark", default=None, help="基准标的代码 (如 sh.000300 沪深300)")
    backtest_parser.add_argument("--output-dir", default="output/charts", help="图表输出目录")
    backtest_parser.add_argument("--with-events", action="store_true", help="启用事件因子（财报、分红、大宗交易等）")

    # 模拟交易命令
    sim_parser = subparsers.add_parser("simulate", help="模拟交易")
    sim_parser.add_argument("--symbols", nargs="+", required=True, help="监控标的列表")
    sim_parser.add_argument("--strategy", default="ma_cross", choices=list(_STRATEGY_REGISTRY.keys()), help="策略名称")
    sim_parser.add_argument("--datasource", default="akshare", help="数据源名称")
    sim_parser.add_argument("--market", default="a_share", choices=list(_MARKET_MAP.keys()), help="市场类型")
    sim_parser.add_argument("--capital", type=float, default=100000, help="初始资金")
    sim_parser.add_argument("--commission", type=float, default=0.0003, help="佣金费率")
    sim_parser.add_argument("--slippage", type=float, default=0.001, help="滑点费率")
    sim_parser.add_argument("--interval", type=int, default=60, help="轮询间隔(秒)")
    sim_parser.add_argument("--max-rounds", type=int, default=0, help="最大轮询次数，0表示无限循环")
    sim_parser.add_argument("--db-path", default="data/openquant.db", help="数据库路径")

    # 批量回测命令
    batch_parser = subparsers.add_parser("batch_backtest", help="批量回测（多标的 × 多策略）")
    batch_parser.add_argument(
        "--stocks", nargs="+", required=True,
        help="标的列表，格式: market:symbol:name (如 hk_stock:00700:腾讯)",
    )
    batch_parser.add_argument(
        "--strategies", nargs="+", default=None,
        help="策略列表（默认使用全部策略），可选: " + ", ".join(_STRATEGY_REGISTRY.keys()),
    )
    batch_parser.add_argument("--start-date", required=True, help="开始日期 (YYYY-MM-DD)")
    batch_parser.add_argument("--end-date", required=True, help="结束日期 (YYYY-MM-DD)")
    batch_parser.add_argument("--datasource", default="akshare", help="数据源名称")
    batch_parser.add_argument("--capital", type=float, default=100000, help="初始资金")
    batch_parser.add_argument("--commission", type=float, default=0.0003, help="佣金费率")
    batch_parser.add_argument("--slippage", type=float, default=0.001, help="滑点费率")
    batch_parser.add_argument("--db-path", default="data/openquant.db", help="数据库路径")
    batch_parser.add_argument("--output-dir", default="output/charts", help="图表输出目录")

    # 参数优化命令
    optimize_parser = subparsers.add_parser("optimize", help="策略参数优化")
    optimize_parser.add_argument("--symbol", required=True, help="标的代码 (如 600519)")
    optimize_parser.add_argument("--start-date", required=True, help="开始日期 (YYYY-MM-DD)")
    optimize_parser.add_argument("--end-date", required=True, help="结束日期 (YYYY-MM-DD)")
    optimize_parser.add_argument("--strategy", required=True, choices=list(_STRATEGY_REGISTRY.keys()), help="策略名称")
    optimize_parser.add_argument(
        "--params", nargs="+", required=True,
        help="参数空间定义，格式: name:type:low:high[:step] 或 name:choice:val1,val2,... "
             "(如 short_window:int:3:20:1 long_window:int:10:60:5 position_ratio:float:0.5:1.0:0.1)",
    )
    optimize_parser.add_argument("--search-method", default="grid", choices=["grid", "random"], help="搜索方式")
    optimize_parser.add_argument("--num-trials", type=int, default=100, help="随机搜索试验次数")
    optimize_parser.add_argument("--target-metric", default="sharpe_ratio", help="优化目标指标 (如 sharpe_ratio, total_return, calmar_ratio)")
    optimize_parser.add_argument("--minimize", action="store_true", help="最小化目标指标（默认最大化）")
    optimize_parser.add_argument("--top-n", type=int, default=10, help="输出 Top N 结果")
    optimize_parser.add_argument("--datasource", default="baostock", help="数据源名称")
    optimize_parser.add_argument("--market", default="a_share", choices=list(_MARKET_MAP.keys()), help="市场类型")
    optimize_parser.add_argument("--capital", type=float, default=100000, help="初始资金")
    optimize_parser.add_argument("--commission", type=float, default=0.0003, help="佣金费率")
    optimize_parser.add_argument("--slippage", type=float, default=0.001, help="滑点费率")
    optimize_parser.add_argument("--benchmark", default=None, help="基准标的代码")
    optimize_parser.add_argument("--output-dir", default="output/optimize", help="结果输出目录")

    args = parser.parse_args()

    if args.command == "backtest":
        run_backtest(args)
    elif args.command == "simulate":
        run_simulation(args)
    elif args.command == "batch_backtest":
        run_batch_backtest(args)
    elif args.command == "optimize":
        run_optimize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
