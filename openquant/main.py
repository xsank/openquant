"""OpenQuant 入口文件

提供命令行接口，支持回测和模拟交易。
"""
from __future__ import annotations

import argparse
import sys

from openquant.core.models import MarketType
from openquant.datasource.factory import DataSourceFactory
from openquant.engine.backtest_engine import BacktestEngine
from openquant.storage.sqlite_storage import SqliteStorage
from openquant.strategy.ma_cross import MACrossStrategy
from openquant.strategy.macd_strategy import MACDStrategy
from openquant.utils.logger import setup_logger

logger = setup_logger()

_STRATEGY_REGISTRY = {
    "ma_cross": MACrossStrategy,
    "macd": MACDStrategy,
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
    data_source = DataSourceFactory.get(args.datasource)

    market = _MARKET_MAP.get(args.market, MarketType.A_SHARE)

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

    # 运行回测
    engine.run()

    # 输出结果
    results = engine.get_results()
    _print_results(results)

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
    print("=" * 60 + "\n")


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

    args = parser.parse_args()

    if args.command == "backtest":
        run_backtest(args)
    elif args.command == "simulate":
        run_simulation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
