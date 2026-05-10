"""股票筛选推荐模块 - 滚动窗口验证版（Walk-Forward Analysis）

基于多策略 × 多次滚动回测，统计信号一致性来评估买入/卖出概率。
核心逻辑：
1. 获取每只股票近35周（约8个月）的历史数据
2. 用15次滚动窗口（训练20周 + 验证1周）逐周滚动
3. 每次滚动中，用所有策略分析验证周末是否有买入/卖出信号
4. 信号一致性 = 策略在N次滚动中发出信号的比率
5. 综合信号一致性、回测绩效指标计算最终买入/卖出概率
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from openquant.core.models import MarketType
from openquant.datasource.factory import DataSourceFactory
from openquant.engine.backtest_engine import BacktestEngine
from openquant.storage.sqlite_storage import SqliteStorage

logger = logging.getLogger(__name__)


@dataclass
class StrategyRollingResult:
    """单个策略在滚动验证中的汇总结果"""
    strategy_name: str
    buy_signal_ratio: float = 0.0       # 买入信号出现比率 (0~1)
    sell_signal_ratio: float = 0.0      # 卖出信号出现比率 (0~1)
    latest_buy_signal: bool = False     # 最新一轮是否有买入信号
    latest_sell_signal: bool = False    # 最新一轮是否有卖出信号
    avg_return: float = 0.0            # 各轮平均收益率
    avg_win_rate: float = 0.0          # 各轮平均胜率
    avg_sharpe: float = 0.0            # 各轮平均夏普比率
    avg_max_drawdown: float = 0.0      # 各轮平均最大回撤
    total_rolls: int = 0               # 总滚动次数


@dataclass
class StockRecommendation:
    """股票推荐结果"""
    symbol: str
    display_name: str
    market: MarketType
    buy_probability: float = 0.0
    sell_probability: float = 0.0
    buy_consistency: float = 0.0       # 买入信号一致性 (0~1)
    sell_consistency: float = 0.0      # 卖出信号一致性 (0~1)
    composite_score: float = 0.0
    total_strategies: int = 0
    rolling_rounds: int = 0            # 滚动验证轮数
    strategy_results: list[StrategyRollingResult] = field(default_factory=list)
    latest_close: float = 0.0


class StockScreener:
    """股票筛选推荐器 - 滚动窗口验证版"""

    def __init__(
        self,
        strategy_registry: dict,
        datasource_name: str = "akshare",
        initial_capital: float = 100000.0,
        train_weeks: int = 20,
        rolling_rounds: int = 15,
    ):
        """
        Args:
            strategy_registry: 策略注册表 {name: class}
            datasource_name: 数据源名称
            initial_capital: 回测初始资金
            train_weeks: 每轮训练窗口周数（默认20周≈5个月）
            rolling_rounds: 滚动验证轮数（默认15轮）
        """
        self.strategy_registry = strategy_registry
        self.datasource_name = datasource_name
        self.initial_capital = initial_capital
        self.train_weeks = train_weeks
        self.rolling_rounds = rolling_rounds
        # 总数据需求 = 训练窗口 + 滚动轮数（每轮步长1周）
        self.total_weeks_needed = train_weeks + rolling_rounds

    def screen_stocks(
        self,
        stock_configs: list[tuple[MarketType, str, str]],
        end_date: str | None = None,
    ) -> list[StockRecommendation]:
        """筛选推荐股票（滚动窗口验证）

        Args:
            stock_configs: 股票配置列表 [(market, symbol, display_name), ...]
            end_date: 结束日期，默认为今天

        Returns:
            按买入概率降序排列的推荐列表
        """
        DataSourceFactory.register_defaults()

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d")
            - timedelta(weeks=self.total_weeks_needed)
        ).strftime("%Y-%m-%d")

        logger.info(
            "股票筛选（滚动验证）：数据区间 %s ~ %s（%d周），"
            "训练窗口=%d周，滚动%d轮",
            start_date, end_date, self.total_weeks_needed,
            self.train_weeks, self.rolling_rounds,
        )

        recommendations: list[StockRecommendation] = []

        for idx, (market, symbol, display_name) in enumerate(stock_configs):
            logger.info("正在分析 %s (%s)...", display_name, symbol)
            recommendation = self._analyze_stock_rolling(
                market, symbol, display_name, start_date, end_date
            )
            if recommendation is not None:
                recommendations.append(recommendation)

            if idx < len(stock_configs) - 1:
                time.sleep(1.5)

        recommendations.sort(key=lambda r: r.buy_probability, reverse=True)
        return recommendations

    def _fetch_with_retry(
        self,
        data_source,
        symbol: str,
        start_date: str,
        end_date: str,
        market: MarketType,
        display_name: str,
        max_retries: int = 3,
        retry_delay: float = 3.0,
    ) -> pd.DataFrame | None:
        """带重试的数据获取"""
        for attempt in range(1, max_retries + 1):
            try:
                df = data_source.fetch_daily_bars(symbol, start_date, end_date, market)
                return df
            except Exception as exc:
                error_msg = str(exc)
                is_connection_error = any(
                    keyword in error_msg
                    for keyword in ["Connection aborted", "RemoteDisconnected", "ConnectionReset", "timeout"]
                )
                if is_connection_error and attempt < max_retries:
                    wait_time = retry_delay * attempt
                    logger.warning(
                        "获取 %s 数据连接失败 (第%d次)，%.1f秒后重试: %s",
                        display_name, attempt, wait_time, exc,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("获取 %s 数据失败: %s", display_name, exc)
                    return None
        return None

    def _analyze_stock_rolling(
        self,
        market: MarketType,
        symbol: str,
        display_name: str,
        start_date: str,
        end_date: str,
    ) -> StockRecommendation | None:
        """对单只股票执行滚动窗口验证分析"""
        datasource_name = self.datasource_name
        if market != MarketType.A_SHARE and datasource_name == "baostock":
            datasource_name = "akshare"

        data_source = DataSourceFactory.get(datasource_name)
        df = self._fetch_with_retry(data_source, symbol, start_date, end_date, market, display_name)

        if df is None:
            return None

        min_required = self.train_weeks * 4  # 至少需要训练窗口的数据量（约按4天/周估算）
        if df.empty or len(df) < min_required:
            logger.warning("数据不足，跳过 %s（需要约%d条，仅%d条）", display_name, min_required, len(df))
            return None

        latest_close = float(df.iloc[-1]["close"])

        # 生成滚动窗口切片
        rolling_slices = self._generate_rolling_slices(df)
        actual_rounds = len(rolling_slices)

        if actual_rounds < 3:
            logger.warning("有效滚动轮数不足，跳过 %s（仅%d轮）", display_name, actual_rounds)
            return None

        # 对每个策略执行滚动分析
        strategies = {k: v for k, v in self.strategy_registry.items() if k != "event_ma_cross"}
        strategy_results: list[StrategyRollingResult] = []

        for strategy_name, strategy_class in strategies.items():
            result = self._run_strategy_rolling(
                strategy_name, strategy_class, symbol, rolling_slices, market
            )
            strategy_results.append(result)

        # 构建推荐结果
        recommendation = StockRecommendation(
            symbol=symbol,
            display_name=display_name,
            market=market,
            latest_close=latest_close,
            total_strategies=len(strategy_results),
            rolling_rounds=actual_rounds,
            strategy_results=strategy_results,
        )

        self._calculate_probabilities(recommendation)
        return recommendation

    def _generate_rolling_slices(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        """生成滚动窗口数据切片

        每个切片包含 train_weeks 周的训练数据（用于回测和信号检测）。
        滚动步长 = 1 周（约5个交易日）。
        """
        total_bars = len(df)
        step_size = 5  # 约1周的交易日
        train_size = self.train_weeks * 5  # 约训练窗口的交易日数

        # 如果数据不够训练窗口，调小训练窗口
        if total_bars < train_size:
            train_size = total_bars - step_size

        slices: list[pd.DataFrame] = []
        # 从末尾往前滚动，最新的一轮排在最后
        end_positions = []
        for roll_idx in range(self.rolling_rounds):
            end_pos = total_bars - roll_idx * step_size
            start_pos = end_pos - train_size
            if start_pos < 0:
                break
            end_positions.append((start_pos, end_pos))

        # 反转，使时间顺序从早到晚
        end_positions.reverse()

        for start_pos, end_pos in end_positions:
            slices.append(df.iloc[start_pos:end_pos].reset_index(drop=True))

        return slices

    def _run_strategy_rolling(
        self,
        strategy_name: str,
        strategy_class,
        symbol: str,
        rolling_slices: list[pd.DataFrame],
        market: MarketType,
    ) -> StrategyRollingResult:
        """对单个策略执行滚动验证，统计信号一致性"""
        buy_signals: list[bool] = []
        sell_signals: list[bool] = []
        returns: list[float] = []
        win_rates: list[float] = []
        sharpes: list[float] = []
        drawdowns: list[float] = []

        for slice_df in rolling_slices:
            buy_signal, sell_signal, metrics = self._evaluate_single_round(
                strategy_name, strategy_class, symbol, slice_df, market
            )
            buy_signals.append(buy_signal)
            sell_signals.append(sell_signal)
            if metrics:
                returns.append(metrics.get("total_return", 0.0))
                win_rates.append(metrics.get("win_rate", 0.0))
                sharpes.append(metrics.get("sharpe_ratio", 0.0))
                drawdowns.append(abs(metrics.get("max_drawdown", 0.0)))

        total_rolls = len(buy_signals)
        if total_rolls == 0:
            return StrategyRollingResult(strategy_name=strategy_name)

        return StrategyRollingResult(
            strategy_name=strategy_name,
            buy_signal_ratio=sum(buy_signals) / total_rolls,
            sell_signal_ratio=sum(sell_signals) / total_rolls,
            latest_buy_signal=buy_signals[-1] if buy_signals else False,
            latest_sell_signal=sell_signals[-1] if sell_signals else False,
            avg_return=sum(returns) / len(returns) if returns else 0.0,
            avg_win_rate=sum(win_rates) / len(win_rates) if win_rates else 0.0,
            avg_sharpe=sum(sharpes) / len(sharpes) if sharpes else 0.0,
            avg_max_drawdown=sum(drawdowns) / len(drawdowns) if drawdowns else 0.0,
            total_rolls=total_rolls,
        )

    def _evaluate_single_round(
        self,
        strategy_name: str,
        strategy_class,
        symbol: str,
        df: pd.DataFrame,
        market: MarketType,
    ) -> tuple[bool, bool, dict]:
        """评估单轮：回测 + 信号检测

        Returns:
            (has_buy_signal, has_sell_signal, metrics_dict)
        """
        from openquant.core.models import Bar, OrderSide, Portfolio

        if len(df) < 10:
            return False, False, {}

        # 回测获取绩效指标
        storage = SqliteStorage(":memory:")
        storage.initialize()
        metrics = {}
        try:
            strategy = strategy_class()
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                commission_rate=0.0003,
                slippage_rate=0.001,
                storage=storage,
            )
            engine.set_strategy(strategy)
            engine.add_data(symbol, df, market)
            engine.run()
            metrics = engine.get_results()
        except Exception as exc:
            logger.debug("策略 %s 单轮回测失败: %s", strategy_name, exc)
        finally:
            storage.close()

        # 信号检测：观察最后3天
        buy_signal = False
        sell_signal = False
        try:
            strategy = strategy_class()
            portfolio = Portfolio(initial_capital=self.initial_capital, cash=self.initial_capital)
            strategy.initialize(portfolio)

            recent_window = min(3, len(df))
            for idx in range(len(df)):
                row = df.iloc[idx]
                bar = Bar(
                    symbol=symbol,
                    datetime=pd.Timestamp(row["datetime"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    market=market,
                )
                orders = strategy.on_bar(bar, portfolio)

                if idx >= len(df) - recent_window:
                    for order in orders:
                        if order.side == OrderSide.BUY:
                            buy_signal = True
                        elif order.side == OrderSide.SELL:
                            sell_signal = True
        except Exception as exc:
            logger.debug("策略 %s 信号检测失败: %s", strategy_name, exc)

        return buy_signal, sell_signal, metrics

    def _calculate_probabilities(self, recommendation: StockRecommendation) -> None:
        """基于滚动验证结果计算买入/卖出概率

        概率公式：
        买入概率 = 信号一致性权重(35%) + 最新信号权重(30%) + 绩效权重(35%)
          - 信号一致性: 各策略买入信号比率的加权平均
          - 最新信号: 最新一轮有多少策略给出买入信号
          - 绩效: 收益率 + 胜率 + 夏普综合
        """
        results = recommendation.strategy_results
        if not results:
            return

        total = len(results)

        # === 买入概率计算 ===
        # 1. 信号一致性：各策略的买入信号比率平均值
        buy_consistency = sum(r.buy_signal_ratio for r in results) / total
        recommendation.buy_consistency = buy_consistency

        # 2. 最新信号覆盖率
        latest_buy_count = sum(1 for r in results if r.latest_buy_signal)
        latest_buy_ratio = latest_buy_count / total

        # 3. 绩效指标（以一致性高的策略为主）
        high_consistency_strategies = [r for r in results if r.buy_signal_ratio >= 0.3]
        if high_consistency_strategies:
            avg_return = sum(r.avg_return for r in high_consistency_strategies) / len(high_consistency_strategies)
            avg_win_rate = sum(r.avg_win_rate for r in high_consistency_strategies) / len(high_consistency_strategies)
            avg_sharpe = sum(r.avg_sharpe for r in high_consistency_strategies) / len(high_consistency_strategies)
        else:
            avg_return = sum(r.avg_return for r in results) / total
            avg_win_rate = sum(r.avg_win_rate for r in results) / total
            avg_sharpe = sum(r.avg_sharpe for r in results) / total

        # 归一化绩效
        return_score = min(max((avg_return + 15) / 30, 0), 1.0)
        win_rate_score = min(max(avg_win_rate / 100, 0), 1.0)
        sharpe_score = min(max((avg_sharpe + 1) / 4, 0), 1.0)
        performance_score = 0.4 * return_score + 0.35 * win_rate_score + 0.25 * sharpe_score

        # 加权综合
        buy_composite = (
            0.35 * buy_consistency
            + 0.30 * latest_buy_ratio
            + 0.35 * performance_score
        )
        recommendation.composite_score = buy_composite

        # 最终概率（一致性低时大幅惩罚）
        if buy_consistency < 0.1 and latest_buy_count == 0:
            recommendation.buy_probability = buy_composite * 0.1 * 100
        else:
            consistency_boost = min(1.0 + buy_consistency, 2.0)
            recommendation.buy_probability = min(buy_composite * consistency_boost * 100, 99.0)

        # === 卖出概率计算 ===
        sell_consistency = sum(r.sell_signal_ratio for r in results) / total
        recommendation.sell_consistency = sell_consistency

        latest_sell_count = sum(1 for r in results if r.latest_sell_signal)
        latest_sell_ratio = latest_sell_count / total

        # 卖出绩效（表现越差越该卖）
        high_sell_strategies = [r for r in results if r.sell_signal_ratio >= 0.3]
        if high_sell_strategies:
            sell_avg_return = sum(r.avg_return for r in high_sell_strategies) / len(high_sell_strategies)
            sell_avg_drawdown = sum(r.avg_max_drawdown for r in high_sell_strategies) / len(high_sell_strategies)
            sell_avg_sharpe = sum(r.avg_sharpe for r in high_sell_strategies) / len(high_sell_strategies)
        else:
            sell_avg_return = sum(r.avg_return for r in results) / total
            sell_avg_drawdown = sum(r.avg_max_drawdown for r in results) / total
            sell_avg_sharpe = sum(r.avg_sharpe for r in results) / total

        loss_score = min(max((-sell_avg_return + 15) / 30, 0), 1.0)
        drawdown_score = min(max(sell_avg_drawdown / 15, 0), 1.0)
        sharpe_inv_score = min(max((1 - sell_avg_sharpe) / 4, 0), 1.0)
        sell_perf = 0.4 * loss_score + 0.35 * drawdown_score + 0.25 * sharpe_inv_score

        sell_composite = (
            0.35 * sell_consistency
            + 0.30 * latest_sell_ratio
            + 0.35 * sell_perf
        )

        if sell_consistency < 0.1 and latest_sell_count == 0:
            recommendation.sell_probability = sell_composite * 0.1 * 100
        else:
            sell_boost = min(1.0 + sell_consistency, 2.0)
            recommendation.sell_probability = min(sell_composite * sell_boost * 100, 99.0)


def print_recommendations(recommendations: list[StockRecommendation]) -> None:
    """格式化输出推荐结果"""
    print("\n" + "=" * 80)
    print("  📊 股票多策略综合分析 - 滚动验证版（Walk-Forward Analysis）")
    print(f"  分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if recommendations:
        rounds = recommendations[0].rolling_rounds
        print(f"  验证方式: {rounds}轮滚动窗口验证")
    print("=" * 80)

    if not recommendations:
        print("  ⚠️ 无推荐结果")
        return

    # 综合表格
    print(f"\n  {'排名':<3} {'股票':<8} {'买入概率':<9} {'卖出概率':<9} {'买入一致性':<10} {'卖出一致性':<10} {'最新价':<9} {'建议':<8}")
    print("  " + "-" * 74)

    for rank, rec in enumerate(recommendations, 1):
        # 操作建议
        if rec.buy_probability >= 50 and rec.sell_probability < 20:
            action = "买入 📈"
        elif rec.sell_probability >= 50 and rec.buy_probability < 20:
            action = "卖出 📉"
        elif rec.buy_probability >= 30 and rec.sell_probability < 30:
            action = "偏多 🔼"
        elif rec.sell_probability >= 30 and rec.buy_probability < 30:
            action = "偏空 🔽"
        else:
            action = "观望 ⏸️"

        # emoji
        buy_emoji = "🟢" if rec.buy_probability >= 60 else ("🟡" if rec.buy_probability >= 40 else "⚪")
        sell_emoji = "🔴" if rec.sell_probability >= 60 else ("🟠" if rec.sell_probability >= 40 else "⚪")

        buy_cons_str = f"{rec.buy_consistency * 100:.0f}%"
        sell_cons_str = f"{rec.sell_consistency * 100:.0f}%"

        print(
            f"  {rank:<3}{rec.display_name:<8}"
            f"{buy_emoji}{rec.buy_probability:>5.1f}%   "
            f"{sell_emoji}{rec.sell_probability:>5.1f}%   "
            f"  {buy_cons_str:<9} {sell_cons_str:<9} "
            f"{rec.latest_close:>8.2f}  {action}"
        )

    # 详细分析
    print("\n" + "-" * 80)
    print("  📋 详细策略信号分析（滚动验证）")
    print("-" * 80)

    top_stocks = [r for r in recommendations if r.buy_probability >= 25][:5]
    if not top_stocks:
        top_stocks = recommendations[:3]

    for rec in top_stocks:
        print(
            f"\n  【{rec.display_name}】({rec.symbol})"
            f" - 买入: {rec.buy_probability:.1f}% | 卖出: {rec.sell_probability:.1f}%"
            f" | 滚动{rec.rolling_rounds}轮验证"
        )

        # 按买入信号一致性排序展示策略
        sorted_results = sorted(rec.strategy_results, key=lambda r: r.buy_signal_ratio, reverse=True)

        high_buy = [r for r in sorted_results if r.buy_signal_ratio >= 0.3]
        if high_buy:
            print(f"    📈 高一致性买入策略（信号比率≥30%）:")
            for sr in high_buy:
                print(
                    f"       - {sr.strategy_name}: "
                    f"信号比率={sr.buy_signal_ratio*100:.0f}%({sr.total_rolls}轮), "
                    f"最新={'✅买入' if sr.latest_buy_signal else '⏸️观望'}, "
                    f"均收益={sr.avg_return:.2f}%, "
                    f"均夏普={sr.avg_sharpe:.2f}"
                )

        high_sell = [r for r in sorted_results if r.sell_signal_ratio >= 0.3]
        if high_sell:
            print(f"    📉 高一致性卖出策略（信号比率≥30%）:")
            for sr in high_sell:
                print(
                    f"       - {sr.strategy_name}: "
                    f"信号比率={sr.sell_signal_ratio*100:.0f}%({sr.total_rolls}轮), "
                    f"最新={'🔴卖出' if sr.latest_sell_signal else '⏸️观望'}, "
                    f"均回撤={sr.avg_max_drawdown:.2f}%"
                )

        low_signal = [r for r in sorted_results if r.buy_signal_ratio < 0.3 and r.sell_signal_ratio < 0.3]
        if low_signal:
            names = ", ".join(r.strategy_name for r in low_signal[:4])
            remaining = len(low_signal) - 4
            if remaining > 0:
                names += f" 等{remaining}个"
            print(f"    ⏸️  低信号策略: {names}")

    # 风险提示
    high_sell_stocks = [r for r in recommendations if r.sell_probability >= 40]
    if high_sell_stocks:
        print(f"\n  ⚠️ 卖出风险提示:")
        for rec in high_sell_stocks:
            sell_strategies = [r for r in rec.strategy_results if r.sell_signal_ratio >= 0.3]
            names = ", ".join(r.strategy_name for r in sell_strategies[:3])
            print(
                f"    🔴 {rec.display_name}: 卖出概率 {rec.sell_probability:.1f}%, "
                f"一致性 {rec.sell_consistency*100:.0f}% (触发: {names})"
            )

    print("\n" + "=" * 80)
    print("  💡 说明: 信号一致性 = 策略在多轮滚动验证中重复给出相同信号的比率")
    print("     一致性越高表示信号越稳定可靠，非偶发性事件")
    print("  ⚠️ 免责声明: 以上分析仅供参考，不构成投资建议")
    print("=" * 80 + "\n")
