"""股票筛选推荐模块

基于多策略回测结果，综合评分筛选出当前最适合买入的股票。
核心逻辑：
1. 获取每只股票近9周的历史数据
2. 用所有策略分别回测，记录每个策略的买入信号
3. 分析最后一个交易日是否处于"买入信号"状态
4. 结合回测收益率、胜率、夏普比率等指标综合评分
5. 输出买入推荐列表及概率
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
class SignalResult:
    """单个策略对单只股票的信号结果"""
    strategy_name: str
    has_buy_signal: bool
    total_return: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    max_drawdown: float = 0.0


@dataclass
class StockRecommendation:
    """股票推荐结果"""
    symbol: str
    display_name: str
    market: MarketType
    buy_probability: float = 0.0
    composite_score: float = 0.0
    signal_count: int = 0
    total_strategies: int = 0
    strategy_signals: list[SignalResult] = field(default_factory=list)
    latest_close: float = 0.0


class StockScreener:
    """股票筛选推荐器"""

    def __init__(
        self,
        strategy_registry: dict,
        datasource_name: str = "akshare",
        initial_capital: float = 100000.0,
        lookback_weeks: int = 9,
    ):
        self.strategy_registry = strategy_registry
        self.datasource_name = datasource_name
        self.initial_capital = initial_capital
        self.lookback_weeks = lookback_weeks

    def screen_stocks(
        self,
        stock_configs: list[tuple[MarketType, str, str]],
        end_date: str | None = None,
    ) -> list[StockRecommendation]:
        """筛选推荐股票

        Args:
            stock_configs: 股票配置列表 [(market, symbol, display_name), ...]
            end_date: 结束日期，默认为今天

        Returns:
            按买入概率降序排列的推荐列表
        """
        DataSourceFactory.register_defaults()

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(weeks=self.lookback_weeks)).strftime("%Y-%m-%d")

        logger.info("股票筛选：回测区间 %s ~ %s（%d周）", start_date, end_date, self.lookback_weeks)

        recommendations: list[StockRecommendation] = []

        for idx, (market, symbol, display_name) in enumerate(stock_configs):
            logger.info("正在分析 %s (%s)...", display_name, symbol)
            recommendation = self._analyze_single_stock(
                market, symbol, display_name, start_date, end_date
            )
            if recommendation is not None:
                recommendations.append(recommendation)

            # 请求间隔，避免被远端断开连接
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
        """带重试的数据获取，处理远端断开连接问题"""
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
                        "获取 %s 数据连接失败 (第%d次)，%0.1f秒后重试: %s",
                        display_name, attempt, wait_time, exc,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("获取 %s 数据失败: %s", display_name, exc)
                    return None
        return None

    def _analyze_single_stock(
        self,
        market: MarketType,
        symbol: str,
        display_name: str,
        start_date: str,
        end_date: str,
    ) -> StockRecommendation | None:
        """分析单只股票"""
        datasource_name = self.datasource_name
        if market != MarketType.A_SHARE and datasource_name == "baostock":
            datasource_name = "akshare"

        data_source = DataSourceFactory.get(datasource_name)

        df = self._fetch_with_retry(data_source, symbol, start_date, end_date, market, display_name)
        if df is None:
            return None

        if df.empty or len(df) < 10:
            logger.warning("数据不足，跳过 %s（仅 %d 条）", display_name, len(df))
            return None

        latest_close = df.iloc[-1]["close"]
        recommendation = StockRecommendation(
            symbol=symbol,
            display_name=display_name,
            market=market,
            latest_close=latest_close,
            total_strategies=len(self.strategy_registry),
        )

        signal_results: list[SignalResult] = []

        for strategy_name, strategy_class in self.strategy_registry.items():
            if strategy_name == "event_ma_cross":
                continue  # 跳过需要事件数据的策略

            signal_result = self._run_strategy_analysis(
                strategy_name, strategy_class, symbol, df, market
            )
            signal_results.append(signal_result)

        recommendation.strategy_signals = signal_results
        recommendation.total_strategies = len(signal_results)
        recommendation.signal_count = sum(1 for s in signal_results if s.has_buy_signal)

        self._calculate_composite_score(recommendation)
        return recommendation

    def _run_strategy_analysis(
        self,
        strategy_name: str,
        strategy_class,
        symbol: str,
        df: pd.DataFrame,
        market: MarketType,
    ) -> SignalResult:
        """运行单个策略的分析"""
        storage = SqliteStorage(":memory:")
        storage.initialize()

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

            results = engine.get_results()

            # 检测最后几天是否有买入信号
            has_buy_signal = self._detect_latest_buy_signal(
                strategy_name, strategy_class, symbol, df, market
            )

            return SignalResult(
                strategy_name=strategy_name,
                has_buy_signal=has_buy_signal,
                total_return=results.get("total_return", 0.0),
                win_rate=results.get("win_rate", 0.0),
                sharpe_ratio=results.get("sharpe_ratio", 0.0),
                total_trades=results.get("total_trades", 0),
                max_drawdown=results.get("max_drawdown", 0.0),
            )
        except Exception as exc:
            logger.warning("策略 %s 回测 %s 失败: %s", strategy_name, symbol, exc)
            return SignalResult(strategy_name=strategy_name, has_buy_signal=False)
        finally:
            storage.close()

    def _detect_latest_buy_signal(
        self,
        strategy_name: str,
        strategy_class,
        symbol: str,
        df: pd.DataFrame,
        market: MarketType,
    ) -> bool:
        """检测最后交易日是否处于买入信号状态

        方法：用数据的最后3天分别作为结束点，观察策略是否产生买入订单
        """
        from openquant.core.models import Bar, OrderSide, Portfolio

        if len(df) < 5:
            return False

        strategy = strategy_class()
        portfolio = Portfolio(initial_capital=self.initial_capital, cash=self.initial_capital)
        strategy.initialize(portfolio)

        buy_signal_in_recent_days = False
        recent_window = min(3, len(df))

        # 模拟喂入所有数据，观察最后几天是否有买入信号
        for idx in range(len(df)):
            row = df.iloc[idx]
            bar = Bar(
                symbol=symbol,
                datetime=pd.Timestamp(row["datetime"]),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                market=market,
            )
            orders = strategy.on_bar(bar, portfolio)

            # 只关心最后几天的信号
            if idx >= len(df) - recent_window:
                for order in orders:
                    if order.side == OrderSide.BUY:
                        buy_signal_in_recent_days = True
                        break

        return buy_signal_in_recent_days

    def _calculate_composite_score(self, recommendation: StockRecommendation) -> None:
        """计算综合评分和买入概率

        评分因素：
        1. 信号覆盖率（多少策略给出买入信号）- 权重40%
        2. 有信号策略的平均收益率 - 权重25%
        3. 有信号策略的平均胜率 - 权重20%
        4. 有信号策略的平均夏普比率 - 权重15%
        """
        signals = recommendation.strategy_signals
        if not signals:
            recommendation.buy_probability = 0.0
            recommendation.composite_score = 0.0
            return

        total = recommendation.total_strategies
        signal_count = recommendation.signal_count

        # 信号覆盖率得分 [0, 1]
        coverage_score = signal_count / total if total > 0 else 0.0

        # 对有买入信号的策略计算加权指标
        bullish_signals = [s for s in signals if s.has_buy_signal]

        if bullish_signals:
            avg_return = sum(s.total_return for s in bullish_signals) / len(bullish_signals)
            avg_win_rate = sum(s.win_rate for s in bullish_signals) / len(bullish_signals)
            avg_sharpe = sum(s.sharpe_ratio for s in bullish_signals) / len(bullish_signals)
        else:
            # 没有买入信号时，用全部策略的平均值（通常偏低）
            avg_return = sum(s.total_return for s in signals) / len(signals)
            avg_win_rate = sum(s.win_rate for s in signals) / len(signals)
            avg_sharpe = sum(s.sharpe_ratio for s in signals) / len(signals)

        # 归一化各指标到 [0, 1] 区间
        return_score = min(max((avg_return + 20) / 40, 0), 1.0)  # [-20%, 20%] -> [0, 1]
        win_rate_score = min(max(avg_win_rate / 100, 0), 1.0)    # [0, 100] -> [0, 1]
        sharpe_score = min(max((avg_sharpe + 1) / 4, 0), 1.0)    # [-1, 3] -> [0, 1]

        # 加权综合评分
        composite_score = (
            0.40 * coverage_score
            + 0.25 * return_score
            + 0.20 * win_rate_score
            + 0.15 * sharpe_score
        )

        recommendation.composite_score = composite_score

        # 买入概率：综合评分 × 信号增强
        # 如果没有任何策略给出买入信号，概率大幅降低
        if signal_count == 0:
            recommendation.buy_probability = composite_score * 0.1 * 100
        else:
            signal_boost = min(1.0 + (signal_count / total) * 0.5, 1.5)
            recommendation.buy_probability = min(composite_score * signal_boost * 100, 99.0)


def print_recommendations(recommendations: list[StockRecommendation]) -> None:
    """格式化输出推荐结果"""
    print("\n" + "=" * 70)
    print("  📊 股票买入推荐 - 基于多策略综合分析")
    print(f"  分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    if not recommendations:
        print("  ⚠️ 无推荐结果")
        return

    print(f"\n  {'排名':<4} {'股票':<12} {'买入概率':<10} {'信号数':<8} {'最新价':<10} {'综合评分':<8}")
    print("  " + "-" * 62)

    for rank, rec in enumerate(recommendations, 1):
        prob_bar = "█" * int(rec.buy_probability / 5) + "░" * (20 - int(rec.buy_probability / 5))
        signal_str = f"{rec.signal_count}/{rec.total_strategies}"

        # 信号强度标识
        if rec.buy_probability >= 60:
            emoji = "🟢"
        elif rec.buy_probability >= 40:
            emoji = "🟡"
        else:
            emoji = "🔴"

        print(
            f"  {emoji} {rank:<3} {rec.display_name:<10} "
            f"{rec.buy_probability:>5.1f}%     {signal_str:<7} "
            f"{rec.latest_close:>8.2f}   {rec.composite_score:.3f}"
        )

    # 详细信号分析
    print("\n" + "-" * 70)
    print("  📋 详细策略信号分析")
    print("-" * 70)

    top_stocks = [r for r in recommendations if r.buy_probability >= 30][:5]
    if not top_stocks:
        top_stocks = recommendations[:3]

    for rec in top_stocks:
        print(f"\n  【{rec.display_name}】({rec.symbol}) - 买入概率: {rec.buy_probability:.1f}%")
        bullish_strategies = [s for s in rec.strategy_signals if s.has_buy_signal]
        if bullish_strategies:
            print(f"    ✅ 发出买入信号的策略:")
            for sig in bullish_strategies:
                print(
                    f"       - {sig.strategy_name}: "
                    f"收益率={sig.total_return:.2f}%, "
                    f"胜率={sig.win_rate:.1f}%, "
                    f"夏普={sig.sharpe_ratio:.3f}"
                )
        else:
            print("    ❌ 无策略发出买入信号")

        bearish_strategies = [s for s in rec.strategy_signals if not s.has_buy_signal and s.total_trades > 0]
        if bearish_strategies:
            names = ", ".join(s.strategy_name for s in bearish_strategies[:4])
            remaining = len(bearish_strategies) - 4
            if remaining > 0:
                names += f" 等{remaining}个"
            print(f"    ⏸️  观望策略: {names}")

    print("\n" + "=" * 70)
    print("  ⚠️ 免责声明: 以上分析仅供参考，不构成投资建议")
    print("=" * 70 + "\n")
