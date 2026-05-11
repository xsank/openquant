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
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from openquant.core.models import MarketType, Position
from openquant.datasource.factory import DataSourceFactory
from openquant.engine.backtest_engine import BacktestEngine
from openquant.risk.stop_loss import StopLossConfig
from openquant.storage.sqlite_storage import SqliteStorage

logger = logging.getLogger(__name__)


def _display_width(text: str) -> int:
    """计算字符串在终端中的显示宽度（中文占2格，emoji占2格）"""
    width = 0
    for char in text:
        code_point = ord(char)
        # emoji 范围（常见 emoji 占2格）
        if code_point >= 0x1F300:
            width += 2
        elif unicodedata.east_asian_width(char) in ("W", "F"):
            width += 2
        else:
            width += 1
    return width


def _pad_to_width(text: str, target_width: int, align: str = "left") -> str:
    """将字符串填充到指定显示宽度"""
    current_width = _display_width(text)
    padding = max(0, target_width - current_width)
    if align == "left":
        return text + " " * padding
    elif align == "right":
        return " " * padding + text
    else:
        left_pad = padding // 2
        return " " * left_pad + text + " " * (padding - left_pad)


@dataclass
class StrategyRollingResult:
    """单个策略在滚动验证中的汇总结果"""
    strategy_name: str
    buy_signal_ratio: float = 0.0       # 买入信号出现比率 (0~1)
    sell_signal_ratio: float = 0.0      # 卖出信号出现比率 (0~1)
    latest_buy_signal: bool = False     # 最新一轮是否有买入信号
    latest_sell_signal: bool = False    # 最新一轮是否有卖出信号
    avg_return: float = 0.0            # 各轮平均收益率
    sum_return: float = 0.0            # 各轮累加总收益率
    avg_win_rate: float = 0.0          # 各轮平均胜率（日胜率）
    avg_sharpe: float = 0.0            # 各轮平均夏普比率
    avg_max_drawdown: float = 0.0      # 各轮平均最大回撤
    total_rolls: int = 0               # 总滚动次数
    avg_trade_win_rate: float = 0.0    # 各轮平均交易胜率 (盈利交易/总交易)
    avg_profit_factor: float = 0.0     # 各轮平均盈亏比 (总盈利/总亏损)
    avg_holding_days: float = 0.0      # 各轮平均持仓天数
    avg_trade_frequency: float = 0.0   # 各轮平均交易频率 (次/周)
    avg_win_pct: float = 0.0           # 平均每笔盈利交易收益率 (%)
    avg_loss_pct: float = 0.0          # 平均每笔亏损交易亏损率 (%)
    expected_value: float = 0.0        # 收益期望 = 胜率*平均盈利 - (1-胜率)*平均亏损


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
    backtest_win_rate: float = 0.0     # 历史回测总胜率 (各策略加权平均)
    backtest_return: float = 0.0       # 历史回测总收益率 (各策略加权平均)
    trade_win_rate: float = 0.0        # 交易胜率 (盈利交易笔数/总交易笔数)
    profit_factor: float = 0.0         # 盈亏比 (总盈利/总亏损)
    avg_holding_days: float = 0.0      # 平均持仓天数
    trade_frequency: float = 0.0       # 交易频率 (次/周)
    best_strategy_name: str = ""       # 最优策略名称
    expected_value: float = 0.0        # 收益期望 (每笔交易的数学期望收益率 %)
    price_data: pd.DataFrame | None = None          # 全量价格数据（供绘图）
    best_strategy_trades: list[dict] = field(default_factory=list)  # 最优策略交易记录


class StockScreener:
    """股票筛选推荐器 - 滚动窗口验证版"""

    def __init__(
        self,
        strategy_registry: dict,
        datasource_name: str = "multi_source",
        initial_capital: float = 100000.0,
        train_weeks: int = 20,
        rolling_rounds: int = 15,
    ):
        self.strategy_registry = strategy_registry
        self.datasource_name = datasource_name
        self.initial_capital = initial_capital
        self.train_weeks = train_weeks
        self.rolling_rounds = rolling_rounds
        self.total_weeks_needed = train_weeks + rolling_rounds

    def screen_stocks(
        self,
        stock_configs: list[tuple[MarketType, str, str]],
        end_date: str | None = None,
    ) -> list[StockRecommendation]:
        """筛选推荐股票（滚动窗口验证）"""
        DataSourceFactory.register_defaults()

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d")
            - timedelta(weeks=self.total_weeks_needed)
        ).strftime("%Y-%m-%d")

        logger.info(
            "股票筛选（滚动验证）：数据区间 %s ~ %s（%d周），训练窗口=%d周，滚动%d轮",
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

        recommendations.sort(key=lambda r: r.expected_value, reverse=True)
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

        min_required = self.train_weeks * 4
        if df.empty or len(df) < min_required:
            logger.warning("数据不足，跳过 %s（需要约%d条，仅%d条）", display_name, min_required, len(df))
            return None

        latest_close = float(df.iloc[-1]["close"])

        rolling_slices = self._generate_rolling_slices(df)
        actual_rounds = len(rolling_slices)

        if actual_rounds < 3:
            logger.warning("有效滚动轮数不足，跳过 %s（仅%d轮）", display_name, actual_rounds)
            return None

        strategies = {k: v for k, v in self.strategy_registry.items() if k != "event_ma_cross"}
        strategy_results: list[StrategyRollingResult] = []

        for strategy_name, strategy_class in strategies.items():
            result = self._run_strategy_rolling(
                strategy_name, strategy_class, symbol, rolling_slices, market
            )
            strategy_results.append(result)

        # 选出最优策略：按收益期望排序，期望相同则按总收益排序
        best_strategy = max(
            strategy_results,
            key=lambda r: (r.expected_value, r.sum_return),
        )
        logger.info(
            "  %s 最优策略: %s (EV=%.4f%%, sum_ret=%.2f%%, twr=%.1f%%)",
            display_name, best_strategy.strategy_name,
            best_strategy.expected_value, best_strategy.sum_return,
            best_strategy.avg_trade_win_rate,
        )

        # 对最优策略做全量数据回测，收集交易记录供 PDF 绘图
        best_trades = self._run_best_strategy_full_backtest(
            best_strategy.strategy_name, symbol, df, market
        )

        recommendation = StockRecommendation(
            symbol=symbol,
            display_name=display_name,
            market=market,
            latest_close=latest_close,
            total_strategies=len(strategy_results),
            rolling_rounds=actual_rounds,
            strategy_results=strategy_results,
            best_strategy_name=best_strategy.strategy_name,
            price_data=df,
            best_strategy_trades=best_trades,
        )

        self._calculate_probabilities(recommendation)
        return recommendation

    def _run_best_strategy_full_backtest(
        self,
        strategy_name: str,
        symbol: str,
        df: pd.DataFrame,
        market: MarketType,
    ) -> list[dict]:
        """对最优策略做全量数据回测，返回交易记录列表"""
        strategy_class = self.strategy_registry.get(strategy_name)
        if not strategy_class:
            return []

        no_stop_profit = StopLossConfig(enabled=False)
        storage = SqliteStorage(":memory:")
        storage.initialize()
        trades = []
        try:
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                commission_rate=0.0003,
                slippage_rate=0.001,
                storage=storage,
                stop_loss_config=no_stop_profit,
            )
            engine.set_strategy(strategy_class())
            engine.add_data(symbol, df, market)
            engine.run()

            for trade in engine._portfolio.trade_history:
                trades.append({
                    "datetime": trade.traded_at,
                    "side": trade.side.name,
                    "price": trade.price,
                    "quantity": trade.quantity,
                })
        except Exception as exc:
            logger.debug("最优策略全量回测失败 %s: %s", strategy_name, exc)
        finally:
            storage.close()
        return trades

    def _generate_rolling_slices(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        """生成滚动窗口数据切片"""
        total_bars = len(df)
        step_size = 5
        train_size = self.train_weeks * 5

        if total_bars < train_size:
            train_size = total_bars - step_size

        end_positions = []
        for roll_idx in range(self.rolling_rounds):
            end_pos = total_bars - roll_idx * step_size
            start_pos = end_pos - train_size
            if start_pos < 0:
                break
            end_positions.append((start_pos, end_pos))

        end_positions.reverse()

        slices: list[pd.DataFrame] = []
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
        """对单个策略执行滚动验证，统计信号一致性

        窗口可以重叠，但交易日期不允许重复：
        跨窗口收集所有交易记录，按(日期+方向)去重后计算总收益和交易级指标。
        """
        buy_signals: list[bool] = []
        sell_signals: list[bool] = []
        returns: list[float] = []
        win_rates: list[float] = []
        sharpes: list[float] = []
        drawdowns: list[float] = []
        trade_win_rates: list[float] = []
        profit_factors: list[float] = []
        holding_days_list: list[float] = []
        trade_frequencies: list[float] = []
        win_pcts: list[float] = []
        loss_pcts: list[float] = []

        # 跨窗口交易记录去重：key = (日期字符串, 方向)
        seen_trade_keys: set[tuple[str, str]] = set()
        deduplicated_trades: list[dict] = []

        for slice_df in rolling_slices:
            buy_signal, sell_signal, metrics, round_trades = self._evaluate_single_round(
                strategy_name, strategy_class, symbol, slice_df, market
            )
            buy_signals.append(buy_signal)
            sell_signals.append(sell_signal)
            if metrics:
                returns.append(metrics.get("total_return", 0.0))
                win_rates.append(metrics.get("win_rate", 0.0))
                sharpes.append(metrics.get("sharpe_ratio", 0.0))
                drawdowns.append(abs(metrics.get("max_drawdown", 0.0)))
                trade_win_rates.append(metrics.get("trade_win_rate", 0.0))
                profit_factors.append(metrics.get("profit_factor", 0.0))
                holding_days_list.append(metrics.get("avg_holding_days", 0.0))
                trade_frequencies.append(metrics.get("trade_frequency", 0.0))
                win_pcts.append(metrics.get("avg_win_pct", 0.0))
                loss_pcts.append(metrics.get("avg_loss_pct", 0.0))

            # 按交易日期+方向去重
            for trade in round_trades:
                trade_date_str = str(trade["datetime"])[:10]
                trade_key = (trade_date_str, trade["side"])
                if trade_key not in seen_trade_keys:
                    seen_trade_keys.add(trade_key)
                    deduplicated_trades.append(trade)

        total_rolls = len(buy_signals)
        if total_rolls == 0:
            return StrategyRollingResult(strategy_name=strategy_name)

        # 用去重后的交易记录计算总收益和交易级指标（传入最新价用于未平仓浮盈）
        last_slice = rolling_slices[-1] if rolling_slices else None
        slice_latest_close = float(last_slice.iloc[-1]["close"]) if last_slice is not None and len(last_slice) > 0 else 0.0
        dedup_metrics = self._calculate_dedup_trade_metrics(deduplicated_trades, latest_close=slice_latest_close)

        mean_trade_win_rate = dedup_metrics["trade_win_rate"]
        mean_win_pct = dedup_metrics["avg_win_pct"]
        mean_loss_pct = dedup_metrics["avg_loss_pct"]
        win_ratio = mean_trade_win_rate / 100.0
        expected_value = win_ratio * mean_win_pct - (1 - win_ratio) * mean_loss_pct

        return StrategyRollingResult(
            strategy_name=strategy_name,
            buy_signal_ratio=sum(buy_signals) / total_rolls,
            sell_signal_ratio=sum(sell_signals) / total_rolls,
            latest_buy_signal=buy_signals[-1] if buy_signals else False,
            latest_sell_signal=sell_signals[-1] if sell_signals else False,
            avg_return=sum(returns) / len(returns) if returns else 0.0,
            sum_return=dedup_metrics["total_return"],
            avg_win_rate=sum(win_rates) / len(win_rates) if win_rates else 0.0,
            avg_sharpe=sum(sharpes) / len(sharpes) if sharpes else 0.0,
            avg_max_drawdown=sum(drawdowns) / len(drawdowns) if drawdowns else 0.0,
            total_rolls=total_rolls,
            avg_trade_win_rate=mean_trade_win_rate,
            avg_profit_factor=dedup_metrics["profit_factor"],
            avg_holding_days=dedup_metrics["avg_holding_days"],
            avg_trade_frequency=sum(trade_frequencies) / len(trade_frequencies) if trade_frequencies else 0.0,
            avg_win_pct=round(mean_win_pct, 4),
            avg_loss_pct=round(mean_loss_pct, 4),
            expected_value=round(expected_value, 4),
        )

    def _calculate_dedup_trade_metrics(self, trades: list[dict], latest_close: float = 0.0) -> dict:
        """基于去重后的交易记录计算总收益和交易级指标

        将 BUY/SELL 按时间顺序配对，计算每笔交易的盈亏。
        若最后一笔 BUY 未配对 SELL 且 latest_close > 0，则按最新价计算浮盈纳入总收益。
        """
        if not trades:
            return {
                "total_return": 0.0, "trade_win_rate": 0.0, "profit_factor": 0.0,
                "avg_holding_days": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            }

        sorted_trades = sorted(trades, key=lambda t: str(t["datetime"]))

        # 配对 BUY → SELL
        paired_trades: list[dict] = []
        pending_buy: dict | None = None
        for trade in sorted_trades:
            if trade["side"] == "BUY" and pending_buy is None:
                pending_buy = trade
            elif trade["side"] == "SELL" and pending_buy is not None:
                buy_price = pending_buy["price"]
                sell_price = trade["price"]
                quantity = min(pending_buy["quantity"], trade["quantity"])
                pnl = (sell_price - buy_price) * quantity
                pnl_pct = (sell_price - buy_price) / buy_price * 100 if buy_price > 0 else 0.0
                buy_dt = pd.Timestamp(pending_buy["datetime"])
                sell_dt = pd.Timestamp(trade["datetime"])
                holding_days = max((sell_dt - buy_dt).days, 1)
                paired_trades.append({
                    "pnl": pnl, "pnl_pct": pnl_pct, "holding_days": holding_days,
                    "buy_price": buy_price, "sell_price": sell_price, "quantity": quantity,
                })
                pending_buy = None

        # 未平仓持仓：按最新价计算浮盈
        if pending_buy is not None and latest_close > 0:
            buy_price = pending_buy["price"]
            quantity = pending_buy["quantity"]
            unrealized_pnl = (latest_close - buy_price) * quantity
            unrealized_pct = (latest_close - buy_price) / buy_price * 100 if buy_price > 0 else 0.0
            buy_dt = pd.Timestamp(pending_buy["datetime"])
            holding_days = max((pd.Timestamp.now() - buy_dt).days, 1)
            paired_trades.append({
                "pnl": unrealized_pnl, "pnl_pct": unrealized_pct, "holding_days": holding_days,
                "buy_price": buy_price, "sell_price": latest_close, "quantity": quantity,
            })

        if not paired_trades:
            return {
                "total_return": 0.0, "trade_win_rate": 0.0, "profit_factor": 0.0,
                "avg_holding_days": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            }

        winning = [t for t in paired_trades if t["pnl"] > 0]
        losing = [t for t in paired_trades if t["pnl"] <= 0]
        total_paired = len(paired_trades)

        trade_win_rate = len(winning) / total_paired * 100 if total_paired > 0 else 0.0

        total_profit = sum(t["pnl"] for t in winning)
        total_loss = abs(sum(t["pnl"] for t in losing))
        if total_loss > 0:
            raw_profit_factor = total_profit / total_loss
            profit_factor = min(raw_profit_factor, 10.0)
        elif total_profit > 0:
            # 全胜无亏损 → 用平均盈利%/1来近似，体现真实收益水平
            avg_win = sum(t["pnl_pct"] for t in winning) / len(winning) if winning else 0.0
            profit_factor = min(round(avg_win / 1.0, 2), 10.0) if avg_win > 0 else 10.0
        else:
            profit_factor = 0.0

        avg_holding_days = sum(t["holding_days"] for t in paired_trades) / total_paired

        # 计算总收益：所有配对交易的累积收益 / 初始资金
        total_pnl = sum(t["pnl"] for t in paired_trades)
        total_return = total_pnl / self.initial_capital * 100

        avg_win_pct = sum(t["pnl_pct"] for t in winning) / len(winning) if winning else 0.0
        avg_loss_pct = abs(sum(t["pnl_pct"] for t in losing) / len(losing)) if losing else 0.0

        return {
            "total_return": round(total_return, 2),
            "trade_win_rate": round(trade_win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_holding_days": round(avg_holding_days, 1),
            "avg_win_pct": round(avg_win_pct, 4),
            "avg_loss_pct": round(avg_loss_pct, 4),
        }

    def _evaluate_single_round(
        self,
        strategy_name: str,
        strategy_class,
        symbol: str,
        df: pd.DataFrame,
        market: MarketType,
    ) -> tuple[bool, bool, dict, list[dict]]:
        """评估单轮：回测 + 信号检测

        返回: (buy_signal, sell_signal, metrics, trade_records)
        trade_records: 该轮产生的交易记录 [{datetime, side, price, quantity}]
        """
        from openquant.core.models import Bar, OrderSide, Portfolio

        if len(df) < 10:
            return False, False, {}, []

        # 回测获取绩效指标和交易记录（禁用默认止盈，让策略自行决定卖出时机）
        no_stop_profit = StopLossConfig(enabled=False)
        storage = SqliteStorage(":memory:")
        storage.initialize()
        metrics = {}
        round_trades: list[dict] = []
        try:
            strategy = strategy_class()
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                commission_rate=0.0003,
                slippage_rate=0.001,
                storage=storage,
                stop_loss_config=no_stop_profit,
            )
            engine.set_strategy(strategy)
            engine.add_data(symbol, df, market)
            engine.run()
            metrics = engine.get_results()
            for trade in engine._portfolio.trade_history:
                round_trades.append({
                    "datetime": trade.traded_at,
                    "side": trade.side.name,
                    "price": trade.price,
                    "quantity": trade.quantity,
                })
        except Exception as exc:
            logger.debug("策略 %s 单轮回测失败: %s", strategy_name, exc)
        finally:
            storage.close()

        # 信号检测：模拟执行订单并观察最后3天
        # 只记录最后一个信号方向，避免同一轮中买卖信号同时为True
        last_signal_direction = None  # "buy" or "sell"
        try:
            strategy = strategy_class()
            portfolio = Portfolio(initial_capital=self.initial_capital, cash=self.initial_capital)
            strategy.initialize(portfolio)

            recent_window = min(3, len(df))
            for idx in range(len(df)):
                row = df.iloc[idx]
                close_price = float(row["close"])
                bar = Bar(
                    symbol=symbol,
                    datetime=pd.Timestamp(row["datetime"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=close_price,
                    volume=float(row["volume"]),
                    market=market,
                )
                orders = strategy.on_bar(bar, portfolio)

                # 模拟执行订单，更新 portfolio 持仓状态
                for order in orders:
                    if order.side == OrderSide.BUY:
                        cost = order.price * order.quantity
                        if portfolio.cash >= cost:
                            portfolio.cash -= cost
                            if symbol in portfolio.positions:
                                pos = portfolio.positions[symbol]
                                total_qty = pos.quantity + order.quantity
                                pos.avg_cost = (pos.avg_cost * pos.quantity + cost) / total_qty
                                pos.quantity = total_qty
                            else:
                                portfolio.positions[symbol] = Position(
                                    symbol=symbol,
                                    quantity=order.quantity,
                                    avg_cost=order.price,
                                    current_price=close_price,
                                    market=market,
                                )
                    elif order.side == OrderSide.SELL:
                        if symbol in portfolio.positions:
                            pos = portfolio.positions[symbol]
                            sell_qty = min(order.quantity, pos.quantity)
                            portfolio.cash += order.price * sell_qty
                            pos.quantity -= sell_qty
                            if pos.quantity <= 0:
                                del portfolio.positions[symbol]

                # 更新持仓当前价格
                if symbol in portfolio.positions:
                    portfolio.positions[symbol].current_price = close_price

                # 只记录最后几天的信号，以最后一个信号为准
                if idx >= len(df) - recent_window:
                    for order in orders:
                        if order.side == OrderSide.BUY:
                            last_signal_direction = "buy"
                        elif order.side == OrderSide.SELL:
                            last_signal_direction = "sell"
        except Exception as exc:
            logger.debug("策略 %s 信号检测失败: %s", strategy_name, exc)

        buy_signal = last_signal_direction == "buy"
        sell_signal = last_signal_direction == "sell"
        return buy_signal, sell_signal, metrics, round_trades

    def _calculate_probabilities(self, recommendation: StockRecommendation) -> None:
        """基于滚动验证结果计算买入/卖出概率

        概率计算逻辑（v3 - 层级递进式）：
        - 基础概率：根据"最新信号策略数"确定基础区间
        - 加成概率：根据"信号一致性 + 绩效"在基础上递增
        - 设计目标：3+策略在最新轮同时给出信号且绩效好时达到70%+
        """
        results = recommendation.strategy_results
        if not results:
            return

        total = len(results)

        # === 买入概率计算 ===
        active_buy_strategies = [r for r in results if r.buy_signal_ratio > 0]
        active_buy_count = len(active_buy_strategies)
        best_buy_consistency = max((r.buy_signal_ratio for r in results), default=0.0)

        if active_buy_strategies:
            recommendation.buy_consistency = sum(r.buy_signal_ratio for r in active_buy_strategies) / len(active_buy_strategies)
        else:
            recommendation.buy_consistency = 0.0

        latest_buy_count = sum(1 for r in results if r.latest_buy_signal)
        latest_buy_ratio = latest_buy_count / total
        buy_coverage = active_buy_count / total if total > 0 else 0.0

        # 绩效指标
        if active_buy_strategies:
            avg_return = sum(r.avg_return for r in active_buy_strategies) / len(active_buy_strategies)
            avg_win_rate = sum(r.avg_win_rate for r in active_buy_strategies) / len(active_buy_strategies)
            avg_sharpe = sum(r.avg_sharpe for r in active_buy_strategies) / len(active_buy_strategies)
        else:
            avg_return = sum(r.avg_return for r in results) / total
            avg_win_rate = sum(r.avg_win_rate for r in results) / total
            avg_sharpe = sum(r.avg_sharpe for r in results) / total

        return_score = min(max((avg_return + 10) / 25, 0), 1.0)
        win_rate_score = min(max(avg_win_rate / 80, 0), 1.0)
        sharpe_score = min(max((avg_sharpe + 0.5) / 3, 0), 1.0)
        performance_score = 0.4 * return_score + 0.35 * win_rate_score + 0.25 * sharpe_score

        # 层级递进概率计算
        # 基础概率由"最新信号策略数"决定区间
        if latest_buy_count >= 4:
            base_probability = 55.0  # 4+策略最新给出买入信号
        elif latest_buy_count >= 3:
            base_probability = 45.0
        elif latest_buy_count >= 2:
            base_probability = 35.0
        elif latest_buy_count >= 1:
            base_probability = 25.0
        elif active_buy_count >= 3:
            base_probability = 15.0  # 历史有信号但最新没有
        elif active_buy_count >= 1:
            base_probability = 8.0
        else:
            base_probability = 0.0

        # 加成概率：一致性 + 覆盖率 + 绩效
        consistency_bonus = best_buy_consistency * 20  # 最高一致性贡献最多20%
        coverage_bonus = buy_coverage * 10  # 覆盖率贡献最多10%
        performance_bonus = performance_score * 15  # 绩效贡献最多15%

        buy_probability = base_probability + consistency_bonus + coverage_bonus + performance_bonus
        recommendation.buy_probability = min(max(buy_probability, 0.0), 99.0)
        recommendation.composite_score = buy_probability / 100.0

        # === 卖出概率计算 ===
        active_sell_strategies = [r for r in results if r.sell_signal_ratio > 0]
        active_sell_count = len(active_sell_strategies)
        best_sell_consistency = max((r.sell_signal_ratio for r in results), default=0.0)

        if active_sell_strategies:
            recommendation.sell_consistency = sum(r.sell_signal_ratio for r in active_sell_strategies) / len(active_sell_strategies)
        else:
            recommendation.sell_consistency = 0.0

        latest_sell_count = sum(1 for r in results if r.latest_sell_signal)
        latest_sell_ratio = latest_sell_count / total
        sell_coverage = active_sell_count / total if total > 0 else 0.0

        # 卖出绩效：只有真正亏损时才给高分
        if active_sell_strategies:
            sell_avg_return = sum(r.avg_return for r in active_sell_strategies) / len(active_sell_strategies)
            sell_avg_drawdown = sum(r.avg_max_drawdown for r in active_sell_strategies) / len(active_sell_strategies)
            sell_avg_sharpe = sum(r.avg_sharpe for r in active_sell_strategies) / len(active_sell_strategies)
        else:
            sell_avg_return = sum(r.avg_return for r in results) / total
            sell_avg_drawdown = sum(r.avg_max_drawdown for r in results) / total
            sell_avg_sharpe = sum(r.avg_sharpe for r in results) / total

        loss_score = min(max(-sell_avg_return / 15, 0), 1.0)
        drawdown_score = min(max((sell_avg_drawdown - 3) / 12, 0), 1.0)
        sharpe_inv_score = min(max(-sell_avg_sharpe / 2, 0), 1.0)
        sell_perf = 0.4 * loss_score + 0.35 * drawdown_score + 0.25 * sharpe_inv_score

        # 卖出层级递进
        if latest_sell_count >= 4:
            sell_base = 55.0
        elif latest_sell_count >= 3:
            sell_base = 45.0
        elif latest_sell_count >= 2:
            sell_base = 35.0
        elif latest_sell_count >= 1:
            sell_base = 25.0
        elif active_sell_count >= 3:
            sell_base = 15.0
        elif active_sell_count >= 1:
            sell_base = 8.0
        else:
            sell_base = 0.0

        sell_consistency_bonus = best_sell_consistency * 20
        sell_coverage_bonus = sell_coverage * 10
        sell_perf_bonus = sell_perf * 15

        sell_probability = sell_base + sell_consistency_bonus + sell_coverage_bonus + sell_perf_bonus
        recommendation.sell_probability = min(max(sell_probability, 0.0), 99.0)

        # === 历史回测绩效汇总（基于最优策略） ===
        best_name = recommendation.best_strategy_name
        best_result = next((r for r in results if r.strategy_name == best_name), None)
        if best_result:
            recommendation.backtest_return = best_result.sum_return
            recommendation.backtest_win_rate = best_result.avg_win_rate
            recommendation.trade_win_rate = best_result.avg_trade_win_rate
            recommendation.profit_factor = best_result.avg_profit_factor
            recommendation.avg_holding_days = best_result.avg_holding_days
            recommendation.trade_frequency = best_result.avg_trade_frequency
            recommendation.expected_value = best_result.expected_value
        else:
            recommendation.backtest_return = 0.0
            recommendation.backtest_win_rate = 0.0
            recommendation.trade_win_rate = 0.0
            recommendation.profit_factor = 0.0
            recommendation.avg_holding_days = 0.0
            recommendation.trade_frequency = 0.0
            recommendation.expected_value = 0.0


def print_recommendations(recommendations: list[StockRecommendation]) -> None:
    """格式化输出推荐结果（中文对齐版）"""
    print("\n" + "=" * 82)
    print("  📊 股票多策略综合分析 - 滚动验证版（Walk-Forward Analysis）")
    print(f"  分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if recommendations:
        rounds = recommendations[0].rolling_rounds
        print(f"  验证方式: {rounds}轮滚动窗口验证")
    print("=" * 82)

    if not recommendations:
        print("  ⚠️ 无推荐结果")
        return

    # 固定列宽定义
    col_widths = {
        "rank": 5,
        "stock": 10,
        "strategy": 16,
        "trade_wr": 10,
        "pf": 8,
        "ev": 10,
        "bt_return": 10,
        "hold_days": 8,
        "freq": 8,
        "price": 10,
        "action": 10,
    }

    header = (
        "  "
        + _pad_to_width("排名", col_widths["rank"])
        + _pad_to_width("股票", col_widths["stock"])
        + _pad_to_width("最优策略", col_widths["strategy"])
        + _pad_to_width("交易胜率", col_widths["trade_wr"])
        + _pad_to_width("盈亏比", col_widths["pf"])
        + _pad_to_width("期望收益", col_widths["ev"])
        + _pad_to_width("总收益", col_widths["bt_return"])
        + _pad_to_width("持仓天", col_widths["hold_days"])
        + _pad_to_width("频率/周", col_widths["freq"])
        + _pad_to_width("最新价", col_widths["price"])
        + "建议"
    )
    total_width = sum(col_widths.values()) + 4
    print(f"\n{header}")
    print("  " + "-" * total_width)

    for rank, rec in enumerate(recommendations, 1):
        # 操作建议：基于最优策略最新一轮的真实执行信号
        best_name = rec.best_strategy_name
        best_result = next((r for r in rec.strategy_results if r.strategy_name == best_name), None)

        if best_result:
            if best_result.latest_buy_signal and not best_result.latest_sell_signal:
                action = "买入 📈"
            elif best_result.latest_sell_signal and not best_result.latest_buy_signal:
                action = "卖出 📉"
            elif best_result.latest_buy_signal and best_result.latest_sell_signal:
                action = "冲突 ⚡"
            elif rec.backtest_return <= 0 and rec.trade_win_rate <= 0:
                action = "观望 👀"
            else:
                action = "持有 ⏸️"
        else:
            action = "观望 👀"

        # 未持仓视角：综合所有策略的最新信号判断入场时机
        buy_signal_count = sum(1 for r in rec.strategy_results if r.latest_buy_signal)
        sell_signal_count = sum(1 for r in rec.strategy_results if r.latest_sell_signal)

        if best_result and best_result.latest_buy_signal:
            fresh_action = "买入"
        elif best_result and best_result.latest_sell_signal:
            fresh_action = "观望"
        elif buy_signal_count >= 2 and rec.expected_value > 0:
            fresh_action = "买入"
        elif buy_signal_count >= 1 and rec.expected_value > 0:
            fresh_action = "关注"
        elif sell_signal_count >= 2:
            fresh_action = "观望"
        elif rec.expected_value > 0 and sell_signal_count == 0:
            fresh_action = "等待信号"
        else:
            fresh_action = "观望"
        action = f"{action}（{fresh_action}）"

        rank_str = str(rank)
        strategy_str = best_name if best_name else "N/A"
        trade_wr_str = f"{rec.trade_win_rate:5.1f}%"
        if rec.profit_factor > 0:
            pf_str = f"{rec.profit_factor:5.2f}"
        elif rec.trade_win_rate > 0 and rec.backtest_return > 0:
            # 有盈利交易但盈亏比为0（全亏不会出现正收益）→ 显示为N/A
            pf_str = "  N/A"
        else:
            pf_str = "  N/A"
        ev_str = f"{rec.expected_value:+6.3f}%"
        bt_return_str = f"{rec.backtest_return:+6.2f}%"
        hold_days_str = f"{rec.avg_holding_days:5.1f}"
        freq_str = f"{rec.trade_frequency:5.2f}"
        price_str = f"{rec.latest_close:>9.2f}"

        row = (
            "  "
            + _pad_to_width(rank_str, col_widths["rank"])
            + _pad_to_width(rec.display_name, col_widths["stock"])
            + _pad_to_width(strategy_str, col_widths["strategy"])
            + _pad_to_width(trade_wr_str, col_widths["trade_wr"])
            + _pad_to_width(pf_str, col_widths["pf"])
            + _pad_to_width(ev_str, col_widths["ev"])
            + _pad_to_width(bt_return_str, col_widths["bt_return"])
            + _pad_to_width(hold_days_str, col_widths["hold_days"])
            + _pad_to_width(freq_str, col_widths["freq"])
            + _pad_to_width(price_str, col_widths["price"])
            + action
        )
        print(row)

    # 详细分析：各股票的策略对比
    print("\n" + "-" * 82)
    print("  📋 各股票策略穷举对比（按期望收益排序）")
    print("-" * 82)

    for rec in recommendations[:5]:
        print(
            f"\n  【{rec.display_name}】({rec.symbol})"
            f" - 最优策略: {rec.best_strategy_name}"
            f" | EV={rec.expected_value:+.3f}%"
            f" | 滚动{rec.rolling_rounds}轮验证"
        )

        sorted_results = sorted(rec.strategy_results, key=lambda r: r.expected_value, reverse=True)
        for sr in sorted_results:
            is_best = "⭐" if sr.strategy_name == rec.best_strategy_name else "  "
            signal_str = ""
            if sr.latest_buy_signal:
                signal_str = " 📈买入"
            elif sr.latest_sell_signal:
                signal_str = " 📉卖出"
            print(
                f"    {is_best} {sr.strategy_name:<16s}"
                f" EV={sr.expected_value:+7.3f}%"
                f" 胜率={sr.avg_trade_win_rate:5.1f}%"
                f" 总收益={sr.sum_return:+7.2f}%"
                f" 盈亏比={sr.avg_profit_factor:5.2f}"
                f"{signal_str}"
            )

    print("\n" + "=" * 82)
    print("  💡 说明: 期望收益(EV) = 胜率×平均盈利% - (1-胜率)×平均亏损%")
    print("     EV>0 表示该策略长期数学期望为正，适合真实交易")
    print("     排名按最优策略的期望收益排序，建议基于最优策略最新信号")
    print("  ⚠️ 免责声明: 以上分析仅供参考，不构成投资建议")
    print("=" * 82 + "\n")


class BacktestValidator:
    """历史滚动回测验证器

    性能优化版：先一次性预计算所有验证点的推荐概率，
    然后对不同阈值组合复用同一份预计算结果（纯数值比较，无重复计算）。
    """

    def __init__(
        self,
        strategy_registry: dict,
        datasource_name: str = "multi_source",
        initial_capital: float = 100000.0,
        train_weeks: int = 20,
        rolling_rounds: int = 15,
    ):
        self.strategy_registry = strategy_registry
        self.datasource_name = datasource_name
        self.initial_capital = initial_capital
        self.train_weeks = train_weeks
        self.rolling_rounds = rolling_rounds

    def validate(
        self,
        stock_configs: list[tuple[MarketType, str, str]],
        validation_weeks: int = 30,
        buy_thresholds: list[float] | None = None,
        sell_thresholds: list[float] | None = None,
    ) -> dict:
        """执行历史回测验证

        优化：先预计算所有验证点的推荐概率，再对不同阈值复用结果。
        """
        if buy_thresholds is None:
            buy_thresholds = [40.0, 50.0, 60.0, 70.0, 80.0]
        if sell_thresholds is None:
            sell_thresholds = [40.0, 50.0, 60.0, 70.0, 80.0]

        from openquant.datasource.factory import DataSourceFactory
        DataSourceFactory.register_defaults()

        total_weeks = self.train_weeks + self.rolling_rounds + validation_weeks
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(weeks=total_weeks)).strftime("%Y-%m-%d")

        logger.info(
            "回测验证：数据区间 %s ~ %s（%d周），验证窗口=%d周",
            start_date, end_date, total_weeks, validation_weeks,
        )

        stock_data = self._fetch_all_data(stock_configs, start_date, end_date)
        if not stock_data:
            logger.error("无法获取任何股票数据")
            return {}

        validation_points = self._generate_validation_points(stock_data, validation_weeks)
        logger.info("生成了 %d 个验证时间点", len(validation_points))

        # 核心优化：一次性预计算所有验证点的推荐概率
        precomputed = self._precompute_all_recommendations(
            stock_data, stock_configs, validation_points
        )
        logger.info("预计算完成，共 %d 个验证点 × %d 只股票", len(validation_points), len(stock_data))

        # 对每个阈值组合进行模拟交易（纯数值计算，秒级完成）
        results = {}
        for buy_threshold in buy_thresholds:
            for sell_threshold in sell_thresholds:
                key = f"buy_{buy_threshold:.0f}_sell_{sell_threshold:.0f}"
                result = self._simulate_with_precomputed(
                    stock_data, precomputed, validation_points,
                    buy_threshold, sell_threshold,
                )
                results[key] = result
                logger.info(
                    "阈值 买入>%.0f%% 卖出>%.0f%%: 收益=%.2f%%, 交易=%d次, 胜率=%.1f%%",
                    buy_threshold, sell_threshold,
                    result["total_return"], result["total_trades"], result["win_rate"],
                )

        return results

    def _fetch_all_data(
        self,
        stock_configs: list[tuple[MarketType, str, str]],
        start_date: str,
        end_date: str,
    ) -> dict[str, tuple[MarketType, str, pd.DataFrame]]:
        """获取所有股票的历史数据"""
        data_source = DataSourceFactory.get(self.datasource_name)
        stock_data = {}

        for market, symbol, display_name in stock_configs:
            try:
                logger.info("获取 %s (%s) 的历史数据...", display_name, symbol)
                df = data_source.fetch_daily_bars(symbol, start_date, end_date, market)
                if df is not None and not df.empty and len(df) >= 60:
                    stock_data[display_name] = (market, symbol, df)
                    logger.info("  → %d 条数据", len(df))
                else:
                    logger.warning("  → 数据不足，跳过 %s", display_name)
                time.sleep(1.5)
            except Exception as exc:
                logger.error("获取 %s 数据失败: %s", display_name, exc)

        return stock_data

    def _generate_validation_points(
        self, stock_data: dict, validation_weeks: int
    ) -> list[int]:
        """生成验证时间点索引"""
        min_length = min(len(df) for _, _, df in stock_data.values())
        points = []
        for week_offset in range(validation_weeks, 0, -1):
            idx = min_length - week_offset * 5
            if idx > self.train_weeks * 5:
                points.append(idx)
        return points

    def _precompute_all_recommendations(
        self,
        stock_data: dict[str, tuple[MarketType, str, pd.DataFrame]],
        stock_configs: list[tuple[MarketType, str, str]],
        validation_points: list[int],
    ) -> dict[int, dict[str, tuple[float, float, float]]]:
        """预计算所有验证点的推荐概率

        Returns:
            {point_idx: {display_name: (buy_prob, sell_prob, price)}}
        """
        strategies = {k: v for k, v in self.strategy_registry.items() if k != "event_ma_cross"}
        precomputed: dict[int, dict[str, tuple[float, float, float]]] = {}

        for point_idx in validation_points:
            point_results: dict[str, tuple[float, float, float]] = {}

            for market, symbol, display_name in stock_configs:
                if display_name not in stock_data:
                    continue
                _, _, full_df = stock_data[display_name]
                if point_idx > len(full_df):
                    continue

                df = full_df.iloc[:point_idx].reset_index(drop=True)
                min_required = self.train_weeks * 4
                if len(df) < min_required:
                    continue

                current_price = float(df.iloc[-1]["close"])

                screener = StockScreener(
                    strategy_registry=strategies,
                    datasource_name=self.datasource_name,
                    initial_capital=self.initial_capital,
                    train_weeks=self.train_weeks,
                    rolling_rounds=self.rolling_rounds,
                )
                rolling_slices = screener._generate_rolling_slices(df)
                if len(rolling_slices) < 3:
                    continue

                strategy_results = []
                for strategy_name, strategy_class in strategies.items():
                    result = screener._run_strategy_rolling(
                        strategy_name, strategy_class, symbol, rolling_slices, market
                    )
                    strategy_results.append(result)

                recommendation = StockRecommendation(
                    symbol=symbol,
                    display_name=display_name,
                    market=market,
                    latest_close=current_price,
                    total_strategies=len(strategy_results),
                    rolling_rounds=len(rolling_slices),
                    strategy_results=strategy_results,
                )
                screener._calculate_probabilities(recommendation)
                point_results[display_name] = (
                    recommendation.buy_probability,
                    recommendation.sell_probability,
                    current_price,
                )

            precomputed[point_idx] = point_results
            logger.info("  验证点 %d/%d 完成", validation_points.index(point_idx) + 1, len(validation_points))

        return precomputed

    def _simulate_with_precomputed(
        self,
        stock_data: dict[str, tuple[MarketType, str, pd.DataFrame]],
        precomputed: dict[int, dict[str, tuple[float, float, float]]],
        validation_points: list[int],
        buy_threshold: float,
        sell_threshold: float,
    ) -> dict:
        """使用预计算结果模拟交易（纯数值计算）"""
        capital = self.initial_capital
        holdings: dict[str, dict] = {}
        trade_log: list[dict] = []

        for point_idx in validation_points:
            point_results = precomputed.get(point_idx, {})

            for display_name, (buy_prob, sell_prob, price) in point_results.items():
                # 卖出逻辑
                if sell_prob >= sell_threshold and display_name in holdings:
                    holding = holdings[display_name]
                    sell_value = holding["shares"] * price
                    profit = sell_value - holding["shares"] * holding["avg_cost"]
                    capital += sell_value
                    trade_log.append({
                        "action": "sell",
                        "stock": display_name,
                        "price": price,
                        "shares": holding["shares"],
                        "profit": profit,
                        "point_idx": point_idx,
                    })
                    del holdings[display_name]

                # 买入逻辑
                elif buy_prob >= buy_threshold and display_name not in holdings:
                    position_size = capital * 0.15
                    if position_size > 1000 and capital > position_size:
                        shares = int(position_size / price)
                        if shares > 0:
                            cost = shares * price
                            capital -= cost
                            holdings[display_name] = {
                                "shares": shares,
                                "avg_cost": price,
                            }
                            trade_log.append({
                                "action": "buy",
                                "stock": display_name,
                                "price": price,
                                "shares": shares,
                                "profit": 0,
                                "point_idx": point_idx,
                            })

        # 计算最终收益（平仓所有持仓）
        final_equity = capital
        for display_name, holding in holdings.items():
            if display_name in stock_data:
                _, _, df = stock_data[display_name]
                final_price = float(df.iloc[-1]["close"])
                final_equity += holding["shares"] * final_price

        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        total_trades = len(trade_log)
        winning_trades = sum(1 for t in trade_log if t["action"] == "sell" and t["profit"] > 0)
        sell_trades = sum(1 for t in trade_log if t["action"] == "sell")
        win_rate = (winning_trades / sell_trades * 100) if sell_trades > 0 else 0.0

        return {
            "total_return": total_return,
            "final_equity": final_equity,
            "total_trades": total_trades,
            "sell_trades": sell_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "trade_log": trade_log,
        }


def print_validation_results(results: dict) -> None:
    """格式化输出回测验证结果"""
    if not results:
        print("  ⚠️ 无验证结果")
        return

    print("\n" + "=" * 82)
    print("  📊 历史回测验证结果 - 概率阈值优化")
    print(f"  验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 82)

    # 按收益率排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]["total_return"], reverse=True)

    header = (
        "  "
        + _pad_to_width("买入阈值", 10)
        + _pad_to_width("卖出阈值", 10)
        + _pad_to_width("总收益率", 12)
        + _pad_to_width("交易次数", 10)
        + _pad_to_width("卖出次数", 10)
        + _pad_to_width("胜率", 10)
        + "评级"
    )
    print(f"\n{header}")
    print("  " + "-" * 72)

    for key, result in sorted_results:
        buy_th = f"{result['buy_threshold']:.0f}%"
        sell_th = f"{result['sell_threshold']:.0f}%"
        total_return = result["total_return"]
        total_trades = result["total_trades"]
        sell_trades = result["sell_trades"]
        win_rate = result["win_rate"]

        if total_return > 10 and win_rate > 60:
            rating = "⭐⭐⭐"
        elif total_return > 5 and win_rate > 50:
            rating = "⭐⭐"
        elif total_return > 0:
            rating = "⭐"
        else:
            rating = "❌"

        row = (
            "  "
            + _pad_to_width(buy_th, 10)
            + _pad_to_width(sell_th, 10)
            + _pad_to_width(f"{total_return:>7.2f}%", 12)
            + _pad_to_width(f"{total_trades:>5d}", 10)
            + _pad_to_width(f"{sell_trades:>5d}", 10)
            + _pad_to_width(f"{win_rate:>5.1f}%", 10)
            + rating
        )
        print(row)

    # 找出最优组合
    if sorted_results:
        best_key, best_result = sorted_results[0]
        print(f"\n  🏆 最优阈值组合:")
        print(f"     买入阈值: {best_result['buy_threshold']:.0f}%")
        print(f"     卖出阈值: {best_result['sell_threshold']:.0f}%")
        print(f"     总收益率: {best_result['total_return']:.2f}%")
        print(f"     胜率: {best_result['win_rate']:.1f}%")
        print(f"     交易次数: {best_result['total_trades']}")

    # 找出70%+阈值的最优组合
    high_threshold_results = [
        (k, v) for k, v in sorted_results
        if v["buy_threshold"] >= 70 or v["sell_threshold"] >= 70
    ]
    if high_threshold_results:
        best_high = max(high_threshold_results, key=lambda x: x[1]["total_return"])
        print(f"\n  📌 70%+高置信度最优组合:")
        print(f"     买入阈值: {best_high[1]['buy_threshold']:.0f}%")
        print(f"     卖出阈值: {best_high[1]['sell_threshold']:.0f}%")
        print(f"     总收益率: {best_high[1]['total_return']:.2f}%")
        print(f"     胜率: {best_high[1]['win_rate']:.1f}%")
    else:
        print("\n  ⚠️ 70%+阈值下无交易发生（信号太少）")

    print("\n" + "=" * 82)
    print("  💡 说明: 高阈值意味着更严格的信号筛选，交易次数更少但置信度更高")
    print("  ⚠️ 免责声明: 历史表现不代表未来收益，以上分析仅供参考")
    print("=" * 82 + "\n")
