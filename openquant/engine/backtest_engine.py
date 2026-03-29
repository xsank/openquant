"""回测引擎

基于历史数据驱动策略执行，模拟交易过程并记录绩效。
支持止损止盈和风控规则。
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime

import pandas as pd

from openquant.core.exceptions import InsufficientFundsError, InsufficientPositionError
from openquant.core.interfaces import EngineInterface, StrategyInterface
from openquant.core.models import (
    Bar,
    FrequencyType,
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    Portfolio,
    Position,
    TradeRecord,
)
from openquant.risk.risk_manager import RiskManager
from openquant.risk.stop_loss import StopLossConfig, StopLossManager
from openquant.storage.sqlite_storage import SqliteStorage
from openquant.utils.metrics import calculate_benchmark_metrics, calculate_metrics

logger = logging.getLogger(__name__)


class BacktestEngine(EngineInterface):
    """历史回测引擎"""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.0003,
        slippage_rate: float = 0.001,
        storage: SqliteStorage | None = None,
        stop_loss_config: StopLossConfig | None = None,
        risk_manager: RiskManager | None = None,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.storage = storage

        self._strategy: StrategyInterface | None = None
        self._portfolio: Portfolio | None = None
        self._equity_curve: list[tuple[datetime, float]] = []
        self._data_feeds: list[tuple[str, pd.DataFrame, MarketType]] = []
        self._stop_loss_manager = StopLossManager(stop_loss_config)
        self._risk_manager = risk_manager
        self._benchmark_data: pd.DataFrame | None = None
        self._benchmark_symbol: str | None = None

    def set_strategy(self, strategy: StrategyInterface) -> None:
        self._strategy = strategy

    def add_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        market: MarketType = MarketType.A_SHARE,
    ) -> None:
        """添加回测数据

        Args:
            symbol: 标的代码
            data: K线数据 DataFrame，需包含 datetime, open, high, low, close, volume 列
            market: 市场类型
        """
        required_columns = {"datetime", "open", "high", "low", "close", "volume"}
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(f"数据缺少必要列: {missing}")
        self._data_feeds.append((symbol, data.sort_values("datetime").reset_index(drop=True), market))

    def set_benchmark(self, symbol: str, data: pd.DataFrame) -> None:
        """设置基准数据用于对比分析

        Args:
            symbol: 基准标的代码（如沪深300指数 000300）
            data: 基准K线数据 DataFrame，需包含 datetime, close 列
        """
        if "datetime" not in data.columns or "close" not in data.columns:
            raise ValueError("基准数据需包含 datetime 和 close 列")
        self._benchmark_symbol = symbol
        self._benchmark_data = data.sort_values("datetime").reset_index(drop=True)

    def run(self) -> Portfolio:
        if self._strategy is None:
            raise ValueError("未设置策略，请先调用 set_strategy()")
        if not self._data_feeds:
            raise ValueError("未添加数据，请先调用 add_data()")

        self._portfolio = Portfolio(
            initial_capital=self.initial_capital,
            cash=self.initial_capital,
        )
        self._equity_curve.clear()
        self._stop_loss_manager.reset()
        if self._risk_manager:
            self._risk_manager.reset()
        self._strategy.initialize(self._portfolio)

        # 合并所有数据并按时间排序
        all_bars = self._build_bar_sequence()

        logger.info(
            "开始回测: 策略=%s, 初始资金=%.2f, 数据条数=%d",
            self._strategy.get_name(), self.initial_capital, len(all_bars),
        )

        for bar in all_bars:
            # 更新持仓市价
            self._update_position_price(bar)

            # 更新风控每日状态
            if self._risk_manager:
                self._risk_manager.update_daily_state(bar.datetime, self._portfolio)

            # 检查止损止盈（优先于策略信号）
            stop_orders = self._stop_loss_manager.check_stop(
                bar, self._portfolio, self._create_sell_order,
            )
            for order in stop_orders:
                self._execute_order(order, bar)

            # 策略生成订单
            orders = self._strategy.on_bar(bar, self._portfolio)

            # 风控检查 + 执行订单
            for order in orders:
                if self._risk_manager:
                    passed, reason = self._risk_manager.check_order(order, bar, self._portfolio)
                    if not passed:
                        order.status = OrderStatus.REJECTED
                        logger.warning("风控拦截订单: %s - %s", order.order_id, reason)
                        continue
                self._execute_order(order, bar)

            # 记录权益曲线
            self._equity_curve.append((bar.datetime, self._portfolio.total_equity))

        self._strategy.on_finish(self._portfolio)

        # 保存结果到数据库
        if self.storage and self._portfolio.trade_history:
            self.storage.save_trade_records(self._portfolio.trade_history)

        logger.info(
            "回测完成: 最终权益=%.2f, 总收益率=%.2f%%",
            self._portfolio.total_equity,
            self._portfolio.total_return * 100,
        )
        return self._portfolio

    def get_results(self) -> dict:
        if not self._equity_curve:
            return {}

        dates = [item[0] for item in self._equity_curve]
        values = [item[1] for item in self._equity_curve]
        equity_series = pd.Series(values, index=pd.DatetimeIndex(dates))

        metrics = calculate_metrics(equity_series)
        metrics["strategy_name"] = self._strategy.get_name() if self._strategy else ""
        metrics["initial_capital"] = self.initial_capital
        metrics["final_equity"] = values[-1] if values else 0
        metrics["total_trades"] = len(self._portfolio.trade_history) if self._portfolio else 0
        metrics["total_commission"] = self._portfolio.total_commission if self._portfolio else 0

        # 基准对比指标
        if self._benchmark_data is not None:
            benchmark_equity = self._build_benchmark_equity(equity_series)
            if benchmark_equity is not None and not benchmark_equity.empty:
                benchmark_metrics = calculate_benchmark_metrics(
                    equity_series, benchmark_equity,
                )
                metrics.update(benchmark_metrics)
                metrics["benchmark_symbol"] = self._benchmark_symbol or ""

        return metrics

    def get_benchmark_equity_curve(self) -> pd.DataFrame:
        """获取基准权益曲线（归一化到与策略相同的初始资金）"""
        if self._benchmark_data is None or not self._equity_curve:
            return pd.DataFrame()

        dates = [item[0] for item in self._equity_curve]
        values = [item[1] for item in self._equity_curve]
        equity_series = pd.Series(values, index=pd.DatetimeIndex(dates))

        benchmark_equity = self._build_benchmark_equity(equity_series)
        if benchmark_equity is None or benchmark_equity.empty:
            return pd.DataFrame()

        return pd.DataFrame({
            "datetime": benchmark_equity.index,
            "equity": benchmark_equity.values,
        })

    def _build_benchmark_equity(self, strategy_equity: pd.Series) -> pd.Series | None:
        """根据基准数据构建与策略对齐的基准权益曲线

        将基准收盘价归一化到与策略相同的初始资金水平。
        """
        if self._benchmark_data is None:
            return None

        benchmark_df = self._benchmark_data.copy()
        benchmark_df["datetime"] = pd.to_datetime(benchmark_df["datetime"])
        benchmark_series = pd.Series(
            benchmark_df["close"].values,
            index=pd.DatetimeIndex(benchmark_df["datetime"]),
        )

        # 对齐到策略的日期范围
        common_index = strategy_equity.index.intersection(benchmark_series.index)
        if len(common_index) < 2:
            # 尝试按日期（忽略时间）对齐
            strategy_dates = strategy_equity.index.normalize()
            benchmark_dates = benchmark_series.index.normalize()

            strategy_equity_daily = strategy_equity.copy()
            strategy_equity_daily.index = strategy_dates
            strategy_equity_daily = strategy_equity_daily[~strategy_equity_daily.index.duplicated(keep="last")]

            benchmark_series_daily = benchmark_series.copy()
            benchmark_series_daily.index = benchmark_dates
            benchmark_series_daily = benchmark_series_daily[~benchmark_series_daily.index.duplicated(keep="last")]

            common_index = strategy_equity_daily.index.intersection(benchmark_series_daily.index)
            if len(common_index) < 2:
                return None

            benchmark_aligned = benchmark_series_daily.loc[common_index]
        else:
            benchmark_aligned = benchmark_series.loc[common_index]

        # 归一化：基准初始值 = 策略初始资金
        initial_benchmark_price = benchmark_aligned.iloc[0]
        if initial_benchmark_price == 0:
            return None

        normalized_benchmark = benchmark_aligned / initial_benchmark_price * self.initial_capital
        return normalized_benchmark

    def get_equity_curve(self) -> pd.DataFrame:
        """获取权益曲线 DataFrame"""
        if not self._equity_curve:
            return pd.DataFrame()
        dates = [item[0] for item in self._equity_curve]
        values = [item[1] for item in self._equity_curve]
        return pd.DataFrame({"datetime": dates, "equity": values})

    def _build_bar_sequence(self) -> list[Bar]:
        """将所有数据源合并为按时间排序的 Bar 序列"""
        all_bars: list[Bar] = []
        for symbol, df, market in self._data_feeds:
            for _, row in df.iterrows():
                bar = Bar(
                    symbol=symbol,
                    datetime=pd.Timestamp(row["datetime"]).to_pydatetime(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    amount=float(row.get("amount", 0)),
                    market=market,
                )
                all_bars.append(bar)
        all_bars.sort(key=lambda b: b.datetime)
        return all_bars

    def _update_position_price(self, bar: Bar) -> None:
        """更新持仓的当前价格"""
        if bar.symbol in self._portfolio.positions:
            self._portfolio.positions[bar.symbol].current_price = bar.close

    def _execute_order(self, order: Order, bar: Bar) -> None:
        """执行订单（模拟撮合）"""
        try:
            if order.side == OrderSide.BUY:
                self._execute_buy(order, bar)
            else:
                self._execute_sell(order, bar)
        except (InsufficientFundsError, InsufficientPositionError) as exc:
            order.status = OrderStatus.REJECTED
            logger.warning("订单被拒绝: %s - %s", order.order_id, exc)

    def _execute_buy(self, order: Order, bar: Bar) -> None:
        """执行买入"""
        # 考虑滑点
        fill_price = order.price * (1 + self.slippage_rate)
        total_cost = fill_price * order.quantity
        commission = total_cost * self.commission_rate

        if total_cost + commission > self._portfolio.cash:
            raise InsufficientFundsError(
                f"资金不足: 需要 {total_cost + commission:.2f}, 可用 {self._portfolio.cash:.2f}"
            )

        # 更新组合
        self._portfolio.cash -= (total_cost + commission)

        if order.symbol in self._portfolio.positions:
            pos = self._portfolio.positions[order.symbol]
            total_quantity = pos.quantity + order.quantity
            pos.avg_cost = (pos.avg_cost * pos.quantity + fill_price * order.quantity) / total_quantity
            pos.quantity = total_quantity
        else:
            self._portfolio.positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=order.quantity,
                avg_cost=fill_price,
                current_price=bar.close,
                market=order.market,
            )

        # 更新订单状态
        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.filled_at = bar.datetime
        order.commission = commission

        # 记录成交
        trade = TradeRecord(
            trade_id=str(uuid.uuid4())[:8],
            order_id=order.order_id,
            symbol=order.symbol,
            side=OrderSide.BUY,
            price=fill_price,
            quantity=order.quantity,
            commission=commission,
            traded_at=bar.datetime,
            market=order.market,
        )
        self._portfolio.trade_history.append(trade)
        self._strategy.on_order_filled(order, self._portfolio)

        # 通知止损管理器
        self._stop_loss_manager.on_order_filled(order.symbol, OrderSide.BUY, fill_price)

        # 通知风控管理器
        if self._risk_manager:
            self._risk_manager.on_trade_result(True)

        logger.debug("买入成交: %s %d股 @ %.2f, 佣金=%.2f", order.symbol, order.quantity, fill_price, commission)

    def _execute_sell(self, order: Order, bar: Bar) -> None:
        """执行卖出"""
        if order.symbol not in self._portfolio.positions:
            raise InsufficientPositionError(f"无持仓: {order.symbol}")

        pos = self._portfolio.positions[order.symbol]
        if pos.quantity < order.quantity:
            raise InsufficientPositionError(
                f"持仓不足: 持有 {pos.quantity}, 卖出 {order.quantity}"
            )

        # 保存成本价（在持仓可能被删除前）
        entry_avg_cost = pos.avg_cost

        # 考虑滑点
        fill_price = order.price * (1 - self.slippage_rate)
        total_revenue = fill_price * order.quantity
        commission = total_revenue * self.commission_rate

        # 更新组合
        self._portfolio.cash += (total_revenue - commission)
        pos.quantity -= order.quantity

        if pos.quantity == 0:
            del self._portfolio.positions[order.symbol]

        # 更新订单状态
        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.filled_at = bar.datetime
        order.commission = commission

        # 记录成交
        trade = TradeRecord(
            trade_id=str(uuid.uuid4())[:8],
            order_id=order.order_id,
            symbol=order.symbol,
            side=OrderSide.SELL,
            price=fill_price,
            quantity=order.quantity,
            commission=commission,
            traded_at=bar.datetime,
            market=order.market,
        )
        self._portfolio.trade_history.append(trade)
        self._strategy.on_order_filled(order, self._portfolio)

        # 通知止损管理器
        self._stop_loss_manager.on_order_filled(order.symbol, OrderSide.SELL, fill_price)

        # 通知风控管理器（判断本次卖出是否盈利）
        if self._risk_manager:
            is_profitable = fill_price > entry_avg_cost
            self._risk_manager.on_trade_result(is_profitable)

        logger.debug("卖出成交: %s %d股 @ %.2f, 佣金=%.2f", order.symbol, order.quantity, fill_price, commission)

    def _create_sell_order(
        self,
        symbol: str,
        price: float,
        quantity: int,
        market: MarketType = MarketType.A_SHARE,
    ) -> Order:
        """创建卖出订单（供止损止盈管理器使用）"""
        return Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=OrderSide.SELL,
            price=price,
            quantity=quantity,
            created_at=datetime.now(),
            market=market,
        )
