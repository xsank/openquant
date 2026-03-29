"""模拟交易引擎

基于实时/准实时数据驱动策略执行，模拟实盘交易过程。
支持定时轮询行情数据，按策略信号生成模拟订单。
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime

import pandas as pd

from openquant.core.exceptions import DataSourceError, InsufficientFundsError, InsufficientPositionError
from openquant.core.interfaces import DataSourceInterface, EngineInterface, StrategyInterface
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
from openquant.storage.sqlite_storage import SqliteStorage
from openquant.utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)


class SimulationEngine(EngineInterface):
    """模拟交易引擎

    使用数据源的实时/历史数据驱动策略，模拟实盘交易。
    支持按固定间隔轮询行情并触发策略。
    """

    def __init__(
        self,
        data_source: DataSourceInterface,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.0003,
        slippage_rate: float = 0.001,
        poll_interval: int = 60,
        max_rounds: int = 0,
        storage: SqliteStorage | None = None,
    ):
        """
        Args:
            data_source: 数据源实例
            initial_capital: 初始资金
            commission_rate: 佣金费率
            slippage_rate: 滑点费率
            poll_interval: 行情轮询间隔（秒）
            max_rounds: 最大轮询次数，0 表示无限循环直到手动停止
            storage: 存储实例（可选）
        """
        self.data_source = data_source
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.poll_interval = poll_interval
        self.max_rounds = max_rounds
        self.storage = storage

        self._strategy: StrategyInterface | None = None
        self._portfolio: Portfolio | None = None
        self._equity_curve: list[tuple[datetime, float]] = []
        self._watch_list: list[tuple[str, MarketType]] = []
        self._running = False

    def set_strategy(self, strategy: StrategyInterface) -> None:
        self._strategy = strategy

    def add_symbol(self, symbol: str, market: MarketType = MarketType.A_SHARE) -> None:
        """添加监控标的"""
        self._watch_list.append((symbol, market))

    def run(self) -> Portfolio:
        """运行模拟交易（阻塞式，按 poll_interval 轮询）"""
        if self._strategy is None:
            raise ValueError("未设置策略，请先调用 set_strategy()")
        if not self._watch_list:
            raise ValueError("未添加监控标的，请先调用 add_symbol()")

        self._portfolio = Portfolio(
            initial_capital=self.initial_capital,
            cash=self.initial_capital,
        )
        self._equity_curve.clear()
        self._strategy.initialize(self._portfolio)
        self._running = True

        logger.info(
            "模拟交易启动: 策略=%s, 初始资金=%.2f, 监控标的=%s",
            self._strategy.get_name(),
            self.initial_capital,
            [s[0] for s in self._watch_list],
        )

        try:
            round_count = 0
            while self._running:
                self._poll_and_execute()
                round_count += 1
                if self.max_rounds > 0 and round_count >= self.max_rounds:
                    logger.info("已达到最大轮询次数 %d，自动停止", self.max_rounds)
                    break
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logger.info("收到中断信号，停止模拟交易")
        finally:
            self._running = False
            self._strategy.on_finish(self._portfolio)
            if self.storage and self._portfolio.trade_history:
                self.storage.save_trade_records(self._portfolio.trade_history)

        return self._portfolio

    def run_once(self) -> Portfolio:
        """执行一次轮询（非阻塞，适合外部调度）"""
        if self._strategy is None:
            raise ValueError("未设置策略")
        if self._portfolio is None:
            self._portfolio = Portfolio(
                initial_capital=self.initial_capital,
                cash=self.initial_capital,
            )
            self._strategy.initialize(self._portfolio)

        self._poll_and_execute()
        return self._portfolio

    def stop(self) -> None:
        """停止模拟交易"""
        self._running = False

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
        return metrics

    def _poll_and_execute(self) -> None:
        """轮询行情并执行策略"""
        now = datetime.now()
        for symbol, market in self._watch_list:
            try:
                quote = self.data_source.fetch_realtime_quote(symbol, market)
                bar = Bar(
                    symbol=symbol,
                    datetime=now,
                    open=quote.get("open", quote["price"]),
                    high=quote.get("high", quote["price"]),
                    low=quote.get("low", quote["price"]),
                    close=quote["price"],
                    volume=quote.get("volume", 0),
                    amount=quote.get("amount", 0),
                    market=market,
                )

                # 更新持仓价格
                if symbol in self._portfolio.positions:
                    self._portfolio.positions[symbol].current_price = bar.close

                # 策略生成订单
                orders = self._strategy.on_bar(bar, self._portfolio)

                # 执行订单
                for order in orders:
                    self._execute_order(order, bar)

            except DataSourceError as exc:
                logger.warning("获取 %s 行情失败: %s", symbol, exc)

        # 记录权益
        self._equity_curve.append((now, self._portfolio.total_equity))

        if self.storage:
            self.storage.save_portfolio_snapshot(self._portfolio, now)

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
        fill_price = order.price * (1 + self.slippage_rate)
        total_cost = fill_price * order.quantity
        commission = total_cost * self.commission_rate

        if total_cost + commission > self._portfolio.cash:
            raise InsufficientFundsError(
                f"资金不足: 需要 {total_cost + commission:.2f}, 可用 {self._portfolio.cash:.2f}"
            )

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

        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.filled_at = bar.datetime
        order.commission = commission

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
        logger.info("模拟买入: %s %d股 @ %.2f", order.symbol, order.quantity, fill_price)

    def _execute_sell(self, order: Order, bar: Bar) -> None:
        """执行卖出"""
        if order.symbol not in self._portfolio.positions:
            raise InsufficientPositionError(f"无持仓: {order.symbol}")

        pos = self._portfolio.positions[order.symbol]
        if pos.quantity < order.quantity:
            raise InsufficientPositionError(
                f"持仓不足: 持有 {pos.quantity}, 卖出 {order.quantity}"
            )

        fill_price = order.price * (1 - self.slippage_rate)
        total_revenue = fill_price * order.quantity
        commission = total_revenue * self.commission_rate

        self._portfolio.cash += (total_revenue - commission)
        pos.quantity -= order.quantity

        if pos.quantity == 0:
            del self._portfolio.positions[order.symbol]

        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.filled_at = bar.datetime
        order.commission = commission

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
        logger.info("模拟卖出: %s %d股 @ %.2f", order.symbol, order.quantity, fill_price)
