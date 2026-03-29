"""MACD 策略

基于 MACD 指标的金叉死叉信号进行交易。
"""
from __future__ import annotations

from openquant.core.models import Bar, Order, Portfolio
from openquant.strategy.base import BaseStrategy
from openquant.utils.indicators import macd


class MACDStrategy(BaseStrategy):
    """MACD 金叉死叉策略"""

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        position_ratio: float = 0.9,
    ):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.position_ratio = position_ratio

    def get_name(self) -> str:
        return f"MACD({self.fast_period},{self.slow_period},{self.signal_period})"

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        self._record_bar(bar)
        orders: list[Order] = []

        close_series = self.get_close_series(bar.symbol)
        min_bars_needed = self.slow_period + self.signal_period + 1
        if len(close_series) < min_bars_needed:
            return orders

        dif, dea, macd_hist = macd(close_series, self.fast_period, self.slow_period, self.signal_period)

        current_hist = macd_hist.iloc[-1]
        prev_hist = macd_hist.iloc[-2]

        has_position = bar.symbol in portfolio.positions and portfolio.positions[bar.symbol].quantity > 0

        # MACD 柱状图由负转正（金叉）买入
        if prev_hist <= 0 < current_hist and not has_position:
            available_cash = portfolio.cash * self.position_ratio
            quantity = self.calculate_max_buyable(bar.close, available_cash)
            if quantity > 0:
                orders.append(self.create_buy_order(bar.symbol, bar.close, quantity, bar.market))

        # MACD 柱状图由正转负（死叉）卖出
        elif prev_hist >= 0 > current_hist and has_position:
            position = portfolio.positions[bar.symbol]
            orders.append(self.create_sell_order(bar.symbol, bar.close, position.quantity, bar.market))

        return orders
