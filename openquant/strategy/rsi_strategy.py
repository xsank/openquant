"""RSI 均值回归策略

当 RSI 跌入超卖区域后回升时买入，涨入超买区域后回落时卖出。
适合天级/周级震荡行情，利用价格的均值回归特性。
"""
from __future__ import annotations

from openquant.core.models import Bar, Order, Portfolio
from openquant.strategy.base import BaseStrategy
from openquant.utils.indicators import rsi


class RSIReversalStrategy(BaseStrategy):
    """RSI 均值回归策略"""

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        position_ratio: float = 0.9,
    ):
        """
        Args:
            rsi_period: RSI 计算周期
            oversold: 超卖阈值，RSI 低于此值视为超卖
            overbought: 超买阈值，RSI 高于此值视为超买
            position_ratio: 仓位比例 (0~1)
        """
        super().__init__()
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.position_ratio = position_ratio

    def get_name(self) -> str:
        return f"RSI_Reversal({self.rsi_period},{self.oversold},{self.overbought})"

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        self._record_bar(bar)
        orders: list[Order] = []

        close_series = self.get_close_series(bar.symbol)
        if len(close_series) < self.rsi_period + 2:
            return orders

        rsi_series = rsi(close_series, self.rsi_period)
        current_rsi = rsi_series.iloc[-1]
        prev_rsi = rsi_series.iloc[-2]

        has_position = (
            bar.symbol in portfolio.positions
            and portfolio.positions[bar.symbol].quantity > 0
        )

        # RSI 从超卖区回升（前一根在超卖区，当前根突破超卖线）→ 买入
        if prev_rsi <= self.oversold < current_rsi and not has_position:
            available_cash = portfolio.cash * self.position_ratio
            quantity = self.calculate_max_buyable(bar.close, available_cash)
            if quantity > 0:
                orders.append(
                    self.create_buy_order(bar.symbol, bar.close, quantity, bar.market)
                )

        # RSI 从超买区回落（前一根在超买区，当前根跌破超买线）→ 卖出
        elif prev_rsi >= self.overbought > current_rsi and has_position:
            position = portfolio.positions[bar.symbol]
            orders.append(
                self.create_sell_order(
                    bar.symbol, bar.close, position.quantity, bar.market
                )
            )

        return orders
