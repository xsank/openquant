"""布林带策略

价格跌破下轨后回到带内时买入（超跌反弹），
价格突破上轨后回到带内时卖出（超涨回落）。
适合天级/周级震荡市，利用价格偏离均值后的回归。
"""
from __future__ import annotations

from openquant.core.models import Bar, Order, Portfolio
from openquant.strategy.base import BaseStrategy
from openquant.utils.indicators import bollinger_bands


class BollingerBandStrategy(BaseStrategy):
    """布林带均值回归策略"""

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        position_ratio: float = 0.9,
    ):
        """
        Args:
            window: 布林带均线周期
            num_std: 标准差倍数
            position_ratio: 仓位比例 (0~1)
        """
        super().__init__()
        self.window = window
        self.num_std = num_std
        self.position_ratio = position_ratio

    def get_name(self) -> str:
        return f"Bollinger({self.window},{self.num_std})"

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        self._record_bar(bar)
        orders: list[Order] = []

        close_series = self.get_close_series(bar.symbol)
        if len(close_series) < self.window + 1:
            return orders

        upper, middle, lower = bollinger_bands(
            close_series, self.window, self.num_std
        )

        current_close = close_series.iloc[-1]
        prev_close = close_series.iloc[-2]
        current_lower = lower.iloc[-1]
        prev_lower = lower.iloc[-2]
        current_upper = upper.iloc[-1]
        prev_upper = upper.iloc[-2]

        has_position = (
            bar.symbol in portfolio.positions
            and portfolio.positions[bar.symbol].quantity > 0
        )

        # 价格从下轨下方回到带内 → 买入信号
        if prev_close <= prev_lower and current_close > current_lower and not has_position:
            available_cash = portfolio.cash * self.position_ratio
            quantity = self.calculate_max_buyable(bar.close, available_cash)
            if quantity > 0:
                orders.append(
                    self.create_buy_order(bar.symbol, bar.close, quantity, bar.market)
                )

        # 价格从上轨上方回到带内 → 卖出信号
        elif prev_close >= prev_upper and current_close < current_upper and has_position:
            position = portfolio.positions[bar.symbol]
            orders.append(
                self.create_sell_order(
                    bar.symbol, bar.close, position.quantity, bar.market
                )
            )

        return orders
