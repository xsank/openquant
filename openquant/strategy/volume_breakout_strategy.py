"""均线 + 成交量突破策略

结合价格突破和成交量确认的趋势策略：
- 买入：价格站上长期均线 + 成交量放大（超过均量 N 倍）
- 卖出：价格跌破长期均线

成交量放大确认突破的有效性，减少假突破带来的损失。
适合天级/周级趋势行情。
"""
from __future__ import annotations

from openquant.core.models import Bar, Order, Portfolio
from openquant.strategy.base import BaseStrategy
from openquant.utils.indicators import moving_average


class VolumeBreakoutStrategy(BaseStrategy):
    """均线 + 成交量突破策略"""

    def __init__(
        self,
        ma_period: int = 20,
        volume_ma_period: int = 20,
        volume_multiplier: float = 1.5,
        position_ratio: float = 0.9,
    ):
        """
        Args:
            ma_period: 价格均线周期
            volume_ma_period: 成交量均线周期
            volume_multiplier: 成交量放大倍数阈值
            position_ratio: 仓位比例 (0~1)
        """
        super().__init__()
        self.ma_period = ma_period
        self.volume_ma_period = volume_ma_period
        self.volume_multiplier = volume_multiplier
        self.position_ratio = position_ratio

    def get_name(self) -> str:
        return f"VolumeBreakout({self.ma_period},vol×{self.volume_multiplier})"

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        self._record_bar(bar)
        orders: list[Order] = []

        close_series = self.get_close_series(bar.symbol)
        volume_series = self.get_volume_series(bar.symbol)
        min_bars = max(self.ma_period, self.volume_ma_period) + 1
        if len(close_series) < min_bars:
            return orders

        price_ma = moving_average(close_series, self.ma_period)
        volume_ma = moving_average(volume_series, self.volume_ma_period)

        current_close = close_series.iloc[-1]
        prev_close = close_series.iloc[-2]
        current_price_ma = price_ma.iloc[-1]
        prev_price_ma = price_ma.iloc[-2]
        current_volume = volume_series.iloc[-1]
        current_volume_ma = volume_ma.iloc[-1]

        has_position = (
            bar.symbol in portfolio.positions
            and portfolio.positions[bar.symbol].quantity > 0
        )

        # 价格从均线下方突破到上方 + 成交量放大 → 买入
        volume_confirmed = (
            current_volume_ma > 0
            and current_volume > current_volume_ma * self.volume_multiplier
        )
        price_breakout = prev_close <= prev_price_ma and current_close > current_price_ma

        if price_breakout and volume_confirmed and not has_position:
            available_cash = portfolio.cash * self.position_ratio
            quantity = self.calculate_max_buyable(bar.close, available_cash)
            if quantity > 0:
                orders.append(
                    self.create_buy_order(bar.symbol, bar.close, quantity, bar.market)
                )

        # 价格跌破均线 → 卖出
        elif current_close < current_price_ma and has_position:
            position = portfolio.positions[bar.symbol]
            orders.append(
                self.create_sell_order(
                    bar.symbol, bar.close, position.quantity, bar.market
                )
            )

        return orders
