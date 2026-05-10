"""EMA趋势跟随策略

使用长周期EMA交叉捕捉中长期趋势：
- 买入：EMA短期上穿EMA长期，且价格站上EMA长期（确认趋势成立）
- 卖出：EMA短期下穿EMA长期

相比 MA_Cross(5,20)，本策略使用 EMA(10,50) 更长周期滤波，
减少震荡市的频繁交易，适合捕捉如 AMD 4月~5月连续上涨等大趋势行情。
"""
from __future__ import annotations

from openquant.core.models import Bar, Order, Portfolio
from openquant.risk.stop_loss import StopLossConfig
from openquant.strategy.base import BaseStrategy
from openquant.utils.indicators import exponential_moving_average


class TrendFollowStrategy(BaseStrategy):
    """EMA趋势跟随策略"""

    def __init__(
        self,
        short_ema_period: int = 10,
        long_ema_period: int = 50,
        position_ratio: float = 0.9,
        stop_loss_config: StopLossConfig | None = None,
    ):
        """
        Args:
            short_ema_period: 短期EMA周期
            long_ema_period: 长期EMA周期
            position_ratio: 仓位比例 (0~1)
            stop_loss_config: 止损止盈配置
        """
        super().__init__(stop_loss_config=stop_loss_config)
        self.short_ema_period = short_ema_period
        self.long_ema_period = long_ema_period
        self.position_ratio = position_ratio

    def get_name(self) -> str:
        return f"TrendFollow({self.short_ema_period},{self.long_ema_period})"

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        self._record_bar(bar)
        orders: list[Order] = []

        close_series = self.get_close_series(bar.symbol)
        if len(close_series) < self.long_ema_period + 2:
            return orders

        short_ema = exponential_moving_average(close_series, self.short_ema_period)
        long_ema = exponential_moving_average(close_series, self.long_ema_period)

        current_short = short_ema.iloc[-1]
        current_long = long_ema.iloc[-1]
        prev_short = short_ema.iloc[-2]
        prev_long = long_ema.iloc[-2]

        has_position = (
            bar.symbol in portfolio.positions
            and portfolio.positions[bar.symbol].quantity > 0
        )

        # EMA短期上穿长期 且 价格站上EMA长期 → 趋势确认买入
        crossover = prev_short <= prev_long and current_short > current_long
        price_above_trend = bar.close > current_long

        if crossover and price_above_trend and not has_position:
            available_cash = portfolio.cash * self.position_ratio
            quantity = self.calculate_max_buyable(bar.close, available_cash)
            if quantity > 0:
                orders.append(
                    self.create_buy_order(bar.symbol, bar.close, quantity, bar.market)
                )

        # EMA短期下穿长期 → 趋势结束卖出
        elif prev_short >= prev_long and current_short < current_long and has_position:
            position = portfolio.positions[bar.symbol]
            orders.append(
                self.create_sell_order(bar.symbol, bar.close, position.quantity, bar.market)
            )

        return orders
