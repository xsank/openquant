"""双重动量策略

结合绝对动量和相对动量的经典策略：
- 绝对动量：当前价格高于 N 日前价格（ROC > 0），确认上升趋势
- 相对动量：短期均线在长期均线之上，确认趋势强度
- 两个条件同时满足时买入，任一条件不满足时卖出

适合周级趋势行情，能有效过滤假突破，减少震荡市的频繁交易。
"""
from __future__ import annotations

from openquant.core.models import Bar, Order, Portfolio
from openquant.risk.stop_loss import StopLossConfig
from openquant.strategy.base import BaseStrategy
from openquant.utils.indicators import moving_average, rate_of_change


class DualMomentumStrategy(BaseStrategy):
    """双重动量策略"""

    def __init__(
        self,
        roc_period: int = 20,
        short_ma_period: int = 10,
        long_ma_period: int = 30,
        position_ratio: float = 0.9,
        stop_loss_config: StopLossConfig | None = None,
    ):
        """
        Args:
            roc_period: ROC 动量回看周期
            short_ma_period: 短期均线周期
            long_ma_period: 长期均线周期
            position_ratio: 仓位比例 (0~1)
            stop_loss_config: 止损止盈配置
        """
        super().__init__(stop_loss_config=stop_loss_config)
        self.roc_period = roc_period
        self.short_ma_period = short_ma_period
        self.long_ma_period = long_ma_period
        self.position_ratio = position_ratio

    def get_name(self) -> str:
        return f"DualMomentum({self.roc_period},{self.short_ma_period},{self.long_ma_period})"

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        self._record_bar(bar)
        orders: list[Order] = []

        close_series = self.get_close_series(bar.symbol)
        min_bars = max(self.roc_period, self.long_ma_period) + 2
        if len(close_series) < min_bars:
            return orders

        roc_series = rate_of_change(close_series, self.roc_period)
        short_ma = moving_average(close_series, self.short_ma_period)
        long_ma = moving_average(close_series, self.long_ma_period)

        current_roc = roc_series.iloc[-1]
        current_short_ma = short_ma.iloc[-1]
        current_long_ma = long_ma.iloc[-1]

        prev_roc = roc_series.iloc[-2]
        prev_short_ma = short_ma.iloc[-2]
        prev_long_ma = long_ma.iloc[-2]

        has_position = (
            bar.symbol in portfolio.positions
            and portfolio.positions[bar.symbol].quantity > 0
        )

        # 绝对动量 + 相对动量同时确认 → 买入
        absolute_momentum_on = current_roc > 0
        relative_momentum_on = current_short_ma > current_long_ma

        prev_absolute = prev_roc > 0
        prev_relative = prev_short_ma > prev_long_ma

        if not has_position:
            # 双重动量同时开启（至少一个是新开启的）
            if absolute_momentum_on and relative_momentum_on:
                was_both_on = prev_absolute and prev_relative
                if not was_both_on:
                    available_cash = portfolio.cash * self.position_ratio
                    quantity = self.calculate_max_buyable(bar.close, available_cash)
                    if quantity > 0:
                        orders.append(
                            self.create_buy_order(
                                bar.symbol, bar.close, quantity, bar.market
                            )
                        )
        else:
            # 任一动量信号消失 → 卖出
            if not absolute_momentum_on or not relative_momentum_on:
                position = portfolio.positions[bar.symbol]
                orders.append(
                    self.create_sell_order(
                        bar.symbol, bar.close, position.quantity, bar.market
                    )
                )

        return orders
