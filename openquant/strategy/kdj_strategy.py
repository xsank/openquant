"""KDJ 策略

基于 KDJ 随机指标的金叉死叉信号进行交易：
- K 线上穿 D 线且处于超卖区域时买入
- K 线下穿 D 线且处于超买区域时卖出

适合天级/周级震荡行情，对短期价格拐点敏感。
"""
from __future__ import annotations

from openquant.core.models import Bar, Order, Portfolio
from openquant.risk.stop_loss import StopLossConfig
from openquant.strategy.base import BaseStrategy
from openquant.utils.indicators import kdj


class KDJStrategy(BaseStrategy):
    """KDJ 金叉死叉策略"""

    def __init__(
        self,
        fastk_period: int = 9,
        slowk_period: int = 3,
        slowd_period: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0,
        position_ratio: float = 0.9,
        stop_loss_config: StopLossConfig | None = None,
    ):
        """
        Args:
            fastk_period: RSV 周期
            slowk_period: K 值平滑周期
            slowd_period: D 值平滑周期
            oversold: 超卖阈值
            overbought: 超买阈值
            position_ratio: 仓位比例 (0~1)
            stop_loss_config: 止损止盈配置
        """
        super().__init__(stop_loss_config=stop_loss_config)
        self.fastk_period = fastk_period
        self.slowk_period = slowk_period
        self.slowd_period = slowd_period
        self.oversold = oversold
        self.overbought = overbought
        self.position_ratio = position_ratio

    def get_name(self) -> str:
        return f"KDJ({self.fastk_period},{self.slowk_period},{self.slowd_period})"

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        self._record_bar(bar)
        orders: list[Order] = []

        close_series = self.get_close_series(bar.symbol)
        high_series = self.get_high_series(bar.symbol)
        low_series = self.get_low_series(bar.symbol)

        min_bars = self.fastk_period + max(self.slowk_period, self.slowd_period) + 1
        if len(close_series) < min_bars:
            return orders

        k_values, d_values, j_values = kdj(
            high_series, low_series, close_series,
            self.fastk_period, self.slowk_period, self.slowd_period,
        )

        current_k = k_values.iloc[-1]
        current_d = d_values.iloc[-1]
        prev_k = k_values.iloc[-2]
        prev_d = d_values.iloc[-2]

        has_position = (
            bar.symbol in portfolio.positions
            and portfolio.positions[bar.symbol].quantity > 0
        )

        # K 上穿 D 且在超卖区域 → 买入
        if (
            prev_k <= prev_d
            and current_k > current_d
            and current_k < self.oversold
            and not has_position
        ):
            available_cash = portfolio.cash * self.position_ratio
            quantity = self.calculate_max_buyable(bar.close, available_cash)
            if quantity > 0:
                orders.append(
                    self.create_buy_order(bar.symbol, bar.close, quantity, bar.market)
                )

        # K 下穿 D 且在超买区域 → 卖出
        elif (
            prev_k >= prev_d
            and current_k < current_d
            and current_k > self.overbought
            and has_position
        ):
            position = portfolio.positions[bar.symbol]
            orders.append(
                self.create_sell_order(
                    bar.symbol, bar.close, position.quantity, bar.market
                )
            )

        return orders
