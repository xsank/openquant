"""均线交叉策略

当短期均线上穿长期均线时买入，下穿时卖出。
"""
from __future__ import annotations

import logging

from openquant.core.models import Bar, Order, Portfolio
from openquant.risk.stop_loss import StopLossConfig
from openquant.strategy.base import BaseStrategy
from openquant.utils.indicators import moving_average

logger = logging.getLogger(__name__)


class MACrossStrategy(BaseStrategy):
    """双均线交叉策略"""

    def __init__(self, short_window: int = 5, long_window: int = 20, position_ratio: float = 0.9, stop_loss_config: StopLossConfig | None = None):
        """
        Args:
            short_window: 短期均线周期
            long_window: 长期均线周期
            position_ratio: 仓位比例 (0~1)
            stop_loss_config: 止损止盈配置
        """
        super().__init__(stop_loss_config=stop_loss_config)
        self.short_window = short_window
        self.long_window = long_window
        self.position_ratio = position_ratio

    def get_name(self) -> str:
        return f"MA_Cross({self.short_window},{self.long_window})"

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        self._record_bar(bar)
        orders: list[Order] = []

        close_series = self.get_close_series(bar.symbol)
        if len(close_series) < self.long_window + 1:
            logger.debug(
                "数据不足，需要至少 %d 根K线，当前仅 %d 根，跳过信号计算",
                self.long_window + 1,
                len(close_series),
            )
            return orders

        short_ma = moving_average(close_series, self.short_window)
        long_ma = moving_average(close_series, self.long_window)

        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]

        has_position = bar.symbol in portfolio.positions and portfolio.positions[bar.symbol].quantity > 0

        # 金叉买入
        if prev_short <= prev_long and current_short > current_long and not has_position:
            available_cash = portfolio.cash * self.position_ratio
            quantity = self.calculate_max_buyable(bar.close, available_cash)
            if quantity > 0:
                orders.append(self.create_buy_order(bar.symbol, bar.close, quantity, bar.market))
            else:
                logger.warning(
                    "金叉信号触发但资金不足: %s 价格=%.2f, 可用资金=%.2f, 买入1手需=%.2f",
                    bar.symbol, bar.close, available_cash, bar.close * 100,
                )

        # 死叉卖出
        elif prev_short >= prev_long and current_short < current_long and has_position:
            position = portfolio.positions[bar.symbol]
            orders.append(self.create_sell_order(bar.symbol, bar.close, position.quantity, bar.market))

        return orders
