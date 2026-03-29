"""海龟交易策略

经典趋势跟踪策略：
- 价格突破 N 日最高价时买入（突破入场）
- 价格跌破 M 日最低价时卖出（突破离场）
- 使用 ATR 进行仓位管理，控制每笔交易的风险

适合天级/周级趋势行情，是最经典的趋势跟踪系统之一。
"""
from __future__ import annotations

from openquant.core.models import Bar, Order, Portfolio
from openquant.strategy.base import BaseStrategy
from openquant.utils.indicators import atr, donchian_channel


class TurtleStrategy(BaseStrategy):
    """海龟交易策略"""

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 20,
        risk_ratio: float = 0.02,
        position_ratio: float = 0.9,
    ):
        """
        Args:
            entry_period: 入场通道周期（突破 N 日最高价买入）
            exit_period: 离场通道周期（跌破 M 日最低价卖出）
            atr_period: ATR 计算周期
            risk_ratio: 单笔风险占总资金比例
            position_ratio: 最大仓位比例 (0~1)
        """
        super().__init__()
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.risk_ratio = risk_ratio
        self.position_ratio = position_ratio

    def get_name(self) -> str:
        return f"Turtle({self.entry_period},{self.exit_period})"

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        self._record_bar(bar)
        orders: list[Order] = []

        close_series = self.get_close_series(bar.symbol)
        high_series = self.get_high_series(bar.symbol)
        low_series = self.get_low_series(bar.symbol)

        min_bars = max(self.entry_period, self.exit_period, self.atr_period) + 1
        if len(close_series) < min_bars:
            return orders

        # 计算入场通道（不含当前 bar，使用前 N 根）
        entry_upper, _, _ = donchian_channel(
            high_series.iloc[:-1], low_series.iloc[:-1], self.entry_period
        )
        # 计算离场通道
        _, _, exit_lower = donchian_channel(
            high_series.iloc[:-1], low_series.iloc[:-1], self.exit_period
        )

        current_close = close_series.iloc[-1]
        entry_high = entry_upper.iloc[-1]
        exit_low = exit_lower.iloc[-1]

        has_position = (
            bar.symbol in portfolio.positions
            and portfolio.positions[bar.symbol].quantity > 0
        )

        if not has_position:
            # 价格突破入场通道上轨 → 买入
            if current_close > entry_high:
                current_atr = atr(high_series, low_series, close_series, self.atr_period).iloc[-1]
                available_cash = portfolio.cash * self.position_ratio

                # 基于 ATR 的仓位计算：风险金额 / ATR = 股数
                if current_atr > 0:
                    risk_amount = portfolio.total_equity * self.risk_ratio
                    atr_based_quantity = int(risk_amount / current_atr)
                    max_quantity = self.calculate_max_buyable(bar.close, available_cash)
                    quantity = min(atr_based_quantity, max_quantity)
                    # 按手数取整
                    quantity = (quantity // 100) * 100
                else:
                    quantity = self.calculate_max_buyable(bar.close, available_cash)

                if quantity > 0:
                    orders.append(
                        self.create_buy_order(bar.symbol, bar.close, quantity, bar.market)
                    )
        else:
            # 价格跌破离场通道下轨 → 卖出
            if current_close < exit_low:
                position = portfolio.positions[bar.symbol]
                orders.append(
                    self.create_sell_order(
                        bar.symbol, bar.close, position.quantity, bar.market
                    )
                )

        return orders
