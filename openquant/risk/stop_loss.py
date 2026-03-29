"""止损止盈管理模块

支持多种止损止盈策略：固定比例、移动止损（Trailing Stop）、ATR 止损。
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime

from openquant.core.models import Bar, Order, OrderSide, Portfolio, Position

logger = logging.getLogger(__name__)


class StopLossType(enum.Enum):
    """止损类型"""
    FIXED_PERCENT = "fixed_percent"
    TRAILING_PERCENT = "trailing_percent"
    ATR_BASED = "atr_based"


@dataclass
class StopLossConfig:
    """止损止盈配置

    Attributes:
        stop_loss_type: 止损类型
        stop_loss_pct: 固定止损比例（如 0.05 表示亏损 5% 止损）
        take_profit_pct: 止盈比例（如 0.10 表示盈利 10% 止盈），None 表示不设止盈
        trailing_pct: 移动止损回撤比例（仅 TRAILING_PERCENT 类型使用）
        atr_multiplier: ATR 倍数（仅 ATR_BASED 类型使用）
        enabled: 是否启用止损止盈
    """
    stop_loss_type: StopLossType = StopLossType.FIXED_PERCENT
    stop_loss_pct: float = 0.05
    take_profit_pct: float | None = 0.10
    trailing_pct: float = 0.05
    atr_multiplier: float = 2.0
    enabled: bool = True


class StopLossManager:
    """止损止盈管理器

    跟踪每个持仓的最高价，根据配置生成止损止盈卖出订单。
    """

    def __init__(self, config: StopLossConfig | None = None):
        self._config = config or StopLossConfig()
        self._highest_prices: dict[str, float] = {}
        self._entry_prices: dict[str, float] = {}
        self._atr_values: dict[str, float] = {}

    @property
    def config(self) -> StopLossConfig:
        return self._config

    @config.setter
    def config(self, value: StopLossConfig) -> None:
        self._config = value

    def reset(self) -> None:
        """重置所有跟踪状态"""
        self._highest_prices.clear()
        self._entry_prices.clear()
        self._atr_values.clear()

    def on_order_filled(self, symbol: str, side: OrderSide, fill_price: float) -> None:
        """订单成交后更新跟踪状态"""
        if side == OrderSide.BUY:
            self._entry_prices[symbol] = fill_price
            self._highest_prices[symbol] = fill_price
        elif side == OrderSide.SELL:
            self._entry_prices.pop(symbol, None)
            self._highest_prices.pop(symbol, None)
            self._atr_values.pop(symbol, None)

    def update_atr(self, symbol: str, atr_value: float) -> None:
        """更新标的的 ATR 值（供 ATR 止损使用）"""
        self._atr_values[symbol] = atr_value

    def check_stop(
        self,
        bar: Bar,
        portfolio: Portfolio,
        create_sell_order_fn,
    ) -> list[Order]:
        """检查是否触发止损止盈，返回需要执行的卖出订单列表

        Args:
            bar: 当前 K 线
            portfolio: 当前组合
            create_sell_order_fn: 创建卖出订单的回调函数

        Returns:
            止损止盈产生的卖出订单列表
        """
        if not self._config.enabled:
            return []

        symbol = bar.symbol
        if symbol not in portfolio.positions:
            return []

        position = portfolio.positions[symbol]
        if position.quantity <= 0:
            return []

        current_price = bar.close

        # 更新最高价跟踪
        if symbol in self._highest_prices:
            self._highest_prices[symbol] = max(self._highest_prices[symbol], current_price)
        else:
            self._highest_prices[symbol] = current_price

        entry_price = self._entry_prices.get(symbol, position.avg_cost)
        if entry_price <= 0:
            return []

        # 检查止盈
        if self._config.take_profit_pct is not None:
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= self._config.take_profit_pct:
                logger.info(
                    "触发止盈: %s 盈利 %.2f%% >= %.2f%%",
                    symbol, profit_pct * 100, self._config.take_profit_pct * 100,
                )
                return [create_sell_order_fn(symbol, current_price, position.quantity, bar.market)]

        # 检查止损
        should_stop = self._check_stop_loss(symbol, current_price, entry_price)
        if should_stop:
            return [create_sell_order_fn(symbol, current_price, position.quantity, bar.market)]

        return []

    def _check_stop_loss(self, symbol: str, current_price: float, entry_price: float) -> bool:
        """根据止损类型检查是否触发止损"""
        stop_type = self._config.stop_loss_type

        if stop_type == StopLossType.FIXED_PERCENT:
            loss_pct = (entry_price - current_price) / entry_price
            if loss_pct >= self._config.stop_loss_pct:
                logger.info(
                    "触发固定止损: %s 亏损 %.2f%% >= %.2f%%",
                    symbol, loss_pct * 100, self._config.stop_loss_pct * 100,
                )
                return True

        elif stop_type == StopLossType.TRAILING_PERCENT:
            highest = self._highest_prices.get(symbol, entry_price)
            drawdown_pct = (highest - current_price) / highest
            if drawdown_pct >= self._config.trailing_pct:
                logger.info(
                    "触发移动止损: %s 从最高价 %.2f 回撤 %.2f%% >= %.2f%%",
                    symbol, highest, drawdown_pct * 100, self._config.trailing_pct * 100,
                )
                return True

        elif stop_type == StopLossType.ATR_BASED:
            atr_value = self._atr_values.get(symbol)
            if atr_value is not None and atr_value > 0:
                stop_price = entry_price - self._config.atr_multiplier * atr_value
                if current_price <= stop_price:
                    logger.info(
                        "触发ATR止损: %s 价格 %.2f <= 止损价 %.2f (入场价 %.2f - %.1f × ATR %.2f)",
                        symbol, current_price, stop_price,
                        entry_price, self._config.atr_multiplier, atr_value,
                    )
                    return True

        return False
