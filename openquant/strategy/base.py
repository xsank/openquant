"""策略基类

提供策略的通用功能，如历史数据管理、下单辅助方法等。
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime

import pandas as pd

from openquant.core.interfaces import StrategyInterface
from openquant.core.models import (
    Bar,
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    Portfolio,
)
from openquant.risk.stop_loss import StopLossConfig, StopLossManager


class BaseStrategy(StrategyInterface):
    """策略基类，提供常用辅助方法"""

    def __init__(self, stop_loss_config: StopLossConfig | None = None):
        self._bar_history: dict[str, list[Bar]] = defaultdict(list)
        self._current_bar: Bar | None = None
        self._stop_loss_config = stop_loss_config

    def initialize(self, portfolio: Portfolio) -> None:
        self._bar_history.clear()

    def on_order_filled(self, order: Order, portfolio: Portfolio) -> None:
        pass

    def _record_bar(self, bar: Bar) -> None:
        """记录K线到历史"""
        self._bar_history[bar.symbol].append(bar)
        self._current_bar = bar

    def get_close_series(self, symbol: str) -> pd.Series:
        """获取指定标的的收盘价序列"""
        bars = self._bar_history.get(symbol, [])
        return pd.Series([b.close for b in bars])

    def get_high_series(self, symbol: str) -> pd.Series:
        """获取指定标的的最高价序列"""
        bars = self._bar_history.get(symbol, [])
        return pd.Series([b.high for b in bars])

    def get_low_series(self, symbol: str) -> pd.Series:
        """获取指定标的的最低价序列"""
        bars = self._bar_history.get(symbol, [])
        return pd.Series([b.low for b in bars])

    def get_open_series(self, symbol: str) -> pd.Series:
        """获取指定标的的开盘价序列"""
        bars = self._bar_history.get(symbol, [])
        return pd.Series([b.open for b in bars])

    def get_volume_series(self, symbol: str) -> pd.Series:
        """获取指定标的的成交量序列"""
        bars = self._bar_history.get(symbol, [])
        return pd.Series([b.volume for b in bars])

    def get_bar_count(self, symbol: str) -> int:
        """获取已记录的K线数量"""
        return len(self._bar_history.get(symbol, []))

    def create_buy_order(
        self,
        symbol: str,
        price: float,
        quantity: int,
        market: MarketType = MarketType.A_SHARE,
    ) -> Order:
        """创建买入订单"""
        return Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=OrderSide.BUY,
            price=price,
            quantity=quantity,
            created_at=self._current_bar.datetime if self._current_bar else datetime.now(),
            market=market,
        )

    def create_sell_order(
        self,
        symbol: str,
        price: float,
        quantity: int,
        market: MarketType = MarketType.A_SHARE,
    ) -> Order:
        """创建卖出订单"""
        return Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=OrderSide.SELL,
            price=price,
            quantity=quantity,
            created_at=self._current_bar.datetime if self._current_bar else datetime.now(),
            market=market,
        )

    def calculate_max_buyable(self, price: float, cash: float, lot_size: int = 100) -> int:
        """计算最大可买数量（按手数取整）"""
        if price <= 0:
            return 0
        max_shares = int(cash / price)
        return (max_shares // lot_size) * lot_size

    @property
    def stop_loss_config(self) -> StopLossConfig | None:
        """获取止损止盈配置"""
        return self._stop_loss_config

    @stop_loss_config.setter
    def stop_loss_config(self, value: StopLossConfig | None) -> None:
        self._stop_loss_config = value
