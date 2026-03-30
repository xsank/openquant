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
    EventFactor,
    EventSentiment,
    EventType,
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
        self._event_store: dict[str, list[EventFactor]] = defaultdict(list)

    def initialize(self, portfolio: Portfolio) -> None:
        self._bar_history.clear()
        self._event_store.clear()

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

    def load_events(self, symbol: str, events: list[EventFactor]) -> None:
        """加载事件因子数据（由引擎在回测前调用）"""
        self._event_store[symbol] = sorted(events, key=lambda e: e.event_date)

    def get_events_on_date(self, symbol: str, target_date: datetime) -> list[EventFactor]:
        """获取指定日期的事件列表"""
        target = target_date.date() if hasattr(target_date, 'date') else target_date
        return [
            e for e in self._event_store.get(symbol, [])
            if e.event_date.date() == target
        ]

    def get_events_in_window(
        self,
        symbol: str,
        end_date: datetime,
        lookback_days: int = 5,
    ) -> list[EventFactor]:
        """获取最近 N 天内的事件列表"""
        end = pd.Timestamp(end_date)
        start = end - pd.Timedelta(days=lookback_days)
        return [
            e for e in self._event_store.get(symbol, [])
            if start <= pd.Timestamp(e.event_date) <= end
        ]

    def compute_event_score(
        self,
        symbol: str,
        current_date: datetime,
        lookback_days: int = 5,
        decay_factor: float = 0.8,
    ) -> float:
        """计算综合事件得分

        将近期事件按时间衰减加权汇总，利多为正、利空为负。
        得分范围大致在 [-3, +3]，0 表示无事件或中性。

        Args:
            symbol: 标的代码
            current_date: 当前日期
            lookback_days: 回看天数
            decay_factor: 每天的衰减系数 (0~1)

        Returns:
            综合事件得分
        """
        recent_events = self.get_events_in_window(symbol, current_date, lookback_days)
        if not recent_events:
            return 0.0

        score = 0.0
        current_ts = pd.Timestamp(current_date)
        for event in recent_events:
            days_ago = (current_ts - pd.Timestamp(event.event_date)).days
            weight = decay_factor ** max(days_ago, 0)

            if event.sentiment == EventSentiment.BULLISH:
                score += event.strength * weight
            elif event.sentiment == EventSentiment.BEARISH:
                score -= event.strength * weight

        return score

    def has_bearish_event(
        self,
        symbol: str,
        current_date: datetime,
        lookback_days: int = 3,
        threshold: float = 0.5,
    ) -> bool:
        """检查近期是否有显著利空事件"""
        return self.compute_event_score(symbol, current_date, lookback_days) < -threshold

    def has_bullish_event(
        self,
        symbol: str,
        current_date: datetime,
        lookback_days: int = 3,
        threshold: float = 0.5,
    ) -> bool:
        """检查近期是否有显著利多事件"""
        return self.compute_event_score(symbol, current_date, lookback_days) > threshold

    @property
    def stop_loss_config(self) -> StopLossConfig | None:
        """获取止损止盈配置"""
        return self._stop_loss_config

    @stop_loss_config.setter
    def stop_loss_config(self, value: StopLossConfig | None) -> None:
        self._stop_loss_config = value
