"""事件增强均线交叉策略

在经典 MA Cross 策略基础上，引入事件因子对交易信号进行修正：
1. 利空事件抑制买入 —— 即使出现金叉，若近期有显著利空事件则推迟买入
2. 利多事件加速买入 —— 金叉 + 利多事件时提高仓位比例
3. 利空事件加速卖出 —— 即使未出现死叉，若持仓期间出现重大利空则提前止损
4. 利多事件延迟卖出 —— 死叉信号出现但近期有强利多事件时，可选择持有观望

通过事件因子的引入，策略可以：
- 避免在财报暴雷、大股东减持等利空期间盲目追涨
- 在业绩超预期、回购增持等利多期间适当加仓
- 整体上修正纯技术面策略的部分错误决策
"""
from __future__ import annotations

import logging

from openquant.core.models import Bar, Order, Portfolio
from openquant.risk.stop_loss import StopLossConfig
from openquant.strategy.base import BaseStrategy
from openquant.utils.indicators import moving_average

logger = logging.getLogger(__name__)


class EventEnhancedMACrossStrategy(BaseStrategy):
    """事件增强均线交叉策略"""

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        position_ratio: float = 0.9,
        event_lookback_days: int = 5,
        bearish_block_threshold: float = -0.5,
        bullish_boost_threshold: float = 0.5,
        bullish_position_boost: float = 0.15,
        event_force_sell_threshold: float = -1.2,
        bullish_hold_threshold: float = 1.0,
        stop_loss_config: StopLossConfig | None = None,
    ):
        """
        Args:
            short_window: 短期均线周期
            long_window: 长期均线周期
            position_ratio: 基础仓位比例 (0~1)
            event_lookback_days: 事件回看天数
            bearish_block_threshold: 利空阻止买入的得分阈值（负数）
            bullish_boost_threshold: 利多加仓的得分阈值（正数）
            bullish_position_boost: 利多事件时额外增加的仓位比例
            event_force_sell_threshold: 强制卖出的利空得分阈值（负数，更极端）
            bullish_hold_threshold: 利多延迟卖出的得分阈值
            stop_loss_config: 止损止盈配置
        """
        super().__init__(stop_loss_config=stop_loss_config)
        self.short_window = short_window
        self.long_window = long_window
        self.position_ratio = position_ratio
        self.event_lookback_days = event_lookback_days
        self.bearish_block_threshold = bearish_block_threshold
        self.bullish_boost_threshold = bullish_boost_threshold
        self.bullish_position_boost = bullish_position_boost
        self.event_force_sell_threshold = event_force_sell_threshold
        self.bullish_hold_threshold = bullish_hold_threshold

    def get_name(self) -> str:
        return f"EventMA_Cross({self.short_window},{self.long_window})"

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        self._record_bar(bar)
        orders: list[Order] = []

        close_series = self.get_close_series(bar.symbol)
        if len(close_series) < self.long_window + 1:
            return orders

        short_ma = moving_average(close_series, self.short_window)
        long_ma = moving_average(close_series, self.long_window)

        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]

        has_position = (
            bar.symbol in portfolio.positions
            and portfolio.positions[bar.symbol].quantity > 0
        )

        event_score = self.compute_event_score(
            bar.symbol, bar.datetime, self.event_lookback_days,
        )

        is_golden_cross = prev_short <= prev_long and current_short > current_long
        is_death_cross = prev_short >= prev_long and current_short < current_long

        # === 买入逻辑 ===
        if is_golden_cross and not has_position:
            if event_score < self.bearish_block_threshold:
                logger.info(
                    "[%s] 金叉信号被利空事件阻止: event_score=%.2f, 阈值=%.2f",
                    bar.datetime.strftime("%Y-%m-%d"), event_score, self.bearish_block_threshold,
                )
            else:
                effective_ratio = self.position_ratio
                if event_score > self.bullish_boost_threshold:
                    effective_ratio = min(
                        self.position_ratio + self.bullish_position_boost, 0.98,
                    )
                    logger.info(
                        "[%s] 金叉 + 利多事件加仓: event_score=%.2f, 仓位比例 %.0f%% -> %.0f%%",
                        bar.datetime.strftime("%Y-%m-%d"), event_score,
                        self.position_ratio * 100, effective_ratio * 100,
                    )

                available_cash = portfolio.cash * effective_ratio
                quantity = self.calculate_max_buyable(bar.close, available_cash)
                if quantity > 0:
                    orders.append(
                        self.create_buy_order(bar.symbol, bar.close, quantity, bar.market),
                    )

        # === 卖出逻辑 ===
        elif has_position:
            position = portfolio.positions[bar.symbol]

            # 情况 1：重大利空事件强制卖出（不等死叉）
            if event_score < self.event_force_sell_threshold:
                logger.info(
                    "[%s] 重大利空事件触发强制卖出: event_score=%.2f",
                    bar.datetime.strftime("%Y-%m-%d"), event_score,
                )
                orders.append(
                    self.create_sell_order(
                        bar.symbol, bar.close, position.quantity, bar.market,
                    ),
                )

            # 情况 2：死叉信号
            elif is_death_cross:
                if event_score > self.bullish_hold_threshold:
                    logger.info(
                        "[%s] 死叉信号被利多事件延迟: event_score=%.2f, 继续持有",
                        bar.datetime.strftime("%Y-%m-%d"), event_score,
                    )
                else:
                    orders.append(
                        self.create_sell_order(
                            bar.symbol, bar.close, position.quantity, bar.market,
                        ),
                    )

        return orders
